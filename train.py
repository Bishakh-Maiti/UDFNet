"""Train and evaluate UDFNet (Algorithm 1 of Wang et al., Information Fusion 2025).

Whole-scene .mat files (e.g. Berlin, HSI + SAR):
    python train.py --hsi data/berlin/data_HS_LR.mat --radar data/berlin/data_SAR_HR.mat \
        --train-gt data/berlin/TrainImage.mat --test-gt data/berlin/TestImage.mat \
        --patch-size 9 --runs 10 --out runs/berlin

Pre-cut patch folder in the MFT layout (e.g. MUUFL, HSI + LiDAR):
    python train.py --patch-dir data/MUUFL --runs 10 --out runs/muufl

Smoke test on a synthetic scene (no data needed):
    python train.py --synthetic --epochs 40 --patch-size 7 --lr 1e-3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from udfnet.data import (PatchDataset, SceneData, load_patch_folder, load_scene, make_splits,
                         synthetic_scene)
from udfnet.losses import LossHistory, udfnet_loss
from udfnet.metrics import classification_report
from udfnet.model import UDFNet

MODALITIES = ("HSI", "radar")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data (pick one of: whole-scene files, --patch-dir, --synthetic)")
    d.add_argument("--hsi", help="whole-scene HSI .mat (H, W, L)")
    d.add_argument("--radar", help="whole-scene SAR/LiDAR .mat (H, W, B) or (H, W)")
    d.add_argument("--train-gt", help="training label map .mat (H, W), 0 = unlabelled")
    d.add_argument("--test-gt", help="test label map .mat (H, W), 0 = unlabelled")
    d.add_argument("--hsi-key"), d.add_argument("--radar-key"), d.add_argument("--train-key"), d.add_argument("--test-key")
    d.add_argument("--patch-dir", help="folder of pre-cut patches in the MFT layout (HSI_Tr.mat, LIDAR_Tr.mat, ...)")
    d.add_argument("--radar-name", help="radar file prefix in --patch-dir (auto-detected, e.g. LIDAR or SAR)")
    d.add_argument("--synthetic", action="store_true", help="use a small toy scene instead of real data")
    d.add_argument("--patch-size", type=int, default=None,
                   help="odd patch size S (default: 11 for scenes, the stored size for --patch-dir; "
                        "smaller values centre-crop pre-cut patches)")
    d.add_argument("--val-fraction", type=float, default=0.2, help="share of training samples used for validation (paper: 20%%)")

    m = p.add_argument_group("model")
    m.add_argument("--channels", type=int, default=128, help="feature width (128 gives the paper's 0.96M params on Berlin)")
    m.add_argument("--blocks", type=int, default=2, help="shared residual blocks (paper: 2)")
    m.add_argument("--heads", type=int, default=4)
    m.add_argument("--tau", type=float, default=1.0, help="energy-score temperature")
    m.add_argument("--exchange-threshold", type=float, default=2e-2)
    m.add_argument("--dropout", type=float, default=0.2)
    m.add_argument("--no-detach-uncertainty", action="store_true",
                   help="let gradients flow through the energy score into the fusion weights")

    t = p.add_argument_group("training (defaults follow Algorithm 1)")
    t.add_argument("--epochs", type=int, default=500)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=1e-5)
    t.add_argument("--seed", type=int, default=430)
    t.add_argument("--runs", type=int, default=1, help="independent repeats, run r uses seed + r (paper: 10)")
    t.add_argument("--gamma", type=float, default=0.1, help="weight of the ranking regularizer L_extra")
    t.add_argument("--sparsity", type=float, default=0.05, help="L1 weight on exchangeable BN scales (paper: 0.05)")
    t.add_argument("--history-start", type=int, default=5, help="epoch L_s from which losses are recorded")
    t.add_argument("--eval-every", type=int, default=5, help="validate every N epochs")
    t.add_argument("--patience", type=int, default=0, help="early-stop after N evals without val gain (0 = off)")

    o = p.add_argument_group("system / output")
    o.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    o.add_argument("--workers", type=int, default=2)
    o.add_argument("--eval-batch-size", type=int, default=512)
    o.add_argument("--out", default="runs/udfnet")
    o.add_argument("--save-map", action="store_true", help="predict every pixel and save a class map (whole-scene only)")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(ds, batch_size, args, pin, shuffle=False, generator=None, drop_last=False):
    return DataLoader(ds, batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=pin,
                      generator=generator, drop_last=drop_last, persistent_workers=args.workers > 0)


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    preds, labels, lambdas, ce_sum = [], [], [], 0.0
    for x_h, x_r, y, _ in dl:
        out = model([x_h.to(device, non_blocking=True), x_r.to(device, non_blocking=True)])
        ce_sum += float(F.cross_entropy(out["fused"], y.to(device), reduction="sum"))
        preds.append(out["fused"].argmax(1).cpu())
        labels.append(y)
        lambdas.append(out["lambdas"].cpu())
    labels = torch.cat(labels)
    return (torch.cat(preds).numpy(), labels.numpy(), torch.cat(lambdas, dim=1).numpy(),
            ce_sum / max(len(labels), 1))


def run_once(args, data, n_classes, in_channels, run: int):
    seed = args.seed + run
    seed_everything(seed)
    device = torch.device(args.device)
    splits = make_splits(data, args.patch_size, args.val_fraction, seed)

    pin = device.type == "cuda"
    g = torch.Generator().manual_seed(seed)
    train_dl = loader(splits.train, args.batch_size, args, pin, shuffle=True, generator=g,
                      drop_last=len(splits.train) > args.batch_size)
    val_dl = loader(splits.val, args.eval_batch_size, args, pin)

    model = UDFNet(
        in_channels=in_channels,
        n_classes=n_classes,
        patch_size=args.patch_size,
        channels=args.channels,
        n_blocks=args.blocks,
        heads=args.heads,
        tau=args.tau,
        exchange_threshold=args.exchange_threshold,
        dropout=args.dropout,
        detach_uncertainty=not args.no_detach_uncertainty,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = LossHistory(len(splits.train), len(in_channels), start_epoch=args.history_start, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"[run {run}] seed={seed} train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} "
          f"classes={n_classes} params={n_params:,} ({n_params / 1e6:.3f}M) device={device} ({dev_name})")

    best_val, best_val_loss, best_epoch, best_state, bad_evals = -1.0, float("inf"), 0, None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = {"total": 0.0, "fused_ce": 0.0, "uni_ce": 0.0, "reg": 0.0}
        lam_sum, n_batches = torch.zeros(len(in_channels), device=device), 0
        for x_h, x_r, y, idx in train_dl:
            x_h, x_r, y, idx = (t.to(device, non_blocking=True) for t in (x_h, x_r, y, idx))
            out = model([x_h, x_r])
            losses = udfnet_loss(out, y, idx, epoch, history, gamma=args.gamma,
                                 sparsity_weight=args.sparsity, sparsity_term=model.sparsity_penalty())
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            opt.step()
            for k in agg:
                agg[k] += losses[k].detach()
            lam_sum += out["lambdas"].detach().mean(1)
            n_batches += 1

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            vp, vy, _, val_loss = predict(model, val_dl, device)
            val_oa = classification_report(vy, vp, n_classes)["OA"]
            nb = max(n_batches, 1)
            a = {k: float(v) / nb for k, v in agg.items()}
            lam = (lam_sum / nb).tolist()
            print(f"[run {run}] epoch {epoch:4d}  loss {a['total']:.4f} (fused {a['fused_ce']:.3f}, "
                  f"uni {a['uni_ce']:.3f}, reg {a['reg']:.3f})  mean lambda "
                  + " ".join(f"{n} {v:.3f}" for n, v in zip(MODALITIES, lam))
                  + f"  val OA {val_oa:.2f} loss {val_loss:.4f}  [{time.time() - t0:.0f}s]", flush=True)
            # Best = highest val OA; ties (common once val OA saturates at 100%) go to the lower val loss.
            if val_oa > best_val or (val_oa == best_val and val_loss < best_val_loss):
                best_val, best_val_loss, best_epoch, bad_evals = val_oa, val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_evals += 1
                if args.patience and bad_evals >= args.patience:
                    print(f"[run {run}] early stop at epoch {epoch}")
                    break
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)
    print(f"[run {run}] best checkpoint: epoch {best_epoch} (val OA {best_val:.2f}, val loss {best_val_loss:.4f})")
    tp, ty, tl, _ = predict(model, loader(splits.test, args.eval_batch_size, args, pin), device)
    report = classification_report(ty, tp, n_classes)
    report.update(best_val_OA=best_val, best_val_loss=best_val_loss, best_epoch=best_epoch, params=n_params, train_seconds=train_seconds,
                  mean_test_lambda={n: float(tl[m].mean()) for m, n in enumerate(MODALITIES)})
    print(f"[run {run}] TEST  OA {report['OA']:.2f}  AA {report['AA']:.2f}  Kappa {report['Kappa']:.2f}  "
          + "mean test lambda " + " ".join(f"{n} {v:.3f}" for n, v in report["mean_test_lambda"].items()))

    os.makedirs(args.out, exist_ok=True)
    torch.save({"state_dict": best_state, "args": vars(args), "n_classes": n_classes,
                "in_channels": list(in_channels)}, os.path.join(args.out, f"udfnet_run{run}.pt"))
    if args.save_map:
        h, w = data.train_gt.shape
        all_coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), -1).reshape(-1, 2)
        map_ds = PatchDataset(data.hsi, data.radar, all_coords, np.zeros(len(all_coords), dtype=np.int64),
                              args.patch_size)
        mp, _, _, _ = predict(model, loader(map_ds, args.eval_batch_size, args, pin), device)
        np.save(os.path.join(args.out, f"class_map_run{run}.npy"), (mp + 1).reshape(h, w))
    return report


def load_data(args):
    scene_args = ("hsi", "radar", "train_gt", "test_gt")
    if args.synthetic:
        data = synthetic_scene(seed=args.seed)
    elif args.patch_dir:
        if args.save_map:
            raise SystemExit("--save-map needs whole-scene data; pre-cut patch folders have no pixel grid")
        data = load_patch_folder(args.patch_dir, args.radar_name, args.patch_size)
        print(f"patch folder {args.patch_dir}: radar prefix '{data.radar_name}', "
              f"HSI train {tuple(data.hsi_train.shape)} test {tuple(data.hsi_test.shape)}, "
              f"radar train {tuple(data.radar_train.shape)} test {tuple(data.radar_test.shape)}")
    else:
        missing = [n for n in scene_args if getattr(args, n) is None]
        if missing:
            raise SystemExit(f"Missing --{', --'.join(m.replace('_', '-') for m in missing)} "
                             f"(or use --patch-dir / --synthetic)")
        data = load_scene(args.hsi, args.radar, args.train_gt, args.test_gt,
                          args.hsi_key, args.radar_key, args.train_key, args.test_key)

    if isinstance(data, SceneData):
        if args.patch_size is None:
            args.patch_size = 11
        print(f"scene: HSI {data.hsi.shape}, radar {data.radar.shape}")
        in_channels = (data.hsi.shape[2], data.radar.shape[2])
        train_labels = data.train_gt[data.train_gt > 0] - 1
    else:
        args.patch_size = data.patch_size
        in_channels = (data.hsi_train.shape[1], data.radar_train.shape[1])
        train_labels = data.y_train
    if args.patch_size % 2 == 0:
        raise SystemExit("--patch-size must be odd")

    n_classes = data.n_classes
    counts = np.bincount(train_labels, minlength=n_classes)
    print(f"classes {n_classes}, patch size {args.patch_size}, training samples per class "
          f"(before the {args.val_fraction:.0%} validation split):")
    for c, n in enumerate(counts, 1):
        print(f"  class {c:2d}: {n}")
    print(f"  total   : {counts.sum()}   (paper: Berlin 2820, Augsburg 761, MUUFL 2683, Trento 819)")
    return data, n_classes, in_channels


def main():
    args = parse_args()
    data, n_classes, in_channels = load_data(args)

    reports = [run_once(args, data, n_classes, in_channels, r) for r in range(args.runs)]
    summary = {}
    for key in ("OA", "AA", "Kappa"):
        vals = np.array([r[key] for r in reports])
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    summary["per_class_mean"] = np.array([r["per_class"] for r in reports]).mean(0).tolist()
    summary["mean_test_lambda"] = {n: float(np.mean([r["mean_test_lambda"][n] for r in reports]))
                                   for n in MODALITIES}
    print("\n=== summary over %d run(s) ===" % args.runs)
    for key in ("OA", "AA", "Kappa"):
        print(f"{key:6s} {summary[key]['mean']:.2f} +/- {summary[key]['std']:.2f}")
    for c, acc in enumerate(summary["per_class_mean"], 1):
        print(f"  class {c:2d}: {acc:.2f}")
    print("mean test lambda: " + ", ".join(f"{n} {v:.3f}" for n, v in summary["mean_test_lambda"].items()))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary, "runs": reports}, f, indent=2)
    print(f"saved results to {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
