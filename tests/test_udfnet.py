"""Sanity checks for the UDFNet reimplementation. Run: python -m pytest tests -q"""

import numpy as np
import torch

from udfnet.data import PatchDataset, labelled_pixels, synthetic_scene
from udfnet.losses import LossHistory, ranking_regularizer, udfnet_loss
from udfnet.metrics import classification_report
from udfnet.model import ImprovedCrissCrossAttention, SharedResidualBlock, UDFNet, energy_score


def test_cca_attends_only_to_row_and_column():
    torch.manual_seed(0)
    s = 7
    cca = ImprovedCrissCrossAttention(channels=32, patch_size=s, heads=4)
    att = cca.attention_weights(torch.randn(2, 32, s, s))  # (B, h, H, W, H+W)
    assert att.shape == (2, 4, s, s, 2 * s)
    assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)
    nonzero = (att > 0).sum(-1)
    assert torch.all(nonzero == 2 * s - 1)  # 2S-1 positions, own position counted once


def test_cca_output_depends_only_on_cross():
    """Changing a pixel outside (i, j)'s row and column must not change output at (i, j)."""
    torch.manual_seed(0)
    s = 7
    cca = ImprovedCrissCrossAttention(channels=32, patch_size=s, heads=4).eval()
    with torch.no_grad():
        cca.eta.fill_(1.0)
        x = torch.randn(1, 32, s, s)
        y1 = cca(x)
        x2 = x.clone()
        x2[:, :, 5, 6] += 10.0  # not in row 2 or column 3
        y2 = cca(x2)
    assert torch.allclose(y1[:, :, 2, 3], y2[:, :, 2, 3], atol=1e-5)
    assert not torch.allclose(y1[:, :, 2, 6], y2[:, :, 2, 6])  # same column -> changes


def test_energy_lower_for_confident_logits():
    confident = torch.tensor([[10.0, 2.0, 1.0]])
    unsure = torch.tensor([[1.0, 0.2, 0.1]])
    assert energy_score(confident) < energy_score(unsure)


def test_fusion_weight_higher_for_lower_energy():
    model = UDFNet(in_channels=(10, 2), n_classes=3, patch_size=5, channels=32, heads=4)
    assert torch.all(model.alpha < 0) and torch.all(model.beta >= 0)
    u = torch.tensor([-10.0, -2.0])  # confident, unsure
    lam = (model.alpha[0] * u + model.beta[0]).clamp_min(0)
    assert lam[0] > lam[1]


def test_channel_exchange_swaps_low_gamma_channels():
    block = SharedResidualBlock(channels=8, exchange_threshold=0.1)
    with torch.no_grad():
        block.bn2[0].weight.fill_(1.0)
        block.bn2[0].weight[1] = 0.0  # HSI channel 1 (eligible half) is redundant
        block.bn2[1].weight.fill_(1.0)
    a, b = torch.zeros(1, 8, 3, 3), torch.ones(1, 8, 3, 3)
    out_a, out_b = block.exchange([a, b])
    assert torch.all(out_a[:, 1] == 1) and torch.all(out_a[:, 0] == 0)
    assert torch.equal(out_b, b)


def test_forward_shapes_and_backward():
    torch.manual_seed(0)
    model = UDFNet(in_channels=(30, 2), n_classes=5, patch_size=7, channels=32, heads=4)
    xs = [torch.randn(6, 30, 7, 7), torch.randn(6, 2, 7, 7)]
    out = model(xs)
    assert out["fused"].shape == (6, 5)
    assert out["unimodal"].shape == (2, 6, 5)
    assert out["lambdas"].shape == (2, 6) and torch.all(out["lambdas"] >= 0)
    hist = LossHistory(n_samples=6, n_modalities=2, start_epoch=1)
    y, idx = torch.randint(0, 5, (6,)), torch.arange(6)
    udfnet_loss(out, y, idx, epoch=1, history=hist)  # epoch 1 fills history
    out = model(xs)
    losses = udfnet_loss(out, y, idx, epoch=2, history=hist, sparsity_weight=0.05,
                         sparsity_term=model.sparsity_penalty())
    losses["total"].backward()
    assert model.alpha_raw.grad is not None and model.beta_raw.grad is not None
    assert losses["reg"] >= 0


def test_ranking_regularizer_penalises_wrong_order():
    hist = torch.tensor([[2.0, 0.5]])  # sample 0 historically worse
    valid = torch.tensor([True, True])
    wrong = ranking_regularizer(torch.tensor([[0.9, 0.1]]), hist, valid)  # worse sample trusted more
    right = ranking_regularizer(torch.tensor([[0.1, 0.9]]), hist, valid)
    assert wrong > right


def test_patch_centre_is_labelled_pixel():
    scene = synthetic_scene(height=20, width=20, hsi_bands=4, radar_bands=1, n_classes=3)
    coords, labels = labelled_pixels(scene.train_gt)
    ds = PatchDataset(scene.hsi, scene.radar, coords, labels, patch_size=5)
    x_h, x_r, y, _ = ds[0]
    r, c = coords[0]
    assert torch.allclose(x_h[:, 2, 2], torch.from_numpy(scene.hsi[r, c]))
    assert x_h.shape == (4, 5, 5) and x_r.shape == (1, 5, 5)


def test_metrics_match_hand_example():
    # 90 forest all right, 10 water with 5 right -> OA 95, AA 75 (example from the explanation)
    y = np.array([0] * 90 + [1] * 10)
    p = np.array([0] * 90 + [1] * 5 + [0] * 5)
    rep = classification_report(y, p, 2)
    assert abs(rep["OA"] - 95) < 1e-9 and abs(rep["AA"] - 75) < 1e-9


def test_patch_folder_format(tmp_path):
    """MFT layout: split test parts are concatenated in numeric order, radar prefix is detected,
    patches become (N, C, S, S) in [0, 1], labels become 0-based, and centre-cropping works."""
    from scipy.io import savemat
    from udfnet.data import load_patch_folder, make_splits

    rng = np.random.default_rng(0)
    s, n_tr, n_te = 7, 12, 20
    hsi_tr, hsi_te = rng.normal(size=(n_tr, s, s, 5)), rng.normal(size=(n_te, s, s, 5))
    hsi_te[:, s // 2, s // 2, 0] = np.arange(n_te)  # order marker at the centre pixel
    savemat(tmp_path / "HSI_Tr.mat", {"Data": hsi_tr})
    for k, part in enumerate(np.array_split(hsi_te, 3), 1):  # Part1..Part3, saved out of order
        savemat(tmp_path / f"HSI_Te_Part{k}.mat", {"Data": part})
    savemat(tmp_path / "LIDAR_Tr.mat", {"Data": rng.normal(size=(n_tr, s, s))})
    savemat(tmp_path / "LIDAR_Te.mat", {"Data": rng.normal(size=(n_te, s, s))})
    savemat(tmp_path / "TrLabel.mat", {"Data": np.repeat([1, 2, 3], 4)[:, None]})
    savemat(tmp_path / "TeLabel.mat", {"Data": rng.integers(1, 4, size=(n_te, 1))})

    data = load_patch_folder(str(tmp_path))
    assert data.radar_name == "LIDAR" and data.patch_size == s and data.n_classes == 3
    assert tuple(data.hsi_train.shape) == (n_tr, 5, s, s) and tuple(data.radar_test.shape) == (n_te, 1, s, s)
    assert data.hsi_train.min() >= 0 and data.hsi_test.max() <= 1
    centre = data.hsi_test[:, 0, s // 2, s // 2]
    assert torch.all(centre[1:] > centre[:-1])  # parts concatenated in order
    assert set(data.y_train.tolist()) == {0, 1, 2}

    cropped = load_patch_folder(str(tmp_path), patch_size=3)
    assert cropped.patch_size == 3
    assert torch.allclose(cropped.hsi_test[:, :, 1, 1], data.hsi_test[:, :, s // 2, s // 2])

    splits = make_splits(data, s, 0.2, seed=0)
    assert len(splits.train) + len(splits.val) == n_tr
    x_h, x_r, y, i = splits.train[0]
    assert x_h.shape == (5, s, s) and x_r.shape == (1, s, s) and i == 0
