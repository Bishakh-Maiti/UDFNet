# UDFNet: unofficial PyTorch reimplementation

This is an **unofficial** reimplementation of **"Uncertainty-Aware Dynamic Fusion Network with Criss-Cross Attention for multimodal remote sensing land cover classification"** (Wang et al., *Information Fusion* 123 (2025) 103249). No official code is public, so this was written from the paper's description; it is not the authors' code. Where the paper is unclear, the choice made here is listed under [Assumptions and deviations](#assumptions-and-deviations-from-the-paper).

**Task:** classify each labelled pixel from an S×S patch centred on it, using two co-registered modalities: a hyperspectral image (HSI, L bands) and a radar image (SAR or LiDAR, B channels). Fusion is at the decision level: each modality predicts on its own, and the logits are combined with per-sample weights λ derived from each branch's energy-based uncertainty.

## Layout

```
udfnet/model.py    FEP (AsyFFNet-style, channel exchange), improved Criss-Cross Attention,
                   classifier heads, energy score, dynamic fusion  -> UDFNet
udfnet/losses.py   LossHistory (Eq. 18), ranking regularizer L_extra (Eq. 19-20), total loss (Eq. 21)
udfnet/data.py     whole-scene .mat loading + on-the-fly patches; MFT pre-cut patch folders;
                   stratified 20% validation split; synthetic_scene()
udfnet/metrics.py  OA, AA, Kappa (x100), per-class accuracy
train.py           Algorithm 1: training, validation-based model selection, repeated runs, results.json
tests/             10 sanity tests
```

## Installation

```bash
pip install -r requirements.txt      # torch, numpy, scipy, h5py, pytest
python -m pytest tests -q            # 10 tests, a few seconds on CPU
```

For a GPU, install the CUDA build of PyTorch first (see pytorch.org). `--device` defaults to `cuda` when it is available.

## Smoke test on synthetic data (no download needed)

```bash
python train.py --synthetic --epochs 40 --patch-size 7 --lr 1e-3 --out runs/synthetic
```

`synthetic_scene()` builds a 60×60 scene with 5 blocky classes, class-specific HSI spectra and radar values, plus noise. In one quadrant the HSI is replaced by pure noise, so radar is the only reliable modality there and the HSI fusion weight should drop in that quadrant.

## Data

### Format A: whole scene (.mat)

You need four files: the HSI `(H, W, L)`, the radar `(H, W, B)` or `(H, W)`, the training label map `(H, W)` and the test label map `(H, W)`. In the label maps, 0 means unlabelled and classes are 1..K.

- Each file is read with `scipy.io.loadmat`, falling back to `h5py` for MATLAB v7.3 files (which are transposed back).
- If a file holds several arrays, choose one with `--hsi-key`, `--radar-key`, `--train-key` or `--test-key`; otherwise the largest array is used.
- Patches are cut on the fly with reflect padding, so Berlin's ~460k test pixels are never stored as patches.

**Berlin and Augsburg** (HSI + SAR) are available at <https://github.com/danfenghong/ISPRS_S2FL>.

```bash
python train.py --hsi data/berlin/data_HS_LR.mat --radar data/berlin/data_SAR_HR.mat \
  --train-gt data/berlin/TrainImage.mat --test-gt data/berlin/TestImage.mat \
  --patch-size 9 --runs 10 --out runs/berlin --save-map

python train.py --hsi data/augsburg/data_HS_LR.mat --radar data/augsburg/data_SAR_HR.mat \
  --train-gt data/augsburg/TrainImage.mat --test-gt data/augsburg/TestImage.mat \
  --patch-size 9 --runs 10 --out runs/augsburg
```

(Adjust the file names to match your download.)

### Format B: pre-cut patch folders (MFT layout)

The MUUFL and Trento pre-cut folders are linked from Google Drive in the README of <https://github.com/AnkurDeria/MFT>. A folder contains:

```
HSI_Tr.mat     HSI_Te.mat  (or HSI_Te_Part1.mat, HSI_Te_Part2.mat, ...; concatenated in numeric order)
LIDAR_Tr.mat   LIDAR_Te.mat   (radar prefix auto-detected, e.g. LIDAR or SAR; override with --radar-name)
TrLabel.mat    TeLabel.mat
```

Each file holds the key `Data`. Patches are `(N, S, S, C)` or `(N, S, S)` and are converted to `(N, C, S, S)`. Labels are 1-based. `--patch-size` defaults to the stored size; a smaller odd size centre-crops the patches.

```bash
python train.py --patch-dir data/MUUFL  --runs 10 --out runs/muufl
python train.py --patch-dir data/Trento --runs 10 --out runs/trento
```

The whole test set is held in memory as float32 patches. For MUUFL at S = 11 that is about 1.7 GB, so make sure you have enough RAM.

### Splits and checks

- Training uses the dataset's fixed train/test split. A stratified 20% of the training samples (at least 1 kept for training per class, seed-dependent) becomes the validation set.
- At startup, `train.py` prints the training samples per class before the validation split. Compare the total with the paper: Berlin 2820, Augsburg 761, MUUFL 2683, Trento 819.

**Suggested patch sizes** (from the paper's Fig. 12): Berlin 9, Augsburg 9, MUUFL 11, Trento 11.

## Default hyperparameters

| Setting | Value | Flag |
|---|---|---|
| Optimizer | Adam, lr 1e-4, weight decay 1e-5 | `--lr`, `--weight-decay` |
| Batch size / max epochs | 64 / 500 | `--batch-size`, `--epochs` |
| Seed | 430 (run *r* uses 430 + *r*) | `--seed`, `--runs` (paper: 10) |
| Validation | 20% of training samples, every 5 epochs, best val OA kept | `--val-fraction`, `--eval-every`, `--patience` |
| Feature width C / residual blocks | 128 / 2 | `--channels`, `--blocks` |
| CCA heads | 4 (C′ = C/8) | `--heads` |
| Channel-exchange threshold | 0.02 | `--exchange-threshold` |
| Sparsity (L1 on BN γ) | 0.05 | `--sparsity` |
| Ranking regularizer γ / history start L_s | 0.1 / epoch 5 | `--gamma`, `--history-start` |
| Energy temperature τ | 1 | `--tau` |
| Dropout in heads | 0.2 | `--dropout` |
| Patch size | 11 (scenes), stored size (patch folders) | `--patch-size` |

Other flags: `--device`, `--workers`, `--eval-batch-size`, `--out`, `--save-map`, `--no-detach-uncertainty`.

**Parameter counts:**

| Dataset | Inputs (HSI + radar) | Classes | S | Parameters |
|---|---|---|---|---|
| Berlin | 244 + 4 | 8 | 9 | 960,150 (paper: 0.96M) |
| Augsburg | 180 + 4 | 7 | 9 | 886,164 |
| MUUFL | 64 + 2 | 11 | 11 | 753,820 |
| Trento | 63 + 1 | 6 | 11 | 750,226 |

**Outputs in `--out`:**

- `results.json`: per-run and mean ± std OA/AA/Kappa, mean per-class accuracy, and mean test λ per modality.
- `udfnet_run<r>.pt`: the best-validation `state_dict`, plus `args`, `n_classes` and `in_channels`.
- `class_map_run<r>.npy`: the full-scene prediction, with `--save-map` (whole-scene format only).

Expect several hours per run on Berlin for 500 epochs. Use `--patience` for early stopping.

## Assumptions and deviations from the paper

1. **Ranking regularizer (Eq. 19) is replaced by the QMF form.** As printed, Eq. 19 puts the sign on λ and the margin on |λ_i − λ_j|. Neither the sign nor the loss history carries a gradient, so the only gradient left pulls the weights together and ranks nothing. This code uses the correctness-ranking form of QMF (Zhang et al., ICML 2023), which the paper builds on: `relu(sign(l_i − l_j)·(λ_i − λ_j) + |l_i − l_j|)`. Here l is the historical unimodal loss, so the historically worse sample must get the lower weight. Each sample is paired with the next one in the batch (`torch.roll`), and the loss is averaged over valid pairs (both have history and l_i ≠ l_j).
2. **Loss history (Eq. 18)** is a per-sample, per-modality running mean of the unimodal cross-entropy, recorded from epoch L_s = 5 onward and indexed by dataset sample index.
3. **Uncertainty is detached** when computing λ = α·u + β. α and β therefore learn from the fused loss, and a branch cannot inflate its own logits to gain weight. Use `--no-detach-uncertainty` to let gradients flow through u.
4. **λ is clamped at ≥ 0**, so a modality can be silenced but never vote against a class. The signs α < 0 and β ≥ 0 are enforced by α = −softplus(a) and β = softplus(b), both initialised to magnitude 0.1.
5. **Channel exchange** follows channel-exchanging networks (CEN). After the second BN of each shared block, a channel whose |γ| < 0.02 is replaced by the other modality's channel. The HSI branch may only exchange the first half of the channels and the radar branch only the second half. The L1 penalty (weight 0.05) is applied only to these exchangeable γ's.
6. **FEP details:**
   - The per-modality stem is Conv3×3 (no bias) → BN → ReLU, with C = 128.
   - The two shared residual blocks are conv3×3 → BN_m → ReLU → conv3×3 → BN_m → exchange → +skip → ReLU. The conv weights are shared and the BNs are per modality.
7. **Classifier head design** is not specified in the paper. Each modality uses global average pooling → Dropout(0.2) → Linear(C, K).
8. **CCA details:**
   - Q and K have C′ = C/8 channels and V has C channels, split over 4 heads.
   - Learnable positional encodings (1, C′, S, S), initialised with truncated-normal std 0.02, are added to Q and K.
   - Each position attends to the 2S − 1 positions in its own row and column, with the duplicated self position masked in the column part. The softmax is over row and column scores together, scaled by 1/√(C′/h).
   - A 1×1 output projection follows. The output is z′ = LayerNorm_channels(z + η·O), with η initialised to 0 (as in CCNet).
   - The positional encodings fix the model to the training patch size.
9. **Energy score:** u = −τ·logsumexp(g/τ) with τ = 1 (Liu et al., NeurIPS 2020).
10. **Preprocessing:**
    - Whole scenes use per-band min-max scaling to [0, 1] over the whole image, with reflect padding at the borders.
    - Patch folders use per-band min-max over train + test patches, computed before any centre crop.
11. **Validation and model selection:** a stratified 20% of the training samples is held out, and the test set is evaluated with the best-validation-OA checkpoint. Ties go to the checkpoint with the lower validation cross-entropy. Without this, validation OA saturates at 100% within about 10 epochs on Trento and the earliest, barely trained checkpoint would be kept. The paper does not say how the checkpoint used for testing was chosen.
12. **Loss weighting:** total = CE(fused) + Σ_m CE(unimodal_m) + 0.1·L_extra + 0.05·Σ|γ|. The sparsity term dominates the printed total early in training (about 12.8 at initialisation, with 256 exchangeable γ's starting at 1) and shrinks as γ decays.

Because of these choices, numbers may differ from the paper's tables. Report them as coming from an unofficial reimplementation.

## Verification done

- `pytest`: 10/10 pass.
  - CCA attention has shape (B, h, S, S, 2S), rows sum to 1, and exactly 2S − 1 entries are > 0.
  - CCA output at (i, j) ignores pixels outside its row and column.
  - Confident logits have lower energy.
  - α < 0, β ≥ 0, and lower energy gives a higher λ.
  - Channel exchange works.
  - A full forward + loss + backward runs, and α and β receive gradients.
  - The ranking regularizer penalises the wrong order.
  - The patch centre is the labelled pixel.
  - Metrics match the worked example (OA 95, AA 75).
  - The MFT folder loader handles Part files, centre-cropping and labels.
- Synthetic run on an RTX 4060 Laptop GPU (40 epochs, S = 7, lr 1e-3, about 11 s): test OA 94.46, AA 93.40, Kappa 92.80. From the checkpoint:

  | Region | mean λ HSI | mean λ radar | HSI share |
  |---|---|---|---|
  | Corrupted-HSI quadrant | 0.347 | 0.683 | 0.34 |
  | Elsewhere | 0.572 | 0.645 | 0.47 |

  The HSI weight drops where the HSI is unreliable, as intended.
