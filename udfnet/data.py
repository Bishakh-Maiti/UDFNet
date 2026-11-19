"""Data loading for HSI + radar (SAR or LiDAR) land cover datasets.

Two input formats are supported.

A. Whole scene (e.g. Berlin / Augsburg from github.com/danfenghong/ISPRS_S2FL):
  * HSI cube           (H, W, L)
  * radar image        (H, W, B) or (H, W)       (SAR: B channels, LiDAR: usually 1-2)
  * training label map (H, W), 0 = unlabelled, classes 1..K
  * test label map     (H, W), 0 = unlabelled, classes 1..K
  Patches of size S x S are cut around each labelled pixel on the fly (reflect padding), so even
  Berlin's ~460k test pixels never have to be held in memory as patches.

B. Pre-cut patches in the MFT repository layout (github.com/AnkurDeria/MFT), one folder with
  HSI_Tr.mat, HSI_Te.mat (or HSI_Te_Part1.mat, HSI_Te_Part2.mat, ...), <RADAR>_Tr.mat,
  <RADAR>_Te.mat (RADAR = LIDAR, SAR, ...), TrLabel.mat, TeLabel.mat, each holding key 'Data'.
  Patches are (N, S, S, C) or (N, S, S); labels are 1-based.

Every dataset returns (hsi_patch [L,S,S], radar_patch [B,S,S], label (0-based), sample_index).
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def load_mat_array(path: str, key: Optional[str] = None) -> np.ndarray:
    """Load one array from a .mat file (v5 via scipy, v7.3 via h5py). If `key` is None, the
    largest non-metadata array is used."""
    try:
        from scipy.io import loadmat

        data = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
        transpose = False
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        import h5py

        with h5py.File(path, "r") as f:
            data = {k: np.array(f[k]) for k in f.keys() if isinstance(f[k], h5py.Dataset)}
        transpose = True
    if not data:
        raise ValueError(f"No arrays found in {path}")
    if key is None:
        key = max(data, key=lambda k: np.asarray(data[k]).size)
    elif key not in data:
        raise KeyError(f"{key!r} not in {path}; available: {sorted(data)}")
    arr = np.asarray(data[key])
    return arr.T if transpose else arr


def to_hwc(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {arr.shape}")
    return arr


def minmax_per_band(img: np.ndarray) -> np.ndarray:
    """ASSUMPTION: per-band min-max scaling to [0, 1] (the paper only says 'after preprocessing')."""
    lo = img.min(axis=(0, 1), keepdims=True)
    hi = img.max(axis=(0, 1), keepdims=True)
    return (img - lo) / np.maximum(hi - lo, 1e-8)


def labelled_pixels(gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinates (N, 2) and 0-based labels (N,) of all non-zero pixels."""
    gt = np.asarray(gt).astype(np.int64)
    rows, cols = np.nonzero(gt)
    return np.stack([rows, cols], axis=1), gt[rows, cols] - 1


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into train/val, keeping at least one training sample per class."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = int(round(len(idx) * val_fraction))
        n_val = min(n_val, len(idx) - 1)
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


class PatchDataset(Dataset):
    """Returns (hsi_patch [L,S,S], radar_patch [B,S,S], label, sample_index)."""

    def __init__(self, hsi: np.ndarray, radar: np.ndarray, coords: np.ndarray, labels: np.ndarray,
                 patch_size: int):
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd so the labelled pixel is the centre")
        r = patch_size // 2
        pad = ((r, r), (r, r), (0, 0))
        self.hsi = torch.from_numpy(np.pad(hsi, pad, mode="reflect")).permute(2, 0, 1).contiguous()
        self.radar = torch.from_numpy(np.pad(radar, pad, mode="reflect")).permute(2, 0, 1).contiguous()
        self.coords = torch.as_tensor(coords, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.size = patch_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        row, col = self.coords[i].tolist()  # padded coords: top-left of the patch
        s = self.size
        return (
            self.hsi[:, row:row + s, col:col + s],
            self.radar[:, row:row + s, col:col + s],
            self.labels[i],
            i,
        )


@dataclass
class SceneData:
    hsi: np.ndarray
    radar: np.ndarray
    train_gt: np.ndarray
    test_gt: np.ndarray

    @property
    def n_classes(self) -> int:
        return int(max(self.train_gt.max(), self.test_gt.max()))


def load_scene(hsi_path, radar_path, train_gt_path, test_gt_path,
               hsi_key=None, radar_key=None, train_key=None, test_key=None) -> SceneData:
    hsi = minmax_per_band(to_hwc(load_mat_array(hsi_path, hsi_key)))
    radar = minmax_per_band(to_hwc(load_mat_array(radar_path, radar_key)))
    train_gt = load_mat_array(train_gt_path, train_key).astype(np.int64)
    test_gt = load_mat_array(test_gt_path, test_key).astype(np.int64)
    if not (hsi.shape[:2] == radar.shape[:2] == train_gt.shape == test_gt.shape):
        raise ValueError(
            f"Spatial sizes differ: hsi {hsi.shape[:2]}, radar {radar.shape[:2]}, "
            f"train {train_gt.shape}, test {test_gt.shape}"
        )
    return SceneData(hsi, radar, train_gt, test_gt)


def synthetic_scene(height=60, width=60, hsi_bands=30, radar_bands=2, n_classes=5, seed=0) -> SceneData:
    """A small toy scene for smoke tests: blocky class regions, class-specific spectra in the HSI,
    class-specific 'height' in the radar, plus noise. One quadrant has corrupted HSI (shadow-like)
    so radar is the reliable modality there."""
    rng = np.random.default_rng(seed)
    block = 10
    class_map = rng.integers(1, n_classes + 1, size=(height // block, width // block))
    class_map = np.kron(class_map, np.ones((block, block), dtype=np.int64))
    spectra = rng.normal(0, 1, size=(n_classes, hsi_bands))
    heights = rng.normal(0, 1, size=(n_classes, radar_bands))
    hsi = spectra[class_map - 1] + 0.8 * rng.normal(size=(height, width, hsi_bands))
    radar = heights[class_map - 1] + 0.8 * rng.normal(size=(height, width, radar_bands))
    hsi[: height // 2, : width // 2] = rng.normal(size=(height // 2, width // 2, hsi_bands)) * 0.3
    split = rng.random((height, width))
    train_gt = np.where(split < 0.15, class_map, 0)
    test_gt = np.where(split >= 0.15, class_map, 0)
    return SceneData(minmax_per_band(hsi.astype(np.float32)), minmax_per_band(radar.astype(np.float32)),
                     train_gt, test_gt)


# ---------------------------------------------------------------------------
# Format B: pre-cut patches (MFT repository layout)
# ---------------------------------------------------------------------------


class PreCutPatchDataset(Dataset):
    """Pre-cut patches held as tensors. Returns (hsi [L,S,S], radar [B,S,S], label, sample_index)."""

    def __init__(self, hsi: torch.Tensor, radar: torch.Tensor, labels: torch.Tensor):
        if not (len(hsi) == len(radar) == len(labels)):
            raise ValueError(f"Length mismatch: hsi {len(hsi)}, radar {len(radar)}, labels {len(labels)}")
        self.hsi, self.radar, self.labels = hsi, radar, torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.hsi[i], self.radar[i], self.labels[i], i


@dataclass
class PatchFolderData:
    hsi_train: torch.Tensor   # (N_tr, L, S, S) float32, scaled to [0, 1]
    radar_train: torch.Tensor  # (N_tr, B, S, S)
    y_train: np.ndarray        # (N_tr,) 0-based
    hsi_test: torch.Tensor
    radar_test: torch.Tensor
    y_test: np.ndarray
    radar_name: str

    @property
    def n_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_test.max())) + 1

    @property
    def patch_size(self) -> int:
        return int(self.hsi_train.shape[-1])


def _numeric_suffix(path: str) -> int:
    m = re.search(r"Part(\d+)\.mat$", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _load_split(folder: str, prefix: str, split: str) -> np.ndarray:
    """Load <prefix>_<split>.mat, or concatenate <prefix>_<split>_Part1.mat, Part2, ... in order."""
    single = os.path.join(folder, f"{prefix}_{split}.mat")
    if os.path.exists(single):
        return np.asarray(load_mat_array(single, "Data"), dtype=np.float32)
    parts = sorted(glob.glob(os.path.join(folder, f"{prefix}_{split}_Part*.mat")), key=_numeric_suffix)
    if not parts:
        raise FileNotFoundError(f"Neither {single} nor {prefix}_{split}_Part*.mat found in {folder}")
    return np.concatenate([np.asarray(load_mat_array(p, "Data"), dtype=np.float32) for p in parts], axis=0)


def _to_nchw(arr: np.ndarray, name: str) -> np.ndarray:
    """(N,S,S,C) or (N,S,S) -> (N,C,S,S)."""
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"{name}: expected patches shaped (N,S,S,C) or (N,S,S), got {arr.shape}")
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


def _centre_crop(arr: np.ndarray, size: int, name: str) -> np.ndarray:
    s = arr.shape[-1]
    if size > s:
        raise ValueError(f"{name}: requested patch size {size} > stored size {s}")
    if (s - size) % 2:
        raise ValueError(f"{name}: cannot centre-crop {s} to {size} (need the same parity)")
    o = (s - size) // 2
    return arr[..., o:o + size, o:o + size] if o else arr


def _minmax_joint(train: np.ndarray, test: np.ndarray) -> None:
    """ASSUMPTION: per-band min-max to [0, 1] over train+test patches, in place (N, C, S, S)."""
    lo = np.minimum(train.min(axis=(0, 2, 3)), test.min(axis=(0, 2, 3)))[None, :, None, None]
    hi = np.maximum(train.max(axis=(0, 2, 3)), test.max(axis=(0, 2, 3)))[None, :, None, None]
    scale = 1.0 / np.maximum(hi - lo, 1e-8)
    for a in (train, test):
        a -= lo
        a *= scale


def detect_radar_name(folder: str) -> str:
    names = set()
    for p in glob.glob(os.path.join(folder, "*_Tr.mat")):
        prefix = os.path.basename(p)[: -len("_Tr.mat")]
        if prefix.upper() != "HSI":
            names.add(prefix)
    if len(names) != 1:
        raise ValueError(f"Could not auto-detect the radar file prefix in {folder} (found {sorted(names)}); "
                         f"pass --radar-name")
    return names.pop()


def _labels(folder: str, name: str) -> np.ndarray:
    y = np.asarray(load_mat_array(os.path.join(folder, f"{name}.mat"), "Data")).astype(np.int64).ravel()
    if y.min() < 1:
        raise ValueError(f"{name}.mat: expected 1-based labels, found min {y.min()}")
    return y - 1


def load_patch_folder(folder: str, radar_name: Optional[str] = None,
                      patch_size: Optional[int] = None) -> PatchFolderData:
    radar_name = radar_name or detect_radar_name(folder)
    arrays = {}
    for mod, prefix in (("hsi", "HSI"), ("radar", radar_name)):
        for split in ("Tr", "Te"):
            arrays[mod, split] = _to_nchw(_load_split(folder, prefix, split), f"{prefix}_{split}")
        # Scale on the stored patches, then crop, so the scaling does not depend on --patch-size.
        _minmax_joint(arrays[mod, "Tr"], arrays[mod, "Te"])
        if patch_size is not None:
            for split in ("Tr", "Te"):
                arrays[mod, split] = np.ascontiguousarray(
                    _centre_crop(arrays[mod, split], patch_size, f"{prefix}_{split}"))
    y_tr, y_te = _labels(folder, "TrLabel"), _labels(folder, "TeLabel")
    for (mod, split), a in arrays.items():
        n = len(y_tr) if split == "Tr" else len(y_te)
        if len(a) != n:
            raise ValueError(f"{mod} {split}: {len(a)} patches but {n} labels")
    t = {k: torch.from_numpy(v) for k, v in arrays.items()}
    return PatchFolderData(t["hsi", "Tr"], t["radar", "Tr"], y_tr, t["hsi", "Te"], t["radar", "Te"], y_te,
                           radar_name)


# ---------------------------------------------------------------------------
# Train / validation / test datasets for either format
# ---------------------------------------------------------------------------


@dataclass
class Splits:
    train: Dataset
    val: Dataset
    test: Dataset
    all_train_labels: np.ndarray  # labels of every training pixel, before the validation split


def make_splits(data: Union[SceneData, PatchFolderData], patch_size: int, val_fraction: float,
                seed: int) -> Splits:
    if isinstance(data, SceneData):
        tr_coords, tr_labels = labelled_pixels(data.train_gt)
        te_coords, te_labels = labelled_pixels(data.test_gt)
        tr_idx, va_idx = stratified_split(tr_labels, val_fraction, seed)
        make = lambda c, l: PatchDataset(data.hsi, data.radar, c, l, patch_size)
        return Splits(make(tr_coords[tr_idx], tr_labels[tr_idx]), make(tr_coords[va_idx], tr_labels[va_idx]),
                      make(te_coords, te_labels), tr_labels)
    tr_idx, va_idx = stratified_split(data.y_train, val_fraction, seed)
    def sub(idx: np.ndarray) -> PreCutPatchDataset:
        t = torch.as_tensor(idx, dtype=torch.long)
        return PreCutPatchDataset(data.hsi_train[t].clone(), data.radar_train[t].clone(), data.y_train[idx])

    return Splits(sub(tr_idx), sub(va_idx),
                  PreCutPatchDataset(data.hsi_test, data.radar_test, data.y_test), data.y_train)
