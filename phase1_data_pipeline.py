"""
=============================================================
STCA-Net  |  Phase 1: Data Pipeline for Bonn EEG Dataset
=============================================================
Dataset:  University of Bonn Epileptic Seizure Recognition
          Andrzejak et al. (2001)  —  5 classes, 500 records
          178 time-points per record, 173.61 Hz sampling rate

Classes:
  1 = Seizure          (set E  — ictal,      from seizure area)
  2 = Seizure-free     (set D  — interictal, from seizure area)
  3 = Seizure-free     (set C  — interictal, from healthy area)
  4 = Healthy eye-closed (set B)
  5 = Healthy eye-open   (set A)

HOW TO USE WITH THE REAL DATASET
---------------------------------
1. Download from:
   https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition
   OR Kaggle: search "Epileptic Seizure Recognition"
2. Place the CSV at:  data/bonn/epileptic_seizure.csv
3. Run:  python3 phase1_data_pipeline.py

The pipeline will:
  a) Load & inspect the raw data
  b) Z-score normalise each record
  c) Create sliding window segments (configurable size & stride)
  d) Build 3 task variants (2-class, 3-class, 5-class)
  e) Perform 10-fold cross-validation splits
  f) Save ready-to-use .npy arrays  +  visualisation plots
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless rendering for saved plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG  —  change these as needed
# ─────────────────────────────────────────────
DATA_PATH   = "data/bonn/epileptic_seizure.csv"
OUTPUT_DIR  = "data/processed"
PLOTS_DIR   = "plots"
RANDOM_SEED = 42
N_FOLDS     = 10        # 10-fold cross-validation as in MASF paper
WINDOW_SIZE = 178       # 1 full record = 1 window (original UCI format)
# For larger datasets or the original 4097-sample files, set WINDOW_SIZE=256
# and STRIDE=128 to get overlapping windows
STRIDE      = 178       # no overlap by default on the 178-point UCI version
FS          = 173.61    # sampling frequency in Hz

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

np.random.seed(RANDOM_SEED)


# =============================================================
# STEP 1: LOAD DATA
# =============================================================
def load_bonn_dataset(path: str) -> pd.DataFrame:
    """
    Load the UCI Bonn Epileptic Seizure Recognition CSV.

    Expected format:
        Column 0  : row index (unnamed)
        X1..X178  : 178 EEG amplitude values
        y         : class label (1–5)
    """
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)

    df = pd.read_csv(path, index_col=0)
    print(f"  Shape          : {df.shape}")
    print(f"  Columns        : {list(df.columns[:5])} ... {list(df.columns[-3:])}")
    print(f"  Class counts   :")
    label_col = 'y' if 'y' in df.columns else df.columns[-1]
    for cls, cnt in df[label_col].value_counts().sort_index().items():
        names = {1:'Seizure (E)', 2:'Interictal-seizure area (D)',
                 3:'Interictal-healthy area (C)',
                 4:'Healthy eyes-closed (B)', 5:'Healthy eyes-open (A)'}
        print(f"    Class {cls} — {names.get(int(cls),'?'):35s} : {cnt} records")

    # Rename label column to 'label' for consistency
    df = df.rename(columns={label_col: 'label'})
    return df


# =============================================================
# STEP 2: NORMALISATION
# =============================================================
def zscore_normalise(df: pd.DataFrame) -> tuple:
    """
    Z-score normalise each record independently.
    Formula: x_norm = (x - mean) / std  per row.

    Why per-record?  EEG amplitude varies across patients and
    recording sessions.  Record-level normalisation removes
    DC offset and amplitude scale differences without leaking
    statistics from the test set.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Z-score normalisation (per record)")
    print("=" * 60)

    feature_cols = [c for c in df.columns if c.startswith('X')]
    X = df[feature_cols].values.astype(np.float32)   # (500, 178)
    y = df['label'].values.astype(np.int32)

    # Per-row z-score
    means = X.mean(axis=1, keepdims=True)
    stds  = X.std(axis=1,  keepdims=True) + 1e-8     # avoid div-by-zero
    X_norm = (X - means) / stds

    print(f"  Raw   — mean: {X.mean():.4f},  std: {X.std():.4f},  "
          f"min: {X.min():.4f},  max: {X.max():.4f}")
    print(f"  Norm  — mean: {X_norm.mean():.4f},  std: {X_norm.std():.4f},  "
          f"min: {X_norm.min():.4f},  max: {X_norm.max():.4f}")

    return X_norm, y, feature_cols


# =============================================================
# STEP 3: WINDOWING  (important for real 4097-point files)
# =============================================================
def create_windows(X: np.ndarray, y: np.ndarray,
                   window_size: int, stride: int) -> tuple:
    """
    Sliding window segmentation.

    For the UCI 178-point version:
        window_size=178, stride=178  → 1 window per record (identity)

    For the original Andrzejak 4097-point .txt files:
        window_size=256, stride=128  → ~31 windows per record with 50% overlap

    Shape out: (N_windows, window_size, 1)  — the trailing '1' is the
               channel dimension expected by Conv1D layers.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Sliding window segmentation")
    print("=" * 60)
    print(f"  Window size : {window_size} pts  = "
          f"{window_size/FS*1000:.1f} ms at {FS} Hz")
    print(f"  Stride      : {stride} pts  = "
          f"{stride/FS*1000:.1f} ms  "
          f"({'no overlap' if stride==window_size else f'{100*(1-stride/window_size):.0f}% overlap'})")

    n_records, record_len = X.shape
    windows, labels = [], []

    for i in range(n_records):
        record = X[i]
        start = 0
        while start + window_size <= record_len:
            windows.append(record[start: start + window_size])
            labels.append(y[i])
            start += stride

    X_win = np.array(windows, dtype=np.float32)
    y_win = np.array(labels,  dtype=np.int32)

    # Add channel dimension: (N, T, 1)
    X_win = X_win[:, :, np.newaxis]

    print(f"  Input records : {n_records}")
    print(f"  Output windows: {len(X_win)}  —  shape {X_win.shape}")

    return X_win, y_win


# =============================================================
# STEP 4: BUILD TASK VARIANTS
# =============================================================
def build_task_variants(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Construct the three classification tasks used in the literature.

    Task A — 2-class  : Seizure (1) vs Non-seizure (2,3,4,5)
    Task B — 3-class  : Seizure (1) vs Pre-ictal/interictal (2,3) vs Healthy (4,5)
    Task C — 5-class  : All five original classes  (same as MASF Bonn result)

    Each task returns (X_task, y_task) with labels re-mapped to 0-based integers.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Building task variants")
    print("=" * 60)

    tasks = {}

    # ── Task A: 2-class ──────────────────────────────────────
    mask_A = np.ones(len(y), dtype=bool)
    y_A    = np.where(y == 1, 1, 0)          # 1=seizure, 0=non-seizure
    tasks['2class'] = {
        'X': X[mask_A],
        'y': y_A[mask_A],
        'n_classes': 2,
        'class_names': ['Non-seizure', 'Seizure'],
        'description': '2-class: Seizure vs Non-seizure'
    }

    # ── Task B: 3-class ──────────────────────────────────────
    # Keep only classes 1, 2/3 (merged as "interictal"), 4/5 (merged as "healthy")
    y_B = np.where(y == 1, 2,
          np.where(y <= 3, 1, 0))            # 0=healthy, 1=interictal, 2=seizure
    tasks['3class'] = {
        'X': X,
        'y': y_B,
        'n_classes': 3,
        'class_names': ['Healthy', 'Interictal', 'Seizure'],
        'description': '3-class: Healthy / Interictal / Seizure'
    }

    # ── Task C: 5-class (original) ───────────────────────────
    y_C = y - 1                               # 0-indexed
    tasks['5class'] = {
        'X': X,
        'y': y_C,
        'n_classes': 5,
        'class_names': ['Seizure', 'D-interictal', 'C-interictal',
                        'Eyes-closed', 'Eyes-open'],
        'description': '5-class: All original Bonn classes (matches MASF)'
    }

    for name, task in tasks.items():
        unique, counts = np.unique(task['y'], return_counts=True)
        dist = {task['class_names'][int(u)]: int(c) for u, c in zip(unique, counts)}
        print(f"  {task['description']}")
        print(f"    Samples: {len(task['y'])}  |  Distribution: {dist}")

    return tasks


# =============================================================
# STEP 5: 10-FOLD CROSS-VALIDATION SPLITS
# =============================================================
def create_cv_splits(tasks: dict, n_folds: int = 10) -> dict:
    """
    Generate 10 stratified fold index pairs for each task.

    Stratified  = each fold preserves the class distribution.
    This matches exactly the protocol used in the MASF paper.
    """
    print("\n" + "=" * 60)
    print(f"STEP 5: Creating {n_folds}-fold stratified CV splits")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_SEED)
    splits = {}

    for task_name, task in tasks.items():
        fold_list = []
        for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(task['X'], task['y'])):
            fold_list.append({'train': train_idx, 'test': test_idx})

        splits[task_name] = fold_list
        n_train = len(fold_list[0]['train'])
        n_test  = len(fold_list[0]['test'])
        print(f"  {task_name:8s} — {n_folds} folds  "
              f"| train ≈ {n_train}, test ≈ {n_test} per fold")

    return splits


# =============================================================
# STEP 6: SAVE PROCESSED DATA
# =============================================================
def save_processed_data(tasks: dict, splits: dict, output_dir: str):
    """
    Save .npy files for each task so Phase 2 (model training)
    can load them without re-running this pipeline.

    Files saved:
        {task}_X.npy          — features  (N, window_size, 1)
        {task}_y.npy          — labels    (N,)
        {task}_splits.npy     — fold indices dict
        {task}_metadata.txt   — human-readable summary
    """
    print("\n" + "=" * 60)
    print("STEP 6: Saving processed data")
    print("=" * 60)

    for task_name, task in tasks.items():
        X_path = os.path.join(output_dir, f"{task_name}_X.npy")
        y_path = os.path.join(output_dir, f"{task_name}_y.npy")
        s_path = os.path.join(output_dir, f"{task_name}_splits.npy")
        m_path = os.path.join(output_dir, f"{task_name}_metadata.txt")

        np.save(X_path, task['X'])
        np.save(y_path, task['y'])
        np.save(s_path, splits[task_name], allow_pickle=True)

        with open(m_path, 'w') as f:
            f.write(f"Task            : {task['description']}\n")
            f.write(f"X shape         : {task['X'].shape}\n")
            f.write(f"y shape         : {task['y'].shape}\n")
            f.write(f"n_classes       : {task['n_classes']}\n")
            f.write(f"class_names     : {task['class_names']}\n")
            f.write(f"n_folds         : {N_FOLDS}\n")
            f.write(f"window_size     : {WINDOW_SIZE}\n")
            f.write(f"stride          : {STRIDE}\n")
            f.write(f"sampling_freq   : {FS} Hz\n")
            f.write(f"normalisation   : z-score per record\n")

        sz = os.path.getsize(X_path) / 1024
        print(f"  {task_name:8s} → {X_path}  ({sz:.1f} KB)")

    print(f"\n  All files saved to: {output_dir}/")


# =============================================================
# STEP 7: VISUALISATION  (EDA plots saved as PNG)
# =============================================================
def plot_class_samples(X_raw: np.ndarray, y_raw: np.ndarray,
                       plots_dir: str):
    """
    Plot 1 representative sample from each of the 5 classes.
    Also shows the power spectral density (PSD) per class.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Generating EDA plots")
    print("=" * 60)

    class_names = {1: 'Seizure (E)', 2: 'Interictal-D',
                   3: 'Interictal-C', 4: 'Eyes-closed (B)',
                   5: 'Eyes-open (A)'}
    colors = ['#D85A30', '#BA7517', '#1D9E75', '#378ADD', '#7F77DD']
    t = np.arange(WINDOW_SIZE) / FS * 1000   # time in ms

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.55, wspace=0.35)

    for idx, cls in enumerate(range(1, 6)):
        # Pick first record of this class
        mask     = y_raw == cls
        sample   = X_raw[mask][0]   # raw (unnormalised) sample

        # ── Time-domain ──
        ax_t = fig.add_subplot(gs[0, idx])
        ax_t.plot(t, sample, color=colors[idx], linewidth=0.8)
        ax_t.set_title(class_names[cls], fontsize=9, fontweight='bold')
        ax_t.set_xlabel('Time (ms)', fontsize=7)
        if idx == 0:
            ax_t.set_ylabel('Amplitude (μV)', fontsize=7)
        ax_t.tick_params(labelsize=7)
        ax_t.spines[['top', 'right']].set_visible(False)

        # ── Power spectral density ──
        ax_p = fig.add_subplot(gs[1, idx])
        freqs, psd = sp_signal.welch(sample, fs=FS, nperseg=min(64, len(sample)))
        ax_p.semilogy(freqs, psd, color=colors[idx], linewidth=0.9)
        ax_p.set_xlabel('Freq (Hz)', fontsize=7)
        if idx == 0:
            ax_p.set_ylabel('PSD (μV²/Hz)', fontsize=7)
        ax_p.tick_params(labelsize=7)
        ax_p.spines[['top', 'right']].set_visible(False)
        ax_p.set_xlim(0, FS / 2)

    fig.suptitle('Bonn EEG Dataset — Time domain & PSD per class',
                 fontsize=12, fontweight='bold', y=1.01)

    path = os.path.join(plots_dir, 'eda_class_samples.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_class_distribution(tasks: dict, plots_dir: str):
    """Bar chart of sample counts per class for all three task variants."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (task_name, task) in zip(axes, tasks.items()):
        unique, counts = np.unique(task['y'], return_counts=True)
        names   = [task['class_names'][int(u)] for u in unique]
        palette = ['#D85A30', '#1D9E75', '#378ADD', '#7F77DD', '#BA7517'][:len(unique)]

        bars = ax.bar(names, counts, color=palette, edgecolor='white',
                      linewidth=0.5)
        ax.set_title(task['description'], fontsize=9, fontweight='bold')
        ax.set_ylabel('Sample count', fontsize=8)
        ax.tick_params(labelsize=7, axis='x', rotation=20)
        ax.spines[['top', 'right']].set_visible(False)

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(cnt),
                    ha='center', va='bottom', fontsize=7)

    fig.suptitle('Class distribution per task variant',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(plots_dir, 'class_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_normalisation_effect(X_raw: np.ndarray, X_norm: np.ndarray,
                               y: np.ndarray, plots_dir: str):
    """Side-by-side raw vs normalised for one seizure and one healthy record."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharey='col')
    pairs = [(1, 'Seizure'), (5, 'Healthy (eyes open)')]
    colors = ['#D85A30', '#7F77DD']
    t = np.arange(WINDOW_SIZE) / FS * 1000

    for row, (cls, cname) in enumerate(pairs):
        mask = y == cls
        raw  = X_raw[mask][0]
        norm = X_norm[mask][0]
        col  = colors[row]

        axes[row, 0].plot(t, raw,  color=col, linewidth=0.9)
        axes[row, 0].set_title(f'{cname} — raw', fontsize=9)
        axes[row, 0].set_ylabel('Amplitude', fontsize=8)
        axes[row, 0].spines[['top', 'right']].set_visible(False)

        axes[row, 1].plot(t, norm, color=col, linewidth=0.9)
        axes[row, 1].set_title(f'{cname} — z-score normalised', fontsize=9)
        axes[row, 1].spines[['top', 'right']].set_visible(False)

    for ax in axes.flat:
        ax.set_xlabel('Time (ms)', fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle('Effect of per-record z-score normalisation',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(plots_dir, 'normalisation_effect.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cv_fold_structure(splits: dict, tasks: dict, plots_dir: str):
    """Visualise how the 10 folds are structured for the 5-class task."""
    task_name = '5class'
    task      = tasks[task_name]
    folds     = splits[task_name]
    n         = len(task['y'])

    fold_grid = np.zeros((N_FOLDS, n), dtype=np.int8)
    for fold_idx, fold in enumerate(folds):
        fold_grid[fold_idx, fold['test']] = 1

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(fold_grid, aspect='auto', cmap='Blues', vmin=0, vmax=1,
              interpolation='none')
    ax.set_xlabel('Sample index', fontsize=9)
    ax.set_ylabel('Fold', fontsize=9)
    ax.set_yticks(range(N_FOLDS))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(N_FOLDS)], fontsize=7)
    ax.set_title(f'10-fold CV split — 5-class task  '
                 f'(blue = test, white = train)', fontsize=10)
    fig.tight_layout()

    path = os.path.join(plots_dir, 'cv_fold_structure.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================
# STEP 8: SUMMARY REPORT
# =============================================================
def print_summary(tasks: dict, splits: dict):
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE — Summary")
    print("=" * 60)
    print(f"  Dataset        : Bonn University EEG")
    print(f"  Sampling rate  : {FS} Hz")
    print(f"  Window size    : {WINDOW_SIZE} pts ({WINDOW_SIZE/FS*1000:.1f} ms)")
    print(f"  Normalisation  : per-record z-score")
    print(f"  CV strategy    : {N_FOLDS}-fold stratified (seed={RANDOM_SEED})")
    print()
    for task_name, task in tasks.items():
        print(f"  [{task_name:7s}]  X={task['X'].shape}  "
              f"y={task['y'].shape}  classes={task['n_classes']}")
    print()
    print("  Ready for Phase 2 — STCA-Net model training")
    print("=" * 60)


# =============================================================
# MAIN
# =============================================================
if __name__ == '__main__':

    # 1. Load
    df = load_bonn_dataset(DATA_PATH)

    # Keep raw X for visualisation (before normalisation)
    feature_cols_raw = [c for c in df.columns if c.startswith('X')]
    X_raw = df[feature_cols_raw].values.astype(np.float32)
    y_raw = df['label'].values.astype(np.int32)

    # 2. Normalise
    X_norm, y, feature_cols = zscore_normalise(df)

    # 3. Windowing
    X_win, y_win = create_windows(X_norm, y, WINDOW_SIZE, STRIDE)

    # 4. Task variants
    tasks = build_task_variants(X_win, y_win)

    # 5. CV splits
    splits = create_cv_splits(tasks, N_FOLDS)

    # 6. Save
    save_processed_data(tasks, splits, OUTPUT_DIR)

    # 7. Plots
    plot_class_samples(X_raw, y_raw, PLOTS_DIR)
    plot_class_distribution(tasks, PLOTS_DIR)
    plot_normalisation_effect(X_raw, X_norm, y, PLOTS_DIR)
    plot_cv_fold_structure(splits, tasks, PLOTS_DIR)

    # 8. Summary
    print_summary(tasks, splits)
