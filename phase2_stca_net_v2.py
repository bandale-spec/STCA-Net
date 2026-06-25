"""
=============================================================
STCA-Net v2  |  Phase 2: Fixed Training + Colab-Ready
=============================================================
Fixes applied over v1:
  - Learning rate warmup scheduler (cosine decay with warmup)
  - Label smoothing (0.1) in cross-entropy loss
  - Stronger regularisation: L2 weight decay + dropout 0.5
  - Gradient clipping (norm=1.0)
  - Larger embed_dim=64 (GPU) / 32 (CPU auto-detect)
  - BatchNorm momentum fixed for small batches
  - Deeper classification head with residual dropout
=============================================================
"""

import os, warnings, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
#  AUTO-DETECT GPU / CPU AND SET DIMS
# ─────────────────────────────────────────────
GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) > 0
EMBED_DIM  = 64  if GPU_AVAILABLE else 32
FF_DIM     = 128 if GPU_AVAILABLE else 64
BATCH_SIZE = 256 if GPU_AVAILABLE else 512
print(f"Device: {'GPU' if GPU_AVAILABLE else 'CPU'}  "
      f"| embed_dim={EMBED_DIM}  batch={BATCH_SIZE}")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "data/processed"
OUTPUT_DIR  = "results"
PLOTS_DIR   = "plots"
RANDOM_SEED = 42

EPOCHS      = 150
LR_MAX      = 3e-4        # peak LR after warmup
LR_MIN      = 1e-6
WARMUP_EP   = 10          # epochs for LR warmup
DROPOUT     = 0.5
NUM_HEADS   = 4
GRU_UNITS   = EMBED_DIM
N_FOLDS     = 10
L2          = 1e-4        # L2 weight decay
LABEL_SMOOTH = 0.1        # label smoothing factor
PATIENCE    = 20          # early stopping patience
LR_PAT      = 10          # LR reduce patience

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# =============================================================
# WARMUP + COSINE DECAY LR SCHEDULE
# =============================================================
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup for WARMUP_EP epochs, then cosine decay to LR_MIN.
    Critical for Transformer-based models — prevents divergence early.
    """
    def __init__(self, lr_max, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_max       = tf.cast(lr_max,       tf.float32)
        self.lr_min       = tf.cast(lr_min,       tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps  = tf.cast(total_steps,  tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.lr_max * (step / self.warmup_steps)
        cos_arg   = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cos_lr    = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0 + tf.math.cos(np.pi * tf.minimum(cos_arg, 1.0)))
        return tf.where(step < self.warmup_steps, warmup_lr, cos_lr)

    def get_config(self):
        return {'lr_max': float(self.lr_max), 'lr_min': float(self.lr_min),
                'warmup_steps': int(self.warmup_steps),
                'total_steps':  int(self.total_steps)}


# =============================================================
# BUILDING BLOCKS  (same novelty, improved regularisation)
# =============================================================

def dsca_block(x, filters, name_prefix='dsca'):
    """
    Novel 1 — Depthwise Separable Conv Attention (DSCA):
    Multi-scale (k=3,5,7) depthwise separable Conv1D +
    CBAM-style joint channel + spatial attention.
    L2 regularisation added for Elsevier-quality training.
    """
    reg = regularizers.l2(L2)
    branches = []
    for k in [3, 5, 7]:
        b = layers.DepthwiseConv1D(
                k, padding='same', depthwise_regularizer=reg,
                name=f'{name_prefix}_dw{k}')(x)
        b = layers.Conv1D(
                filters, 1, kernel_regularizer=reg,
                name=f'{name_prefix}_pw{k}')(b)
        b = layers.BatchNormalization(momentum=0.9,
                name=f'{name_prefix}_bn{k}')(b)
        b = layers.Activation('relu', name=f'{name_prefix}_act{k}')(b)
        branches.append(b)
    f = layers.Add(name=f'{name_prefix}_add')(branches)

    # Channel attention
    ca = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gap')(f)
    ca = layers.Dense(max(1, filters//4), activation='relu',
                      kernel_regularizer=reg,
                      name=f'{name_prefix}_ca1')(ca)
    ca = layers.Dense(filters, activation='sigmoid',
                      kernel_regularizer=reg,
                      name=f'{name_prefix}_ca2')(ca)
    ca = layers.Reshape((1, filters), name=f'{name_prefix}_car')(ca)
    f  = layers.Multiply(name=f'{name_prefix}_cam')([f, ca])

    # Spatial attention
    sa_a = layers.Lambda(
               lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
               name=f'{name_prefix}_saa')(f)
    sa_m = layers.Lambda(
               lambda t: tf.reduce_max(t,  axis=-1, keepdims=True),
               name=f'{name_prefix}_sam')(f)
    sa   = layers.Concatenate(name=f'{name_prefix}_sac')([sa_a, sa_m])
    sa   = layers.Conv1D(
               1, 7, padding='same', activation='sigmoid',
               kernel_regularizer=reg,
               name=f'{name_prefix}_sav')(sa)
    f    = layers.Multiply(name=f'{name_prefix}_sam2')([f, sa])

    # Residual
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, kernel_regularizer=reg,
                          name=f'{name_prefix}_res')(x)
    return layers.Add(name=f'{name_prefix}_out')([f, x])


def temporal_pyramid_pooling(x, name_prefix='tpp'):
    """
    Novel 2 — Temporal Pyramid Pooling (TPP):
    3 global descriptors at scales {1, 4, 8} → concat → Dense.
    Captures fast spike (~6ms) and slow wave (~46ms) context.
    """
    reg     = regularizers.l2(L2)
    filters = int(x.shape[-1])

    d1 = layers.GlobalAveragePooling1D(name=f'{name_prefix}_d1')(x)
    p4 = layers.AveragePooling1D(4, strides=4, padding='same',
                                  name=f'{name_prefix}_p4')(x)
    d4 = layers.GlobalAveragePooling1D(name=f'{name_prefix}_d4')(p4)
    p8 = layers.AveragePooling1D(8, strides=8, padding='same',
                                  name=f'{name_prefix}_p8')(x)
    d8 = layers.GlobalAveragePooling1D(name=f'{name_prefix}_d8')(p8)

    cat = layers.Concatenate(name=f'{name_prefix}_cat')([d1, d4, d8])
    out = layers.Dense(filters, activation='relu',
                       kernel_regularizer=reg,
                       name=f'{name_prefix}_proj')(cat)
    out = layers.Dropout(0.3, name=f'{name_prefix}_drop')(out)
    return out   # (batch, filters)


def compact_temporal_encoder(x, embed_dim, num_heads, ff_dim,
                              name_prefix='cte'):
    """
    Novel 3 — Compact Temporal Encoder (CTE):
    Conv1D(stride=4) patch embedding → MHA → Feed-forward → GAP.
    Captures long-range temporal dependencies at 16× lower cost
    than applying MHA directly to 178 timesteps.
    """
    reg = regularizers.l2(L2)

    # Patch embedding: 178 → ~45 timesteps
    t = layers.Conv1D(embed_dim, kernel_size=8, strides=4,
                      padding='same', kernel_regularizer=reg,
                      name=f'{name_prefix}_embed')(x)
    t = layers.BatchNormalization(momentum=0.9,
                                   name=f'{name_prefix}_bn')(t)
    t = layers.Activation('relu', name=f'{name_prefix}_act')(t)

    # Multi-head self-attention
    attn = layers.MultiHeadAttention(
               num_heads=num_heads,
               key_dim=max(1, embed_dim // num_heads),
               dropout=0.1,
               name=f'{name_prefix}_mha')(t, t)
    t = layers.LayerNormalization(epsilon=1e-6,
                                   name=f'{name_prefix}_ln1')(t + attn)

    # Feed-forward
    ff = layers.Dense(ff_dim, activation='gelu',
                      kernel_regularizer=reg,
                      name=f'{name_prefix}_ff1')(t)
    ff = layers.Dropout(0.1, name=f'{name_prefix}_ffdrop')(ff)
    ff = layers.Dense(embed_dim, kernel_regularizer=reg,
                      name=f'{name_prefix}_ff2')(ff)
    t  = layers.LayerNormalization(epsilon=1e-6,
                                    name=f'{name_prefix}_ln2')(t + ff)

    # Global average pool → compact descriptor
    out = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gap')(t)
    out = layers.Dense(embed_dim, activation='relu',
                       kernel_regularizer=reg,
                       name=f'{name_prefix}_proj')(out)
    return out   # (batch, embed_dim)


def cross_attention_fusion(sp_vec, tp_vec, embed_dim, num_heads,
                            name_prefix='caf'):
    """
    Novel 4 — Bidirectional Cross-Attention Fusion (CAF):
    Spatial → Temporal  and  Temporal → Spatial cross-attention.
    Learnable fusion vs simple concatenation in MASF.
    Each branch selectively attends to what the other found.
    """
    reg = regularizers.l2(L2)

    def _expand(v, name):
        v = layers.Dense(embed_dim, kernel_regularizer=reg,
                          name=f'{name_prefix}_{name}_proj')(v)
        return layers.Reshape((1, embed_dim),
                               name=f'{name_prefix}_{name}_rs')(v)

    sq = _expand(sp_vec, 'sq'); tk = _expand(tp_vec, 'tk')
    tq = _expand(tp_vec, 'tq'); sk = _expand(sp_vec, 'sk')

    s2t = layers.MultiHeadAttention(
              num_heads=num_heads,
              key_dim=max(1, embed_dim // num_heads),
              name=f'{name_prefix}_s2t')(sq, tk)
    s2t = layers.LayerNormalization(epsilon=1e-6,
                                     name=f'{name_prefix}_ln_s2t')(sq + s2t)

    t2s = layers.MultiHeadAttention(
              num_heads=num_heads,
              key_dim=max(1, embed_dim // num_heads),
              name=f'{name_prefix}_t2s')(tq, sk)
    t2s = layers.LayerNormalization(epsilon=1e-6,
                                     name=f'{name_prefix}_ln_t2s')(tq + t2s)

    fused = layers.Add(name=f'{name_prefix}_add')([s2t, t2s])
    fused = layers.LayerNormalization(epsilon=1e-6,
                                       name=f'{name_prefix}_ln_out')(fused)
    return layers.Flatten(name=f'{name_prefix}_flat')(fused)


# =============================================================
# FULL STCA-Net
# =============================================================
def build_stca_net(input_shape, n_classes):
    reg    = regularizers.l2(L2)
    inputs = keras.Input(shape=input_shape, name='eeg_input')

    # ── Spatial branch ────────────────────────────────────────
    sp     = dsca_block(inputs, filters=EMBED_DIM, name_prefix='dsca1')
    sp     = dsca_block(sp,     filters=EMBED_DIM, name_prefix='dsca2')
    sp_vec = temporal_pyramid_pooling(sp, name_prefix='tpp')
    # sp_vec: (batch, EMBED_DIM)

    # ── Temporal branch ───────────────────────────────────────
    tp_vec = compact_temporal_encoder(
                 inputs, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 ff_dim=FF_DIM, name_prefix='cte')
    # tp_vec: (batch, EMBED_DIM)

    # ── Cross-Attention Fusion ────────────────────────────────
    fused = cross_attention_fusion(
                sp_vec, tp_vec, embed_dim=EMBED_DIM,
                num_heads=NUM_HEADS, name_prefix='caf')
    # fused: (batch, EMBED_DIM)

    # ── Residual combination ──────────────────────────────────
    sp_r     = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                             name='sp_rp')(sp_vec)
    tp_r     = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                             name='tp_rp')(tp_vec)
    combined = layers.Add(name='res_add')([fused, sp_r, tp_r])
    combined = layers.LayerNormalization(epsilon=1e-6,
                                          name='final_ln')(combined)

    # ── Classification head ───────────────────────────────────
    x = layers.Dense(128, activation='gelu',
                     kernel_regularizer=reg, name='fc1')(combined)
    x = layers.Dropout(DROPOUT, name='drop1')(x)
    x = layers.Dense(64, activation='gelu',
                     kernel_regularizer=reg, name='fc2')(x)
    x = layers.Dropout(DROPOUT * 0.6, name='drop2')(x)

    if n_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        outputs = layers.Dense(n_classes, activation='softmax',
                               name='output')(x)

    return Model(inputs=inputs, outputs=outputs, name='STCA-Net-v2')


# =============================================================
# LABEL-SMOOTHED LOSS
# =============================================================
def smoothed_loss(y_true, y_pred, n_classes, cw_tensor,
                  smooth=LABEL_SMOOTH):
    """
    Sparse categorical cross-entropy with:
      - Label smoothing  (reduces overconfidence)
      - Per-sample class weighting  (handles imbalance)
    """
    y_true_int = tf.cast(y_true, tf.int32)
    y_onehot   = tf.one_hot(y_true_int, n_classes)

    # Label smoothing
    y_smooth   = y_onehot * (1 - smooth) + smooth / n_classes
    per_sample = -tf.reduce_sum(
        y_smooth * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)),
        axis=-1)

    w    = tf.gather(cw_tensor, y_true_int)
    return tf.reduce_mean(per_sample * w)


def binary_smoothed_loss(y_true, y_pred, cw_tensor, smooth=LABEL_SMOOTH):
    y_f    = tf.cast(y_true, tf.float32)
    y_s    = y_f * (1 - smooth) + smooth * 0.5
    bce    = -(y_s * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
               + (1 - y_s) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-7, 1.0)))
    w      = tf.gather(cw_tensor, tf.cast(y_true, tf.int32))
    return tf.reduce_mean(bce * w)


# =============================================================
# METRICS
# =============================================================
def compute_metrics(y_true, y_pred_proba, n_classes):
    if n_classes == 2:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        avg = 'binary'
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
        avg = 'weighted'
    return {
        'accuracy' : accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred,
                                     average=avg, zero_division=0) * 100,
        'recall'   : recall_score(y_true, y_pred,
                                  average=avg, zero_division=0) * 100,
        'f1'       : f1_score(y_true, y_pred,
                              average=avg, zero_division=0) * 100,
        'mcc'      : matthews_corrcoef(y_true, y_pred) * 100,
        'y_pred'   : y_pred,
    }


# =============================================================
# 10-FOLD CV TRAINING  (fast tf.function loop)
# =============================================================
def train_with_cv(task_name, n_folds=N_FOLDS):
    print("\n" + "=" * 60)
    print(f"Training STCA-Net v2  —  Task: {task_name}")
    print("=" * 60)

    X      = np.load(os.path.join(DATA_DIR, f"{task_name}_X.npy"))
    y      = np.load(os.path.join(DATA_DIR, f"{task_name}_y.npy"))
    splits = np.load(os.path.join(DATA_DIR, f"{task_name}_splits.npy"),
                     allow_pickle=True)

    n_classes   = len(np.unique(y))
    input_shape = X.shape[1:]
    binary      = (n_classes == 2)
    print(f"  X: {X.shape}  y: {y.shape}  classes: {n_classes}")

    fold_results, all_y_true, all_y_pred = [], [], []
    first_history = None

    for fold_idx in range(n_folds):
        t0_fold = time.time()
        print(f"\n  ── Fold {fold_idx+1}/{n_folds} ──────────────────────────")

        tr = splits[fold_idx]['train']
        te = splits[fold_idx]['test']
        vc = int(len(tr) * 0.9)

        X_tr  = X[tr[:vc]].astype('float32');  y_tr  = y[tr[:vc]]
        X_val = X[tr[vc:]].astype('float32');  y_val = y[tr[vc:]]
        X_te  = X[te].astype('float32');       y_te  = y[te]

        cw   = compute_class_weight('balanced',
                                     classes=np.unique(y_tr), y=y_tr)
        cw_t = tf.constant(cw, dtype=tf.float32)

        # Steps per epoch for LR schedule
        steps_per_ep = max(1, len(X_tr) // BATCH_SIZE)
        total_steps  = EPOCHS * steps_per_ep
        warmup_steps = WARMUP_EP * steps_per_ep

        schedule = WarmupCosineDecay(LR_MAX, LR_MIN,
                                     warmup_steps, total_steps)
        optimizer = keras.optimizers.AdamW(
                        learning_rate=schedule,
                        weight_decay=L2,
                        clipnorm=1.0)

        ds_tr = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                 .shuffle(len(X_tr), seed=RANDOM_SEED + fold_idx)
                 .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
        ds_val = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                  .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

        model = build_stca_net(input_shape, n_classes)
        if fold_idx == 0:
            print(f"    Parameters: {model.count_params():,}")

        # ── Compiled training step ─────────────────────────────
        @tf.function
        def train_step(xb, yb):
            with tf.GradientTape() as tape:
                pred = model(xb, training=True)
                if binary:
                    pred_s = tf.squeeze(pred, -1)
                    loss   = binary_smoothed_loss(yb, pred_s, cw_t)
                else:
                    loss   = smoothed_loss(yb, pred, n_classes, cw_t)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        @tf.function
        def val_loss_step(xb, yb):
            pred = model(xb, training=False)
            if binary:
                pred_s = tf.squeeze(pred, -1)
                yf = tf.cast(yb, tf.float32)
                return tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(yf, pred_s))
            return tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(yb, pred))

        # ── Training loop ──────────────────────────────────────
        best_val     = np.inf
        patience_ctr = 0
        best_weights = model.get_weights()
        tr_losses, vl_losses, tr_accs, vl_accs = [], [], [], []

        for ep in range(EPOCHS):
            ep_losses = [train_step(xb, yb).numpy() for xb, yb in ds_tr]
            tl = np.mean(ep_losses)

            vl_vals = [val_loss_step(xb, yb).numpy() for xb, yb in ds_val]
            vl = np.mean(vl_vals)

            # Accuracy (subset for speed on CPU)
            sub = min(512, len(X_tr))
            p_tr = model.predict(X_tr[:sub], verbose=0, batch_size=sub)
            p_vl = model.predict(X_val,       verbose=0, batch_size=BATCH_SIZE)
            if binary:
                a_tr = accuracy_score(y_tr[:sub], (p_tr>0.5).astype(int).flatten())
                a_vl = accuracy_score(y_val,      (p_vl>0.5).astype(int).flatten())
            else:
                a_tr = accuracy_score(y_tr[:sub], np.argmax(p_tr, 1))
                a_vl = accuracy_score(y_val,      np.argmax(p_vl, 1))

            tr_losses.append(tl); vl_losses.append(vl)
            tr_accs.append(a_tr); vl_accs.append(a_vl)

            if (ep + 1) % 10 == 0 or ep == 0:
                elapsed = time.time() - t0_fold
                print(f"    ep={ep+1:3d}  tl={tl:.4f}  vl={vl:.4f}  "
                      f"val_acc={a_vl*100:.1f}%  ({elapsed:.0f}s)")

            if vl < best_val - 1e-4:
                best_val     = vl
                best_weights = model.get_weights()
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"    Early stop at epoch {ep+1}")
                    break

        model.set_weights(best_weights)
        n_ep = ep + 1

        if fold_idx == 0:
            first_history = {
                'loss': tr_losses, 'val_loss': vl_losses,
                'accuracy': tr_accs, 'val_accuracy': vl_accs
            }

        proba = model.predict(X_te, verbose=0, batch_size=BATCH_SIZE)
        m_res = compute_metrics(y_te, proba, n_classes)
        fold_results.append(m_res)
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(m_res['y_pred'].tolist())

        elapsed = time.time() - t0_fold
        print(f"\n    ✓ Fold {fold_idx+1}:  "
              f"Acc={m_res['accuracy']:.2f}%  "
              f"Prec={m_res['precision']:.2f}%  "
              f"Rec={m_res['recall']:.2f}%  "
              f"F1={m_res['f1']:.2f}%  "
              f"MCC={m_res['mcc']:.2f}%  "
              f"({elapsed:.0f}s, {n_ep} epochs)")

        keras.backend.clear_session()

    # ── Aggregate across folds ─────────────────────────────────
    mn  = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
    agg = {k: {'mean': np.mean([r[k] for r in fold_results]),
               'std':  np.std( [r[k] for r in fold_results])}
           for k in mn}

    print(f"\n  {'─'*50}")
    print(f"  {'Metric':<12} {'Mean (%)':>10}  {'±Std':>8}")
    print(f"  {'─'*50}")
    for k in mn:
        print(f"  {k.capitalize():<12} "
              f"{agg[k]['mean']:>9.2f}%  ±{agg[k]['std']:>5.2f}%")
    print(f"  {'─'*50}")

    _save_results(task_name, agg, fold_results)
    _save_confusion_matrix(all_y_true, all_y_pred, task_name, n_classes)
    _save_per_fold_bar(fold_results, task_name)
    if first_history:
        _save_training_curve(first_history, task_name)

    return agg


# =============================================================
# ABLATION STUDY  (for paper Table)
# =============================================================
def run_ablation(task_name='5class', n_folds=5):
    """
    Test each novel component independently.
    Uses 5-fold (half) for speed; full 10-fold for final model.

    Variants tested:
      A) Baseline CNN (no attention, no transformer)
      B) DSCA only (spatial branch only, no TPP, no CTE, no CAF)
      C) CTE only  (temporal branch only)
      D) DSCA + CTE, concatenated  (no cross-attention)
      E) Full STCA-Net (DSCA + TPP + CTE + CAF)
    """
    print("\n" + "=" * 60)
    print(f"Ablation Study  —  Task: {task_name}  ({n_folds}-fold)")
    print("=" * 60)

    X      = np.load(os.path.join(DATA_DIR, f"{task_name}_X.npy"))
    y      = np.load(os.path.join(DATA_DIR, f"{task_name}_y.npy"))
    splits = np.load(os.path.join(DATA_DIR, f"{task_name}_splits.npy"),
                     allow_pickle=True)
    n_classes   = len(np.unique(y))
    input_shape = X.shape[1:]

    def _build_variant(variant):
        reg    = regularizers.l2(L2)
        inputs = keras.Input(shape=input_shape)

        if variant == 'A_baseline':
            # Simple CNN baseline
            x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
            x = layers.Conv1D(32, 5, padding='same', activation='relu')(x)
            x = layers.GlobalAveragePooling1D()(x)

        elif variant == 'B_dsca_only':
            x = dsca_block(inputs, EMBED_DIM, 'ab_d1')
            x = dsca_block(x,      EMBED_DIM, 'ab_d2')
            x = layers.GlobalAveragePooling1D()(x)

        elif variant == 'C_cte_only':
            x = compact_temporal_encoder(inputs, EMBED_DIM,
                                         NUM_HEADS, FF_DIM, 'ab_c')

        elif variant == 'D_dsca_cte_concat':
            sp = dsca_block(inputs, EMBED_DIM, 'ab_d1')
            sp = layers.GlobalAveragePooling1D()(sp)
            tp = compact_temporal_encoder(inputs, EMBED_DIM,
                                          NUM_HEADS, FF_DIM, 'ab_t')
            x  = layers.Concatenate()([sp, tp])

        else:  # E = full STCA-Net
            return build_stca_net(input_shape, n_classes)

        x = layers.Dense(64, activation='relu',
                          kernel_regularizer=reg)(x)
        x = layers.Dropout(DROPOUT)(x)
        if n_classes == 2:
            out = layers.Dense(1, activation='sigmoid')(x)
        else:
            out = layers.Dense(n_classes, activation='softmax')(x)
        return Model(inputs=inputs, outputs=out)

    variants = {
        'A_Baseline_CNN':   'A_baseline',
        'B_DSCA_only':      'B_dsca_only',
        'C_CTE_only':       'C_cte_only',
        'D_DSCA+CTE_concat':'D_dsca_cte_concat',
        'E_Full_STCA-Net':  'full',
    }

    abl_rows = []
    for v_name, v_key in variants.items():
        fold_accs, fold_f1s = [], []
        for fold_idx in range(n_folds):
            tr = splits[fold_idx]['train']
            te = splits[fold_idx]['test']
            vc = int(len(tr)*0.9)
            X_tr = X[tr[:vc]].astype('float32'); y_tr = y[tr[:vc]]
            X_te = X[te].astype('float32');      y_te = y[te]

            cw  = compute_class_weight('balanced',
                                        classes=np.unique(y_tr), y=y_tr)
            cwd = dict(enumerate(cw))

            m  = _build_variant(v_key)
            lf = 'binary_crossentropy' if n_classes==2 \
                 else 'sparse_categorical_crossentropy'
            m.compile(optimizer=keras.optimizers.Adam(3e-4),
                      loss=lf, metrics=['accuracy'])
            m.fit(X_tr, y_tr, validation_split=0.1, epochs=50,
                  batch_size=BATCH_SIZE, class_weight=cwd, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(
                      patience=10, restore_best_weights=True)])
            proba = m.predict(X_te, verbose=0, batch_size=BATCH_SIZE)
            res   = compute_metrics(y_te, proba, n_classes)
            fold_accs.append(res['accuracy'])
            fold_f1s.append(res['f1'])
            keras.backend.clear_session()

        abl_rows.append({
            'Variant': v_name,
            'Accuracy (%)': f"{np.mean(fold_accs):.2f}±{np.std(fold_accs):.2f}",
            'F1-score (%)': f"{np.mean(fold_f1s):.2f}±{np.std(fold_f1s):.2f}",
        })
        print(f"  {v_name:<28}  "
              f"Acc={np.mean(fold_accs):.2f}%  F1={np.mean(fold_f1s):.2f}%")

    df = pd.DataFrame(abl_rows)
    p  = os.path.join(OUTPUT_DIR, f'ablation_{task_name}.csv')
    df.to_csv(p, index=False)
    print(f"\n  Ablation saved: {p}")
    return df


# =============================================================
# PLOTS
# =============================================================
def _save_training_curve(history, task_name):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history['loss'])+1)
    a1.plot(ep, history['loss'],     color='#D85A30', label='Train', lw=1.5)
    a1.plot(ep, history['val_loss'], color='#378ADD', label='Val',
            linestyle='--', lw=1.5)
    a1.set_title(f'Loss — {task_name}', fontsize=11, fontweight='bold')
    a1.set_xlabel('Epoch'); a1.set_ylabel('Loss')
    a1.legend(); a1.spines[['top','right']].set_visible(False)

    a2.plot(ep, [a*100 for a in history['accuracy']],
            color='#D85A30', label='Train', lw=1.5)
    a2.plot(ep, [a*100 for a in history['val_accuracy']],
            color='#378ADD', label='Val', linestyle='--', lw=1.5)
    a2.set_title(f'Accuracy — {task_name}', fontsize=11, fontweight='bold')
    a2.set_xlabel('Epoch'); a2.set_ylabel('Accuracy (%)')
    a2.legend(); a2.spines[['top','right']].set_visible(False)

    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, f'training_curve_{task_name}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Plot: {p}")


def _save_confusion_matrix(y_true, y_pred, task_name, n_classes):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    names  = {
        '2class': ['Non-seizure', 'Seizure'],
        '3class': ['Healthy', 'Interictal', 'Seizure'],
        '5class': ['Seizure', 'D-interictal', 'C-interictal',
                   'Eyes-closed', 'Eyes-open']
    }.get(task_name, [str(i) for i in range(n_classes)])

    fig, ax = plt.subplots(figsize=(max(5, n_classes*2),
                                    max(4, n_classes*1.8)))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=names, yticklabels=names,
                linewidths=0.5, ax=ax, cbar_kws={'label': '%'})
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(f'Confusion matrix (%) — {task_name} — 10-fold avg',
                 fontsize=11, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, f'confusion_matrix_{task_name}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Confusion matrix: {p}")


def _save_per_fold_bar(fold_results, task_name):
    accs  = [r['accuracy'] for r in fold_results]
    f1s   = [r['f1']       for r in fold_results]
    x, w  = np.arange(len(fold_results)), 0.35
    folds = [f'F{i+1}' for i in range(len(fold_results))]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x-w/2, accs, w, label='Accuracy', color='#378ADD', edgecolor='white')
    ax.bar(x+w/2, f1s,  w, label='F1-score', color='#D85A30', edgecolor='white')
    ax.axhline(np.mean(accs), color='#378ADD', ls='--', lw=1, alpha=0.8,
               label=f'μAcc={np.mean(accs):.1f}%')
    ax.axhline(np.mean(f1s),  color='#D85A30', ls='--', lw=1, alpha=0.8,
               label=f'μF1={np.mean(f1s):.1f}%')
    ax.set_xticks(x); ax.set_xticklabels(folds, fontsize=8)
    ax.set_ylabel('Score (%)'); ax.set_ylim(max(0, min(accs+f1s)-10), 105)
    ax.set_title(f'Per-fold results — {task_name}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, f'per_fold_{task_name}.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Per-fold: {p}")


def _save_results(task_name, agg, fold_results):
    rows = [{'fold': i+1, **{k: f"{r[k]:.2f}"
             for k in ['accuracy','precision','recall','f1','mcc']}}
            for i, r in enumerate(fold_results)]
    rows.append({'fold': 'Mean±Std', **{
        k: f"{agg[k]['mean']:.2f}±{agg[k]['std']:.2f}"
        for k in ['accuracy','precision','recall','f1','mcc']}})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, f'results_{task_name}.csv'), index=False)
    print(f"  Results: {OUTPUT_DIR}/results_{task_name}.csv")


def save_comparison_table(all_results):
    print("\n" + "=" * 60)
    print("Final comparison  (STCA-Net v2 vs MASF)")
    print("=" * 60)

    # MASF paper Table 5 & 6 values (10-fold CV on Bonn)
    masf_baselines = [
        ('DNN',          '5class', '40.84±1.99', '40.33±1.70', '26.29±2.53'),
        ('CNN',          '5class', '59.30±1.86', '58.92±1.48', '49.77±2.55'),
        ('CNN-RNN',      '5class', '47.97±2.01', '47.63±1.67', '35.12±2.59'),
        ('KNN',          '5class', '47.58±1.73', '46.19±2.06', '36.72±2.17'),
        ('MASF',         '5class', '72.50±1.45', '72.62±1.47', '66.25±2.56'),
    ]

    rows = []
    for task, agg in all_results.items():
        rows.append({
            'Method': 'STCA-Net (ours)', 'Task': task,
            'Accuracy': f"{agg['accuracy']['mean']:.2f}±{agg['accuracy']['std']:.2f}",
            'F1-score': f"{agg['f1']['mean']:.2f}±{agg['f1']['std']:.2f}",
            'MCC':      f"{agg['mcc']['mean']:.2f}±{agg['mcc']['std']:.2f}",
        })
    for name, task, acc, f1, mcc in masf_baselines:
        rows.append({'Method': name, 'Task': task,
                     'Accuracy': acc, 'F1-score': f1, 'MCC': mcc})

    df = pd.DataFrame(rows)
    p  = os.path.join(OUTPUT_DIR, 'comparison_table.csv')
    df.to_csv(p, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n  Saved: {p}")
    return df


# =============================================================
# MAIN
# =============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("STCA-Net v2 — Full Training Pipeline")
    print(f"TensorFlow {tf.__version__}  |  "
          f"{'GPU' if GPU_AVAILABLE else 'CPU'}")
    print("=" * 60)

    all_results = {}
    for task in ['5class', '3class', '2class']:
        all_results[task] = train_with_cv(task)

    save_comparison_table(all_results)

    # Ablation on 5-class with 5 folds (efficient)
    print("\nRunning ablation study...")
    run_ablation('5class', n_folds=5)

    print("\n" + "=" * 60)
    print("Phase 2 complete. Results in results/ and plots/")
    print("=" * 60)
