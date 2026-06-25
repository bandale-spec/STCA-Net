"""
=============================================================
STCA-Net  |  Phase 6: Save Model + Inference Engine
=============================================================
This file does TWO things:

PART A — Save Model
  Re-trains STCA-Net on the FULL dataset (all 10 folds combined)
  for each task and saves the best model as .keras file.
  Why full dataset? Because the saved model used in the app
  should learn from ALL available data, not just 9/10 folds.

PART B — Inference Engine
  A clean, self-contained inference function that:
    1. Takes a raw EEG CSV or numpy array as input
    2. Runs preprocessing (z-score normalisation, windowing)
    3. Runs prediction on all 3 task variants simultaneously
    4. Returns prediction + confidence + class probabilities
    5. This is what Phase 7 (Streamlit app) will call

HOW TO RUN:
  python phase6_save_model.py

Output:
  models/stca_net_2class.keras
  models/stca_net_3class.keras
  models/stca_net_5class.keras
  inference_engine.py  (auto-generated, used by Phase 7)
=============================================================
"""

import os
import sys
import time
import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
DATA_DIR  = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  CONFIG  (must match Phase 2)
# ─────────────────────────────────────────────
EMBED_DIM  = 32
NUM_HEADS  = 4
FF_DIM     = 64
DROPOUT    = 0.5
L2         = 1e-4
LR_MAX     = 3e-4
LR_MIN     = 1e-6
WARMUP_EP  = 10
EPOCHS     = 150
BATCH_SIZE = 512
PATIENCE   = 20
LABEL_SMOOTH = 0.1
RANDOM_SEED  = 42


# =============================================================
# MODEL ARCHITECTURE  (identical to Phase 2)
# =============================================================

def dsca_block(x, filters, pfx):
    reg = regularizers.l2(L2)
    branches = []
    for k in [3, 5, 7]:
        b = layers.DepthwiseConv1D(
                k, padding='same', depthwise_regularizer=reg,
                name=f'{pfx}_dw{k}')(x)
        b = layers.Conv1D(
                filters, 1, kernel_regularizer=reg,
                name=f'{pfx}_pw{k}')(b)
        b = layers.BatchNormalization(
                momentum=0.9, name=f'{pfx}_bn{k}')(b)
        b = layers.Activation('relu', name=f'{pfx}_a{k}')(b)
        branches.append(b)
    f = layers.Add(name=f'{pfx}_add')(branches)
    ca = layers.GlobalAveragePooling1D(name=f'{pfx}_gap')(f)
    ca = layers.Dense(max(1, filters//4), activation='relu',
                      kernel_regularizer=reg, name=f'{pfx}_c1')(ca)
    ca = layers.Dense(filters, activation='sigmoid',
                      kernel_regularizer=reg, name=f'{pfx}_c2')(ca)
    ca = layers.Reshape((1, filters), name=f'{pfx}_cr')(ca)
    f  = layers.Multiply(name=f'{pfx}_cm')([f, ca])
    sa_a = layers.Lambda(
               lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
               name=f'{pfx}_saa')(f)
    sa_m = layers.Lambda(
               lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
               name=f'{pfx}_sam')(f)
    sa   = layers.Concatenate(name=f'{pfx}_sac')([sa_a, sa_m])
    sa   = layers.Conv1D(1, 7, padding='same', activation='sigmoid',
                         kernel_regularizer=reg, name=f'{pfx}_sv')(sa)
    f    = layers.Multiply(name=f'{pfx}_sm')([f, sa])
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, kernel_regularizer=reg,
                          name=f'{pfx}_res')(x)
    return layers.Add(name=f'{pfx}_out')([f, x])


def tpp_block(x, pfx):
    reg = regularizers.l2(L2)
    flt = int(x.shape[-1])
    d1  = layers.GlobalAveragePooling1D(name=f'{pfx}_d1')(x)
    p4  = layers.AveragePooling1D(4, strides=4, padding='same',
                                   name=f'{pfx}_p4')(x)
    d4  = layers.GlobalAveragePooling1D(name=f'{pfx}_d4')(p4)
    p8  = layers.AveragePooling1D(8, strides=8, padding='same',
                                   name=f'{pfx}_p8')(x)
    d8  = layers.GlobalAveragePooling1D(name=f'{pfx}_d8')(p8)
    cat = layers.Concatenate(name=f'{pfx}_cat')([d1, d4, d8])
    out = layers.Dense(flt, activation='relu',
                       kernel_regularizer=reg,
                       name=f'{pfx}_proj')(cat)
    return layers.Dropout(0.3, name=f'{pfx}_drop')(out)


def cte_block(x, pfx):
    reg  = regularizers.l2(L2)
    t    = layers.Conv1D(EMBED_DIM, 8, strides=4, padding='same',
                          kernel_regularizer=reg,
                          name=f'{pfx}_emb')(x)
    t    = layers.BatchNormalization(momentum=0.9,
                                      name=f'{pfx}_bn')(t)
    t    = layers.Activation('relu', name=f'{pfx}_act')(t)
    attn = layers.MultiHeadAttention(
               NUM_HEADS, max(1, EMBED_DIM//NUM_HEADS),
               dropout=0.1, name=f'{pfx}_mha')(t, t)
    t    = layers.LayerNormalization(
               epsilon=1e-6, name=f'{pfx}_ln1')(t + attn)
    ff   = layers.Dense(FF_DIM, activation='gelu',
                         kernel_regularizer=reg,
                         name=f'{pfx}_ff1')(t)
    ff   = layers.Dropout(0.1, name=f'{pfx}_ffd')(ff)
    ff   = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                         name=f'{pfx}_ff2')(ff)
    t    = layers.LayerNormalization(
               epsilon=1e-6, name=f'{pfx}_ln2')(t + ff)
    out  = layers.GlobalAveragePooling1D(name=f'{pfx}_gap')(t)
    return layers.Dense(EMBED_DIM, activation='relu',
                         kernel_regularizer=reg,
                         name=f'{pfx}_proj')(out)


def caf_block(sp, tp, pfx):
    reg = regularizers.l2(L2)
    kd  = max(1, EMBED_DIM // NUM_HEADS)

    def _expand(v, n):
        v = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                          name=f'{pfx}_{n}')(v)
        return layers.Reshape((1, EMBED_DIM),
                               name=f'{pfx}_{n}r')(v)

    sq = _expand(sp, 'sq'); tk = _expand(tp, 'tk')
    tq = _expand(tp, 'tq'); sk = _expand(sp, 'sk')

    s2t = layers.MultiHeadAttention(
              NUM_HEADS, kd, name=f'{pfx}_s2t')(sq, tk)
    s2t = layers.LayerNormalization(
              epsilon=1e-6, name=f'{pfx}_ls')(sq + s2t)
    t2s = layers.MultiHeadAttention(
              NUM_HEADS, kd, name=f'{pfx}_t2s')(tq, sk)
    t2s = layers.LayerNormalization(
              epsilon=1e-6, name=f'{pfx}_lt')(tq + t2s)
    fused = layers.LayerNormalization(
                epsilon=1e-6, name=f'{pfx}_lo')(
                layers.Add(name=f'{pfx}_add')([s2t, t2s]))
    return layers.Flatten(name=f'{pfx}_flat')(fused)


def build_stca_net(input_shape, n_classes):
    reg    = regularizers.l2(L2)
    inputs = keras.Input(shape=input_shape, name='eeg_input')
    sp     = dsca_block(inputs, EMBED_DIM, 'd1')
    sp     = dsca_block(sp,     EMBED_DIM, 'd2')
    sp_vec = tpp_block(sp, 'tpp')
    tp_vec = cte_block(inputs, 'cte')
    fused  = caf_block(sp_vec, tp_vec, 'caf')
    sp_r   = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                           name='spr')(sp_vec)
    tp_r   = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                           name='tpr')(tp_vec)
    c      = layers.LayerNormalization(
                 epsilon=1e-6, name='fln')(
                 layers.Add(name='ra')([fused, sp_r, tp_r]))
    x = layers.Dropout(DROPOUT, name='dr1')(
            layers.Dense(128, activation='gelu',
                          kernel_regularizer=reg, name='fc1')(c))
    x = layers.Dropout(DROPOUT * 0.6, name='dr2')(
            layers.Dense(64, activation='gelu',
                          kernel_regularizer=reg, name='fc2')(x))
    out = (layers.Dense(1, activation='sigmoid', name='out')(x)
           if n_classes == 2
           else layers.Dense(n_classes, activation='softmax',
                              name='out')(x))
    return Model(inputs=inputs, outputs=out, name='STCA-Net')


# =============================================================
# LR SCHEDULE  (identical to Phase 2)
# =============================================================

class WarmupCosineDecay(
        keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, warmup_steps, total_steps):
        super().__init__()
        self.lr_max       = tf.cast(lr_max,       tf.float32)
        self.lr_min       = tf.cast(lr_min,       tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps  = tf.cast(total_steps,  tf.float32)

    def __call__(self, step):
        step    = tf.cast(step, tf.float32)
        wlr     = self.lr_max * (step / self.warmup_steps)
        ca      = ((step - self.warmup_steps) /
                   (self.total_steps - self.warmup_steps))
        clr     = (self.lr_min + 0.5 *
                   (self.lr_max - self.lr_min) *
                   (1.0 + tf.math.cos(
                       np.pi * tf.minimum(ca, 1.0))))
        return tf.where(step < self.warmup_steps, wlr, clr)

    def get_config(self):
        return {
            'lr_max':       float(self.lr_max),
            'lr_min':       float(self.lr_min),
            'warmup_steps': int(self.warmup_steps),
            'total_steps':  int(self.total_steps)
        }


# =============================================================
# PART A — TRAIN ON FULL DATASET AND SAVE
# =============================================================

def train_and_save(task_name):
    """
    Train STCA-Net on the complete dataset (90% train / 10% val)
    and save the best model to models/stca_net_{task}.keras

    This is different from Phase 2 which used 10-fold CV for
    evaluation. Here we use ALL data to get the strongest
    possible model for deployment in the clinical app.
    """
    print(f"\n{'='*55}")
    print(f"Training full model for task: {task_name}")
    print(f"{'='*55}")

    X_path = os.path.join(DATA_DIR, f'{task_name}_X.npy')
    y_path = os.path.join(DATA_DIR, f'{task_name}_y.npy')

    if not os.path.exists(X_path):
        print(f"  [!] Data not found: {X_path}")
        print(f"      Run Phase 1 first.")
        return None

    X = np.load(X_path).astype('float32')
    y = np.load(y_path).astype('int32')
    n_classes = len(np.unique(y))
    print(f"  Dataset: {X.shape}  Classes: {n_classes}")

    # 90/10 train/val split
    np.random.seed(RANDOM_SEED)
    idx     = np.random.permutation(len(X))
    val_cut = int(len(idx) * 0.9)
    tr_idx  = idx[:val_cut]
    vl_idx  = idx[val_cut:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_vl, y_vl = X[vl_idx], y[vl_idx]

    print(f"  Train: {len(X_tr)}  Val: {len(X_vl)}")

    # Class weights
    cw   = compute_class_weight('balanced',
                                 classes=np.unique(y_tr),
                                 y=y_tr)
    cw_t = tf.constant(cw, dtype=tf.float32)

    # LR schedule
    spe          = max(1, len(X_tr) // BATCH_SIZE)
    total_steps  = EPOCHS * spe
    warmup_steps = WARMUP_EP * spe
    schedule     = WarmupCosineDecay(
                       LR_MAX, LR_MIN, warmup_steps, total_steps)
    optimizer    = keras.optimizers.AdamW(
                       learning_rate=schedule,
                       weight_decay=L2,
                       clipnorm=1.0)

    # Datasets
    ds_tr = (tf.data.Dataset
             .from_tensor_slices((X_tr, y_tr))
             .shuffle(len(X_tr), seed=RANDOM_SEED)
             .batch(BATCH_SIZE)
             .prefetch(tf.data.AUTOTUNE))
    ds_vl = (tf.data.Dataset
             .from_tensor_slices((X_vl, y_vl))
             .batch(BATCH_SIZE)
             .prefetch(tf.data.AUTOTUNE))

    binary = (n_classes == 2)
    model  = build_stca_net((178, 1), n_classes)
    print(f"  Parameters: {model.count_params():,}")

    # ── Training step ─────────────────────────────────────────
    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            pred = model(xb, training=True)
            if binary:
                ps   = tf.squeeze(pred, -1)
                yoh  = tf.cast(yb, tf.float32)
                ys   = yoh * (1-LABEL_SMOOTH) + LABEL_SMOOTH*0.5
                bce  = -(ys * tf.math.log(
                             tf.clip_by_value(ps, 1e-7, 1.0)) +
                         (1-ys) * tf.math.log(
                             tf.clip_by_value(1-ps, 1e-7, 1.0)))
                w    = tf.gather(cw_t, tf.cast(yb, tf.int32))
                loss = tf.reduce_mean(bce * w)
            else:
                y_oh  = tf.one_hot(tf.cast(yb, tf.int32), n_classes)
                ys    = y_oh*(1-LABEL_SMOOTH)+LABEL_SMOOTH/n_classes
                ps    = -(tf.reduce_sum(
                              ys * tf.math.log(
                                  tf.clip_by_value(pred, 1e-7, 1.0)),
                              axis=-1))
                w     = tf.gather(cw_t, tf.cast(yb, tf.int32))
                loss  = tf.reduce_mean(ps * w)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(xb, yb):
        pred = model(xb, training=False)
        if binary:
            return tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.cast(yb, tf.float32),
                    tf.squeeze(pred, -1)))
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                yb, pred))

    # ── Training loop ──────────────────────────────────────────
    best_val     = np.inf
    patience_ctr = 0
    best_weights = model.get_weights()
    t0           = time.time()

    print(f"\n  {'Ep':>4}  {'TrainLoss':>10}  "
          f"{'ValLoss':>10}  {'ValAcc':>8}  {'Time':>6}")
    print(f"  {'-'*48}")

    for ep in range(EPOCHS):
        tl = np.mean([train_step(xb, yb).numpy()
                      for xb, yb in ds_tr])
        vl = np.mean([val_step(xb, yb).numpy()
                      for xb, yb in ds_vl])

        # Val accuracy
        pv = model.predict(X_vl, verbose=0, batch_size=BATCH_SIZE)
        if binary:
            yp = (pv > 0.5).astype(int).flatten()
        else:
            yp = np.argmax(pv, axis=1)
        va = accuracy_score(y_vl, yp) * 100

        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  {ep+1:>4}  {tl:>10.4f}  "
                  f"{vl:>10.4f}  {va:>7.1f}%  "
                  f"{time.time()-t0:>5.0f}s")

        # Early stopping
        if vl < best_val - 1e-4:
            best_val     = vl
            best_weights = model.get_weights()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {ep+1}")
                break

    model.set_weights(best_weights)

    # Final validation accuracy
    pv = model.predict(X_vl, verbose=0, batch_size=BATCH_SIZE)
    if binary:
        yp = (pv > 0.5).astype(int).flatten()
    else:
        yp = np.argmax(pv, axis=1)
    final_acc = accuracy_score(y_vl, yp) * 100
    print(f"\n  Final val accuracy: {final_acc:.2f}%")

    # Save model
    save_path = os.path.join(MODEL_DIR,
                              f'stca_net_{task_name}.keras')
    model.save(save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"  Saved: {save_path}  ({size_mb:.2f} MB)")

    return model, final_acc


# =============================================================
# PART B — WRITE INFERENCE ENGINE
# =============================================================

INFERENCE_ENGINE_CODE = '''"""
=============================================================
STCA-Net  |  Inference Engine
=============================================================
Self-contained inference module used by Phase 7 (Streamlit app).
Load once, call predict() as many times as needed.

Usage:
    from inference_engine import STCANetInference
    engine = STCANetInference()
    result = engine.predict(eeg_array)   # (178,) or (N, 178)
    print(result)
=============================================================
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────
#  CLASS AND LABEL DEFINITIONS
# ─────────────────────────────────────────────
TASK_CONFIG = {
    '2class': {
        'model_file': 'models/stca_net_2class.keras',
        'n_classes':  2,
        'class_names': ['Non-seizure', 'Seizure'],
        'description': 'Binary seizure detection',
        'colors': ['#2AAA6E', '#D85A30'],
    },
    '3class': {
        'model_file': 'models/stca_net_3class.keras',
        'n_classes':  3,
        'class_names': ['Healthy', 'Interictal', 'Seizure'],
        'description': 'Three-class classification',
        'colors': ['#2AAA6E', '#E8902A', '#D85A30'],
    },
    '5class': {
        'model_file': 'models/stca_net_5class.keras',
        'n_classes':  5,
        'class_names': [
            'Seizure (ictal)',
            'Interictal — seizure area (D)',
            'Interictal — healthy area (C)',
            'Healthy — eyes closed (B)',
            'Healthy — eyes open (A)'
        ],
        'description': 'Five-class full classification',
        'colors': ['#D85A30','#E8902A','#2AAA6E','#378ADD','#7F77DD'],
    },
}

# Clinical interpretation text per class
CLINICAL_TEXT = {
    '2class': {
        0: ("No seizure activity detected. EEG patterns appear "
            "within normal or interictal range."),
        1: ("Active seizure activity detected. High-amplitude "
            "irregular discharges identified. Immediate clinical "
            "review recommended."),
    },
    '3class': {
        0: ("EEG appears consistent with a healthy background "
            "pattern. No seizure or interictal activity detected."),
        1: ("Interictal activity detected. Epileptiform discharges "
            "present between seizures. Monitoring recommended."),
        2: ("Active seizure (ictal) activity detected. Abnormal "
            "electrical discharge pattern identified. Immediate "
            "clinical attention recommended."),
    },
    '5class': {
        0: ("Ictal activity detected. Pattern consistent with an "
            "active epileptic seizure."),
        1: ("Interictal activity from seizure onset zone. "
            "Epileptiform discharges from the epileptogenic area."),
        2: ("Interictal activity from non-seizure area. "
            "Seizure-free recording from contralateral region."),
        3: ("Healthy EEG — eyes closed. Alpha-wave dominant "
            "pattern consistent with relaxed wakefulness."),
        4: ("Healthy EEG — eyes open. Low-amplitude pattern "
            "consistent with eyes-open alert state."),
    },
}

URGENCY_LEVEL = {
    '2class':  {0: 'LOW',    1: 'HIGH'},
    '3class':  {0: 'LOW',    1: 'MEDIUM', 2: 'HIGH'},
    '5class':  {0: 'HIGH',   1: 'MEDIUM', 2: 'LOW',
                3: 'LOW',    4: 'LOW'},
}

URGENCY_COLORS = {
    'LOW':    '#2AAA6E',
    'MEDIUM': '#E8902A',
    'HIGH':   '#D85A30',
}


class STCANetInference:
    """
    Main inference class. Load once, predict many times.

    Example:
        engine = STCANetInference(model_dir='models')
        result = engine.predict(eeg_segment)
        print(result['5class']['predicted_class'])
        print(result['5class']['confidence'])
    """

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models    = {}
        self._load_all_models()

    def _load_all_models(self):
        """Load all three task models at startup."""
        print("Loading STCA-Net models...")
        for task, cfg in TASK_CONFIG.items():
            path = os.path.join(self.model_dir,
                                os.path.basename(cfg['model_file']))
            if os.path.exists(path):
                self.models[task] = keras.models.load_model(path)
                print(f"  [OK] {task}: {path}")
            else:
                print(f"  [!] {task}: model not found at {path}")
                print(f"      Run Phase 6 first to generate saved models.")
        print(f"Loaded {len(self.models)}/3 models.\\n")

    def preprocess(self, eeg_input):
        """
        Preprocess raw EEG input for inference.

        Accepts:
          - (178,)     numpy array — single segment
          - (N, 178)   numpy array — batch of segments
          - (178, 1)   numpy array — already shaped

        Returns:
          - (N, 178, 1) float32 array, z-score normalised
        """
        arr = np.array(eeg_input, dtype=np.float32)

        # Handle shape
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]        # (1, 178)
        if arr.ndim == 2 and arr.shape[-1] != 1:
            pass                            # (N, 178) — correct
        if arr.ndim == 3:
            arr = arr[:, :, 0]             # (N, 178, 1) → (N, 178)

        # Trim or pad to 178 samples
        if arr.shape[1] > 178:
            arr = arr[:, :178]
        elif arr.shape[1] < 178:
            pad = 178 - arr.shape[1]
            arr = np.pad(arr, ((0,0),(0,pad)), mode='edge')

        # Per-record z-score normalisation
        means = arr.mean(axis=1, keepdims=True)
        stds  = arr.std(axis=1,  keepdims=True) + 1e-8
        arr   = (arr - means) / stds

        # Add channel dimension
        return arr[:, :, np.newaxis]       # (N, 178, 1)

    def predict_single_task(self, eeg_processed, task):
        """
        Run inference for one task on preprocessed input.

        Returns dict with:
          predicted_class, class_name, confidence,
          probabilities, clinical_text, urgency
        """
        if task not in self.models:
            return None

        cfg    = TASK_CONFIG[task]
        model  = self.models[task]
        binary = cfg['n_classes'] == 2

        proba = model.predict(
            eeg_processed, verbose=0, batch_size=256)

        if binary:
            proba = proba.flatten()        # (N,)
            pred  = (proba > 0.5).astype(int)
            probs_full = np.stack(
                [1 - proba, proba], axis=1)  # (N, 2)
        else:
            probs_full = proba              # (N, n_classes)
            pred       = np.argmax(proba, axis=1)

        # For single segment, return scalar results
        if len(pred) == 1:
            p      = int(pred[0])
            conf   = float(probs_full[0, p]) * 100
            probs  = {cfg['class_names'][i]: float(probs_full[0, i])
                      for i in range(cfg['n_classes'])}
            return {
                'predicted_class':  p,
                'class_name':       cfg['class_names'][p],
                'confidence':       round(conf, 2),
                'probabilities':    probs,
                'clinical_text':    CLINICAL_TEXT[task][p],
                'urgency':          URGENCY_LEVEL[task][p],
                'urgency_color':    URGENCY_COLORS[
                                        URGENCY_LEVEL[task][p]],
                'colors':           cfg['colors'],
                'description':      cfg['description'],
            }
        else:
            # Batch result
            return {
                'predicted_classes': pred.tolist(),
                'class_names': [cfg['class_names'][p]
                                for p in pred],
                'confidences':  [float(probs_full[i, pred[i]])*100
                                 for i in range(len(pred))],
                'probabilities': probs_full.tolist(),
            }

    def predict(self, eeg_input):
        """
        Main prediction function. Runs all 3 task models.

        Args:
            eeg_input: raw EEG segment, shape (178,) or (N,178)

        Returns:
            dict with keys '2class', '3class', '5class'
            each containing full prediction results
        """
        processed = self.preprocess(eeg_input)
        results   = {}
        for task in ['2class', '3class', '5class']:
            results[task] = self.predict_single_task(
                processed, task)
        return results

    def predict_from_csv(self, csv_path, signal_col=None):
        """
        Load EEG from a CSV file and predict.

        CSV format supported:
          - Single column of 178 values (no header)
          - Multiple columns where signal_col names the EEG column
          - Standard Bonn format (X1..X178 columns)

        Returns prediction results dict.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        if signal_col and signal_col in df.columns:
            # Named column
            eeg = df[signal_col].values[:178]
        elif all(f'X{i}' in df.columns for i in range(1, 10)):
            # Bonn format — X1..X178
            xcols = [c for c in df.columns if c.startswith('X')]
            xcols.sort(key=lambda x: int(x[1:]))
            eeg   = df[xcols].values[0]        # first row
        elif df.shape[1] == 1:
            # Single column
            eeg = df.iloc[:178, 0].values
        elif df.shape[0] == 1:
            # Single row
            eeg = df.iloc[0, :178].values
        else:
            # Take first 178 values of first column
            eeg = df.iloc[:178, 0].values

        return self.predict(eeg.astype(np.float32))

    def get_summary(self, results):
        """
        Generate a plain-text summary of prediction results.
        Used by the app for display and report generation.
        """
        lines = ["=" * 50,
                 "STCA-Net EEG Analysis Report",
                 "=" * 50]

        for task, res in results.items():
            if res is None:
                continue
            cfg = TASK_CONFIG[task]
            lines += [
                f"\\n[{cfg['description']}]",
                f"  Predicted : {res['class_name']}",
                f"  Confidence: {res['confidence']:.1f}%",
                f"  Urgency   : {res['urgency']}",
                f"  Clinical  : {res['clinical_text']}",
                "  Probabilities:",
            ]
            for cname, prob in res['probabilities'].items():
                bar = int(prob * 20) * "█"
                lines.append(f"    {cname:<40} "
                              f"{prob*100:>5.1f}% {bar}")

        lines += ["\\n" + "=" * 50,
                  "DISCLAIMER: This is an AI-assisted screening "
                  "tool only. It does not constitute a clinical "
                  "diagnosis. Always consult a qualified "
                  "neurologist for medical decisions.",
                  "=" * 50]
        return "\\n".join(lines)


# ─────────────────────────────────────────────
#  Quick test when run directly
# ─────────────────────────────────────────────
if __name__ == '__main__':
    engine = STCANetInference()
    test   = np.random.randn(178).astype(np.float32)
    result = engine.predict(test)

    for task, res in result.items():
        if res:
            print(f"{task}: {res['class_name']} "
                  f"({res['confidence']:.1f}%) "
                  f"[{res['urgency']}]")

    print("\\n" + engine.get_summary(result))
'''


def write_inference_engine():
    """Write the inference engine as a standalone Python file."""
    path = "inference_engine.py"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(INFERENCE_ENGINE_CODE)
    print(f"\n  Inference engine written: {path}")
    print(f"  Size: {os.path.getsize(path)/1024:.1f} KB")


# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    print("=" * 55)
    print("STCA-Net — Phase 6: Save Model + Inference Engine")
    print(f"TensorFlow: {tf.__version__}")
    print("=" * 55)

    results_summary = []

    # ── Train and save all 3 task models ──────────────────────
    for task in ['2class', '3class', '5class']:
        out = train_and_save(task)
        if out:
            model, acc = out
            results_summary.append(
                {'Task': task, 'Val Accuracy': f'{acc:.2f}%',
                 'Saved': f'models/stca_net_{task}.keras'})
        keras.backend.clear_session()

    # ── Write inference engine ─────────────────────────────────
    print(f"\n{'='*55}")
    print("Writing Inference Engine...")
    write_inference_engine()

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Phase 6 Complete — Summary")
    print(f"{'='*55}")
    for r in results_summary:
        print(f"  {r['Task']:10s}  "
              f"ValAcc={r['Val Accuracy']:>8}  "
              f"→  {r['Saved']}")

    print(f"\n  Next step: run Phase 5 with --mode saved")
    print(f"  Then:      run Phase 7 (Streamlit app)")
    print(f"{'='*55}")