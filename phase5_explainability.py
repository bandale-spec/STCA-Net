"""
=============================================================
STCA-Net  |  Phase 5: Explainability (XAI)
=============================================================
Runs AFTER Phase 2 (trained model) and Phase 6 (saved model).

What this file produces:
  1. Grad-CAM heatmaps     — which TIME POINTS drove the decision
  2. Attention weight maps  — how CAF branches interacted
  3. Saliency maps          — raw gradient sensitivity per sample
  4. Per-class EEG overlays — one plot per class showing signal
                              with importance highlighted
  5. Summary XAI report     — all classes side by side (paper fig)

HOW TO RUN:
  Option A (uses saved .keras model from Phase 6):
      python phase5_explainability.py --mode saved
  Option B (rebuilds model weights on the fly from Phase 2):
      python phase5_explainability.py --mode rebuild

Output folder: plots/explainability/
=============================================================
"""

import os
import sys
import warnings
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers

tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
#  PATHS  (adjust if your folder structure differs)
# ─────────────────────────────────────────────
DATA_DIR    = "data/processed"
MODEL_DIR   = "models"
PLOTS_DIR   = "plots/explainability"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  CONSTANTS  (must match Phase 2 config)
# ─────────────────────────────────────────────
EMBED_DIM  = 32
NUM_HEADS  = 4
FF_DIM     = 64
DROPOUT    = 0.5
L2         = 1e-4
FS         = 173.61   # Hz — Bonn dataset sampling rate

CLASS_NAMES_5 = ['Seizure (E)', 'D-interictal', 'C-interictal',
                 'Eyes-closed (B)', 'Eyes-open (A)']
CLASS_NAMES_3 = ['Healthy', 'Interictal', 'Seizure']
CLASS_NAMES_2 = ['Non-seizure', 'Seizure']

# Colours per class (consistent across all plots)
CLASS_COLORS = {
    0: '#D85A30',   # seizure     — red
    1: '#E8902A',   # D-interictal — orange
    2: '#2AAA6E',   # C-interictal — green
    3: '#378ADD',   # eyes-closed  — blue
    4: '#7F77DD',   # eyes-open    — purple
}


# =============================================================
# REBUILD STCA-Net  (copied from Phase 2 so Phase 5 is self-contained)
# =============================================================

def dsca_block(x, filters, pfx):
    reg = regularizers.l2(L2)
    branches = []
    for k in [3, 5, 7]:
        b = layers.DepthwiseConv1D(k, padding='same',
                                    depthwise_regularizer=reg,
                                    name=f'{pfx}_dw{k}')(x)
        b = layers.Conv1D(filters, 1, kernel_regularizer=reg,
                          name=f'{pfx}_pw{k}')(b)
        b = layers.BatchNormalization(momentum=0.9,
                                       name=f'{pfx}_bn{k}')(b)
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
        lambda t: tf.reduce_max(t,  axis=-1, keepdims=True),
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
                       kernel_regularizer=reg, name=f'{pfx}_proj')(cat)
    return layers.Dropout(0.3, name=f'{pfx}_drop')(out)


def cte_block(x, pfx):
    reg = regularizers.l2(L2)
    t   = layers.Conv1D(EMBED_DIM, 8, strides=4, padding='same',
                         kernel_regularizer=reg, name=f'{pfx}_emb')(x)
    t   = layers.BatchNormalization(momentum=0.9, name=f'{pfx}_bn')(t)
    t   = layers.Activation('relu', name=f'{pfx}_act')(t)
    attn = layers.MultiHeadAttention(
               NUM_HEADS, max(1, EMBED_DIM//NUM_HEADS),
               dropout=0.1, name=f'{pfx}_mha')(t, t)
    t   = layers.LayerNormalization(epsilon=1e-6, name=f'{pfx}_ln1')(t + attn)
    ff  = layers.Dense(FF_DIM, activation='gelu',
                        kernel_regularizer=reg, name=f'{pfx}_ff1')(t)
    ff  = layers.Dropout(0.1, name=f'{pfx}_ffd')(ff)
    ff  = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                        name=f'{pfx}_ff2')(ff)
    t   = layers.LayerNormalization(epsilon=1e-6, name=f'{pfx}_ln2')(t + ff)
    out = layers.GlobalAveragePooling1D(name=f'{pfx}_gap')(t)
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

    s2t = layers.MultiHeadAttention(NUM_HEADS, kd,
                                     name=f'{pfx}_s2t')(sq, tk)
    s2t = layers.LayerNormalization(epsilon=1e-6,
                                     name=f'{pfx}_ls')(sq + s2t)
    t2s = layers.MultiHeadAttention(NUM_HEADS, kd,
                                     name=f'{pfx}_t2s')(tq, sk)
    t2s = layers.LayerNormalization(epsilon=1e-6,
                                     name=f'{pfx}_lt')(tq + t2s)
    fused = layers.LayerNormalization(epsilon=1e-6, name=f'{pfx}_lo')(
                layers.Add(name=f'{pfx}_add')([s2t, t2s]))
    return layers.Flatten(name=f'{pfx}_flat')(fused)


def build_stca_net(input_shape, n_classes):
    reg    = regularizers.l2(L2)
    inputs = keras.Input(shape=input_shape, name='eeg_input')

    sp     = dsca_block(inputs, EMBED_DIM, 'd1')
    sp     = dsca_block(sp,     EMBED_DIM, 'd2')
    sp_vec = tpp_block(sp,  'tpp')
    tp_vec = cte_block(inputs, 'cte')
    fused  = caf_block(sp_vec, tp_vec, 'caf')

    sp_r = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                         name='spr')(sp_vec)
    tp_r = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                         name='tpr')(tp_vec)
    c    = layers.LayerNormalization(epsilon=1e-6, name='fln')(
               layers.Add(name='ra')([fused, sp_r, tp_r]))

    x = layers.Dropout(DROPOUT, name='dr1')(
            layers.Dense(128, activation='gelu',
                          kernel_regularizer=reg, name='fc1')(c))
    x = layers.Dropout(DROPOUT * 0.6, name='dr2')(
            layers.Dense(64,  activation='gelu',
                          kernel_regularizer=reg, name='fc2')(x))
    out = (layers.Dense(1,        activation='sigmoid', name='out')(x)
           if n_classes == 2
           else layers.Dense(n_classes, activation='softmax', name='out')(x))

    return Model(inputs=inputs, outputs=out, name='STCA-Net')


# =============================================================
# LOAD DATA  (one sample per class for visualisation)
# =============================================================

def load_samples(task='5class', n_per_class=5):
    """
    Load n_per_class representative samples per class.
    Returns X (samples, 178, 1), y (samples,), class_names list.
    """
    X_path = os.path.join(DATA_DIR, f'{task}_X.npy')
    y_path = os.path.join(DATA_DIR, f'{task}_y.npy')

    if not os.path.exists(X_path):
        print(f"  [!] Data not found at {X_path}")
        print(f"      Run Phase 1 first to generate processed data.")
        sys.exit(1)

    X = np.load(X_path)
    y = np.load(y_path)
    n_classes = len(np.unique(y))

    samples_X, samples_y = [], []
    for cls in range(n_classes):
        idx = np.where(y == cls)[0]
        chosen = idx[:n_per_class]
        samples_X.append(X[chosen])
        samples_y.append(y[chosen])

    return (np.concatenate(samples_X).astype('float32'),
            np.concatenate(samples_y).astype('int32'),
            n_classes)


# =============================================================
# XAI METHOD 1 — GRAD-CAM
# Highlights which time steps drove the prediction
# =============================================================

def compute_gradcam(model, eeg_sample, class_idx, layer_name):
    """
    Compute 1D Grad-CAM for a single EEG sample.

    Args:
        model      : trained Keras model
        eeg_sample : (178, 1) numpy array
        class_idx  : which output class to explain
        layer_name : name of the convolutional layer to hook into

    Returns:
        heatmap : (178,) numpy array of importance scores 0-1
    """
    # Build a sub-model that outputs both the conv layer and predictions
    try:
        grad_model = Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(layer_name).output,
                       model.output]
        )
    except ValueError:
        print(f"  [!] Layer '{layer_name}' not found. "
              f"Available layers: {[l.name for l in model.layers[:10]]}...")
        return np.ones(eeg_sample.shape[0])

    inp = tf.cast(eeg_sample[np.newaxis], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inp)
        conv_outputs, predictions = grad_model(inp, training=False)

        if predictions.shape[-1] == 1:
            # Binary
            loss = predictions[0, 0]
        else:
            loss = predictions[0, class_idx]

    # Gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)   # (1, T, C)

    # Pool gradients over the channel dimension → importance per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (C,)

    # Weight the conv outputs by their gradient importance
    conv_out  = conv_outputs[0]                          # (T, C)
    heatmap   = tf.reduce_sum(
        conv_out * pooled_grads[tf.newaxis, :], axis=-1) # (T,)

    # ReLU + normalise to [0,1]
    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Resize to original signal length (178) if needed
    if len(heatmap) != eeg_sample.shape[0]:
        heatmap = np.interp(
            np.linspace(0, 1, eeg_sample.shape[0]),
            np.linspace(0, 1, len(heatmap)),
            heatmap)

    return heatmap


# =============================================================
# XAI METHOD 2 — SALIENCY MAP (vanilla gradient)
# Raw gradient of loss w.r.t. input signal
# =============================================================

def compute_saliency(model, eeg_sample, class_idx):
    """
    Vanilla saliency: gradient of the predicted class score
    with respect to the input EEG signal.

    Returns: (178,) importance array, normalised 0-1.
    """
    inp = tf.Variable(eeg_sample[np.newaxis].astype('float32'))

    with tf.GradientTape() as tape:
        pred = model(inp, training=False)
        if pred.shape[-1] == 1:
            loss = pred[0, 0]
        else:
            loss = pred[0, class_idx]

    grads    = tape.gradient(loss, inp).numpy()[0, :, 0]  # (178,)
    saliency = np.abs(grads)

    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency


# =============================================================
# XAI METHOD 3 — INTEGRATED GRADIENTS
# More stable than vanilla saliency — averages gradients along
# a path from a baseline (zeros) to the actual input
# =============================================================

def compute_integrated_gradients(model, eeg_sample, class_idx,
                                  n_steps=50):
    """
    Integrated Gradients attribution for one EEG sample.
    Baseline = zero signal (silence / no activity).

    Returns: (178,) attribution array.
    """
    baseline = np.zeros_like(eeg_sample)
    alphas   = np.linspace(0, 1, n_steps)

    grads_list = []
    for alpha in alphas:
        interp = baseline + alpha * (eeg_sample - baseline)
        inp    = tf.Variable(interp[np.newaxis].astype('float32'))

        with tf.GradientTape() as tape:
            pred = model(inp, training=False)
            if pred.shape[-1] == 1:
                loss = pred[0, 0]
            else:
                loss = pred[0, class_idx]

        g = tape.gradient(loss, inp).numpy()[0, :, 0]
        grads_list.append(g)

    avg_grads  = np.mean(grads_list, axis=0)               # (178,)
    integ_grad = (eeg_sample[:, 0] - baseline[:, 0]) * avg_grads

    # Normalise
    abs_ig = np.abs(integ_grad)
    if abs_ig.max() > 0:
        abs_ig = abs_ig / abs_ig.max()

    return abs_ig


# =============================================================
# PLOT 1 — Grad-CAM per class (main result figure)
# =============================================================

def plot_gradcam_per_class(model, X, y, n_classes, class_names,
                            task_name, layer_name='d2_add'):
    """
    One row per class. Each row shows:
      Left  — EEG signal with Grad-CAM heatmap overlaid
      Right — Grad-CAM importance curve
    """
    n_rows = n_classes
    fig    = plt.figure(figsize=(16, 3.5 * n_rows))
    gs     = gridspec.GridSpec(n_rows, 2, figure=fig,
                                hspace=0.55, wspace=0.25)

    # Custom red heatmap
    cmap_heat = LinearSegmentedColormap.from_list(
        'eeg_heat', ['#FFFFFF', '#FFF3CD', '#FF8C00', '#D85A30'], N=256)

    t = np.arange(178) / FS * 1000   # time in ms

    for cls in range(n_classes):
        idx    = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        sample  = X[idx[0]]           # (178, 1)
        heatmap = compute_gradcam(model, sample, cls, layer_name)
        color   = CLASS_COLORS.get(cls, '#333333')
        cname   = class_names[cls]

        # ── Left: signal + heatmap overlay ──────────────────────
        ax1 = fig.add_subplot(gs[cls, 0])
        signal = sample[:, 0]

        # Background heatmap (colour fill)
        for i in range(len(t) - 1):
            ax1.axvspan(t[i], t[i+1], alpha=float(heatmap[i]) * 0.6,
                        color='#FF8C00', linewidth=0)

        ax1.plot(t, signal, color=color, lw=1.2, zorder=3)
        ax1.set_title(f'{cname} — EEG with Grad-CAM overlay',
                      fontsize=9, fontweight='bold')
        ax1.set_xlabel('Time (ms)', fontsize=8)
        ax1.set_ylabel('Amplitude (z-score)', fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.spines[['top', 'right']].set_visible(False)

        # ── Right: importance curve ──────────────────────────────
        ax2 = fig.add_subplot(gs[cls, 1])
        ax2.fill_between(t, heatmap, alpha=0.4, color='#FF8C00')
        ax2.plot(t, heatmap, color='#D85A30', lw=1.5)
        ax2.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.7,
                    label='Threshold 0.5')

        # Annotate top-3 most important regions
        peaks = np.argsort(heatmap)[-3:]
        for pk in peaks:
            ax2.annotate(f'{t[pk]:.0f}ms',
                         xy=(t[pk], heatmap[pk]),
                         xytext=(0, 8), textcoords='offset points',
                         ha='center', fontsize=7, color='#D85A30',
                         fontweight='bold')

        ax2.set_title(f'{cname} — Grad-CAM importance',
                      fontsize=9, fontweight='bold')
        ax2.set_xlabel('Time (ms)', fontsize=8)
        ax2.set_ylabel('Importance', fontsize=8)
        ax2.set_ylim(0, 1.15)
        ax2.tick_params(labelsize=7)
        ax2.legend(fontsize=7)
        ax2.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'STCA-Net Grad-CAM Explanations — {task_name} task\n'
                 f'Highlighted regions indicate time points most '
                 f'influential for classification',
                 fontsize=11, fontweight='bold', y=1.01)

    p = os.path.join(PLOTS_DIR, f'gradcam_{task_name}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


# =============================================================
# PLOT 2 — Integrated Gradients comparison across classes
# =============================================================

def plot_integrated_gradients(model, X, y, n_classes,
                               class_names, task_name):
    """
    Side-by-side IG attribution for one sample per class.
    Shows the EEG signal and its IG attribution in one plot.
    """
    fig, axes = plt.subplots(n_classes, 1,
                              figsize=(14, 3 * n_classes),
                              sharex=True)
    if n_classes == 1:
        axes = [axes]

    t = np.arange(178) / FS * 1000

    for cls in range(n_classes):
        idx    = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        sample  = X[idx[0]]
        ig      = compute_integrated_gradients(model, sample, cls,
                                                n_steps=30)
        signal  = sample[:, 0]
        color   = CLASS_COLORS.get(cls, '#333333')
        cname   = class_names[cls]
        ax      = axes[cls]

        # Normalise signal for same-axis plotting
        sig_n = (signal - signal.min()) / (np.ptp(signal) + 1e-8)

        ax.plot(t, sig_n, color=color, lw=1.2, alpha=0.7,
                label='EEG (normalised)', zorder=3)
        ax.fill_between(t, ig, alpha=0.5, color='#1A6FBF',
                         label='IG attribution', zorder=2)
        ax.plot(t, ig, color='#1A6FBF', lw=1.0, zorder=4)

        ax.set_ylabel(cname, fontsize=8, fontweight='bold',
                      color=color)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        if cls == 0:
            ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Time (ms)', fontsize=9)
    fig.suptitle(f'STCA-Net Integrated Gradients — {task_name}\n'
                 f'Blue fill = signal regions that pushed '
                 f'the prediction toward the true class',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    p = os.path.join(PLOTS_DIR, f'integrated_gradients_{task_name}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


# =============================================================
# PLOT 3 — Saliency map comparison (all classes, one figure)
# =============================================================

def plot_saliency_comparison(model, X, y, n_classes,
                              class_names, task_name):
    """
    Heatmap grid showing saliency for multiple samples per class.
    Rows = classes, columns = individual samples.
    Good for seeing consistency of explanations.
    """
    n_samples = 5
    fig, axes = plt.subplots(n_classes, n_samples,
                              figsize=(15, 2.8 * n_classes))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    t = np.arange(178) / FS * 1000

    for cls in range(n_classes):
        idx = np.where(y == cls)[0]
        for si in range(min(n_samples, len(idx))):
            sample   = X[idx[si]]
            saliency = compute_saliency(model, sample, cls)
            ax       = axes[cls, si]

            # Top half: EEG signal
            sig = sample[:, 0]
            ax.plot(t, (sig - sig.min())/(np.ptp(sig)+1e-8),
                    color=CLASS_COLORS.get(cls, '#333'),
                    lw=0.9, alpha=0.8)
            # Bottom: saliency as image strip
            ax.imshow(saliency[np.newaxis, :],
                      extent=[t[0], t[-1], -0.05, 0],
                      aspect='auto', cmap='hot', vmin=0, vmax=1,
                      zorder=3)

            if si == 0:
                ax.set_ylabel(class_names[cls], fontsize=8,
                              fontweight='bold',
                              color=CLASS_COLORS.get(cls, '#333'))
            ax.set_xlabel('ms', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(f'S{si+1}', fontsize=7)
            ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'STCA-Net Saliency Maps — {task_name}\n'
                 f'Bright = high sensitivity to input change at '
                 f'that time point',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()

    p = os.path.join(PLOTS_DIR, f'saliency_{task_name}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


# =============================================================
# PLOT 4 — XAI Summary Figure (paper-ready, all 3 methods)
# =============================================================

def plot_xai_summary(model, X, y, n_classes, class_names, task_name,
                     layer_name='d2_add'):
    """
    One sample from each class.
    3 columns: Raw EEG | Grad-CAM | Integrated Gradients
    This is the figure to include in the paper.
    """
    fig = plt.figure(figsize=(17, 3.2 * n_classes))
    gs  = gridspec.GridSpec(n_classes, 3, figure=fig,
                             hspace=0.6, wspace=0.32)
    t   = np.arange(178) / FS * 1000
    col_titles = ['Raw EEG Signal',
                  'Grad-CAM (temporal importance)',
                  'Integrated Gradients (attribution)']

    for cls in range(n_classes):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        sample  = X[idx[0]]
        signal  = sample[:, 0]
        color   = CLASS_COLORS.get(cls, '#333333')
        cname   = class_names[cls]

        gradcam = compute_gradcam(model, sample, cls, layer_name)
        ig      = compute_integrated_gradients(
                      model, sample, cls, n_steps=25)

        # ── Col 0: Raw EEG ──────────────────────────────────────
        ax0 = fig.add_subplot(gs[cls, 0])
        ax0.plot(t, signal, color=color, lw=1.2)
        ax0.set_ylabel(cname, fontsize=8, fontweight='bold',
                       color=color)
        ax0.set_xlabel('Time (ms)', fontsize=7)
        ax0.tick_params(labelsize=6)
        ax0.spines[['top', 'right']].set_visible(False)
        if cls == 0:
            ax0.set_title(col_titles[0], fontsize=9,
                          fontweight='bold', pad=8)

        # ── Col 1: Grad-CAM overlay ──────────────────────────────
        ax1 = fig.add_subplot(gs[cls, 1])
        for i in range(len(t) - 1):
            ax1.axvspan(t[i], t[i+1],
                        alpha=float(gradcam[i]) * 0.65,
                        color='#FF8C00', linewidth=0)
        ax1.plot(t, signal, color=color, lw=1.0, alpha=0.8, zorder=3)
        ax1.set_xlabel('Time (ms)', fontsize=7)
        ax1.tick_params(labelsize=6)
        ax1.spines[['top', 'right']].set_visible(False)
        if cls == 0:
            ax1.set_title(col_titles[1], fontsize=9,
                          fontweight='bold', pad=8)

        # ── Col 2: IG ────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[cls, 2])
        sig_n = (signal - signal.min()) / (np.ptp(signal) + 1e-8)
        ax2.fill_between(t, ig, alpha=0.45, color='#1A6FBF')
        ax2.plot(t, sig_n, color=color, lw=0.9, alpha=0.7,
                 label='EEG')
        ax2.plot(t, ig,   color='#1A6FBF', lw=1.2,
                 label='IG')
        ax2.set_xlabel('Time (ms)', fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.spines[['top', 'right']].set_visible(False)
        if cls == 0:
            ax2.set_title(col_titles[2], fontsize=9,
                          fontweight='bold', pad=8)
            ax2.legend(fontsize=6, loc='upper right')

    fig.suptitle(
        f'STCA-Net Explainability Analysis — {task_name} task\n'
        f'Each row = one representative sample from that class.\n'
        f'Orange overlay (Grad-CAM) = critical time points. '
        f'Blue fill (IG) = positive attribution toward true class.',
        fontsize=10, fontweight='bold', y=1.02)

    p = os.path.join(PLOTS_DIR, f'xai_summary_{task_name}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


# =============================================================
# PLOT 5 — Attention weight visualisation (CAF module)
# =============================================================

def plot_attention_analysis(model, X, y, n_classes,
                             class_names, task_name):
    """
    Visualises the cross-attention weights inside the CAF module.
    Since CAF operates on (1,1) sequence tensors, this shows
    how much the spatial branch attended to temporal and vice versa
    across different classes.
    """
    # Get spatial and temporal branch outputs for each class
    sp_model = Model(
        inputs  = model.inputs,
        outputs = model.get_layer('tpp_proj').output,
        name    = 'sp_extractor'
    )
    tp_model = Model(
        inputs  = model.inputs,
        outputs = model.get_layer('cte_proj').output,
        name    = 'tp_extractor'
    )

    sp_means, tp_means = [], []
    for cls in range(n_classes):
        idx = np.where(y == cls)[0][:20]
        sps = sp_model.predict(X[idx], verbose=0, batch_size=32)
        tps = tp_model.predict(X[idx], verbose=0, batch_size=32)
        sp_means.append(sps.mean(axis=0))
        tp_means.append(tps.mean(axis=0))

    sp_matrix = np.array(sp_means)   # (n_classes, EMBED_DIM)
    tp_matrix = np.array(tp_means)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── Spatial branch activations per class ────────────────────
    sns.heatmap(sp_matrix, ax=axes[0], cmap='Blues',
                xticklabels=False,
                yticklabels=class_names,
                cbar_kws={'label': 'Mean activation'})
    axes[0].set_title('Spatial Branch (TPP)\nMean activations per class',
                      fontweight='bold', fontsize=10)
    axes[0].set_xlabel('Embedding dimension', fontsize=9)

    # ── Temporal branch activations per class ───────────────────
    sns.heatmap(tp_matrix, ax=axes[1], cmap='Oranges',
                xticklabels=False,
                yticklabels=class_names,
                cbar_kws={'label': 'Mean activation'})
    axes[1].set_title('Temporal Branch (CTE)\nMean activations per class',
                      fontweight='bold', fontsize=10)
    axes[1].set_xlabel('Embedding dimension', fontsize=9)

    # ── Cross-correlation between spatial and temporal ───────────
    corr = np.corrcoef(
        np.concatenate([sp_matrix, tp_matrix], axis=1))[:n_classes,
                                                         n_classes:]
    sns.heatmap(corr, ax=axes[2], cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True, fmt='.2f', annot_kws={'size': 8},
                cbar_kws={'label': 'Correlation'})
    axes[2].set_title('Spatial vs Temporal\nCross-branch correlation',
                      fontweight='bold', fontsize=10)

    fig.suptitle(
        f'STCA-Net Branch Activation Analysis — {task_name}\n'
        f'Shows how spatial and temporal branches represent '
        f'different EEG classes internally',
        fontsize=11, fontweight='bold')
    fig.tight_layout()

    p = os.path.join(PLOTS_DIR, f'branch_analysis_{task_name}.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


# =============================================================
# QUANTITATIVE XAI TABLE
# Mean Grad-CAM score per temporal band per class
# (for paper Table)
# =============================================================

def compute_xai_table(model, X, y, n_classes, class_names,
                       task_name, layer_name='d2_add'):
    """
    Computes mean Grad-CAM importance in 4 temporal bands:
      0-250ms, 250-500ms, 500-750ms, 750-1025ms
    for each class.
    Saves as CSV for the paper.
    """
    bands = [(0, 44), (44, 88), (88, 132), (132, 178)]
    band_labels = ['0-250ms', '250-500ms', '500-750ms', '750-1025ms']

    rows = []
    for cls in range(n_classes):
        idx = np.where(y == cls)[0][:10]
        heatmaps = []
        for i in idx:
            h = compute_gradcam(model, X[i], cls, layer_name)
            heatmaps.append(h)
        mean_h = np.mean(heatmaps, axis=0)

        row = {'Class': class_names[cls]}
        for (start, end), label in zip(bands, band_labels):
            row[label] = f"{mean_h[start:end].mean():.3f}"
        row['Peak time (ms)'] = f"{np.argmax(mean_h) / FS * 1000:.0f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    p  = os.path.join(RESULTS_DIR, f'xai_gradcam_table_{task_name}.csv')
    df.to_csv(p, index=False)
    print(f"\n  Grad-CAM quantitative table:")
    print(df.to_string(index=False))
    print(f"  Saved: {p}")
    return df


# =============================================================
# MAIN
# =============================================================

def run_explainability(task='5class', mode='rebuild',
                       layer_name='d2_add'):
    """
    Full explainability pipeline for one task variant.

    Args:
        task       : '5class', '3class', or '2class'
        mode       : 'saved'   = load from models/ folder
                     'rebuild' = rebuild architecture + random weights
                                 (use only for testing the XAI pipeline)
        layer_name : Grad-CAM hook layer (default 'd2_add' = 2nd DSCA output)
    """
    print(f"\n{'='*60}")
    print(f"Phase 5: Explainability — Task: {task}")
    print(f"{'='*60}")

    n_cls_map = {'5class': 5, '3class': 3, '2class': 2}
    names_map  = {
        '5class': CLASS_NAMES_5,
        '3class': CLASS_NAMES_3,
        '2class': CLASS_NAMES_2
    }
    n_classes   = n_cls_map[task]
    class_names = names_map[task]

    # ── Load data ────────────────────────────────────────────────
    print(f"\n[1/6] Loading samples...")
    X, y, _ = load_samples(task, n_per_class=10)
    print(f"  X: {X.shape}  y: {y.shape}")

    # ── Load or build model ──────────────────────────────────────
    print(f"\n[2/6] Loading model ({mode} mode)...")
    model_path = os.path.join(MODEL_DIR, f'stca_net_{task}.keras')

    if mode == 'saved' and os.path.exists(model_path):
        # Try a safer approach: rebuild the architecture and load weights.
        # This avoids Lambda deserialization issues across Keras versions.
        try:
            model = build_stca_net((178, 1), n_classes)
            model.load_weights(model_path)
            print(f"  Loaded weights into rebuilt model from: {model_path}")
        except Exception:
            # Fallback: attempt to load the full model (unsafe deserialization)
            keras.config.enable_unsafe_deserialization()
            model = keras.models.load_model(model_path, safe_mode=False)
            print(f"  Loaded full model from: {model_path}")
    else:
        if mode == 'saved':
            print(f"  [!] Saved model not found at {model_path}")
            print(f"      Run Phase 6 first to save the model.")
            print(f"      Falling back to rebuild mode (untrained)...")
        model = build_stca_net((178, 1), n_classes)
        print(f"  Built fresh model ({model.count_params():,} params)")
        print(f"  NOTE: Results will show random-weight behaviour.")
        print(f"        Run Phase 6 first for meaningful XAI.")

    # ── Check layer exists ───────────────────────────────────────
    layer_names = [l.name for l in model.layers]
    if layer_name not in layer_names:
        # Fall back to first available Add layer
        add_layers = [n for n in layer_names if 'add' in n]
        if add_layers:
            layer_name = add_layers[0]
            print(f"  [!] Default layer not found. "
                  f"Using fallback: {layer_name}")

    print(f"  Grad-CAM hook layer: {layer_name}")

    # ── Generate all XAI plots ───────────────────────────────────
    print(f"\n[3/6] Generating Grad-CAM plots...")
    plot_gradcam_per_class(model, X, y, n_classes,
                            class_names, task, layer_name)

    print(f"\n[4/6] Generating Integrated Gradients plots...")
    plot_integrated_gradients(model, X, y, n_classes,
                               class_names, task)

    print(f"\n[5/6] Generating saliency maps...")
    plot_saliency_comparison(model, X, y, n_classes,
                              class_names, task)

    print(f"\n[5b] Generating XAI summary figure (paper-ready)...")
    plot_xai_summary(model, X, y, n_classes, class_names,
                     task, layer_name)

    print(f"\n[5c] Generating branch activation analysis...")
    plot_attention_analysis(model, X, y, n_classes,
                             class_names, task)

    print(f"\n[6/6] Computing quantitative XAI table...")
    compute_xai_table(model, X, y, n_classes, class_names,
                      task, layer_name)

    print(f"\n{'='*60}")
    print(f"Phase 5 complete for task: {task}")
    print(f"All outputs saved to: {PLOTS_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='STCA-Net Phase 5: Explainability')
    parser.add_argument('--task', default='5class',
                        choices=['5class', '3class', '2class'],
                        help='Which task to explain')
    parser.add_argument('--mode', default='saved',
                        choices=['saved', 'rebuild'],
                        help='saved=load .keras file, '
                             'rebuild=build fresh architecture')
    parser.add_argument('--all_tasks', action='store_true',
                        help='Run explainability for all 3 tasks')
    args = parser.parse_args()

    if args.all_tasks:
        for task in ['5class', '3class', '2class']:
            run_explainability(task, args.mode)
    else:
        run_explainability(args.task, args.mode)