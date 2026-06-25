"""
=============================================================
STCA-Net  |  Phase 7: Clinical EEG Analysis App
=============================================================
Streamlit web app for doctors to upload EEG signals and
receive a full AI-assisted diagnostic report.

HOW TO RUN:
    streamlit run app/streamlit_app.py

Features:
  - Upload EEG as CSV file
  - Auto-preprocessing pipeline
  - Runs all 3 STCA-Net models simultaneously
  - Grad-CAM visualisation of important time points
  - Probability charts per class
  - Urgency indicator
  - Downloadable PDF clinical report
  - Patient information form
=============================================================
"""

import os
import sys
import warnings
import io
import datetime
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Add parent directory to path so we can import inference_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers

tf.random.set_seed(42)

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title    = "STCA-Net | EEG Seizure Analysis",
    page_icon     = "🧠",
    layout        = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FS          = 173.61   # Bonn dataset sampling rate Hz
MODEL_DIR   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models')
EMBED_DIM   = 32
NUM_HEADS   = 4
FF_DIM      = 64
DROPOUT     = 0.5
L2          = 1e-4

TASK_CONFIG = {
    '2class': {
        'label':       'Binary Detection',
        'description': 'Seizure vs Non-seizure',
        'n_classes':   2,
        'class_names': ['Non-seizure', 'Seizure'],
        'colors':      ['#2AAA6E', '#D85A30'],
        'icon':        '🔴',
    },
    '3class': {
        'label':       'Three-Class',
        'description': 'Healthy / Interictal / Seizure',
        'n_classes':   3,
        'class_names': ['Healthy', 'Interictal', 'Seizure'],
        'colors':      ['#2AAA6E', '#E8902A', '#D85A30'],
        'icon':        '🟡',
    },
    '5class': {
        'label':       'Five-Class',
        'description': 'Full Bonn Classification',
        'n_classes':   5,
        'class_names': [
            'Seizure (ictal)',
            'Interictal — seizure area',
            'Interictal — healthy area',
            'Healthy — eyes closed',
            'Healthy — eyes open'
        ],
        'colors': ['#D85A30','#E8902A','#2AAA6E','#378ADD','#7F77DD'],
        'icon':   '🔵',
    },
}

CLINICAL_TEXT = {
    '2class': {
        0: "No active seizure detected. EEG pattern is within normal or interictal range.",
        1: "Active seizure activity detected. High-amplitude irregular discharges identified. Immediate clinical review recommended.",
    },
    '3class': {
        0: "EEG consistent with healthy background pattern. No seizure or interictal activity detected.",
        1: "Interictal activity detected. Epileptiform discharges present between seizures. Monitoring recommended.",
        2: "Active seizure (ictal) activity detected. Abnormal electrical discharge pattern identified. Immediate clinical attention required.",
    },
    '5class': {
        0: "Ictal activity detected. Pattern consistent with an active epileptic seizure.",
        1: "Interictal activity from seizure onset zone. Epileptiform discharges from the epileptogenic area.",
        2: "Interictal activity from non-seizure area. Seizure-free recording from contralateral region.",
        3: "Healthy EEG — eyes closed. Alpha-wave dominant pattern consistent with relaxed wakefulness.",
        4: "Healthy EEG — eyes open. Low-amplitude pattern consistent with eyes-open alert state.",
    },
}

URGENCY = {
    '2class': {0: ('LOW',    '#2AAA6E'), 1: ('HIGH',   '#D85A30')},
    '3class': {0: ('LOW',    '#2AAA6E'), 1: ('MEDIUM', '#E8902A'),
               2: ('HIGH',   '#D85A30')},
    '5class': {0: ('HIGH',   '#D85A30'), 1: ('MEDIUM', '#E8902A'),
               2: ('LOW',    '#2AAA6E'), 3: ('LOW',    '#2AAA6E'),
               4: ('LOW',    '#2AAA6E')},
}


# =============================================================
# MODEL ARCHITECTURE  (identical to Phase 2/6)
# =============================================================

def dsca_block(x, filters, pfx):
    reg = regularizers.l2(L2)
    branches = []
    for k in [3, 5, 7]:
        b = layers.DepthwiseConv1D(k, padding='same',
                depthwise_regularizer=reg, name=f'{pfx}_dw{k}')(x)
        b = layers.Conv1D(filters, 1, kernel_regularizer=reg,
                name=f'{pfx}_pw{k}')(b)
        b = layers.BatchNormalization(momentum=0.9,
                name=f'{pfx}_bn{k}')(b)
        b = layers.Activation('relu', name=f'{pfx}_a{k}')(b)
        branches.append(b)
    f  = layers.Add(name=f'{pfx}_add')(branches)
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
                kernel_regularizer=reg, name=f'{pfx}_proj')(cat)
    return layers.Dropout(0.3, name=f'{pfx}_drop')(out)


def cte_block(x, pfx):
    reg  = regularizers.l2(L2)
    t    = layers.Conv1D(EMBED_DIM, 8, strides=4, padding='same',
                kernel_regularizer=reg, name=f'{pfx}_emb')(x)
    t    = layers.BatchNormalization(momentum=0.9,
                name=f'{pfx}_bn')(t)
    t    = layers.Activation('relu', name=f'{pfx}_act')(t)
    attn = layers.MultiHeadAttention(NUM_HEADS,
                max(1, EMBED_DIM//NUM_HEADS), dropout=0.1,
                name=f'{pfx}_mha')(t, t)
    t    = layers.LayerNormalization(epsilon=1e-6,
                name=f'{pfx}_ln1')(t + attn)
    ff   = layers.Dense(FF_DIM, activation='gelu',
                kernel_regularizer=reg, name=f'{pfx}_ff1')(t)
    ff   = layers.Dropout(0.1, name=f'{pfx}_ffd')(ff)
    ff   = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                name=f'{pfx}_ff2')(ff)
    t    = layers.LayerNormalization(epsilon=1e-6,
                name=f'{pfx}_ln2')(t + ff)
    out  = layers.GlobalAveragePooling1D(name=f'{pfx}_gap')(t)
    return layers.Dense(EMBED_DIM, activation='relu',
                kernel_regularizer=reg, name=f'{pfx}_proj')(out)


def caf_block(sp, tp, pfx):
    reg = regularizers.l2(L2)
    kd  = max(1, EMBED_DIM // NUM_HEADS)
    def _e(v, n):
        v = layers.Dense(EMBED_DIM, kernel_regularizer=reg,
                name=f'{pfx}_{n}')(v)
        return layers.Reshape((1, EMBED_DIM),
                name=f'{pfx}_{n}r')(v)
    sq = _e(sp, 'sq'); tk = _e(tp, 'tk')
    tq = _e(tp, 'tq'); sk = _e(sp, 'sk')
    s2t = layers.MultiHeadAttention(NUM_HEADS, kd,
                name=f'{pfx}_s2t')(sq, tk)
    s2t = layers.LayerNormalization(epsilon=1e-6,
                name=f'{pfx}_ls')(sq + s2t)
    t2s = layers.MultiHeadAttention(NUM_HEADS, kd,
                name=f'{pfx}_t2s')(tq, sk)
    t2s = layers.LayerNormalization(epsilon=1e-6,
                name=f'{pfx}_lt')(tq + t2s)
    fused = layers.LayerNormalization(epsilon=1e-6,
                name=f'{pfx}_lo')(
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
    c      = layers.LayerNormalization(epsilon=1e-6, name='fln')(
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
# MODEL LOADING  (cached so it only loads once)
# =============================================================

@st.cache_resource
def load_all_models():
    """Load all 3 saved models. Cached after first load."""
    models = {}
    for task, cfg in TASK_CONFIG.items():
        path = os.path.join(MODEL_DIR, f'stca_net_{task}.keras')
        try:
            keras.config.enable_unsafe_deserialization()
            m = keras.models.load_model(path, safe_mode=False)
            models[task] = m
        except Exception as e:
            # Fallback: rebuild architecture and load weights
            try:
                m = build_stca_net((178, 1), cfg['n_classes'])
                m.load_weights(path)
                models[task] = m
            except Exception as e2:
                models[task] = None
    return models


# =============================================================
# PREPROCESSING
# =============================================================

def preprocess_eeg(eeg_input):
    """Z-score normalise and reshape to (1, 178, 1)."""
    arr = np.array(eeg_input, dtype=np.float32).flatten()
    if len(arr) > 178:
        arr = arr[:178]
    elif len(arr) < 178:
        arr = np.pad(arr, (0, 178 - len(arr)), mode='edge')
    mean = arr.mean()
    std  = arr.std() + 1e-8
    arr  = (arr - mean) / std
    return arr, arr[np.newaxis, :, np.newaxis]


# =============================================================
# INFERENCE
# =============================================================

def run_inference(models, processed):
    """Run all 3 models and return structured results."""
    results = {}
    for task, cfg in TASK_CONFIG.items():
        model = models.get(task)
        if model is None:
            results[task] = None
            continue
        proba = model.predict(processed, verbose=0)
        binary = cfg['n_classes'] == 2
        if binary:
            p      = float(proba[0, 0])
            probs  = [1 - p, p]
            pred   = int(p > 0.5)
        else:
            probs  = proba[0].tolist()
            pred   = int(np.argmax(probs))
        conf         = probs[pred] * 100
        urgency, urg_color = URGENCY[task][pred]
        results[task] = {
            'predicted':     pred,
            'class_name':    cfg['class_names'][pred],
            'confidence':    round(conf, 2),
            'probabilities': probs,
            'clinical_text': CLINICAL_TEXT[task][pred],
            'urgency':       urgency,
            'urgency_color': urg_color,
        }
    return results


# =============================================================
# GRAD-CAM  (for visualisation in app)
# =============================================================

def compute_gradcam_app(model, eeg_processed, class_idx,
                         layer_name='d2_out'):
    """Compute Grad-CAM heatmap for one EEG segment."""
    try:
        grad_model = Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(layer_name).output,
                       model.output])
    except Exception:
        return np.zeros(178)

    inp = tf.cast(eeg_processed, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        conv_out, preds = grad_model(inp, training=False)
        loss = (preds[0, 0] if preds.shape[-1] == 1
                else preds[0, class_idx])

    grads       = tape.gradient(loss, conv_out)
    pooled      = tf.reduce_mean(grads, axis=(0, 1))
    heatmap     = tf.reduce_sum(
        conv_out[0] * pooled[tf.newaxis, :], axis=-1)
    heatmap     = tf.nn.relu(heatmap).numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Resize to 178
    if len(heatmap) != 178:
        heatmap = np.interp(
            np.linspace(0, 1, 178),
            np.linspace(0, 1, len(heatmap)),
            heatmap)
    return heatmap


# =============================================================
# PLOTLY CHARTS
# =============================================================

def plot_eeg_signal(signal, title="EEG Signal"):
    """Interactive EEG waveform plot."""
    t   = np.arange(len(signal)) / FS * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal,
        mode='lines',
        line=dict(color='#1A6FBF', width=1.5),
        name='EEG'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Time (ms)',
        yaxis_title='Amplitude (z-score)',
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#EEEEEE')
    fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE')
    return fig


def plot_eeg_with_gradcam(signal, heatmap, class_name, color):
    """EEG signal with Grad-CAM overlay."""
    t   = np.arange(len(signal)) / FS * 1000
    fig = make_subplots(rows=2, cols=1,
                         shared_xaxes=True,
                         row_heights=[0.7, 0.3],
                         vertical_spacing=0.05)

    # EEG signal
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode='lines',
        line=dict(color=color, width=1.5),
        name='EEG signal'), row=1, col=1)

    # Grad-CAM heatmap
    fig.add_trace(go.Scatter(
        x=t, y=heatmap,
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.35)',
        line=dict(color='#FF8C00', width=1.2),
        name='Grad-CAM importance'), row=2, col=1)

    # Mark peak importance
    peak_t = t[np.argmax(heatmap)]
    fig.add_vline(x=peak_t, line_dash='dash',
                  line_color='#D85A30', line_width=1.5)
    fig.add_annotation(
        x=peak_t, y=1.05, yref='y2',
        text=f'Peak: {peak_t:.0f}ms',
        showarrow=False, font=dict(size=10, color='#D85A30'))

    fig.update_layout(
        title=f'Grad-CAM Explanation — {class_name}',
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', y=1.08),
    )
    fig.update_xaxes(title_text='Time (ms)', row=2, col=1,
                     showgrid=True, gridcolor='#EEEEEE')
    fig.update_yaxes(title_text='Amplitude', row=1, col=1,
                     showgrid=True, gridcolor='#EEEEEE')
    fig.update_yaxes(title_text='Importance', row=2, col=1,
                     showgrid=True, gridcolor='#EEEEEE',
                     range=[0, 1.1])
    return fig


def plot_probability_bar(probabilities, class_names, colors, title):
    """Horizontal probability bar chart."""
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probabilities],
        y=class_names,
        orientation='h',
        marker_color=colors,
        text=[f'{p*100:.1f}%' for p in probabilities],
        textposition='outside',
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Probability (%)',
        xaxis_range=[0, 115],
        height=max(180, len(class_names) * 55),
        margin=dict(l=160, r=60, t=45, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#EEEEEE')
    return fig


def plot_confidence_gauge(confidence, urgency_color, title):
    """Confidence gauge chart."""
    fig = go.Figure(go.Indicator(
        mode  = 'gauge+number',
        value = confidence,
        title = {'text': title, 'font': {'size': 13}},
        number = {'suffix': '%', 'font': {'size': 28}},
        gauge = {
            'axis': {'range': [0, 100],
                     'tickwidth': 1, 'tickcolor': '#888'},
            'bar':  {'color': urgency_color},
            'bgcolor': 'white',
            'borderwidth': 1,
            'bordercolor': '#CCCCCC',
            'steps': [
                {'range': [0,  50], 'color': '#F0F0F0'},
                {'range': [50, 75], 'color': '#FFF3CD'},
                {'range': [75,100], 'color': '#FCE4EC'},
            ],
            'threshold': {
                'line': {'color': '#333', 'width': 3},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white',
    )
    return fig


# =============================================================
# PDF REPORT GENERATOR
# =============================================================

def generate_pdf_report(patient_info, signal, results,
                          gradcam_fig=None):
    """
    Generate a downloadable PDF clinical report.
    Uses reportlab for clean professional output.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                     Spacer, Table, TableStyle,
                                     HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=20*mm, leftMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Header ────────────────────────────────────────────────
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        fontSize=18, textColor=colors.HexColor('#1A6FBF'),
        spaceAfter=4)
    sub_style = ParagraphStyle(
        'Sub', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#555555'),
        spaceAfter=2)
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, spaceAfter=4, leading=14)
    bold_style = ParagraphStyle(
        'Bold', parent=styles['Normal'],
        fontSize=10, fontName='Helvetica-Bold', spaceAfter=2)

    story.append(Paragraph("STCA-Net Clinical EEG Report", title_style))
    story.append(Paragraph(
        "AI-Assisted Epileptic Seizure Classification", sub_style))
    story.append(HRFlowable(width='100%', thickness=1,
                              color=colors.HexColor('#1A6FBF')))
    story.append(Spacer(1, 6*mm))

    # ── Patient Information ────────────────────────────────────
    story.append(Paragraph("Patient Information", bold_style))
    now = datetime.datetime.now()
    pat_data = [
        ['Patient Name',  patient_info.get('name', 'N/A'),
         'Date',          now.strftime('%Y-%m-%d')],
        ['Age',           patient_info.get('age', 'N/A'),
         'Time',          now.strftime('%H:%M:%S')],
        ['Patient ID',    patient_info.get('pid', 'N/A'),
         'Referring Dr',  patient_info.get('doctor', 'N/A')],
        ['Gender',        patient_info.get('gender', 'N/A'),
         'Department',    patient_info.get('dept', 'Neurology')],
    ]
    pat_table = Table(pat_data, colWidths=[35*mm,55*mm,35*mm,55*mm])
    pat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#D5E8F0')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#D5E8F0')),
        ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(pat_table)
    story.append(Spacer(1, 6*mm))

    # ── EEG Signal Info ────────────────────────────────────────
    story.append(Paragraph("EEG Recording Details", bold_style))
    eeg_data = [
        ['Sampling Rate', f'{FS} Hz',
         'Segment Length', '178 samples (~1.025 s)'],
        ['Channels',      '1 (single channel)',
         'Normalisation', 'Per-record z-score'],
        ['Model',         'STCA-Net v2',
         'Parameters',    '~47,000'],
    ]
    eeg_table = Table(eeg_data, colWidths=[35*mm,55*mm,35*mm,55*mm])
    eeg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#D5E8F0')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#D5E8F0')),
        ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(eeg_table)
    story.append(Spacer(1, 6*mm))

    # ── Classification Results ─────────────────────────────────
    story.append(Paragraph("Classification Results", bold_style))
    story.append(HRFlowable(width='100%', thickness=0.5,
                              color=colors.HexColor('#CCCCCC')))
    story.append(Spacer(1, 3*mm))

    urgency_colors_pdf = {
        'HIGH':   colors.HexColor('#D85A30'),
        'MEDIUM': colors.HexColor('#E8902A'),
        'LOW':    colors.HexColor('#2AAA6E'),
    }

    for task, cfg in TASK_CONFIG.items():
        res = results.get(task)
        if res is None:
            continue
        urgency     = res['urgency']
        urg_col     = urgency_colors_pdf.get(
            urgency, colors.HexColor('#333333'))

        # Task header
        task_header = ParagraphStyle(
            f'th_{task}', parent=styles['Normal'],
            fontSize=11, fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1A6FBF'), spaceAfter=2)
        story.append(Paragraph(
            f"{cfg['label']} — {cfg['description']}", task_header))

        # Result row
        result_data = [[
            'Prediction', res['class_name'],
            'Confidence', f"{res['confidence']:.1f}%",
            'Urgency',    urgency
        ]]
        r_table = Table(result_data,
                         colWidths=[28*mm,52*mm,28*mm,28*mm,22*mm,22*mm])
        r_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), colors.HexColor('#D5E8F0')),
            ('BACKGROUND', (2,0), (2,0), colors.HexColor('#D5E8F0')),
            ('BACKGROUND', (4,0), (4,0), colors.HexColor('#D5E8F0')),
            ('BACKGROUND', (5,0), (5,0), urg_col),
            ('TEXTCOLOR',  (5,0), (5,0), colors.white),
            ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
            ('FONTNAME',   (1,0), (1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('GRID',       (0,0), (-1,-1), 0.5,
             colors.HexColor('#CCCCCC')),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(r_table)

        # Probabilities
        prob_data = [cfg['class_names']]
        prob_vals = [f"{p*100:.1f}%" for p in res['probabilities']]
        prob_data.append(prob_vals)
        col_w = 170 * mm / len(cfg['class_names'])
        p_table = Table(prob_data,
                         colWidths=[col_w]*len(cfg['class_names']))
        p_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F5F5F5')),
            ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',   (0,0), (-1,-1), 8),
            ('GRID',       (0,0), (-1,-1), 0.5,
             colors.HexColor('#CCCCCC')),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(Spacer(1, 1*mm))
        story.append(p_table)

        # Clinical text
        clin_style = ParagraphStyle(
            f'clin_{task}', parent=styles['Normal'],
            fontSize=9, textColor=colors.HexColor('#444444'),
            leftIndent=5, spaceAfter=2, leading=13)
        story.append(Spacer(1, 1*mm))
        story.append(Paragraph(
            f"Clinical note: {res['clinical_text']}", clin_style))
        story.append(Spacer(1, 4*mm))

    # ── Disclaimer ────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=1,
                              color=colors.HexColor('#CCCCCC')))
    story.append(Spacer(1, 3*mm))
    disclaimer_style = ParagraphStyle(
        'disc', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#888888'),
        leading=11)
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-assisted "
        "screening tool (STCA-Net) trained on the University of "
        "Bonn EEG dataset. It does NOT constitute a clinical "
        "diagnosis and should NOT be used as a sole basis for "
        "medical decisions. All results must be reviewed and "
        "confirmed by a qualified neurologist or epileptologist. "
        "Model accuracy: 98.17% (binary), 93.57% (3-class), "
        "75.65% (5-class) on 10-fold cross-validation.",
        disclaimer_style))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"Report generated: {now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"| Model: STCA-Net v2 | "
        f"Framework: TensorFlow {tf.__version__}",
        disclaimer_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# =============================================================
# LOAD CSV HELPER
# =============================================================

def parse_uploaded_csv(uploaded_file):
    """
    Parse uploaded CSV into a 178-sample EEG array.
    Handles multiple formats:
      - Single column of values
      - Bonn format (X1..X178 columns)
      - Row of 178 values
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Bonn format
        xcols = sorted([c for c in df.columns
                         if c.startswith('X') and
                         c[1:].isdigit()],
                        key=lambda x: int(x[1:]))
        if len(xcols) >= 100:
            return df[xcols].values[0].astype(np.float32), None

        # Single row, many columns
        if df.shape[0] == 1 and df.shape[1] >= 50:
            return df.iloc[0].values[:178].astype(np.float32), None

        # Single column
        if df.shape[1] == 1:
            return df.iloc[:178, 0].values.astype(np.float32), None

        # Multiple columns — try first numeric column
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            return df[num_cols[0]].values[:178].astype(np.float32), None

        return None, "Could not parse CSV format."
    except Exception as e:
        return None, str(e)


# =============================================================
# CUSTOM CSS
# =============================================================

def inject_css():
    st.markdown("""
    <style>
    /* ===== PROFESSIONAL COLOR SCHEME ===== */
    :root {
        --primary: #1A1A1A;
        --primary-light: #2D2D2D;
        --primary-dark: #000000;
        --accent: #0066CC;
        --accent-light: #E6F2FF;
        --success: #2AAA6E;
        --warning: #E8902A;
        --danger: #D85A30;
        --bg-white: #FFFFFF;
        --bg-light: #F5F7FA;
        --text-dark: #1A1A1A;
        --text-light: #666666;
        --border-light: #E0E0E0;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.10);
        --shadow-lg: 0 8px 20px rgba(0,0,0,0.12);
    }

    /* ===== MAIN APP ===== */
    .stApp {
        background-color: #FFFFFF;
    }

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #1A1A1A;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    p, span, label {
        color: #1A1A1A;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #F5F7FA;
        border-right: 1px solid #E0E0E0;
    }
    
    [data-testid="stSidebar"] * {
        color: #1A1A1A !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #1A1A1A;
        font-weight: 700;
        margin-bottom: 16px;
        font-size: 16px;
    }

    /* ===== FORM INPUTS ===== */
    .stTextInput input,
    .stNumberInput input,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #FFFFFF;
        color: #1A1A1A;
        border: 1px solid #D0D0D0;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 14px;
        box-shadow: var(--shadow-sm);
    }

    .stTextInput input::placeholder,
    .stNumberInput input::placeholder {
        color: #999999;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    [data-testid="stSidebar"] .stTextInput input:focus,
    [data-testid="stSidebar"] .stNumberInput input:focus {
        border-color: #0066CC;
        box-shadow: 0 0 0 3px rgba(0,102,204,0.1);
        outline: none;
    }

    /* ===== SELECT BOXES ===== */
    .stSelectbox select,
    [data-testid="stSidebar"] .stSelectbox select {
        background-color: #FFFFFF;
        color: #1A1A1A;
        border: 1px solid #D0D0D0;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 14px;
        box-shadow: var(--shadow-sm);
    }

    .stSelectbox select:focus {
        border-color: #0066CC;
        box-shadow: 0 0 0 3px rgba(0,102,204,0.1);
    }

    /* ===== BUTTONS ===== */
    .stButton button {
        background-color: #1A1A1A;
        color: #FFFFFF;
        border: 2px solid #1A1A1A;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        width: 100%;
    }

    .stButton button:hover {
        background-color: #2D2D2D;
        border-color: #2D2D2D;
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }

    .stButton button:active {
        background-color: #000000;
        border-color: #000000;
        transform: translateY(0);
    }

    /* Primary Action Button (Upload/Demo) */
    .stButton button[kind="primary"] {
        background-color: #0066CC;
        border-color: #0066CC;
        color: white;
    }

    .stButton button[kind="primary"]:hover {
        background-color: #0052A3;
        border-color: #0052A3;
    }

    /* ===== CARDS & CONTAINERS ===== */
    .metric-card {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid #0066CC;
        margin-bottom: 12px;
        border: 1px solid #E0E0E0;
    }

    .metric-card h3 {
        margin: 0 0 8px 0;
        font-size: 13px;
        color: #666666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card .value {
        font-size: 26px;
        font-weight: 700;
        color: #1A1A1A;
        margin: 4px 0 0 0;
    }

    /* ===== URGENCY BADGES ===== */
    .urgency-high {
        background: #FFF0ED;
        color: #D85A30;
        border: 1px solid #F4CCC8;
        border-radius: 20px;
        padding: 6px 14px;
        font-weight: 600;
        display: inline-block;
        font-size: 13px;
    }

    .urgency-medium {
        background: #FFF8F0;
        color: #E8902A;
        border: 1px solid #F5E6D3;
        border-radius: 20px;
        padding: 6px 14px;
        font-weight: 600;
        display: inline-block;
        font-size: 13px;
    }

    .urgency-low {
        background: #F0F9F4;
        color: #2AAA6E;
        border: 1px solid #D5F0E1;
        border-radius: 20px;
        padding: 6px 14px;
        font-weight: 600;
        display: inline-block;
        font-size: 13px;
    }

    /* ===== HEADER BANNER ===== */
    .app-header {
        background: linear-gradient(135deg, #1A1A1A 0%, #2D2D2D 100%);
        padding: 28px 32px;
        border-radius: 8px;
        margin-bottom: 28px;
        color: white;
        box-shadow: var(--shadow-md);
        border: 1px solid rgba(0,0,0,0.1);
    }

    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 32px;
        font-weight: 700;
    }

    .app-header p {
        color: rgba(255,255,255,0.90);
        margin: 8px 0 0 0;
        font-size: 15px;
        line-height: 1.4;
    }

    /* ===== RESULT SECTIONS ===== */
    .result-section {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 22px 24px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 18px;
        border: 1px solid #E0E0E0;
    }

    .result-section h3 {
        margin: 0 0 12px 0;
        font-size: 16px;
        color: #1A1A1A;
        font-weight: 600;
    }

    /* ===== ALERTS & MESSAGES ===== */
    .stAlert {
        background-color: var(--bg-light);
        border-radius: 6px;
        border-left: 4px solid #0066CC;
    }

    .stSuccess {
        background-color: #F0F9F4;
        border-left-color: #2AAA6E;
        color: #1A1A1A;
    }

    .stWarning {
        background-color: #FFF8F0;
        border-left-color: #E8902A;
        color: #1A1A1A;
    }

    .stError {
        background-color: #FFF0ED;
        border-left-color: #D85A30;
        color: #1A1A1A;
    }

    /* ===== DISCLAIMER BOX ===== */
    .disclaimer {
        background: #FFFBF0;
        border: 1px solid #FFE4CC;
        border-radius: 6px;
        padding: 14px 16px;
        font-size: 13px;
        color: #4A4A4A;
        margin-top: 18px;
        line-height: 1.5;
    }

    /* ===== TABS & EXPANDABLE ===== */
    [data-testid="stTabs"] button {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: #1A1A1A;
        font-weight: 500;
        padding: 8px 16px;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom-color: #0066CC;
        color: #0066CC;
    }

    /* ===== HIDE STREAMLIT ELEMENTS ===== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ===== SEPARATOR ===== */
    hr { 
        border: none;
        border-top: 1px solid #E0E0E0;
        margin: 20px 0;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F5F7FA;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #CCCCCC;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #999999;
    }

    </style>
    """, unsafe_allow_html=True)



# =============================================================
# MAIN APP
# =============================================================

def main():
    inject_css()

    # ── Header banner ──────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>🧠 STCA-Net &nbsp;|&nbsp; EEG Seizure Analysis</h1>
        <p>AI-assisted epileptic seizure classification
           &nbsp;·&nbsp; Spatiotemporal Cross-Attention Network
           &nbsp;·&nbsp; University of Bonn EEG Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar — Patient Info ──────────────────────────────────
    with st.sidebar:
        st.markdown("### 👤 Patient Information")
        pat_name   = st.text_input("Patient Name",
                                    placeholder="Full name")
        pat_id     = st.text_input("Patient ID",
                                    placeholder="e.g. P-2024-001")
        pat_age    = st.text_input("Age",
                                    placeholder="e.g. 28")
        pat_gender = st.selectbox("Gender",
                                   ["Select", "Male", "Female", "Other"])
        pat_doctor = st.text_input("Referring Doctor",
                                    placeholder="Dr. Name")
        pat_dept   = st.text_input("Department",
                                    value="Neurology")
        st.markdown("---")
        st.markdown("### ⚙️ Model Info")
        st.markdown("""
        **Architecture:** STCA-Net v2
        **Parameters:** ~47K
        **Tasks:** Binary / 3-class / 5-class
        **Dataset:** Univ. of Bonn EEG

        **Accuracy:**
        - Binary: 98.17%
        - 3-class: 93.57%
        - 5-class: 75.65%
        """)
        st.markdown("---")
        st.markdown(
            "<small style='color:rgba(255,255,255,0.6)'>"
            "STCA-Net | Research Prototype<br>"
            "Not for clinical use</small>",
            unsafe_allow_html=True)

    patient_info = {
        'name':   pat_name,
        'pid':    pat_id,
        'age':    pat_age,
        'gender': pat_gender,
        'doctor': pat_doctor,
        'dept':   pat_dept,
    }

    # ── Load models (cached) ────────────────────────────────────
    with st.spinner("Loading STCA-Net models..."):
        models = load_all_models()

    loaded = sum(1 for m in models.values() if m is not None)
    if loaded == 0:
        st.error("No models found in `models/` folder. "
                 "Run `python phase6_save_model.py` first.")
        return
    elif loaded < 3:
        st.warning(f"Only {loaded}/3 models loaded. "
                   f"Some tasks may be unavailable.")
    else:
        st.success("All 3 STCA-Net models loaded successfully.")

    st.markdown("---")

    # ── File Upload ─────────────────────────────────────────────
    st.markdown("## 📁 Upload EEG Signal")
    st.markdown(
        "Upload a CSV file containing a single-channel EEG segment "
        "(178 samples at 173.61 Hz). Supports Bonn dataset format "
        "(X1..X178 columns) or a single column of values.")

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Choose EEG CSV file",
            type=['csv'],
            help="CSV with 178 EEG amplitude values")
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.button("🎲 Use Demo Signal",
                              use_container_width=True,
                              help="Generate a synthetic demo EEG")

    # ── Process signal ──────────────────────────────────────────
    raw_signal = None

    if use_demo:
        # Generate a synthetic seizure-like signal
        np.random.seed(7)
        t  = np.linspace(0, 1.025, 178)
        raw_signal = (
            3.5 * np.sin(2 * np.pi * 8 * t) +
            2.0 * np.sin(2 * np.pi * 15 * t) +
            0.5 * np.random.randn(178) +
            np.exp(-((t - 0.5)**2) / 0.02) * 5
        ).astype(np.float32)
        st.info("Using synthetic demo signal (seizure-like pattern).")

    elif uploaded is not None:
        raw_signal, err = parse_uploaded_csv(uploaded)
        if err:
            st.error(f"CSV parse error: {err}")
            return

    if raw_signal is None:
        st.markdown("""
        <div style='background:#FFFFFF; border-radius:8px;
                    padding:48px 32px; text-align:center;
                    border:2px dashed #D0D0D0; color:#999999;
                    margin-top:20px;'>
            <h3 style='color:#999999; font-size:18px; 
                       font-weight:500; margin:0;'>
            📤 Upload a CSV or click <strong>"Use Demo Signal"</strong> 
            to begin analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Show raw signal ─────────────────────────────────────────
    st.markdown("## 📈 EEG Signal")
    signal_norm, processed = preprocess_eeg(raw_signal)
    st.plotly_chart(
        plot_eeg_signal(signal_norm, "Uploaded EEG — z-score normalised"),
        use_container_width=True)

    st.markdown("---")

    # ── Run inference ───────────────────────────────────────────
    st.markdown("## 🤖 AI Analysis Results")

    with st.spinner("Running STCA-Net inference..."):
        results = run_inference(models, processed)

    # ── Summary cards ───────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, (task, cfg) in zip([c1, c2, c3], TASK_CONFIG.items()):
        res = results.get(task)
        if res is None:
            continue
        urg_cls = f"urgency-{res['urgency'].lower()}"
        with col:
            st.markdown(f"""
<div class="result-section">
    <h4 style='color:#1A6FBF; margin:0 0 8px 0;
               font-size:14px;'>{cfg['icon']}
        {cfg['label']}</h4>
    <div style='font-size:18px; font-weight:700;
                color:#1A3A5C; margin-bottom:6px;'>
        {res['class_name']}</div>
    <div style='font-size:13px; color:#666;
                margin-bottom:8px;'>
        Confidence: <b>{res['confidence']:.1f}%</b></div>
    <span class="{urg_cls}">{res['urgency']}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Detailed results per task ───────────────────────────────
    st.markdown("## 📊 Detailed Results")
    tabs = st.tabs([
        f"{TASK_CONFIG[t]['icon']} {TASK_CONFIG[t]['label']}"
        for t in TASK_CONFIG])

    for tab, (task, cfg) in zip(tabs, TASK_CONFIG.items()):
        res = results.get(task)
        if res is None:
            with tab:
                st.warning("Model not available for this task.")
            continue

        with tab:
            col_l, col_r = st.columns([1, 1])

            # Left: probability chart
            with col_l:
                st.plotly_chart(
                    plot_probability_bar(
                        res['probabilities'],
                        cfg['class_names'],
                        cfg['colors'],
                        f"Class Probabilities — {cfg['label']}"),
                    use_container_width=True)

            # Right: confidence gauge + clinical text
            with col_r:
                st.plotly_chart(
                    plot_confidence_gauge(
                        res['confidence'],
                        res['urgency_color'],
                        "Confidence"),
                    use_container_width=True)

                urg_cls = f"urgency-{res['urgency'].lower()}"
                st.markdown(f"""<div style='background:#F8FAFC; border-radius:8px; padding:14px; margin-top:8px;'>  <b>Predicted:</b> {res['class_name']}<br>  <b>Urgency:</b>  <span class="{urg_cls}">{res['urgency']}</span>  <br><br>
    <b>Clinical Note:</b><br>
    <span style='color:#444; font-size:13px;'>
    {res['clinical_text']}</span>
</div>
""", unsafe_allow_html=True)
            st.markdown("#### Grad-CAM Explanation")
            with st.spinner("Computing Grad-CAM..."):
                model = models.get(task)
                if model:
                    heatmap = compute_gradcam_app(
                        model, processed, res['predicted'])
                    st.plotly_chart(
                        plot_eeg_with_gradcam(
                            signal_norm, heatmap,
                            res['class_name'],
                            cfg['colors'][res['predicted']]),
                        use_container_width=True)
                    peak_ms = np.argmax(heatmap) / FS * 1000
                    st.caption(
                        f"Peak importance at **{peak_ms:.0f} ms** — "
                        f"the model focused most on this time point "
                        f"when predicting '{res['class_name']}'.")

    st.markdown("---")

    # ── Download PDF Report ─────────────────────────────────────
    st.markdown("## 📄 Download Clinical Report")

    if not pat_name:
        st.info("Enter patient information in the sidebar "
                "to personalise the report.")

    with st.spinner("Generating PDF report..."):
        pdf_bytes = generate_pdf_report(
            patient_info, signal_norm, results)

    fname = (f"STCA_Net_Report_"
             f"{pat_name.replace(' ','_') or 'Patient'}_"
             f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
             f".pdf")

    st.download_button(
        label     = "⬇️ Download PDF Report",
        data      = pdf_bytes,
        file_name = fname,
        mime      = "application/pdf",
        use_container_width = True,
    )

    # ── Disclaimer ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="disclaimer">
            <b>Disclaimer:</b> This is an AI-assisted research prototype
            based on the STCA-Net model trained on the University of Bonn
            EEG dataset. It is intended for research and educational
            purposes only. Results do NOT constitute a clinical diagnosis
            and must be reviewed by a qualified neurologist before any
            medical decision is made.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
