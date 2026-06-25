"""
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
        print(f"Loaded {len(self.models)}/3 models.\n")

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
                f"\n[{cfg['description']}]",
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

        lines += ["\n" + "=" * 50,
                  "DISCLAIMER: This is an AI-assisted screening "
                  "tool only. It does not constitute a clinical "
                  "diagnosis. Always consult a qualified "
                  "neurologist for medical decisions.",
                  "=" * 50]
        return "\n".join(lines)


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

    print("\n" + engine.get_summary(result))
