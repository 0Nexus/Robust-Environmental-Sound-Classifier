"""
Robust Environmental Sound Classifier
======================================
Builds a classifier for environmental sounds (subset of ESC-50), then
diagnoses how performance collapses under realistic audio degradations
(additive noise, low-pass "muffling", downsampling), and applies a
mitigation (noise-augmented training) to recover accuracy.

This mirrors the real-world workflow of: explore -> diagnose limitation ->
devise work-around -> measure improvement.

Dataset: ESC-50 (https://github.com/karoldvl/ESC-50)
Run from a directory containing the cloned esc50_repo/ folder.
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

RNG = 42
np.random.seed(RNG)

DATA_DIR = "esc50_repo"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_PATH = os.path.join(DATA_DIR, "meta", "esc50.csv")

# 8 classes spanning urban / household / alert sounds — good mix of
# tonal (siren, clock_alarm), broadband (rain, vacuum_cleaner) and
# transient (glass_breaking, door_wood_knock) signal characteristics.
SELECTED_CLASSES = [
    "siren",
    "dog",
    "car_horn",
    "glass_breaking",
    "clock_alarm",
    "vacuum_cleaner",
    "rain",
    "door_wood_knock",
]

SR = 22050  # target sample rate


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
def extract_features(y, sr=SR):
    """Summarise an audio clip as a fixed-length feature vector using
    MFCCs + spectral descriptors (mean & std pooled over time)."""
    if len(y) < sr // 10:
        y = np.pad(y, (0, sr // 10 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    feats = []
    for block in (mfcc, centroid, bandwidth, rolloff, zcr, rms):
        feats.append(block.mean(axis=1))
        feats.append(block.std(axis=1))
    return np.concatenate(feats)


# ---------------------------------------------------------------------
# Degradation functions — simulate real-world capture conditions
# ---------------------------------------------------------------------
def add_white_noise(y, snr_db=5):
    """Add white noise at a target signal-to-noise ratio (dB)."""
    sig_power = np.mean(y ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise


def low_pass_muffle(y, sr=SR, cutoff=1000):
    """Simulate a muffled/through-a-wall recording via a simple FFT
    low-pass filter."""
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    Y[freqs > cutoff] = 0
    return np.fft.irfft(Y, n=len(y))


def downsample_roundtrip(y, sr=SR, target_sr=8000):
    """Simulate a low-quality capture device by downsampling then
    upsampling back (loses high-frequency detail irreversibly)."""
    y_low = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return librosa.resample(y_low, orig_sr=target_sr, target_sr=sr)


def degrade(y, sr=SR):
    """Apply a realistic *combined* degradation: muffling + downsampling
    + moderate noise, simulating a cheap mic in a noisy environment."""
    y = low_pass_muffle(y, sr, cutoff=3000)
    y = downsample_roundtrip(y, sr, target_sr=8000)
    y = add_white_noise(y, snr_db=5)
    return y


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_dataset():
    meta = pd.read_csv(META_PATH)
    meta = meta[meta["category"].isin(SELECTED_CLASSES)].reset_index(drop=True)
    print(f"Loaded metadata: {len(meta)} clips across {meta['category'].nunique()} classes")

    audio_clean, labels = [], []
    for _, row in meta.iterrows():
        path = os.path.join(AUDIO_DIR, row["filename"])
        y, _ = librosa.load(path, sr=SR)
        audio_clean.append(y)
        labels.append(row["category"])
    return audio_clean, labels


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main():
    print("Loading ESC-50 subset...")
    audio_clean, labels = load_dataset()

    print("Splitting train/test (stratified, 75/25)...")
    idx = np.arange(len(audio_clean))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.25, stratify=labels, random_state=RNG
    )

    y_train = [labels[i] for i in idx_train]
    y_test = [labels[i] for i in idx_test]

    # ---- Baseline model: trained on clean audio only ----
    print("Extracting features (clean train set)...")
    X_train_clean = np.array([extract_features(audio_clean[i]) for i in idx_train])

    print("Training baseline RandomForest (clean audio only)...")
    clf_baseline = RandomForestClassifier(n_estimators=300, random_state=RNG)
    clf_baseline.fit(X_train_clean, y_train)

    # ---- Test conditions ----
    print("Building test sets: clean / noisy / degraded...")
    X_test_clean = np.array([extract_features(audio_clean[i]) for i in idx_test])
    X_test_noisy = np.array(
        [extract_features(add_white_noise(audio_clean[i], snr_db=5)) for i in idx_test]
    )
    X_test_degraded = np.array(
        [extract_features(degrade(audio_clean[i])) for i in idx_test]
    )

    acc_clean = accuracy_score(y_test, clf_baseline.predict(X_test_clean))
    acc_noisy = accuracy_score(y_test, clf_baseline.predict(X_test_noisy))
    acc_degraded = accuracy_score(y_test, clf_baseline.predict(X_test_degraded))

    print(f"\nBaseline model (trained on clean audio only):")
    print(f"  Accuracy on clean test audio     : {acc_clean:.3f}")
    print(f"  Accuracy on noisy test audio      : {acc_noisy:.3f}")
    print(f"  Accuracy on degraded test audio   : {acc_degraded:.3f}")

    # ---- Mitigation: noise-augmented training ----
    print("\nApplying mitigation: augmenting training set with degraded copies...")
    X_train_aug = []
    y_train_aug = []
    for i in idx_train:
        y_clean = audio_clean[i]
        X_train_aug.append(extract_features(y_clean))
        y_train_aug.append(labels[i])
        # add one noisy and one fully-degraded copy of each training clip
        X_train_aug.append(extract_features(add_white_noise(y_clean, snr_db=5)))
        y_train_aug.append(labels[i])
        X_train_aug.append(extract_features(degrade(y_clean)))
        y_train_aug.append(labels[i])
    X_train_aug = np.array(X_train_aug)

    clf_robust = RandomForestClassifier(n_estimators=300, random_state=RNG)
    clf_robust.fit(X_train_aug, y_train_aug)

    acc_clean_r = accuracy_score(y_test, clf_robust.predict(X_test_clean))
    acc_noisy_r = accuracy_score(y_test, clf_robust.predict(X_test_noisy))
    acc_degraded_r = accuracy_score(y_test, clf_robust.predict(X_test_degraded))

    print(f"\nRobust model (trained with noise/degradation augmentation):")
    print(f"  Accuracy on clean test audio     : {acc_clean_r:.3f}")
    print(f"  Accuracy on noisy test audio      : {acc_noisy_r:.3f}")
    print(f"  Accuracy on degraded test audio   : {acc_degraded_r:.3f}")

    # ---- Save results ----
    results = {
        "classes": SELECTED_CLASSES,
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "baseline": {
            "clean": acc_clean,
            "noisy": acc_noisy,
            "degraded": acc_degraded,
        },
        "robust_augmented": {
            "clean": acc_clean_r,
            "noisy": acc_noisy_r,
            "degraded": acc_degraded_r,
        },
        "recovery": {
            "noisy_recovery_pp": round((acc_noisy_r - acc_noisy) * 100, 1),
            "degraded_recovery_pp": round((acc_degraded_r - acc_degraded) * 100, 1),
        },
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed classification report (degraded audio, robust model):")
    print(classification_report(y_test, clf_robust.predict(X_test_degraded)))

    print("\nSaved results.json")


if __name__ == "__main__":
    main()
