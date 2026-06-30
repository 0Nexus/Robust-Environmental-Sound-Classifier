import json
import matplotlib.pyplot as plt
import numpy as np

with open("results.json") as f:
    r = json.load(f)

conditions = ["Clean", "Noisy\n(SNR 5dB)", "Degraded\n(muffled+downsampled+noise)"]
baseline = [r["baseline"]["clean"], r["baseline"]["noisy"], r["baseline"]["degraded"]]
robust = [r["robust_augmented"]["clean"], r["robust_augmented"]["noisy"], r["robust_augmented"]["degraded"]]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, baseline, width, label="Trained on clean audio only", color="#d9534f")
bars2 = ax.bar(x + width/2, robust, width, label="Trained with noise/degradation augmentation", color="#5cb85c")

ax.set_ylabel("Classification Accuracy")
ax.set_title("Robustness of Environmental Sound Classifier to Audio Degradation")
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 1.0)
ax.legend(loc="lower left")
ax.grid(axis="y", alpha=0.3)

for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.0%}", (b.get_x() + b.get_width()/2, h), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("results_chart.png", dpi=150)
print("Saved results_chart.png")
