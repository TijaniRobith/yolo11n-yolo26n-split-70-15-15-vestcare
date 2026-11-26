import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C11, C26 = "#1f77b4", "#ff7f0e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
import os
OUT = os.path.dirname(os.path.abspath(__file__)) + os.sep
w = 0.38


def label(ax, bars, fmt="{:.3f}", dy=0.008):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=8)


# 1) Perbandingan utama (test, mask) -- 2 panel
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 2.2]})
bars = a.bar(["yolo11n-seg", "yolo26n-seg"], [0.3001, 0.3214], color=[C11, C26], width=0.6)
label(a, bars, "{:.4f}")
a.set_ylim(0, 0.40)
a.set_ylabel("test mask mAP50-95")
a.set_title("Perbandingan model (split 70:15:15, 1 seed)", fontsize=10)
ind = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
m11 = [0.626, 0.591, 0.608, 0.5668, 0.3001]
m26 = [0.759, 0.549, 0.637, 0.5991, 0.3214]
x = np.arange(len(ind))
r1 = b.bar(x - w / 2, m11, w, label="yolo11n-seg", color=C11)
r2 = b.bar(x + w / 2, m26, w, label="yolo26n-seg", color=C26)
label(b, r1, "{:.2f}", 0.01)
label(b, r2, "{:.2f}", 0.01)
b.set_xticks(x)
b.set_xticklabels(ind)
b.set_ylim(0, 1.0)
b.set_title("Semua indikator mask (test)", fontsize=10)
b.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(OUT + "perbandingan_model.png", dpi=150)
plt.close()

# 2) Per kelas (test, mask)
kelas = ["pohon", "lubang", "orang", "motor", "mobil"]
ap11 = [0.570, 0.101, 0.211, 0.243, 0.375]
ap26 = [0.575, 0.129, 0.229, 0.306, 0.368]
a50_11 = [0.889, 0.226, 0.422, 0.548, 0.748]
a50_26 = [0.943, 0.310, 0.416, 0.607, 0.720]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, d11, d26, t, ymax in ((axs[0], ap11, ap26, "mask mAP50-95 per kelas (test)", 0.75),
                              (axs[1], a50_11, a50_26, "mask mAP50 per kelas (test)", 1.05)):
    x = np.arange(len(kelas))
    r1 = ax.bar(x - w / 2, d11, w, label="yolo11n-seg", color=C11)
    r2 = ax.bar(x + w / 2, d26, w, label="yolo26n-seg", color=C26)
    label(ax, r1, "{:.2f}", 0.01)
    label(ax, r2, "{:.2f}", 0.01)
    ax.set_xticks(x)
    ax.set_xticklabels(kelas)
    ax.set_ylim(0, ymax)
    ax.set_title(t, fontsize=10)
axs[0].legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT + "per_kelas.png", dpi=150)
plt.close()

# 3) Val vs test
fig, ax = plt.subplots(figsize=(8.5, 4.2))
grup = ["mask mAP50-95\nval", "mask mAP50-95\ntest", "box mAP50-95\nval", "box mAP50-95\ntest"]
g11 = [0.292, 0.3001, 0.384, 0.378]
g26 = [0.285, 0.3214, 0.364, 0.390]
x = np.arange(len(grup))
r1 = ax.bar(x - w / 2, g11, w, label="yolo11n-seg", color=C11)
r2 = ax.bar(x + w / 2, g26, w, label="yolo26n-seg", color=C26)
label(ax, r1, "{:.3f}", 0.006)
label(ax, r2, "{:.3f}", 0.006)
ax.set_xticks(x)
ax.set_xticklabels(grup)
ax.set_ylim(0, 0.5)
ax.set_title("Konsistensi val vs test (1 seed)", fontsize=10)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT + "val_vs_test.png", dpi=150)
plt.close()

# 4) Efisiensi
fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
ax = axs[0]
x = np.arange(3)
r1 = ax.bar(x - w / 2, [5.12, 5.31, 10.60], w, label="yolo11n-seg", color=C11)
r2 = ax.bar(x + w / 2, [4.17, 5.36, 5.03], w, label="yolo26n-seg", color=C26)
label(ax, r1, "{:.2f}", 0.05)
label(ax, r2, "{:.2f}", 0.05)
ax.set_xticks(x)
ax.set_xticklabels(["skrip 07", "laporan #1", "laporan #2"])
ax.set_ylim(0, 13)
ax.set_title("Inferensi test (ms/gambar)", fontsize=9)
ax.legend(frameon=False, fontsize=7, loc="upper left")
data = [("Parameter fused (juta)", [2.850, 2.705], "{:.3f}"),
        ("GFLOPs @640 (fused)", [9.7, 9.1], "{:.1f}"),
        ("Ukuran best.pt (MB)", [6.0, 6.6], "{:.1f}")]
for ax, (t, vals, f) in zip(axs[1:], data):
    bars = ax.bar(["11n", "26n"], vals, color=[C11, C26], width=0.6)
    label(ax, bars, f, max(vals) * 0.01)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title(t, fontsize=9)
plt.tight_layout()
plt.savefig(OUT + "efisiensi.png", dpi=150)
plt.close()
print("ok")
