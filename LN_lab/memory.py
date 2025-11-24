# Plot GPU memory usage for methods using the provided IBM palette.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log10

# --- Data ---
methods = ["LN-TTA (Ours)", "BFTT3D", "DUA", "TENT"]
memory_values = [1468, 7443, 76000.45, 2890]  # GB

# IBM Design Library palette
palette = [
    "#0F62FE",  # IBM Blue
    "#8A3FFC",  # IBM Purple
    "#EE5396",  # IBM Magenta
    "#FF832B",  # IBM Orange
    "#F1C21B",  # IBM Yellow (unused)
]
color_map = {
    "LN-TTA (Ours)": palette[0],
    "DUA": palette[1],
    "BFTT3D": palette[2],
    "TENT": palette[3],
}

colors = [color_map[m] for m in methods]

# Helper to annotate bars with values (e.g., "7,443 GB")
def annotate_bars(ax, bars, fmt="{:,.2f} GB"):
    for rect, val in zip(bars, memory_values):
        height = rect.get_height()
        label = f"{val:,.2f} GB" if isinstance(val, float) and not float(val).is_integer() else f"{val:,.0f} GB"
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=16, rotation=0)

# --- Figure 2: Log scale bar chart (to reveal order-of-magnitude gaps) ---
plt.figure(figsize=(6, 3))
ax = plt.gca()
x = np.arange(len(methods))
bars_log = plt.bar(x, memory_values, color=colors, edgecolor="black", linewidth=1.2)
for i, rect in enumerate(bars_log):
    if methods[i] == "LN-TTA (Ours)":
        rect.set_linewidth(2.5)
        rect.set_hatch("//")
plt.yscale("log")
plt.xticks(x, methods, fontsize=12)
plt.ylabel("Memory (GB, log scale)", fontsize=16)
# plt.title("GPU Memory Usage (GB, log scale) — Lower is better", fontsize=13)
plt.grid(axis="y", linestyle=":", alpha=0.5, which="both")
# Place annotations slightly above bars (handle log scale)
for rect, val in zip(bars_log, memory_values):
    label = f"{val:,.2f} GB" if isinstance(val, float) and not float(val).is_integer() else f"{val:,.0f} GB"
    ax = plt.gca()
    ax.annotate(label,
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=14, rotation=0)

plt.tight_layout()
log_path = "memory_usage_log.png"
plt.subplots_adjust(top=0.94, bottom=0.20)  # tweak to taste
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Optional: also disable ticks on those sides
ax.tick_params(top=False, right=False)

plt.tight_layout()
plt.savefig("memory_usage_log.png", dpi=200)
plt.show()
