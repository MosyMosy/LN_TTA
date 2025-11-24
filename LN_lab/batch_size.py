# Re-create the plot using the provided IBM Design Library palette
import matplotlib.pyplot as plt
import numpy as np

# Data for LN-TTA and TENT methods
batch_sizes = np.array([2, 4, 8, 16, 32])
ln_tta_acc = np.array([42.58, 42.05, 45.955, 63.29, 64.38])
tent_acc   = np.array([4.58,  8.80, 19.70, 36.86, 54.73])

# IBM Design Library palette
palette = [
    "#0F62FE",  # IBM Blue
    "#8A3FFC",  # IBM Purple
    "#EE5396",  # IBM Magenta
    "#FF832B",  # IBM Orange
    "#F1C21B",  # IBM Yellow
]

# Choose distinct colors from the palette
color_ln_tta = palette[0]  # IBM Blue
color_tent   = palette[2]  # IBM Magenta

# Define font size variables
axis_label_fontsize = 18
axis_number_fontsize = 14
legend_fontsize = 18

# Create the plot
plt.figure(figsize=(6, 3))

# Plot LN-TTA (solid line, circle markers)
plt.plot(
    batch_sizes, ln_tta_acc,
    marker='o', linestyle='-', linewidth=2, markersize=6,
    label='LN-TTA', color=color_ln_tta
)

# Plot TENT (dashed line, square markers)
plt.plot(
    batch_sizes, tent_acc,
    marker='s', linestyle='--', linewidth=2, markersize=6,
    label='TENT', color=color_tent
)

# Labels and ticks
plt.xlabel("Batch Size", fontsize=axis_label_fontsize)
plt.ylabel("Top-1 Acc (%)", fontsize=axis_label_fontsize)
plt.xticks(fontsize=axis_number_fontsize)
plt.yticks(fontsize=axis_number_fontsize)

# Legend
plt.legend(fontsize=legend_fontsize, loc='lower right')

# Log scale for x-axis
plt.xscale("log", base=2)

# Grid and layout
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save and show
out_path = "batch_size.pdf"
plt.savefig(out_path, bbox_inches="tight")
plt.show()

out_path
