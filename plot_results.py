import matplotlib.pyplot as plt
import numpy as np

# Data from the evaluation results
labels = [
    'MS Cross-Attn\n(Dec30)',
    'VDiff FiLM\n(Jan05)',
    'VDiff MS Cross-Attn\n(Jan06)',
    'Reference\nPrimitive'
]

# Success rates by difficulty
success_easy = [0.8436, 0.9231, 0.9344, 1.0]
success_medium = [0.8253, 0.9022, 0.9213, 1.0]
success_hard = [0.5295, 0.6366, 0.7099, 1.0]

# Median search times (ms) by difficulty
time_easy_ms = [706.45, 604.56, 722.12, 431.55]
time_medium_ms = [1027.19, 906.18, 1023.15, 1252.88]
time_hard_ms = [2198.67, 2360.24, 2375.91, 6545.99]

# Convert to seconds
time_easy_s = [t / 1000 for t in time_easy_ms]
time_medium_s = [t / 1000 for t in time_medium_ms]
time_hard_s = [t / 1000 for t in time_hard_ms]

# Create figure with 2 rows x 3 columns
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Colors
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

difficulties = ['Easy', 'Medium', 'Hard']
success_data = [success_easy, success_medium, success_hard]
time_data = [time_easy_s, time_medium_s, time_hard_s]

x = np.arange(len(labels))

# Row 1: Success Rates
for i, (diff, success) in enumerate(zip(difficulties, success_data)):
    ax = axes[0, i]
    bars = ax.bar(x, success, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Success Rate', fontsize=11)
    ax.set_title(f'{diff} - Success Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Row 2: Median Search Times
for i, (diff, times) in enumerate(zip(difficulties, time_data)):
    ax = axes[1, i]
    bars = ax.bar(x, times, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Search Time (seconds)', fontsize=11)
    ax.set_title(f'{diff} - Median Search Time', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('evaluation_results_by_difficulty.png', dpi=150, bbox_inches='tight')
plt.savefig('evaluation_results_by_difficulty.pdf', bbox_inches='tight')
print("Saved: evaluation_results_by_difficulty.png and evaluation_results_by_difficulty.pdf")
plt.show()
