import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats

# --- Data Definition ---
# Structure: [Mean, Std, Count]
data_with = {
    # Solo (N=15)
    "Transparency (Solo)": [4.27, 1.00, 15],
    "Understanding (Solo)": [3.73, 1.06, 15],
    "Enjoyment (Solo)": [3.27, 1.12, 15],
    "Intent to Use (Solo)": [3.87, 0.88, 15],
    "Clarity (Solo)": [3.93, 1.00, 15],

    # Group (N=16)
    "Transparency (Group)": [3.69, 1.26, 16],
    "Understanding (Group)": [3.63, 1.17, 16],
    "Fairness (Group)": [3.63, 1.32, 16],
    "Objectivity (Group)": [3.69, 1.16, 16],
    "Enjoyment (Group)": [3.69, 1.04, 16],
    "Intent to Use (Group)": [3.75, 1.20, 16],
    "Clarity (Group)": [3.81, 1.07, 16]
}

data_without = {
    # Solo (N=14)
    "Transparency (Solo)": [1.79, 1.08, 14],
    "Understanding (Solo)": [2.14, 1.06, 14],
    "Enjoyment (Solo)": [1.64, 0.89, 14],
    "Intent to Use (Solo)": [2.07, 1.16, 14],
    "Clarity (Solo)": [3.07, 1.28, 14],

    # Group (N=14)
    "Transparency (Group)": [1.79, 1.26, 14],
    "Understanding (Group)": [1.64, 0.81, 14],
    "Fairness (Group)": [1.64, 0.97, 14],
    "Objectivity (Group)": [1.79, 1.01, 14],
    "Enjoyment (Group)": [1.64, 0.89, 14],
    "Intent to Use (Group)": [1.93, 1.03, 14],
    "Clarity (Group)": [2.86, 1.36, 14]
}

# --- 1. Hypothesis Testing (Including Clarity) ---
results_list = []
# Testing Clarity specifically to confirm values
for metric in ["Clarity (Solo)", "Clarity (Group)"]:
    m1, s1, n1 = data_with[metric]
    m2, s2, n2 = data_without[metric]
    t_stat, p_val = ttest_ind_from_stats(mean1=m1, std1=s1, nobs1=n1,
                                         mean2=m2, std2=s2, nobs2=n2,
                                         equal_var=False)
    results_list.append({"Metric": metric, "p-value": p_val, "Significant": p_val < 0.05})

print("Clarity T-Test Results:", results_list)

# --- 2. Plotting (5 Hypotheses) ---
groups = [
    ("H1: Transparency\n& Understanding",
     ["Transparency (Solo)", "Transparency (Group)", "Understanding (Solo)", "Understanding (Group)"]),
    ("H2: Trust\n(Group)", ["Fairness (Group)", "Objectivity (Group)"]),
    ("H3: Enjoyment", ["Enjoyment (Solo)", "Enjoyment (Group)"]),
    ("H4: Intent\nto Use", ["Intent to Use (Solo)", "Intent to Use (Group)"]),
    ("H5: Clarity", ["Clarity (Solo)", "Clarity (Group)"])
]

all_metrics_ordered = []
labels_clean = []
for g_name, metrics in groups:
    for m in metrics:
        all_metrics_ordered.append(m)
        labels_clean.append(m.replace(" (Solo)", "\n(Solo)").replace(" (Group)", "\n(Group)"))

means_w = [data_with[m][0] for m in all_metrics_ordered]
se_w = [data_with[m][1] / np.sqrt(data_with[m][2]) for m in all_metrics_ordered]
means_no = [data_without[m][0] for m in all_metrics_ordered]
se_no = [data_without[m][1] / np.sqrt(data_without[m][2]) for m in all_metrics_ordered]

x = np.arange(len(all_metrics_ordered))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 7))
rects1 = ax.bar(x - width / 2, means_w, width, label='With Explanation', yerr=se_w, capsize=5, color='#4e79a7',
                edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width / 2, means_no, width, label='No Explanation', yerr=se_no, capsize=5, color='#e15759',
                edgecolor='black', alpha=0.9)

ax.set_ylabel('User Rating (1-5)', fontsize=12, fontweight='bold')
ax.set_title('Study Results: 5 Hypotheses', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels_clean, rotation=0, fontsize=9)
ax.set_ylim(0, 5.5)
ax.legend(fontsize=10, loc='upper left')
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Visual Separators
dividers = [3.5, 5.5, 7.5, 9.5]
for d in dividers:
    ax.axvline(x=d, color='gray', linestyle='--', linewidth=1)

# Group Labels
# Positions: 0-3 (avg 1.5), 4-5 (avg 4.5), 6-7 (avg 6.5), 8-9 (avg 8.5), 10-11 (avg 10.5)
group_positions = [1.5, 4.5, 6.5, 8.5, 10.5]
group_names = [g[0] for g in groups]

for pos, name in zip(group_positions, group_names):
    ax.text(pos, 5.2, name, ha='center', fontsize=11, fontweight='bold', color='#333')

plt.tight_layout()
plt.savefig('five_hypotheses_results.png')