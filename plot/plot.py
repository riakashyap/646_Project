import matplotlib.pyplot as plt
import numpy as np
import json
import os

files = {
    "20251203T204417--metrics__ragarorig_200.json":          "RAGAR",
    "20251203T211209--metrics__200.json":                    "Custom",
    "20251203T222336--metrics__madrverdict_200.json":        "Custom + MADR verdict",
    "20251204T143233--metrics__rerank_200.json":             "Custom + Reranker",
    "20251204T193623--metrics__rerank_madrverdict_200.json": "Custom + Reranker + MADR verdict",
    "20251204T215602--metrics__madrstop_200.json":           "Custom + MADR stop check"
}

def get_data_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return {
        "support_f1": data.get("support_f1", 0),
        "refute_f1": data.get("refute_f1", 0),
        "weighted_f1": data.get("weighted_f1", 0)
    }

labels = []
support_scores = []
refute_scores = []
weighted_scores = []

for filename, custom_label in files.items():
    labels.append(custom_label)
    data = get_data_from_file(filename)
    support_scores.append(data['support_f1'])
    refute_scores.append(data['refute_f1'])
    weighted_scores.append(data['weighted_f1'])

x = np.arange(len(labels))  
width = 0.25  
fig, ax = plt.subplots(figsize=(12, 8))

rects1 = ax.bar(x - width, support_scores, width, label='Support F1', color="#6c90ca")
rects2 = ax.bar(x, refute_scores, width, label='Refute F1', color='#dd8452')
rects3 = ax.bar(x + width, weighted_scores, width, label='Weighted F1', color="#f1d780")

ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15) 
ax.set_ylim(0, 1.0) 
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3) 
ax.grid(axis='y', linestyle='--', alpha=0.7)

def label_top(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(
                f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', 
                va='bottom', 
                fontsize=8
            )

label_top(rects1)
label_top(rects2)
label_top(rects3)

plt.tight_layout()
plt.show()