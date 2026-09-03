# create_final_report.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

langs = ['Hindi', 'English', 'Japanese']
kl_matrix = np.array([
    [0.0,    20.9432, 20.8830], 
    [25.4036, 0.0,    16.0528], 
    [24.8487, 20.2926, 0.0    ]  
])
entropy_data = {
    'Language': ['Hindi', 'Hindi', 'English', 'English', 'Japanese', 'Japanese'],
    'Source': ['Real Corpus', 'Baseline', 'Real Corpus', 'Baseline', 'Real Corpus', 'Baseline'],
    'Entropy': [9.1554, 9.0687, 8.8581, 8.7871, 10.1760, 9.9832]
}
entropy_df = pd.DataFrame(entropy_data)
sns.set_theme(style="white", palette="muted")
fig, axes = plt.subplots(1, 2, figsize=(18, 7)) 

sns.heatmap(kl_matrix, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=langs, yticklabels=langs, ax=axes[0],
            annot_kws={"size": 14})
axes[0].set_title('A) Dissimilarity Between Language Corpora\n(Global KL Divergence)', fontsize=16, pad=20)
axes[0].set_ylabel('P (Source Distribution)', fontsize=12)
axes[0].set_xlabel('Q (Target Distribution)', fontsize=12)

# Plot 2 (Right): Bar Chart of Entropy Comparison
sns.barplot(x='Language', y='Entropy', hue='Source', data=entropy_df, ax=axes[1])
axes[1].set_title('B) Real vs. Baseline Word Order Entropy', fontsize=16, pad=20)
axes[1].set_xlabel('Language', fontsize=12)
axes[1].set_ylabel('Word Order Entropy (bits)', fontsize=12)
axes[1].legend(title='Data Source')
# Add text labels on top of the bars
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=12)

# Final adjustments and saving
fig.suptitle('Project Findings: Comparing Real and Baseline Language Models', fontsize=22, y=1.03)
plt.tight_layout()
plt.savefig('final_project_report.png', dpi=300, bbox_inches='tight')
print("\nGenerated final report: 'final_project_report.png'")

plt.show()