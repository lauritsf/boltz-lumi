"""Plot warm inference wall times before and after optimization."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


STAGES = {
    'pairformer_s': 'Pairformer',
    'alignment_s': 'Sequence alignments',
    'templates_s': 'Templates',
    'diffusion_conditioning_s': 'Diffusion conditioning',
    'diffusion_sampling_s': 'Diffusion sampling',
    'confidence_s': 'Confidence',
    'other_s': 'Other',
}


def seconds_label(seconds):
    decimals = 1 if seconds < 60 else 0
    return f"{seconds:,.{decimals}f} s"


def plot(results):
    rows = {(row['case'], row['recipe']): row for row in results['stage_times'] if row['mode'] == 'warm'}
    cases = list(dict.fromkeys(case for case, _ in rows))
    sns.set_theme(style='darkgrid', palette='muted')
    fig, axes = plt.subplots(1, len(cases), sharey=False, figsize=(12, 4), layout='constrained',
                             gridspec_kw={'wspace': 0.25})
    for ax, case in zip(np.atleast_1d(axes), cases):
        before, after = rows[case, 'upstream'], rows[case, 'optimized']
        bottom = np.zeros(2)
        for color, (key, label) in enumerate(STAGES.items()):
            values = np.array([before[key], after[key]])
            ax.bar(['Before', 'After'], values, bottom=bottom, color=f'C{color}', label=label)
            bottom += values
        size = results['input_sizes'][case]
        biology = f"{size['protein_residues']:,} residues"
        if size['ligand_heavy_atoms']:
            biology += f" + {size['ligand_heavy_atoms']} ligand atoms"
        ax.set_title(f"{before['label']}\n{biology}\n{before['tokens']:,} tokens", fontsize=11)
        ax.set_ylim(0, before['total_s'])
        ax.set_yticks([])
        ax.set_xticks([0, 1], [f"Before\n{seconds_label(before['total_s'])}", f"After\n{seconds_label(after['total_s'])}"])
        ax.grid(False)
        sns.despine(ax=ax, left=True, bottom=True)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc='outside lower center',
               ncols=4, frameon=False)
    return fig


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    outputs = root / 'outputs'
    results = json.loads((outputs / 'benchmark-results.json').read_text())
    plot(results).savefig(outputs / 'benchmark-timings.svg', metadata={'Date': None})
