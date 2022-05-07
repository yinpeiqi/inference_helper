from matplotlib import pyplot as plt
import numpy as np


base = [[13.49, 1446.35, 853.81, 6.84, 926.46, 490.73, 14.12, 1981.97, 916.31],
        [1, 1, 1, 1, 1, 1, 1, 1, 1] ]

dgir = [[12.99, 1421.95, 839.6, 7.04, 873.62, 478.29, 12.76, 1810.26, 903.95], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1] ]
base_re = [[8.98, 1208.82, 699, 5.91, 690.11, 365.2, 10.38, 1712.02, 819.95], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1] ]
dgi = [[8.67, 985.93, 605.68, 5.91, 612.54, 318.16, 9.44, 1271.45, 664.42], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1] ]
models = ["GAT", "GCN", "JKNet"]
datasets = ["ogbn-products", "friendster", "ogbn-papers100M"]
hardware = ["16GB GPU", "32GB GPU"]

fig = plt.figure(figsize=[20, 6])
width = 0.2
x_labels = datasets

for row in range(2):
    for col in range(3):
        x1 = x2 = x3 = x4 = np.arange(0, 3, 1)
        y1, y2, y3, y4 = [], [], [], []
        for i in range(col*3, col*3 + 3):
            y1.append(1.0)
            y2.append(base[row][i] / dgir[row][i])
            y3.append(base[row][i] / base_re[row][i])
            y4.append(base[row][i] / dgi[row][i])

        ax = fig.add_subplot(2, 3, row*3 + col + 1)

        ax.bar(x1 - 1.5 * width, y1, width, label='bottom-up', hatch='...', color='aliceblue', edgecolor='black')
        ax.bar(x2 - 0.5 * width, y2, width, label='DGI', hatch='///', color='lightskyblue', edgecolor='black')
        ax.bar(x3 + 0.5 * width, y3, width, label='bottom-up-rcmk', hatch='\\\\\\', color='royalblue', edgecolor='black')
        ax.bar(x4 + 1.5 * width, y4, width, label='DGI-rcmk', hatch='xxx', color='black', edgecolor='black')

        ax.set_ylabel('Speedup', fontsize=14)
        ax.set_xticks(x1, x_labels, fontsize=14)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0.8, 1.7)
        ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6], ['0.8', '1.0', '1.2', '1.4', '1.6'], fontsize=14)
        ax.tick_params(axis='both', direction='in')
        ax.tick_params(axis='x', size=0)
        ax.set_title(models[col] + " on " + hardware[row], y=-0.25, fontsize=16)


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=14, ncol=4, frameon=False, loc=[0, 0], bbox_to_anchor=[0.3, 1])

fig.tight_layout(pad=0, h_pad=1, w_pad=1)
fig.savefig('main-result.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
