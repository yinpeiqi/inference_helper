from matplotlib import pyplot as plt
import numpy as np

st25 = [[22.8, 1622.05, 949.3, 13.26, 1052.53, 626.05, 31.72, 2158.99, 1269.36],
        [10.4, 1156.84, 765.05, 7.15, 615.69, 413.75, 17.11, 1168.89, 721.97] ]
st50 = [[17.39, 1498.59, 915.33, 10.29, 982.33, 555.01, 20.42, 2093.31, 1101.66],
        [8.47, 1106.33, 781.55, 6.01, 580.25, 419.71, 12.75, 1100.86, 734.34] ]
st75 = [[14.93, 1478.58, 881.18, 9.27, 947.18, 535.5, 17.02, 2010.97, 1096.72],
        [7.77, 1072.22, 781.71, 5.85, 558.56, 414.85, 10.84, 1054.54, 738.53] ]
st = [[13.49, 1442.02, 853.81, 8.41, 926.46, 490.73, 15.20, 1981.97, 1083.52],
        [7.39, 1049.29, 759.33, 5.51, 532.34, 411.72, 10.07, 1026, 718.18] ]
dgi = [[12.18, 1361.91, 839.6, 7.34, 873.62, 478.29, 13.13, 1810.26, 1079.89],
        [7.40, 1050.73, 744.44, 4.81, 420.79, 402.79, 7.57, 877.75, 700.98] ]
dgi_re = [[8.85, 985.93, 605.68, 6.1, 612.54, 318.16, 9.27, 1271.45, 664.42],
        [5.65, 718.63, 505.53, 3.69, 328.37, 240.13, 5.79, 657.88, 433.45] ]

dgi_re_2 = [['8.85', '985', '605', '6.10', '612', '318', '9.27', '1270', '664'],
        ['5.65', '718', '505', '3.69', '328', '240', '5.79', '657', '433'] ]
# base = [[13.49, 1446.35, 853.81, 6.84, 926.46, 490.73, 14.12, 1981.97, 916.31],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1] ]

# dgir = [[12.99, 1421.95, 839.6, 7.04, 873.62, 478.29, 12.76, 1810.26, 903.95], 
#         [1, 1, 1, 1, 1, 1, 1, 1, 1] ]
# base_re = [[8.98, 1208.82, 699, 5.91, 690.11, 365.2, 10.38, 1712.02, 819.95], 
#         [1, 1, 1, 1, 1, 1, 1, 1, 1] ]
# dgi = [[8.67, 985.93, 605.68, 5.91, 612.54, 318.16, 9.44, 1271.45, 664.42], 
#         [1, 1, 1, 1, 1, 1, 1, 1, 1] ]

models = ["GAT", "GCN", "JKNet"]
datasets = ["products", "friendster", "papers100M"]
hardware = ["16GB GPU", "32GB GPU"]

fig = plt.figure(figsize=[20, 10])
width = 0.15
x_labels = datasets

tot = 0
for row in range(2):
    for col in [1, 0, 2]:
        tot += 1
        x1 = x2 = x3 = x4 = x5 = np.arange(0, 3, 1)
        y1, y2, y3, y4, y5 = [], [], [], [], []
        labels = []
        for i in range(col*3, col*3 + 3):
        #     y1.append(st[row][i] / st50[row][i])
        #     y2.append(st[row][i] / st75[row][i])

            y1.append(st[row][i] / st25[row][i])
            y2.append(st[row][i] / st50[row][i])

            y3.append(st[row][i] / st[row][i])
            y4.append(st[row][i] / dgi[row][i])
            y5.append(st[row][i] / dgi_re[row][i])
            labels.append(x_labels[i%3] + "\n" + str(dgi_re_2[row][i]))

        ax = fig.add_subplot(2, 3, tot)

        ax.bar(x1 - 2 * width, y1, width, label='0.25 ST', color='#60acfc', edgecolor='black')
        ax.bar(x2 - 1 * width, y2, width, label='0.5 ST', color='#32d3eb', edgecolor='black')
        ax.bar(x3, y3, width, label='ST', color='#5bc49f', edgecolor='black')
        ax.bar(x4 + 1 * width, y4, width, label='DGI-reorder', color='#feb64d', edgecolor='black')
        ax.bar(x5 + 2 * width, y5, width, label='DGI', color='#ff7c7c', edgecolor='black')

        ax.set_ylabel('Speedup over ST', fontsize=21)
        ax.set_xticks(x1, labels, fontsize=19)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.85)
        ax.set_yticks([0, 0.5, 1, 1.5], ['0', '0.5', '1.0', '1.5'], fontsize=20)
        ax.tick_params(axis='both', direction='in')
        ax.tick_params(axis='x', size=0)
        ax.set_title(models[col] + " on " + hardware[row], y=0.9, fontsize=22)


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=22, ncol=5, frameon=False, loc=[0, 0], bbox_to_anchor=[0.22, 1])

fig.tight_layout(pad=0, h_pad=1, w_pad=1)
fig.savefig('ablation_study.jpg', transparent=True, bbox_inches='tight', pad_inches=0.1)
fig.savefig('ablation_study.pdf', transparent=True, bbox_inches='tight', pad_inches=0.1)
