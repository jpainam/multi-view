import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# style.use('seaborn-whitegrid')
# print(style.available)
N = 8
ind = np.arange(N)  # the x locations for the groups
width = 0.15  # the width of the bars
# plt.rcParams.update({'font.size': 20})
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')
# baseline = [83.49, 84.35, 85.18]
# baseline = ax.bar(ind, baseline, width, edgecolor='black')

results = {'market1501': [2017, 1709, 2707, 920, 2338, 3245, 0, 0],
           'duke': [2809, 3009, 1088, 1395, 1685, 3700, 1330, 1506],
           'cuhk03': [7566, 4020, 594, 510, 423, 0, 0, 0],
           #'viper': [316, 316, 0, 0, 0, 0, 0, 0],
           'cuhk01': [970, 970, 0, 0, 0, 0, 0, 0]}

pos1 = ax.bar(ind + width, results['market1501'], width, edgecolor='black')
pos2 = ax.bar(ind + width * 2, results['duke'], width, edgecolor='black')
pos3 = ax.bar(ind + width * 3, results['cuhk03'], width, edgecolor='black')
#pos4 = ax.bar(ind + width * 4, results['cuhk01'], width, edgecolor='black')
#pos5 = ax.bar(ind + width * 5, results['cuhk01'], width, edgecolor='black')

ax.set_ylabel('Number of images')
ax.set_xlabel('Camera views')
ax.set_xticks(ind + .3 + width)
# ax.set_xticks([])
ax.set_yticks(np.arange(0, 10000, 2000))
ax.set_ylim(0, 8000)
ax.set_xticklabels(np.arange(1, 9, 1))
ax.legend((pos1[0], pos2[0], pos3[0]),
          ('Market-1501', 'DukeMTMC-ReID', 'CUHK03',), prop={'size': 10})

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()
