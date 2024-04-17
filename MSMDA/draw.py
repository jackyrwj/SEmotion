
import matplotlib.pyplot as plt
import numpy as np






batch_sizes = [1.15, 1.2, 1.25, 1.3, 1.35]
CSESN_S3_means = [87.78, 89.53, 88.02, 87.56, 87.19]
CSESN_S3_std = [5, 4, 4, 3, 3]

CSUB_S3_means = [87.64, 90.67, 85.74, 85.61, 88.66]
CSUB_S3_std = [6, 5, 5, 4, 4]

CSESN_S4_means = [60.26, 61.26, 61.6, 64.11, 59.85]
CSESN_S4_std = [10.07, 10.6, 13.45, 11.08, 12.8]

CSUB_S4_means = [45.71, 60.43, 59.85, 62.6, 58.39]
CSUB_S4_std = [9.21, 9.76, 8.08, 9.86, 9.25]




fig, ax = plt.subplots()

bar_width = 0.25 
index = np.arange(len(batch_sizes)) * 1.3  

fig, ax = plt.subplots(figsize=(10, 6))  

plt.grid(axis='y', linestyle='-', alpha=0.4, zorder=-100)
bar_colors = ['#555b6e', '#89b0ae', '#bee3db', '#d0c8c8']  
error_attr = {'elinewidth': 1, 'capsize': 3, 'capthick': 0.5}

rects1 = ax.bar(index - bar_width*1.5, CSESN_S3_means, bar_width, yerr=CSESN_S3_std,
                color=bar_colors[0], label='KFCT_SEED', error_kw=error_attr)
rects2 = ax.bar(index - bar_width/2, CSUB_S3_means, bar_width, yerr=CSUB_S3_std,
                color=bar_colors[1], label='LOSO_SEED', error_kw=error_attr)
rects3 = ax.bar(index + bar_width/2, CSESN_S4_means, bar_width, yerr=CSESN_S4_std,
                color=bar_colors[2], label='KFCT_SEEDIV', error_kw=error_attr)
rects4 = ax.bar(index + bar_width*1.5, CSUB_S4_means, bar_width, yerr=CSUB_S4_std,
                color=bar_colors[3], label='LOSO_SEEDIV', error_kw=error_attr)

ax.set_ylim([10, 100]) 
ax.set_yticklabels(np.arange(10, 101, 10), fontsize = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('b (a=0.85)',fontsize=20)
ax.set_ylabel('Accuracy',fontsize=20)
# ax.set_title('Accuracy with different')
ax.set_xticks(index)
ax.set_xticklabels(batch_sizes,fontsize = 20)
# ax.legend()
ax.legend(loc='lower left',fontsize=16)

plt.tight_layout()
plt.show()
plt.savefig('epoch.png')


