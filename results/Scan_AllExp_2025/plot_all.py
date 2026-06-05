import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Load the data from the CSV file
df = pd.read_csv('NR_scan_cluster_etau.csv')

grid_LZ = df.pivot(index='eps', columns='eta', values='Tstat_LZ')
grid_XNT = df.pivot(index='eps', columns='eta', values='Tstat_XNT')
grid_PANDA = df.pivot(index='eps', columns='eta', values='Tstat_PANDA')

X = grid_LZ.columns.values
Y = grid_LZ.index.values

fig, ax = plt.subplots(figsize=(10, 6))

threshold = 2.71

contour_lz = plt.contour(X, Y, grid_LZ.values, levels=[threshold], colors='red', linestyles='solid', linewidths=2)
contour_xnt = plt.contour(X, Y, grid_XNT.values, levels=[threshold], colors='blue', linestyles='dashed', linewidths=2)
contour_panda = plt.contour(X, Y, grid_PANDA.values, levels=[threshold], colors='green', linestyles='dotted', linewidths=2)


plt.clabel(contour_lz, inline=True, fontsize=10, fmt="LZ")
plt.clabel(contour_xnt, inline=True, fontsize=10, fmt="XNT")
plt.clabel(contour_panda, inline=True, fontsize=10, fmt="PandaX")


upper_bound = 1e6 

plt.contourf(X, Y, grid_LZ.values, levels=[threshold, upper_bound], colors=['red'], alpha=0.15)
plt.contourf(X, Y, grid_XNT.values, levels=[threshold, upper_bound], colors=['blue'], alpha=0.15)
plt.contourf(X, Y, grid_PANDA.values, levels=[threshold, upper_bound], colors=['green'], alpha=0.15)


custom_lines = [Line2D([0], [0], color='red', lw=2, linestyle='solid'),
                Line2D([0], [0], color='blue', lw=2, linestyle='dashed'),
                Line2D([0], [0], color='green', lw=2, linestyle='dotted')]
plt.legend(custom_lines, ['LZ (2025)', 'XENONnT', 'PandaX-4T'], loc='upper right', fontsize=12)

ax.set_yscale('symlog', linthresh=0.05, linscale=0.01)
# ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax.set_xticks(np.pi*np.array([-1./2, -1./4, 0, 1./4, 1./2]), [r'-$\pi/2$', r'-$\pi/4$', '0', r'$\pi/4$', r'$\pi/2$'])
ax.set_yticks(np.outer(np.array([-0.01, -0.1, -1, 0.01, 0.1, 1]),np.array([2, 3, 4 , 5, 6, 7, 8, 9])).flatten(), minor=True)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.set_ylim(-3, 3)
plt.ylabel(r'$\varepsilon_{e\tau}$', size=14)
plt.xlabel(r'$\eta$', size=14)
plt.title('Exclusion Limits for $\epsilon_{e\\tau}$', fontsize=16)

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('All_Experiments_Limits_etau.png')
plt.show()