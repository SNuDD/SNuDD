import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file generated on the cluster
df = pd.read_csv('NR_scan_cluster_emu.csv')

grid_LZ = df.pivot(index='eps', columns='eta', values='Tstat_LZ')

X = grid_LZ.columns.values
Y = grid_LZ.index.values
Z = grid_LZ.values

plt.figure(figsize=(10, 6))

contour = plt.contourf(X, Y, Z, levels=30, cmap='viridis')
plt.colorbar(contour, label='Test Statistic (Tstat_LZ)')


limit_line = plt.contour(X, Y, Z, levels=[4.61], colors='red', linestyles='dashed')
plt.clabel(limit_line, inline=True, fontsize=12, fmt="90%% C.L.")

plt.xlabel(r'Angle $\eta$ (rad)', fontsize=14)
plt.ylabel(r'$\epsilon_{e\mu}$', fontsize=14) 
plt.title('LZ (2025) Sensitivity for $\epsilon_{e\mu}$', fontsize=16)

plt.tight_layout()
plt.savefig('LZ_Scan_Plot.pdf') 
plt.show()