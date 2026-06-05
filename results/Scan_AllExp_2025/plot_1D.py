import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('NR_scan_cluster_ee.csv')

eta_target = -np.pi/8
eta_exact = df.iloc[(df['eta'] - eta_target).abs().argsort()[:1]]['eta'].values[0]
print(f"Selected eta value for 1D plot: {eta_exact:.4f}")

slice = df[df['eta'] == eta_exact]

X = slice['eps']
Y_LZ = slice['Tstat_LZ']
Y_XNT = slice['Tstat_XNT']
Y_PANDA = slice['Tstat_PANDA']

plt.figure(figsize=(10, 6))

plt.plot(X, Y_LZ, label='LZ', color='red', linewidth=2)
plt.plot(X, Y_XNT, label='XENONnT', color='blue', linewidth=2, linestyle='--')
plt.plot(X, Y_PANDA, label='PandaX-4T', color='green', linewidth=2, linestyle=':')

plt.axhline(y=2.71, color='k', linestyle='--', alpha=0.5, label='90% C.L. Limit')

plt.xlabel(r'$\epsilon_{ee}$', fontsize=14)
plt.ylabel('Test Statistic', fontsize=14)
plt.ylim(0, 10)
plt.title(r'Test Statistic vs $\epsilon_{ee}$ (with $\eta = -\pi/8$)', fontsize=16)

plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Tstat_vs_Eps_1D_ee.png')
plt.show()