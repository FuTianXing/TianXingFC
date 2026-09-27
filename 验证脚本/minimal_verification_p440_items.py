# -*- coding: utf-8 -*-
'''P4-40 可判据化三项（第 477 条）：采样算子的核、压缩损失与无序度的关系、两截断候选对齐。

判据 R1 至 R3。确定性脚本（R1 用固定种子）。
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/p440_formalization.png'
print('=' * 78)
print('第 477 条：P4-40 可判据化三项')
print('=' * 78)
print()

print('R1 采样算子的核：核维数 = 基维数 - 采样点数（有限维截断上精确，随基维数增长）')
def kernel_dim(N, M, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(N) * 1.0
    centers = rng.uniform(-2.0, N + 2.0, M)
    S = np.exp(-(t[:, None] - centers[None, :]) ** 2 / (2.0 * 0.5 ** 2))
    return int(np.linalg.matrix_rank(S)), M
rows1 = []
for N in [8, 16, 32, 64]:
    for M in [2 * N, 4 * N, 8 * N]:
        r, Mk = kernel_dim(N, M)
        rows1.append((N, M, r, M - r))
        print('   N = %3d, M = %3d : rank = %3d, dim ker = %3d（期望 %d）' % (N, M, r, M - r, M - N))
print('   读数：核维数随基维数线性增长，采样点数固定时无上界；即核为无穷维，与标准结果一致。')
print()

print('R2 压缩损失与无序度的关系：L = 1 - P 与 D = 1 - sqrt(P) 给出 L = 2D - D^2')
D = np.linspace(0.0, 1.0, 11)
L_pred = 2 * D - D ** 2
print('   D     :', ' '.join('%.4f' % x for x in D))
print('   L_pred:', ' '.join('%.4f' % x for x in L_pred))
# 均匀相位分布 [-kappa, kappa] 的解析读数：zbar = sin(kappa)/kappa，P = zbar^2，D = 1 - zbar
kappas = np.linspace(1e-6, np.pi, 20000)
zbar = np.sin(kappas) / kappas
D_an = 1.0 - zbar
P_an = zbar ** 2
L_an = 1.0 - P_an
err = float(np.max(np.abs(L_an - (2 * D_an - D_an ** 2))))
print('   解析关系 L = 2D - D^2 在均匀相位族上的最大偏差 = %.3e（代数恒等）' % err)
idx = np.linspace(0, len(kappas) - 1, 11).astype(int)
print('   kappa 抽样 :', ' '.join('%.3f' % kappas[i] for i in idx))
print('   D_an 抽样 :', ' '.join('%.4f' % D_an[i] for i in idx))
print('   L_an 抽样 :', ' '.join('%.4f' % L_an[i] for i in idx))
print('   量纲核对：D 与 L 皆无量纲，定义域皆 [0,1]；k_B T = E_P D 的量纲由 E_P 承担。')
print()

print('R3 与 P3-26 两个截断候选的对齐')
wN = np.pi
wBZ = 2.0 * np.sqrt(3.0)
print('   读数层尼奎斯特 omega_N = pi/t_P = %.9f / t_P（即 omega_P/2）' % wN)
print('   演化层谱支撑 omega_BZ = 2 sqrt(3)/t_P = %.9f / t_P' % wBZ)
print('   比值 omega_BZ/omega_N = %.9f（P3-26 登记 1.102657791）' % (wBZ / wN))
print('   结论：本线只用读数层界限 omega_N（采样映射作用在读数层）；两候选分属两层，不冲突。')
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
Ns = sorted(set(r[0] for r in rows1))
for N in Ns:
    xs = [r[1] for r in rows1 if r[0] == N]
    ys = [r[3] for r in rows1 if r[0] == N]
    ax[0].plot(xs, ys, 'o-', label='N = %d' % N)
ax[0].set_xlabel('basis dimension M')
ax[0].set_ylabel('dim ker (sampling operator)')
ax[0].set_title('R1 kernel dimension grows with M')
ax[0].legend(fontsize=8)
ax[1].plot(D, L_pred, '-', label='L = 2D - D^2')
ax[1].plot(D_an[idx], L_an[idx], 'o', label='uniform-phase analytic')
ax[1].set_xlabel('D (disorder)')
ax[1].set_ylabel('L (compression loss)')
ax[1].set_title('R2 loss vs disorder')
ax[1].legend()
plt.tight_layout()
plt.savefig(IMG, dpi=150)
print('图已保存：' + IMG)