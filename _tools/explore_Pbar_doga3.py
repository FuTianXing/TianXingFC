# -*- coding: utf-8 -*-
"""探索 3（2026-09-07）：A 下限敏感性
探索 2 发现 P = sqrt(O*G/A) 在 A→0 时发散。本探索：给 A 加下限项（零点异化/噪声底），
看 P 稳态是否有限、量级如何、与 D（eta 代理）的关联形态。
模型：方案 B 种子 + A 方程加下限项 A_min*(1-A)（A 被推向 A_min 而非 0）。
"""
import numpy as np
from scipy.integrate import solve_ivp

def doga_C(u, y, alpha, beta, gamma, delta, f0, u0, w, A_min):
    D, O, G, A = y
    f = f0 * np.exp(-((u-u0)/w)**2)
    dD = alpha*D*(1-D) - beta*A*D + gamma*O*G + f
    dO = alpha*O*(1-O) - beta*A*O + gamma*G*D + 0.5*f
    dG = alpha*G*(1-G) - beta*A*G + gamma*D*O + 0.5*f
    dA = alpha*A*(1-A) - delta*A*D + A_min*(1-A)   # 下限项
    return [dD, dO, dG, dA]

def run(A_min, T=150.0):
    sol = solve_ivp(doga_C, [0, T], [0,0,0,0.9], args=(1.0,1.0,0.5,1.0,0.3,5.0,1.5,A_min),
                    method='RK45', rtol=1e-9, atol=1e-12, max_step=0.05)
    u, D, O, G, A = sol.t, *sol.y
    P = np.sqrt(np.maximum(O*G,0)/np.maximum(A,1e-12))
    return u, D, O, G, A, P

print("=== A 下限敏感性 ===")
for A_min in [1e-2, 1e-3, 1e-4, 1e-5]:
    u, D, O, G, A, P = run(A_min)
    print(f"A_min={A_min:.0e}: A_ss={A[-1]:.2e} D_ss={D[-1]:.4f} O_ss={O[-1]:.4f} G_ss={G[-1]:.4f} P_ss={P[-1]:.4f}  1/sqrt(A_min)={1/np.sqrt(A_min):.2e}")

print("\n=== A_min=1e-3 时 P vs D 形态 ===")
u, D, O, G, A, P = run(1e-3)
mask = (u > 15) & (D > 0.1) & (P > 1e-3)
idxs = np.where(mask)[0][::max(1,len(np.where(mask)[0])//10)]
for i in idxs:
    print(f"D={D[i]:.4f} P={P[i]:.4f} P/P_ss={P[i]/P[-1]:.4f}")
x = np.log(D[mask]); y = np.log(P[mask]/P[-1])
q, b = np.polyfit(x, y, 1)
print(f"拟合 P/P_ss ~ D^q：q={q:.3f}（线性 q=1 候选）")
