# -*- coding: utf-8 -*-
"""探索 2（2026-09-07）：O、G 需要自身种子才能离开原点（探索 1 发现）
方案 A：O、G 加小常数注入（噪声底 eps）
方案 B：O、G 与 D 同加高斯脉冲种子
看 P(u) = sqrt(O*G/A) 的演化与稳态。
"""
import numpy as np
from scipy.integrate import solve_ivp

def doga_A(u, y, alpha, beta, gamma, delta, f0, u0, w, eps):
    D, O, G, A = y
    f = f0 * np.exp(-((u-u0)/w)**2)
    dD = alpha*D*(1-D) - beta*A*D + gamma*O*G + f
    dO = alpha*O*(1-O) - beta*A*O + gamma*G*D + eps   # 噪声底
    dG = alpha*G*(1-G) - beta*A*G + gamma*D*O + eps
    dA = alpha*A*(1-A) - delta*A*D
    return [dD, dO, dG, dA]

def doga_B(u, y, alpha, beta, gamma, delta, f0, u0, w):
    D, O, G, A = y
    f = f0 * np.exp(-((u-u0)/w)**2)
    dD = alpha*D*(1-D) - beta*A*D + gamma*O*G + f
    dO = alpha*O*(1-O) - beta*A*O + gamma*G*D + 0.5*f   # 弱种子
    dG = alpha*G*(1-G) - beta*A*G + gamma*D*O + 0.5*f
    dA = alpha*A*(1-A) - delta*A*D
    return [dD, dO, dG, dA]

def run(fn, y0, args, T=120.0):
    sol = solve_ivp(fn, [0, T], y0, args=args,
                    method='RK45', rtol=1e-9, atol=1e-12, max_step=0.05)
    u, D, O, G, A = sol.t, *sol.y
    P = np.sqrt(np.maximum(O*G, 0)/np.maximum(A, 1e-12))
    return u, D, O, G, A, P

print("=== 方案 A：O、G 噪声底 eps=0.01 ===")
u, D, O, G, A, P = run(doga_A, [0,0,0,0.9], (1.0,1.0,0.5,1.0,0.3,5.0,1.5,0.01))
for t in [5, 10, 20, 40, 60, 90, 120]:
    i = np.argmin(np.abs(u-t))
    print(f"u={u[i]:5.0f} D={D[i]:.4f} O={O[i]:.4f} G={G[i]:.4f} A={A[i]:.4f} P={P[i]:.4f}")
print(f"P 稳态={P[-1]:.4f}  P_max={P.max():.4f}")

print("\n=== 方案 B：O、G 半强度脉冲种子 ===")
u, D, O, G, A, P = run(doga_B, [0,0,0,0.9], (1.0,1.0,0.5,1.0,0.3,5.0,1.5))
for t in [5, 10, 15, 20, 40, 80, 120]:
    i = np.argmin(np.abs(u-t))
    print(f"u={u[i]:5.0f} D={D[i]:.4f} O={O[i]:.4f} G={G[i]:.4f} A={A[i]:.4f} P={P[i]:.4f}")
print(f"P 稳态={P[-1]:.4f}  P_max={P.max():.4f}")

print("\n=== 方案 B 的 P vs D 形态（eta 代理 D）===")
mask = (u > 12) & (D > 0.05) & (P > 1e-3)
if mask.sum() > 5:
    idxs = np.where(mask)[0][::max(1,len(np.where(mask)[0])//12)]
    for i in idxs:
        print(f"D={D[i]:.4f} P={P[i]:.4f} P/P_ss={P[i]/P[-1]:.4f}")
    x = np.log(D[mask]); y = np.log(P[mask]/P[-1])
    q, b = np.polyfit(x, y, 1)
    print(f"拟合 P/P_ss ~ D^q：q={q:.3f}")

print("\n=== 敏感性：eps 与 f0 ===")
for eps in [0.001, 0.01, 0.05]:
    _, _, _, _, _, P2 = run(doga_A, [0,0,0,0.9], (1.0,1.0,0.5,1.0,0.3,5.0,1.5,eps))
    print(f"eps={eps}: P 稳态={P2[-1]:.4f}")
