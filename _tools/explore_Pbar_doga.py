# -*- coding: utf-8 -*-
"""探索：Pbar 标定候选方向 1（2026-09-07，未闭合探索）
DOGA 简化模型下 P(u) = sqrt(O*G/A) 的演化形态、稳态量级、与序参量(D,O)关系。
模型：对称参数 DOGA 四方程 + 种子注入（高斯脉冲），从混沌态 (D,O,G)=(0,0,0), A 高位 启动。
探索问题：P 与 eta 代理（D 或 O）的曲线形态，是否支持 Pbar = P0*eta 线性候选。
"""
import numpy as np
from scipy.integrate import solve_ivp

def doga(u, y, alpha, beta, gamma, delta, f0, u0, w):
    D, O, G, A = y
    f = f0 * np.exp(-((u-u0)/w)**2)  # 高斯种子注入（作用于 D，天行力校准）
    dD = alpha*D*(1-D) - beta*A*D + gamma*O*G + f
    dO = alpha*O*(1-O) - beta*A*O + gamma*G*D
    dG = alpha*G*(1-G) - beta*A*G + gamma*D*O
    dA = alpha*A*(1-A) - delta*A*D
    return [dD, dO, dG, dA]

def run(alpha=1.0, beta=1.0, gamma=0.5, delta=1.0, f0=0.3, u0=5.0, w=1.5,
        A0=0.9, T=80.0):
    y0 = [0.0, 0.0, 0.0, A0]
    sol = solve_ivp(doga, [0, T], y0, args=(alpha, beta, gamma, delta, f0, u0, w),
                    method='RK45', rtol=1e-9, atol=1e-12, max_step=0.05)
    u, D, O, G, A = sol.t, *sol.y
    P = np.sqrt(np.maximum(O*G, 0)/np.maximum(A, 1e-12))
    return u, D, O, G, A, P

# 主探索
u, D, O, G, A, P = run()
print("=== 演化关键点 ===")
for t_target in [0, 5, 10, 15, 20, 30, 40, 60, 80]:
    i = np.argmin(np.abs(u - t_target))
    print(f"u={u[i]:6.1f} D={D[i]:.4f} O={O[i]:.4f} G={G[i]:.4f} A={A[i]:.4f} P={P[i]:.4f}")

print("\n=== 稳态量级 ===")
print(f"P 最大值 P_max={P.max():.4f}  末值 P_end={P[-1]:.4f}")
P0 = P[-1]  # 稳态 Pbar
print(f"稳态 P0 = {P0:.4f}")

print("\n=== P vs D 曲线形态（eta 代理 D）===")
# 采样若干点看 P 与 D 的关系
idxs = np.arange(0, len(u), 50)
for i in idxs[:20]:
    print(f"D={D[i]:.4f}  P={P[i]:.4f}  P/P0={P[i]/P0:.4f}  D^1={D[i]:.4f}")

print("\n=== 候选形式检验（低温端 P ~ P0 * D^q）===")
# 用 D 在 (0.3, 0.9) 区间的点拟合 q：ln(P/P0) vs ln(D)
mask = (D > 0.3) & (D < 0.9) & (P > 1e-3)
if mask.sum() > 10:
    x = np.log(D[mask]); y = np.log(P[mask]/P0)
    q, b = np.polyfit(x, y, 1)
    print(f"拟合 q = {q:.3f}（线性候选 q=1；若 q≈0.5 则 P~sqrt(D)）")
    # 线性候选残差
    res_lin = np.mean((P[mask] - P0*D[mask])**2)
    res_sqrt = np.mean((P[mask] - P0*np.sqrt(D[mask]))**2)
    print(f"线性候选残差={res_lin:.2e}  开方候选残差={res_sqrt:.2e}")

print("\n=== 参数敏感性（f0, gamma, A0 扰动）===")
for f0_, g_, A0_ in [(0.3, 0.5, 0.9), (0.1, 0.5, 0.9), (0.3, 0.3, 0.9), (0.3, 0.5, 0.5)]:
    _, D2, O2, G2, A2, P2 = run(f0=f0_, gamma=g_, A0=A0_)
    print(f"f0={f0_} gamma={g_} A0={A0_}: P_end={P2[-1]:.4f} P_max={P2.max():.4f}")
