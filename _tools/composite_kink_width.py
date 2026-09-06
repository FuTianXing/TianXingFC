# -*- coding: utf-8 -*-
"""
复合模型探路（终版）：FK 链 kink = 均匀缠绕背景（W=1）上的局域位错
E = Σ[½k(Δu-1)² + K(1-cos 2πu)]（conversion_wave 同模型）
边界：u_0 = 0、u_{N-1} = N（均匀缠绕 = W=1 扇区极小），初条件含位错
kink = Δu 从 1 到 2 的局域位错；宽度 = 相位梯度能量密度分布
扫描 r = K/k，测 ℓ*，对照 ℓ >= 2 色散窗口（色散验证：K=0.0005→7.12、K=0.02→1.13）
结果只打印终端，不写任何文档。
"""
import numpy as np

NS = 400
K_EL = 1.0  # k（弹性）

def relax_kink(r, steps=120000):
    K_pn = r * K_EL
    # 初条件：均匀缠绕 u_n = n + 位错台阶（中部 0→1）
    u = np.arange(NS, dtype=float)
    half = NS // 2
    u[half:] += 1.0
    u[half] = half + 0.5
    eta = 0.005
    for t in range(steps):
        # 弹性梯度（平方项 ½k(Δu-1)²）：dE/du_n = k(2u_n - u_{n-1} - u_{n+1})
        grad = np.zeros(NS)
        grad[1:-1] += K_EL * (2.0 * u[1:-1] - u[:-2] - u[2:])
        # PN 梯度：K·2π sin(2πu)
        grad += K_pn * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        u[1:-1] -= eta * grad[1:-1]
        # 边界（均匀缠绕 + 位错 = W=1）
        u[0] = 0.0
        u[-1] = float(NS)
    E = 0.5 * K_EL * np.sum((np.diff(u) - 1.0) ** 2) \
        + K_pn * np.sum(1.0 - np.cos(2.0 * np.pi * u))
    # 宽度：位错能量密度（Δu - 1 的平方，弹性超量）的惯性半径
    du = np.diff(u)
    dens = 0.5 * K_EL * (du - 1.0) ** 2
    idx = np.arange(NS - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    r2 = (dens * (idx - c0) ** 2).sum() / dens.sum()
    width = 2.0 * np.sqrt(r2)
    return width, E

print("=" * 60)
print("复合模型探路（终版）：FK kink = 均匀缠绕上的位错")
print(f"1D 链 N = {NS}，弹性 k = {K_EL}（Δu 自然长度 1），扫描 r = K/k")
print("=" * 60)
print("\n  r=K/k       kink宽度 ℓ*   能量 E      判据（ℓ* >= 2）")
for r in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    w, E = relax_kink(r)
    ok = "✓ 落窗（色散友好）" if w >= 2.0 else "✗ 窄（未落窗）"
    print(f"  {r:8.4f}   {w:10.2f}   {E:10.2f}   {ok}")

print("\n对照：色散验证宽度趋势 K = 0.0005→ℓ=7.12、K=0.02→ℓ=1.13（同族 FK 链）")
print("复合模型完成（仅打印）。")
