# -*- coding: utf-8 -*-
"""探索：V2 端点不自洽的能量预算解法（2026-09-08）
问题：eta=0, xi=0 给 E=1.5 hbar*omega_P 超上限。
解决方向：E = hbar*omega_P[(1-eta) + 1/2(1-xi)] <= hbar*omega_P 为能量预算，
T_P 是预算饱和温度（饱和线 w + p/2 = 1 上所有态 T = T_P）；
双侧同时最大无序超预算，物理上由 T_P 上限排除（非线性完成区）。
验证：
1. 各分布族的 (eta, xi) 值域与 E 值
2. 超预算态确认（数学上存在）
3. 饱和线可达性（eta=0, xi=1；平衡终点 eta=1/2, xi=0）
4. 平衡路径 p=2w -> w<=1/2 预算约束
5. 波粒互根能量预算表述
"""
import numpy as np

rng = np.random.default_rng(20260908)
N = 2000000

def stats(P, phi):
    eta = np.abs(np.mean(np.exp(1j*phi)))
    xi = np.mean(P)**2 / np.mean(P**2)
    w, p = 1-eta, 1-xi
    E = w + 0.5*p   # 单位 hbar*omega_P
    return eta, xi, w, p, E

print("="*70)
print("1. 各分布族的 (eta, xi) 与能量 E/hbar*omega_P = w + p/2")
print("="*70)
# a) 各向同性高斯（平衡路径）：sigma_a = sigma_b
for s in [0.05, 0.1, 0.2, 0.5]:
    a = rng.normal(0, s, N); b = rng.normal(0, s, N)
    P = 1.0 + a; phi = b
    eta, xi, w, p, E = stats(P, phi)
    print(f"各向同性 s={s}: eta={eta:.5f} xi={xi:.5f} w={w:.5f} p={p:.5f} p/w={p/max(w,1e-12):.3f} E={E:.5f}")
# b) 常数 P + 均匀相位（eta=0, xi=1 饱和线态）
P = np.ones(N); phi = rng.uniform(-np.pi, np.pi, N)
eta, xi, w, p, E = stats(P, phi)
print(f"常数P+均匀相位: eta={eta:.5f} xi={xi:.5f} w={w:.5f} p={p:.5f} E={E:.5f}  <- 饱和线态(eta=0,xi=1)")
# c) 二值 P（定居率 q）+ 均匀相位（q→0 时 eta=0, xi=0 超预算态）
for q in [0.5, 0.1, 0.01, 1e-4]:
    P = np.where(rng.random(N) < q, 1.0, 0.0)
    phi = rng.uniform(-np.pi, np.pi, N)
    eta, xi, w, p, E = stats(P, phi)
    print(f"二值q={q}+均匀相位: eta={eta:.5f} xi={xi:.5f} w={w:.5f} p={p:.5f} E={E:.5f}")
# d) 常数 P + von Mises 相位（xi=1, eta 可变 -> 饱和线扫描）
from scipy.special import i0, i1
for kappa in [0.1, 1.0, 5.0]:
    mu = rng.uniform(0, 2*np.pi, 1)[0]
    # 用拒绝采样生成 von Mises（简单实现）
    phi = np.random.vonmises(mu, kappa, N)
    P = np.ones(N)
    eta, xi, w, p, E = stats(P, phi)
    print(f"常数P+vonMises kappa={kappa}: eta={eta:.5f} xi={xi:.5f} w={w:.5f} p={p:.5f} E={E:.5f}  <- 饱和线扫描(xi=1)")

print("="*70)
print("2. 平衡终点：w=1/2, p=1（eta=1/2, xi=0）是否可达")
print("="*70)
# 二值 P q→0（xi→0）+ 相位分布使 eta≈1/2：混合相位（一半锁定一半随机）
q = 1e-4
P = np.where(rng.random(N) < q, 1.0, 0.0)
mix = rng.random(N)
phi = np.where(mix < 0.5, 0.0, rng.uniform(-np.pi, np.pi, N))
eta, xi, w, p, E = stats(P, phi)
print(f"二值q=1e-4+混合相位(半锁定半随机): eta={eta:.5f} xi={xi:.5f} w={w:.5f} p={p:.5f} E={E:.5f}  <- 期望 eta≈0.5, xi≈q")

print("="*70)
print("3. 超预算态与预算约束")
print("="*70)
# 均匀相位 + 二值 P 的 E 值域扫描（数学上 E_max 可达 1.5）
for q in [1.0, 0.5, 0.1, 1e-2, 1e-4, 0.0]:
    if q == 0:
        P = np.zeros(N); P[0] = 1.0  # 极限：单次定居
    else:
        P = np.where(rng.random(N) < q, 1.0, 0.0)
    phi = rng.uniform(-np.pi, np.pi, N)
    eta, xi, w, p, E = stats(P, phi)
    print(f"q={q:7.1e}: eta={eta:.5f} xi={xi:.5f} E={E:.5f}  {'超预算' if E > 1.0 else '预算内'}")

print("="*70)
print("4. 平衡路径 + 预算约束：p=2w -> E=2w <= 1 -> w<=1/2")
print("="*70)
for s in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
    a = rng.normal(0, s, N); b = rng.normal(0, s, N)
    P = 1.0 + a; phi = b
    eta, xi, w, p, E = stats(P, phi)
    print(f"各向同性 s={s}: w={w:.5f} p/2={p/2:.5f} E={E:.5f} {'预算内' if E<=1 else '超预算'} (平衡终点 w=1/2 时 E=1)")

print("="*70)
print("结论：能量预算约束 E<=hbar*omega_P 排除 eta=0,xi=0 超预算态；")
print("T_P 为预算饱和温度；推论一精确化为'eta=0 且 xi=1（粒侧定居）给 T_P'")
print("="*70)
