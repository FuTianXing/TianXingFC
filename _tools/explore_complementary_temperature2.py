# -*- coding: utf-8 -*-
"""v2 修订：复振幅正交分解下的双侧无序度与运动学定理（探索级）

模型：锁定态 z = Pbar（实轴），涨落 a（径向）、b（切向）为独立高斯。
判据（小涨落展开的理论期望）：
1. 相位边缘无序度：1 - eta = <b^2>/(2 Pbar^2)（仅切向进入）
2. 径向（粒侧）无序度：1 - xi = <a^2>/Pbar^2，xi = <P>^2/<P^2>（仅径向进入）
3. 运动学定理（各向同性 a、b 同方差，与平衡无关）：1 - xi = 2(1 - eta)
4. 各向同性下弹性总能量 E = (kappa/2)(<a^2>+<b^2>) = kappa Pbar^2 (1-eta)
   = hbar omega_P (1-eta) + (hbar omega_P/2)(1-xi)（kappa Pbar^2 = hbar omega_P）
   即双侧公式 kBT = hbar omega_P [(1-eta) + (1-xi)/2] = 2 hbar omega_P (1-eta) = hbar omega_P (1-xi)
5. 对照：复功率相干 |<z>|^2/<|z|^2> 给 1-xi2 = 2 sigma^2/Pbar^2（双重计入，非温度锚）
6. 双侧独立性（各向异性）：径向与切向无序度各自独立，p=w 需要负径向方差，不可能
"""
import numpy as np

rng = np.random.default_rng(20260908)
N = 4000000

print("=" * 72)
print("1-3. 各向同性模型（sigma_a = sigma_b = sigma）：运动学定理 1-xi = 2(1-eta)")
print("=" * 72)
Pbar = 1.0
for sigma in [0.02, 0.05, 0.1, 0.2]:
    a = rng.normal(0, sigma, N)
    b = rng.normal(0, sigma, N)
    X = Pbar + a
    Y = b
    P = np.sqrt(X * X + Y * Y)
    Phi = np.arctan2(Y, X)
    eta = np.abs(np.mean(np.exp(1j * Phi)))
    xi = np.mean(P) ** 2 / np.mean(P ** 2)
    w, p = 1 - eta, 1 - xi
    print(f"sigma={sigma:5.2f}  1-eta={w:.8f} (期望 {sigma ** 2 / 2: .8f})  "
          f"1-xi={p:.8f} (期望 {sigma ** 2: .8f})  p/w={p / w:.4f} (期望 2)")

print("=" * 72)
print("4. 双侧公式与弹性能量的一致性（代数对照，各向同性平衡）")
print("=" * 72)
kappaPbar2 = 1.0  # kappa Pbar^2 = hbar omega_P（归一化）
for sigma in [0.02, 0.05, 0.1]:
    a = rng.normal(0, sigma, N)
    b = rng.normal(0, sigma, N)
    E_elastic = 0.5 * kappaPbar2 * (np.mean(a ** 2) + np.mean(b ** 2))  # 单位 hbar omega_P
    eta = np.abs(np.mean(np.exp(1j * np.arctan2(b, Pbar + a))))
    P = np.sqrt((Pbar + a) ** 2 + b ** 2)
    xi = np.mean(P) ** 2 / np.mean(P ** 2)
    T_two_sector = (1 - eta) + 0.5 * (1 - xi)   # kBT / hbar omega_P
    T_xi_anchor = (1 - xi)                       # 粒侧读出
    T_wave_2x = 2 * (1 - eta)                    # 波侧读出（因子 2）
    print(f"sigma={sigma:5.2f}  E={E_elastic:.6f}  双侧公式={T_two_sector:.6f}  "
          f"粒侧读出={T_xi_anchor:.6f}  2x波侧={T_wave_2x:.6f}")

print("=" * 72)
print("5. 对照：复功率相干 |<z>|^2/<|z|^2>（双重计入，排除作为温度锚）")
print("=" * 72)
for sigma in [0.05, 0.1]:
    a = rng.normal(0, sigma, N)
    b = rng.normal(0, sigma, N)
    z = (Pbar + a) + 1j * b
    xi2 = np.abs(np.mean(z)) ** 2 / np.mean(np.abs(z) ** 2)
    print(f"sigma={sigma:5.2f}  1-xi2={1 - xi2:.8f} (期望 {2 * sigma ** 2: .8f}，"
          f"= 2x(1-xi)，双重计入)")

print("=" * 72)
print("6. 各向异性：径向/切向无序度独立（1-xi 只看径向，1-eta 只看切向）")
print("=" * 72)
for sx, sy in [(0.1, 0.1), (0.05, 0.1), (0.1, 0.05), (0.02, 0.1)]:
    a = rng.normal(0, sx, N)
    b = rng.normal(0, sy, N)
    P = np.sqrt((Pbar + a) ** 2 + b ** 2)
    eta = np.abs(np.mean(np.exp(1j * np.arctan2(b, Pbar + a))))
    xi = np.mean(P) ** 2 / np.mean(P ** 2)
    w, p = 1 - eta, 1 - xi
    print(f"sigma_a={sx:5.2f} sigma_b={sy:5.2f}  1-eta={w:.8f} (期望 {sy ** 2 / 2: .8f})  "
          f"1-xi={p:.8f} (期望 {sx ** 2: .8f})")

print("=" * 72)
print("结论：判据 1-6 通过，则 v2 双侧结构（正交分解 + 运动学定理）成立")
print("=" * 72)
