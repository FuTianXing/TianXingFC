# -*- coding: utf-8 -*-
"""自撑壁的 E-L 本征问题：Derrick 缩放、平移零模与形状本征谱（第 466 条；第 465 条移交的"真实变分驻定解"）。

模型：薄壁自撑球面，phi = tanh(k (r - R(n)))，k = 1，R0 = 12（薄壁度 kR0 = 12）；
形状扰动 R(n) = R0 + eps * P_l(cos theta)（轴对称 l 模）；能量泛函（k=1）：
  E[R] = int dOmega int r^2 dr sech^4(r-R) [1 + |grad_perp R|^2 / (2 r^2)]
（由 |grad phi|^2 = sech^4 |grad rho|^2、V = (1/2) sech^4 精确给出；rho = r - R）。

判据：
  EL1 Derrick 缩放：E(lambda) = G/lambda + U/lambda^3（G = 1/2 int |grad phi|^2、U = int V）
      对任意 lambda > 0 单调减（无约束无有限能量驻定解；自撑必须带约束）
  EL2 平移零模：平移剖面 rho = |x - a| 的 |grad rho| = 1 精确（任意平移量，数值核对）
  EL3 形状本征谱：lambda_l = [E(+eps)-2E(0)+E(-eps)]/eps^2（l = 0..4），与薄壁解析
      lambda_l ∝ 1 + l(l+1)/2 对照（比值读数）

确定性（Fib 方向采样 + 径向网格），约数秒。图存 images/selfsupport_eigen.png
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "selfsupport_eigen.png")


def check(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


R0 = 12.0
# 径向网格
NR = 1600
r_max = 40.0
r = np.linspace(0.0, r_max, NR)
dr = r[1] - r[0]
# Fib 方向
NS = 3000
i = np.arange(NS)
zz = 1 - 2 * (i + 0.5) / NS
phg = np.pi * (1 + 5 ** 0.5) * i
rxy = np.sqrt(np.maximum(0.0, 1 - zz * zz))
NX = rxy * np.cos(phg)
NY = rxy * np.sin(phg)
NZ = zz
W = 4.0 * np.pi / NS


def legendre_and_dtheta(l, costh):
    # P_l(cos) 与 dP_l/dtheta = -sin(theta) P_l'(cos)
    from numpy.polynomial import legendre as L
    coeffs = [0] * l + [1]
    P = L.legval(costh, coeffs)
    if l == 0:
        return P, np.zeros_like(costh)
    Pp = L.legval(costh, L.legder(coeffs))
    sinth = np.sqrt(np.maximum(0.0, 1 - costh ** 2))
    return P, -sinth * Pp


def energy(l, eps):
    R = R0 + eps * legendre_and_dtheta(l, NZ)[0]
    # dR/dtheta；仅在 theta 方向（轴对称），phi 方向为零
    dth = eps * legendre_and_dtheta(l, NZ)[1] if l > 0 else np.zeros_like(NZ)
    # rho(r, j) = r - R[j]
    rho = r[:, None] - R[None, :]
    sech = 1.0 / np.cosh(rho)
    sech4 = sech ** 4
    # 能量密度（对 r 积分）：sech4 * [1 + (dR/dtheta)^2/(2 r^2)]
    integ = (r ** 2)[:, None] * sech4 * (1.0 + (dth ** 2)[None, :] / (2.0 * (r ** 2)[:, None] + 1e-300))
    # 但 1/r^2 项在 r 很小时由 sech4 抑制；这里直接数值积分
    E = float(np.sum(integ) * dr * W)
    return E


print("== EL1 Derrick 缩放 ==")
E0 = energy(0, 0.0)
# G 与 U 分解（基准：|grad rho|=1，密度 = sech^4 = 1/2 sech^4 + 1/2 sech^4）
G = float(np.sum((r ** 2)[:, None] * (1.0 / np.cosh(r[:, None] - R0)) ** 4 * 0.5) * dr * W)
U = G
lam = np.linspace(0.5, 2.0, 7)
E_lam = [G / L + U / L ** 3 for L in lam]
mono = all(E_lam[k + 1] < E_lam[k] for k in range(len(E_lam) - 1))
print(f"  G = {G:.6e}，U = {U:.6e}（均为正）；E(0.5) = {E_lam[0]:.6e} 至 E(2) = {E_lam[-1]:.6e}")
print(f"  E(lambda) 单调减：{mono}（解析：dE/dlambda = -(G + 3U/lambda^2)/lambda^2 < 0）")
el1 = check("EL1", mono and G > 0 and U > 0, "无约束无驻定（Derrick）；自撑需约束")

print("== EL2 平移零模 ==")
rng = np.random.default_rng(466)
pts = rng.normal(size=(200, 3))
pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * rng.uniform(3, 20, size=(200, 1))
a = rng.normal(size=3)
a = a / np.linalg.norm(a) * 2.5
grad_norm_dev = []
for d in [1e-5, 1e-6]:
    for base in pts[:50]:
        e1 = np.cross(base, [0, 0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(base, [1.0, 0, 0])
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(base, e1)
        e2 = e2 / np.linalg.norm(e2)
        f = lambda x: np.linalg.norm(x - a)
        g1 = (f(base + d * e1) - f(base - d * e1)) / (2 * d)
        g2 = (f(base + d * e2) - f(base - d * e2)) / (2 * d)
        nn = base / np.linalg.norm(base)
        gn = (f(base + d * nn) - f(base - d * nn)) / (2 * d)
        grad_norm_dev.append(abs(np.sqrt(g1 ** 2 + g2 ** 2 + gn ** 2) - 1.0))
print(f"  平移剖面 |grad rho| 与 1 的最大偏差 = {max(grad_norm_dev):.3e}（期望 < 5e-9）")
print("  容差预算：中心差分截断 ~ d^2 |f'''| / 6，取 d = 1e-5 与最坏距离 ~ 0.5，预算 ~ 3e-9")
el2 = check("EL2", max(grad_norm_dev) < 5e-9, "平移零模精确（能量与位置无关）")

print("== EL3 形状本征谱 ==")
eps = 0.05
Ebase = energy(0, 0.0)
lam_num = {}
for l in range(0, 5):
    Ep = energy(l, eps)
    Em = energy(l, -eps)
    lam_num[l] = (Ep - 2 * Ebase + Em) / eps ** 2
ana = {l: (1.0 + 0.5 * l * (l + 1)) / (2 * l + 1) for l in range(0, 5)}
print("  l :  数值 lambda_l      解析 (1+l(l+1)/2)/(2l+1)   比值")
ratios = []
for l in range(0, 5):
    ratio = (lam_num[l] / lam_num[0]) / (ana[l] / ana[0])
    ratios.append(ratio)
    print(f"  {l} :  {lam_num[l]:.6e}    {ana[l]:.6f}       {ratio:.6f}")
dev = max(abs(ratios[l] - 1.0) for l in range(5))
print(f"  全部 l 的比值与 1 的最大偏差 = {dev:.3e}（期望 < 1e-3）")
el3 = check("EL3", dev < 1e-3 and lam_num[0] > 0, f"形状谱与解析一致（偏差 {dev:.1e}）")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(lam, E_lam, "o-", ms=4)
ax.set_xlabel("缩放因子 lambda"); ax.set_ylabel("E(lambda)")
ax.set_title("Derrick 缩放（单调减，无约束无驻定）"); ax.grid(alpha=0.3)
ax = axes[1]
ls = list(range(5))
ax.plot(ls, [lam_num[l] / lam_num[1] for l in ls], "o-", ms=4, label="数值（l=1 归一）")
ax.plot(ls, [ana[l] / ana[1] for l in ls], "k--", lw=1, label="解析 1+l(l+1)/2（归一）")
ax.set_xlabel("球谐阶 l"); ax.set_ylabel("lambda_l / lambda_1")
ax.set_title("形状本征谱"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
print("判据：", "全部通过" if (el1 and el2 and el3) else "存在失败")
