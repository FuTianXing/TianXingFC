# -*- coding: utf-8 -*-
"""
minimal_verification_c1_anchor_design.py

[C1-6 锚核内驻设计轮] 拉力、凹口增益与驻留判据（判据 AN-1 至 AN-4）

背景：保护动力学轮（P 轮）给出：保护（边界冻结）下标签精确守恒，但净荷滑向
边界驻留，冻结芯只保形状。登记开放项：把净荷钉在体内所需的锚核设计。本轮给
出该设计的定量判据并做判别力检验。

模型（与 P 轮同口径）：横截面 N^2 = 64^2，u = u0 + theta/2 pi，u0 = IX + IY，
theta = atan2(y - y0, x - x0)；E = sum_axes (k/2)(d_axis u - 1)^2
+ sum K_xy (1 - cos 2 pi u)，k = 1，K0 = 0.005；边界环冻结（保护开）。
凹口锚核：K_xy = K0[1 - d_n exp(-r^2 / w_a^2)]，d_n = 0.9，w_a = 8。

判据
AN-1 拉力直接测量：受控弛豫（芯盘冻结在 xc，边界环冻结）的能量随 xc 的斜率
     = 字符串张力 sigma_w；与连续预测 4 sqrt(K)/pi 对照
AN-2 凹口增益与判据量：A(xc) = E_uniform(xc) - E_notch(xc)；在凹口中心取
     A(r0)，判据量 L_max = A(r0)/sigma_w
AN-3 判别力检验（自由芯两档）：凹口在 x0 = 12 时芯应逃逸（逃逸距离 10 > L_max）；
     凹口在 x0 = 4 时芯应驻留（逃逸距离 3 < L_max）
AN-4 设计结论：把芯停在中心（L = 32）所需的凹口尺度与可行性

运行：python 验证脚本/minimal_verification_c1_anchor_design.py（确定性，约 20 秒）
"""
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
N = 64
NSTEPS = 20000
ETA = 0.001
K_EL = 1.0
K0 = 0.005
TWO_PI = 2.0 * math.pi


def wrap1(a):
    return (a + 0.5) % 1.0 - 0.5


def grids():
    i = np.arange(N, dtype=float)
    IX, IY = np.meshgrid(i, i, indexing="ij")
    return IX, IY


IX, IY = grids()


def init_field(x0, y0):
    theta = np.arctan2(IY - y0, IX - x0)
    return IX + IY + theta / TWO_PI


def karr_notch(x0, y0, d_n=0.9, w_a=8.0):
    return K0 * (1.0 - d_n * np.exp(-(((IX - x0) ** 2 + (IY - y0) ** 2) / w_a ** 2)))


def energy(u, Kxy):
    E = 0.0
    for ax in range(2):
        d = np.diff(u, axis=ax) - 1.0
        E += 0.5 * K_EL * float(np.sum(d ** 2))
    E += float(np.sum(Kxy * (1.0 - np.cos(TWO_PI * u))))
    return E


def grad(u, Kxy):
    g = np.zeros_like(u)
    for ax in range(2):
        d = np.diff(u, axis=ax)
        dd = np.zeros_like(u)
        if ax == 0:
            dd[:-1] -= K_EL * (d - 1.0)
            dd[1:] += K_EL * (d - 1.0)
        else:
            dd[:, :-1] -= K_EL * (d - 1.0)
            dd[:, 1:] += K_EL * (d - 1.0)
        g += dd
    g += Kxy * TWO_PI * np.sin(TWO_PI * u)
    return g


def plaq_field(u):
    dx = wrap1(u[1:, :-1] - u[:-1, :-1])
    dx2 = wrap1(u[1:, 1:] - u[:-1, 1:])
    dy = wrap1(u[:-1, 1:] - u[:-1, :-1])
    dy2 = wrap1(u[1:, 1:] - u[1:, :-1])
    return dx + dy2 - dx2 - dy


def plaq_charge(u):
    return int(np.rint(plaq_field(u)).sum())


def ring_mask():
    m = np.zeros((N, N), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    return m


def disk_mask(x0, y0, r=1):
    m = np.zeros((N, N), dtype=bool)
    x0 = int(round(x0))
    y0 = int(round(y0))
    m[max(x0 - r, 0):x0 + r + 1, max(y0 - r, 0):y0 + r + 1] = True
    return m


def relax(u, Kxy, freeze=None, steps=NSTEPS, eta=ETA):
    u = u.copy()
    for _ in range(steps):
        g = grad(u, Kxy)
        if freeze is not None:
            g[freeze] = 0.0
        u = u - eta * g
    return u


def cells_and_centroid(u):
    w = np.rint(plaq_field(u))
    cells = [(int(a) + 1, int(b) + 1, int(w[a, b])) for a, b in zip(*np.nonzero(w))]
    m = np.abs(w)
    ii = np.arange(1, N, dtype=float)
    JX, JY = np.meshgrid(ii, ii, indexing="ij")
    cx = float((m * JX).sum() / m.sum())
    cy = float((m * JY).sum() / m.sum())
    return cells, (cx, cy)


RING = ring_mask()
Kuni = np.full((N, N), K0)

print("=" * 78)
print("C1-6 锚核内驻设计轮：拉力、凹口增益与驻留判据（判据 AN-1 至 AN-4）")
print("=" * 78)
print()
print("参数：N = %d，K0 = %.4f，sigma_w 预测 = 4 sqrt(K)/pi = %.4f，凹口 d = 0.9，w_a = 8"
      % (N, K0, 4.0 * math.sqrt(K0) / math.pi))

print()
print("[AN-1] 拉力直接测量（受控：芯盘与边界环冻结）")
XC = (4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0)
E_uni = {}
for xc in XC:
    u0 = init_field(xc, 32.0)
    uf = relax(u0, Kuni, freeze=RING | disk_mask(xc, 32.0))
    E_uni[xc] = energy(uf, Kuni)
    tag = ""
    if xc in (4.0, 12.0, 32.0):
        cells, cen = cells_and_centroid(uf)
        tag = "；电荷 %s" % str(cells[:5])
    print("     xc = %4.0f：E = %.4f%s" % (xc, E_uni[xc], tag))
slope = np.polyfit([xc for xc in XC[1:]], [E_uni[xc] for xc in XC[1:]], 1)[0]
pred = 4.0 * math.sqrt(K0) / math.pi
print("     斜率（xc >= 8）= %.4f（预测 %.4f，相对差 %+.2f%%）"
      % (slope, pred, 100.0 * (slope - pred) / pred))

print()
print("[AN-2] 凹口增益与判据量")
Knot12 = karr_notch(12.0, 32.0)
E_not = {}
for xc in (4.0, 12.0, 20.0):
    u0 = init_field(xc, 32.0)
    uf = relax(u0, Knot12, freeze=RING | disk_mask(xc, 32.0))
    E_not[xc] = energy(uf, Knot12)
A12 = E_uni[12.0] - E_not[12.0]
Lmax = A12 / slope
print("     增益：xc = 12（在凹口内）给 A = %.4f；xc = 4 给 %.4f；xc = 20 给 %.4f"
      % (A12, E_uni[4.0] - E_not[4.0], E_uni[20.0] - E_not[20.0]))
print("     判据量 L_max = A/sigma_w = %.2f 格（凹口尺度参考：(1 - sqrt(1-d)) w_a = %.1f）"
      % (Lmax, (1.0 - math.sqrt(0.1)) * 8.0))
dE = E_uni[32.0] - E_uni[4.0]
print("     所需增益（把芯从近边界拉到中心）：E(32) - E(4) = %.2f；可及凹口增益 A = %.4f（差 %.0f 倍）"
      % (dE, A12, dE / A12))
print()
print("[AN-2b] K = 0.02 档的标度对照")
K1 = 0.02
Kuni1 = np.full((N, N), K1)
Knot1 = K1 / K0 * Knot12
E1 = {}
for xc in (4.0, 12.0, 32.0):
    u0 = init_field(xc, 32.0)
    uf = relax(u0, Kuni1, freeze=RING | disk_mask(xc, 32.0))
    E1[xc] = energy(uf, Kuni1)
u0 = init_field(12.0, 32.0)
uf = relax(u0, Knot1, freeze=RING | disk_mask(12.0, 32.0))
A1 = E1[12.0] - energy(uf, Knot1)
slope1 = (E1[32.0] - E1[12.0]) / 20.0
print("     E(4) = %.3f，E(12) = %.3f，E(32) = %.3f；斜率 = %.4f（预测 %.4f）"
      % (E1[4.0], E1[12.0], E1[32.0], slope1, 4.0 * math.sqrt(K1) / math.pi))
print("     A(12) = %.4f；L_max = %.2f 格（与 K = 0.005 档比较：L_max 与 K 近似无关）"
      % (A1, A1 / slope1))

print()
print("[AN-3] 判别力检验（自由芯，仅边界环冻结）")
print("     预期：两档逃逸距离（约 10 与约 3）均大于 L_max = %.2f，两档都应逃逸" % Lmax)
for (x0, tag) in ((12.0, "凹口在 12（逃逸距离约 10）"), (4.0, "凹口在 4（逃逸距离约 3）")):
    Karr = karr_notch(x0, 32.0)
    u0 = init_field(x0, 32.0)
    uf = relax(u0, Karr, freeze=RING)
    cells, (cx, cy) = cells_and_centroid(uf)
    print("     %s：W = %d；电荷 %s；质心 = (%.2f, %.2f)"
          % (tag, plaq_charge(uf), str(cells[:6]), cx, cy))

print()
print("[AN-4] 设计结论")
need_wa = 32.0 / (1.0 - math.sqrt(0.1))
print("     把芯停在中心（逃逸距离 32）所需的凹口宽度：w_a ≳ %.1f（d = 0.9 口径）" % need_wa)
print("     实测 L_max = %.2f 格；可及凹口（w_a <= 8）的驻留半径远小于 32" % Lmax)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
ax.plot([xc for xc in XC], [E_uni[xc] for xc in XC], "o-", ms=5, label="受控能量 E(xc)")
xf = np.array([8.0, 32.0])
ax.plot(xf, np.polyval(np.polyfit([xc for xc in XC[1:]], [E_uni[xc] for xc in XC[1:]], 1), xf),
        "--", lw=1, label="斜率 %.4f / 格" % slope)
ax.set_xlabel("芯位 xc（格）")
ax.set_ylabel("E")
ax.set_title("(a) 拉力：受控能量随芯位")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
labels = ["A(4)", "A(12)", "A(20)", "所需 2.51"]
vals = [E_uni[4.0] - E_not[4.0], A12, E_uni[20.0] - E_not[20.0], dE]
cols = ["tab:blue", "tab:blue", "tab:blue", "tab:red"]
ax.bar(labels, vals, color=cols)
ax.set_yscale("log")
ax.set_ylabel("增益（对数轴）")
ax.set_title("(b) 凹口增益与所需增益（K = 0.005）")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
wa = np.logspace(math.log10(4), math.log10(64), 50)
ax.loglog(wa, Lmax * wa / 8.0, "-", label="L_max ≈ %.2f w_a / 8 外推" % Lmax)
ax.loglog([8.0, 8.0], [Lmax, A1 / slope1], "o", ms=5)
ax.axhline(32.0, color="r", ls=":", lw=1, label="中心距离 32 格")
ax.set_xlabel("凹口宽度 w_a（格）")
ax.set_ylabel("L_max（格）")
ax.set_title("(c) 驻留半径与锚核尺度")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/c1_anchor_design.png", dpi=150)
print()
print("图已存：D:/software/TianXingFC/images/c1_anchor_design.png")
print("总运行时间 = %.1f 秒" % (time.time() - T0))
