# -*- coding: utf-8 -*-
"""P4-45 判据第 673 号：真空涨落谱三点判据的形式化。

模型（命题 9 的计数通道）：折返事件严格周期，周期 t_P；每次折返持续 w 的
时间（物理量，不随采样率变），以登记概率 eta 被独立登记。时间轴离散化，
每 t_P 分 M 个 bin，共 N 个周期。观测量为登记计数序列的功率谱，频率单位
取 1/t_P，能量换算 E_P = hbar * 2 pi / t_P。

序列分解（本判据的核心结构）：
  x = B * p，B 为伯努利门（概率 eta），p 为周期矩形脉冲（宽 w）
  相干部分：梳线，间隔 1/t_P，包络为 sinc^2(f w)，第一零点 f = 1/w
  非相干部分：门噪声，白谱底，高度 2 eta(1-eta) w/M，铺满至采样 Nyquist

由此三点判据各归其位：
  低频无红外发散：成立（白底 + 梳线，无 1/f）
  谱厚度由 t_P 定：仅梳线间隔成立；线宽由观测窗长定，包络宽度由 w 定
  高频在 E_P 硬截断：不成立。瞬时折返（w -> 0）下梳线无零点、延伸至 Nyquist；
    门噪声更是白底铺满全带。w = t_P 时梳线整体消失（信号为常值），
    「截断在 E_P」与「有非零涨落」不能同时成立。
"""

import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"D:\software\TianXingFC"
IMG = os.path.join(ROOT, "images")
os.makedirs(IMG, exist_ok=True)

SEED = 20261017
N_SEED = 32


def sequence(eta, M, N, w_bins, seed=SEED, poisson=False):
    """登记计数序列：每周期 M 个 bin，前 w_bins 个为开，按 eta 伯努利保留。

    poisson=True 时把脉冲位置随机化（间隔指数分布、宽度不变），
    用以拆开「周期性」与「脉冲宽度」两个贡献。
    """
    rng = np.random.default_rng(seed)
    if poisson:
        L = N * M
        p_on = eta * w_bins / M
        x = np.zeros(L)
        t = 0.0
        while True:
            t += rng.exponential(1.0 / p_on)
            if t >= L:
                break
            i = int(t)
            x[i:i + w_bins] = 1.0
        return x
    x = np.zeros((N, M))
    x[:, :w_bins] = 1.0
    keep = rng.random(N) < eta
    x[~keep, :] = 0.0
    return x.ravel()


def power(x):
    """单边功率谱，归一化使 sum(P) = sum(x^2)。"""
    L = x.size
    P = np.abs(np.fft.rfft(x)) ** 2 / L
    P[1:-1] *= 2.0
    return P


def comb(P, N, M):
    """梳线幅度 P[k*N]，k = 0..M/2。"""
    return np.array([P[k * N] for k in range(M // 2 + 1) if k * N < P.size])


def floor_mask(P, N):
    j = np.arange(P.size)
    return (j % N != 0) & (j > 0)


def floor_level(P, N, f_lo=None, f_hi=None):
    """非梳线频点的平均功率，可限定频率窗（单位 1/t_P）。"""
    j = np.arange(P.size)
    m = floor_mask(P, N)
    if f_lo is not None:
        m &= (j / N >= f_lo)
    if f_hi is not None:
        m &= (j / N < f_hi)
    return float(np.mean(P[m]))


def first_null(P, N, M, rel=1e-6):
    """包络第一零点：从 k=1 起第一个幅度低于直流 rel 倍的谐波。"""
    c = comb(P, N, M)
    for k in range(1, c.size):
        if c[k] < rel * c[0]:
            return k
    return None


def line_width(P, N, M, k):
    """k 阶梳线的 -3 dB 全宽，单位 1/t_P。k 须为非零点。"""
    j0 = k * N
    if j0 >= P.size or P[j0] <= 0.0:
        return float("nan")
    thr = P[j0] / 2.0
    lo, hi = j0, j0
    while lo > 0 and P[lo] > thr:
        lo -= 1
    while hi < P.size - 1 and P[hi] > thr:
        hi += 1
    return (hi - lo) / N


print("=" * 78)
print("P4-45 判据第 673 号：真空涨落谱三点判据的形式化")
print("=" * 78)

N0, M0, ETA0 = 1024, 32, 0.5
W0 = M0 // 2

# ---------------- 冒烟 ----------------
print("\n[冒烟]")
x1 = sequence(1.0, M0, N0, W0)
P1 = power(x1)
M1 = abs(float(np.sum(P1)) - float(np.sum(x1 ** 2))) / float(np.sum(x1 ** 2)) < 1e-10
print("  M1 Parseval：谱和 %.3f，时域 %.3f → %s"
      % (float(np.sum(P1)), float(np.sum(x1 ** 2)), "PASS" if M1 else "FAIL"))

c1 = comb(P1, N0, M0)
M2 = c1[0] > 0 and c1[2] < 1e-9 * c1[0]
print("  M2 梳线零点落在 k=2（占空比 1/2）：P(0)=%.3e，P(2)=%.3e → %s"
      % (c1[0], c1[2], "PASS" if M2 else "FAIL"))

b1 = floor_level(power(sequence(1.0, M0, N0, W0)), N0)
b05 = floor_level(power(sequence(0.5, M0, N0, W0)), N0)
M3 = b1 < 1e-12 * b05
print("  M3 eta=1 谱底归零：底 %.3e（eta=0.5 时 %.3e）→ %s"
      % (b1, b05, "PASS" if M3 else "FAIL"))

# ---------------- C 组：相干部分（梳线） ----------------
print("\n[C 组：相干部分，梳线间隔与包络]")

print("  C1 梳线间隔严格为 1/t_P（四档 M）：")
c1ok = True
for M in [8, 16, 32, 64]:
    P = power(sequence(ETA0, M, N0, M // 2))
    c = comb(P, N0, M)
    peaks = [k for k in range(1, c.size) if c[k] > 1e-3 * c[0]]
    step = set(np.diff(peaks))
    c1ok &= step == {2}
    print("      M=%2d：峰位 %s，间隔集合 %s" % (M, peaks[:6], sorted(step)))
C1 = c1ok
print("      → %s（间隔 2/t_P 为占空比 1/2 的零点结构，基周期仍为 t_P）" % ("PASS" if C1 else "FAIL"))

print("  C2 包络第一零点 f = 1/w（M=32 固定，变 w）：")
c2ok = True
for wf in [0.5, 0.25, 0.125]:
    w = int(M0 * wf)
    P = power(sequence(ETA0, M0, N0, w))
    k0 = first_null(P, N0, M0)
    c2ok &= (k0 == int(round(1 / wf)))
    print("      w = %2d bin（占空比 %.3f）：第一零点 k=%s，1/w 预测 %d"
          % (w, wf, k0, int(round(1 / wf))))
C2 = c2ok
print("      → %s" % ("PASS" if C2 else "FAIL"))

print("  C3 第一零点不随采样率移动（占空比 1/2，四档 M）：")
c3ok = True
for M in [8, 16, 32, 64]:
    P = power(sequence(ETA0, M, N0, M // 2))
    k0 = first_null(P, N0, M)
    c3ok &= (k0 == 2)
    print("      M=%2d：第一零点 k=%s，Nyquist k=%d" % (M, k0, M // 2))
C3 = c3ok
print("      → %s（零点由 w 定，与采样 Nyquist 分离）" % ("PASS" if C3 else "FAIL"))

print("  C4 瞬时折返（w = 1 bin）下无零点，带宽随采样扩张：")
c4ok = True
for M in [32, 64, 128]:
    P = power(sequence(ETA0, M, N0, 1))
    c = comb(P, N0, M)
    nz = [k for k in range(1, c.size) if c[k] > 1e-6 * c[0]]
    c4ok &= (len(nz) >= 0.9 * (c.size - 1))
    print("      M=%3d：带内非零谐波 %d / %d，首个零点 %s"
          % (M, len(nz), c.size - 1, first_null(P, N0, M)))
C4 = c4ok
print("      → %s（无 E_P 截断；有效带宽由采样定，属标度伪影）" % ("PASS" if C4 else "FAIL"))

print("  C5 截断恰在 E_P（w = t_P）时相干部分整体消失：")
c5ok = True
for M in [8, 32]:
    P = power(sequence(ETA0, M, N0, M))
    c = comb(P, N0, M)
    tail = float(np.sum(c[1:])) / float(c[0])
    c5ok &= tail < 1e-12
    print("      M=%2d：非零阶谐波总功率 / 直流 = %.3e" % (M, tail))
C5 = c5ok
print("      → %s（常值信号，无涨落；「截断在 E_P」与「非零涨落」不可同时成立）"
      % ("PASS" if C5 else "FAIL"))

# ---------------- L 组：线宽 ----------------
print("\n[L 组：谱线线宽]")

print("  L1 线宽 vs 观测窗长 N（M=32，占空比 1/2，k=1）：")
l1 = []
for N in [256, 512, 1024, 2048]:
    P = power(sequence(1.0, M0, N, W0))
    hw = line_width(P, N, M0, 1)
    l1.append((N, hw, hw * N))
    print("      N=%4d：-3 dB 宽 %.5f，宽*N = %.3f（矩形窗主瓣 1.772，网格量化到 2）"
          % (N, hw, hw * N))
L1 = all(1.5 < p < 2.5 for *_, p in l1) and (l1[-1][1] < l1[0][1] / 7)
print("      → %s（线宽 ∝ 1/N，由观测窗定，非 t_P）" % ("PASS" if L1 else "FAIL"))

print("  L2 线宽与脉冲宽度 w 无关（N=1024，k=1）：")
l2 = []
for wf in [0.5, 0.25, 0.125]:
    w = int(M0 * wf)
    P = power(sequence(1.0, M0, N0, w))
    hw = line_width(P, N0, M0, 1)
    l2.append(hw * N0)
    print("      占空比 %.3f：-3 dB 宽*N = %.3f" % (wf, hw * N0))
L2 = (max(l2) - min(l2)) / np.mean(l2) < 0.05
print("      → %s" % ("PASS" if L2 else "FAIL"))

# ---------------- E 组：非相干部分与恒真排除 ----------------
print("\n[E 组：非相干部分、恒真排除]")

etas = [0.8, 0.5, 0.3, 0.15, 0.05]
ratios = []
for e in etas:
    vals = [floor_level(power(sequence(e, M0, N0, W0, seed=SEED + s)), N0)
            for s in range(N_SEED)]
    ratios.append(float(np.mean(vals)) / (e * (1 - e)))
rel = (max(ratios) - min(ratios)) / np.mean(ratios)
E1 = rel < 0.05
print("  E1 谱底 ∝ eta(1-eta)：底/[eta(1-eta)] 五档 %s，相对极差 %.4f → %s"
      % (" ".join("%.4f" % r for r in ratios), rel, "PASS" if E1 else "FAIL"))
print("     理论值 2w/M = %.4f（w=16, M=32）" % (2.0 * W0 / M0))

Pp = power(sequence(ETA0, M0, N0, W0, poisson=True))
cp = comb(Pp, N0, M0)
Pc = power(sequence(ETA0, M0, N0, W0))
cc = comb(Pc, N0, M0)
r_poi = float(np.mean(cp[1:])) / floor_level(Pp, N0)
r_per = float(np.mean(cc[1:])) / floor_level(Pc, N0)
E2 = r_poi < 0.05 * r_per
print("  E2 泊松反例（同宽同占空比，位置随机）：峰/底 = %.2f（周期档 %.1f）→ %s"
      % (r_poi, r_per, "PASS" if E2 else "FAIL"))

print("  E3 谱底低频不发散（无 1/f 上升）：")
e3ok = True
for eta in [0.5, 0.15]:
    P = power(sequence(eta, M0, N0, W0))
    vlo = floor_level(P, N0, 0.02, 0.06)
    vmid = floor_level(P, N0, 0.10, 0.30)
    e3ok &= (vlo / vmid < 1.5)
    print("      eta=%.2f：f in [0.02,0.06] 底 / f in [0.10,0.30] 底 = %.4f（1/f 时应 >>1）"
          % (eta, vlo / vmid))
E3 = e3ok
print("      → %s（低频有限且随 f 下降，与 1/f 相反）" % ("PASS" if E3 else "FAIL"))

print("  E4 底的零点与线的零点同位，同为 sinc^2(f w)：")
e4ok = True
for wf in [0.5, 0.25]:
    w = int(M0 * wf)
    P = power(sequence(0.5, M0, N0, w))
    inb = floor_level(P, N0, 0.05, 0.5 / wf)
    # 零点取在 f = 1/w 处，用远小于零点间距（1/w）的绝对窗，避开梳线本身
    f0 = 1.0 / wf
    near_null = floor_level(P, N0, f0 - 0.02, f0 + 0.02)
    ratio = near_null / inb
    e4ok &= (ratio < 1e-2)
    print("      占空比 %.2f（零点 f=%.1f）：f=%.2f±0.02 处底/带内底 = %.3e"
          % (wf, f0, f0, ratio))
E4 = e4ok
print("      → %s（底与线共享同一包络，截断由 w 定，非采样 Nyquist）" % ("PASS" if E4 else "FAIL"))

# ---------------- 汇总 ----------------
flags = [M1, M2, M3, C1, C2, C3, C4, C5, L1, L2, E1, E2, E3, E4]
names = ["M1", "M2", "M3", "C1", "C2", "C3", "C4", "C5", "L1", "L2", "E1", "E2", "E3", "E4"]
print("\n[汇总] %d/%d" % (sum(flags), len(flags)))
print("  " + "  ".join("%s:%s" % (n, "P" if x else "F") for n, x in zip(names, flags)))

print("\n[判语]")
print("  三点中「低频无红外发散」成立（E3：低频底有限，随 f 下降）。")
print("  「厚度由 t_P 定」只在梳线间隔意义上成立（C1）；线宽由观测窗长定（L1、L2），")
print("  包络宽度由折返持续时间 w 定（C2、E4），二者都与 t_P 无关。")
print("  「高频在 E_P 硬截断」不成立（C4、C5）：瞬时折返下梳线与谱底都延伸至采样")
print("  Nyquist；让包络零点恰落在 E_P 需 w = t_P，而该极限下信号为常值、")
print("  相干涨落整体消失。硬截断与「有涨落」在本模型内不能同时成立。")

print("\n[边界声明]")
print("  本判据只检验命题 9 的计数通道，不检验体系真空是否真的由该通道承载。")
print("  若真实真空涨落另有连续自由度（场模式），则本轮的判否不迁移到该自由度上。")
print("  「厚度由 t_P 而非耦合常数定」的对照半句在本模型内不可判：模型不含耦合常数。")
print("  本判据不检验 eta 的动力学，P4-45 的硬缺口（eta 演化方程）不受本轮影响。")

# ---------------- 出图 ----------------
fig, ax = plt.subplots(figsize=(9.5, 5))
for wf, c in [(0.5, "C0"), (0.25, "C2"), (0.125, "C3")]:
    w = int(M0 * wf)
    P = power(sequence(1.0, M0, N0, w))
    cc2 = comb(P, N0, M0)
    kk = np.arange(cc2.size)
    amp = cc2 / cc2[0]
    ax.semilogy(kk, np.maximum(amp, 1e-14), "o-", ms=4, color=c,
                label="占空比 %.3f（第一零点 k=%d）" % (wf, int(round(1 / wf))))
ax.axvline(M0 / 2.0, ls=":", color="0.5", label="采样 Nyquist（k=%d）" % (M0 // 2))
ax.set_xlabel("谐波阶 k（f = k/t_P）")
ax.set_ylabel("归一化梳线幅度")
ax.set_title("P4-45：包络零点由折返持续时间定，非 E_P")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
ax.set_ylim(1e-14, 2)
fig.tight_layout()
p1 = os.path.join(IMG, "p445_vacuum_envelope.png")
fig.savefig(p1, dpi=150)
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
NN = [256, 512, 1024, 2048]
ww = [line_width(power(sequence(1.0, M0, N, W0)), N, M0, 1) for N in NN]
ax[0].loglog(NN, ww, "o-", label="实测线宽")
ax[0].loglog(NN, [2.0 / n for n in NN], "--", label="2/N")
ax[0].set_xlabel("周期数 N")
ax[0].set_ylabel("-3 dB 线宽（1/t_P）")
ax[0].set_title("线宽由观测窗定")
ax[0].legend(fontsize=9)
ax[0].grid(alpha=0.3, which="both")

MM = [8, 16, 32, 64]
nulls = []
for M in MM:
    P = power(sequence(ETA0, M, N0, M // 2))
    k0 = first_null(P, N0, M)
    nulls.append(k0 if k0 else M / 2)
ax[1].plot(MM, nulls, "s-", label="包络第一零点")
ax[1].plot(MM, [m / 2.0 for m in MM], ":", color="0.5", label="采样 Nyquist")
ax[1].set_xlabel("每周期 bin 数 M")
ax[1].set_ylabel("f（1/t_P）")
ax[1].set_title("零点不随采样率移动")
ax[1].legend(fontsize=9)
ax[1].grid(alpha=0.3)

ee = np.array(etas)
fl = [floor_level(power(sequence(e, M0, N0, W0)), N0) for e in ee]
ax[2].loglog(ee * (1 - ee), fl, "o", label="实测谱底")
xx = np.linspace((ee * (1 - ee)).min(), 0.25, 10)
ax[2].loglog(xx, fl[-1] * xx / (ee[-1] * (1 - ee[-1])), "--", label="∝ eta(1-eta)")
ax[2].set_xlabel("eta(1-eta)")
ax[2].set_ylabel("谱底高度")
ax[2].set_title("白底强度")
ax[2].legend(fontsize=9)
ax[2].grid(alpha=0.3, which="both")
fig.tight_layout()
p2 = os.path.join(IMG, "p445_vacuum_width_cutoff.png")
fig.savefig(p2, dpi=150)
plt.close(fig)

print("\n[图] %s" % p1)
print("[图] %s" % p2)
