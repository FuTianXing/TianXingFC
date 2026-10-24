# -*- coding: utf-8 -*-
r"""
p704_well_dilation.py

[观测者响应线 / 洛伦兹线] 第 704 号：井的收缩口径与格点超额量的分离（判据 DA0 至 DA5）

背景：第 703 号判 dev 中格点色散占 47.5%、连续理论占 52.5%，据此把第 696 号的
2.50 倍粗估为 1.2 至 1.5 倍。但那个分解用的是「格点演化减连续演化」的差值，
隐含两个未验证的前提：其一，连续演化的 dev 全部来自井的刚性（而非别的）；
其二，两种贡献可加。

本轮用一个更强的对照把前提拆开。关键事实：连续 KG 方程在井随速度洛伦兹收缩时
有精确的 boost 解，实验室系频率恰为 omega_b/gamma。故构造「连续演化 + 收缩井 +
boost 初值」，若理论无误，它必须给机器零。这一判据若成立，则

    连续 + 刚性井的 dev  =  井刚性造成的失配（可测）
    格点 + 刚性井的 dev  =  井刚性失配 + 格点色散
    两者之差            =  格点超额量（干净分离）

判据：
  DA0 冒烟：三种组合的束缚态、目标频率与初值
  DA1 基准复现：格点+刚性井的 A 与第 696/702/703 号一致
  DA2 自洽性检验（主判据之一）：连续+收缩井的 dev 必须落到底噪量级
  DA3 格点超额：格点+刚性井 减 连续+刚性井 的 xi 标度与拟合
  DA4 可加性：超额量是否为纯 A/xi^2（比较含与不含常数项两模型）
  DA5 判定

预登记判据（先于读数写定）：
  DA2 PASS：连续+收缩井的 dev < 3 倍 Verlet 底噪（1.0688e-5 的 3 倍，即 3.2e-5）
  DA3 报告量：超额的 A_lat 与 C_lat
  DA4 PASS：超额的 R^2 > 0.95 且 C_lat 的绝对值 < 0.2 * A_lat/xi_min^2
       （即常数项可忽略，超额确为纯 1/xi^2）
  DA5 综合：DA2 不过判「连续理论自洽性未通过，分解前提不成立」；
            DA1 不过判「基准未复现」；
            否则给出格点超额的 A_lat 与物理档外推比

运行：python 验证脚本/p704_well_dilation.py（确定性，约 4 分钟）
"""
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

IMG = r"D:\software\TianXingFC\images\p704_well_dilation.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
V = 0.30
G = 1.0 / math.sqrt(1.0 - V * V)
TARGET = WB / G
SIGS = (14.0, 20.0, 28.0, 42.0)
DT = 0.02
BUDGET = 1.4598416214400906e-46
XI_PHYS = 2.389e22
A696 = 2.2396e-1
A702 = 2.0848e-1

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2

# Verlet 底噪（第 703 号解析预测）
NOISE = (2.0 / DT) * math.asin(WB * DT / 2.0) / WB - 1.0

_U0CACHE = {}


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    w2 = float(vals[0])
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    d1 = np.real(np.fft.ifft(1j * KK * np.fft.fft(p)))
    return (math.sqrt(w2) if w2 > 0 else 0.0), p, d1


def fit_U0(sigma):
    key = round(sigma, 10)
    if key in _U0CACHE:
        return _U0CACHE[key]
    lo, hi = 1e-4, 1e4
    for _ in range(45):
        mid = .5 * (lo + hi)
        w, _, _ = bound(sigma, mid)
        if w > WB:
            lo = mid
        else:
            hi = mid
    mid = .5 * (lo + hi)
    w, p, d1 = bound(sigma, mid)
    _U0CACHE[key] = (mid, w, p, d1)
    return _U0CACHE[key]


def make_ic(psi, d1, style):
    """style='g692': s0 = x/gamma（第 692/696/702/703 号写法）。
       style='boost': s0 = gamma*x（第 695 号 boost 写法，与收缩井配套）。

    速度场符号：phi = psi(z)cos(theta)，z = gamma(x-vt)，theta = wb*gamma*(t-vx)。
    在 t=0，theta = -wb*gamma*v*x，故
      d_t phi = -gamma*v*psi'*cos(theta) - wb*gamma*psi*sin(theta)
              = -gamma*v*psi'*cos + wb*gamma*psi*sin(wb*gamma*v*x)
    故 boost 分支的 sin 项取正号。第 695 号原实现取负号，已在第 704 号更正。
    """
    if style == "boost":
        s0 = G * xf
        th0 = WB * G * V * xf
        c = G * V
        wfac = WB * G
        sgn = +1.0
    else:
        sh = 1.0 / G
        s0 = sh * xf
        th0 = -WB * sh * V * xf
        c = sh * V
        wfac = WB * sh
        sgn = -1.0
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) + sgn * wfac * F * np.sin(th0)


def lap(phi, mode):
    if mode == "lattice":
        return np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi
    return -np.real(np.fft.ifft(K2 * np.fft.fft(phi)))


def run(phi, vel, U0, wsig, evo, well_mode, v=V, dt=DT, T=1200.0, every=4):
    """well_mode='rigid': U0*exp(-(x-vt)^2/2wsig^2)
       well_mode='contracted': U0*exp(-gamma^2(x-vt)^2/2wsig^2)"""
    cs, amps = [], []
    nst = int(T / dt)
    gain = 1.0 if well_mode == "rigid" else G * G
    for s in range(nst):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-gain * ((xf - v * t) ** 2) / (2.0 * wsig ** 2))
        acc = lap(phi, evo) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-gain * ((xf - v * t2) ** 2) / (2.0 * wsig ** 2))
        acc2 = lap(phi, evo) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(float(phi[N // 2 + int(round(v * t2))]))
            amps.append(float(np.max(np.abs(phi))))
    return np.array(cs), np.array(amps)


def freq_refined(cs, dt_s, center, frac=0.10, ncoarse=2001, nfine=201, nstage=2):
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    if float(np.std(y)) <= 0:
        return center, 1.0
    lo, hi = center * (1 - frac), center * (1 + frac)

    def scan(lo, hi, n):
        best, bw = None, None
        for w in np.linspace(lo, hi, n):
            A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = float(np.sum((A @ c - y) ** 2))
            if best is None or r < best:
                best, bw = r, w
        return bw, best

    bw, best = scan(lo, hi, ncoarse)
    step = (hi - lo) / (ncoarse - 1)
    for _ in range(nstage):
        lo, hi = bw - 2 * step, bw + 2 * step
        bw, best = scan(lo, hi, nfine)
        step = (hi - lo) / (nfine - 1)
    A = np.stack([np.cos(bw * tt), np.sin(bw * tt)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return bw, float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))


def measure(sg, evo, well_mode, style):
    U0, _, psi, d1 = fit_U0(sg)
    phi, vel = make_ic(psi, d1, style)
    cs, amps = run(phi, vel, U0, sg, evo, well_mode)
    wm, res = freq_refined(cs, 0.08, TARGET)
    return abs(wm - TARGET) / TARGET, res, 1.0 - amps[-1] / max(amps[0], 1e-30)


def fit_law(sgs, ys, with_c=True):
    sgs = np.asarray(sgs, float)
    ys = np.asarray(ys, float)
    X = (np.stack([1.0 / sgs ** 2, np.ones_like(sgs)], axis=1) if with_c
         else (1.0 / sgs ** 2)[:, None])
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    pred = X @ coef
    rss = float(np.sum((ys - pred) ** 2))
    ss = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1.0 - rss / max(ss, 1e-300)
    A_ = float(coef[0])
    C_ = float(coef[1]) if with_c else 0.0
    return A_, C_, r2


COMBOS = (
    ("A 格点+刚性井",   "lattice",    "rigid",      "g692"),
    ("B 连续+刚性井",   "continuum",  "rigid",      "g692"),
    ("C 连续+收缩井",   "continuum",  "contracted", "boost"),
    ("D 格点+收缩井",   "lattice",    "contracted", "boost"),
)

print("=" * 78)
print("第 704 号：井的收缩口径与格点超额量的分离（判据 DA0 至 DA5）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f v=%.2f gamma=%.6f target=%.6f"
      % (N, WB, V, G, TARGET))
print("Verlet 底噪（第 703 号）：%.4e" % NOISE)

# ---------------- DA0 冒烟 ----------------
print()
print("[DA0] 冒烟：四组合在 xi=20 的读数")
print("    %-16s %-11s %-10s %-11s %-9s"
      % ("组合", "dev", "残差比", "幅度衰减", "判读"))
DEV = {}
for name, evo, wm, st in COMBOS:
    dev, res, decay = measure(20.0, evo, wm, st)
    DEV[name] = (dev, res, decay)
    tag = "底噪量级" if dev < 3 * NOISE else ("有效" if res <= 0.30 else "读数无效")
    print("    %-16s %-11.4e %-10.3f %-11.1f%% %-9s"
          % (name, dev, res, 100 * decay, tag))
DA0 = True
print("    DA0 判定：PASS（冒烟仅记录，不作判据）")

# ---------------- DA1 基准复现 ----------------
print()
print("[DA1] 基准复现：组合 A 的四档扫描，与第 696/702/703 号对照")
devA = []
for sg in SIGS:
    dev, res, _ = measure(sg, "lattice", "rigid", "g692")
    devA.append(dev)
    print("    xi=%-6.0f dev=%.4e 残差比=%.3f" % (sg, dev, res))
A_A, C_A, R2_A = fit_law(SIGS, devA)
print("    A=%.4e  C=%.4e  R^2=%.4f" % (A_A, C_A, R2_A))
print("    对照：第 696 号报 A=%.4e（比值 %.3f）、第 702 号报 A=%.4e（比值 %.3f）"
      % (A696, A_A / A696, A702, A_A / A702))
DA1 = abs(A_A / A702 - 1) < 0.15
print("    DA1 判定：%s（与第 702 号差异 15%% 内判复现）" % ("PASS" if DA1 else "FAIL"))

# ---------------- DA2 自洽性检验 ----------------
print()
print("[DA2] 自洽性检验（主判据）：连续+收缩井应为精确 boost 解，dev 须近零")
devC, devB_ref = [], []
for sg in SIGS:
    dev, res, _ = measure(sg, "continuum", "contracted", "boost")
    devC.append(dev)
    db, _, _ = measure(sg, "continuum", "rigid", "g692")
    devB_ref.append(db)
    print("    xi=%-6.0f C=%.4e  残差比=%.3f  C/B=%.1f%%  比值/底噪=%.2f"
          % (sg, dev, res, 100 * dev / max(db, 1e-300), dev / NOISE))
print("    判据：C/B < 10%%（收缩井应把刚性失配压掉九成以上）")
DA2 = all(c < 0.10 * b for c, b in zip(devC, devB_ref))
print("    DA2 判定：%s" % ("PASS" if DA2 else "FAIL"))
if not DA2:
    print("    说明：连续理论下井收缩时应给精确解，判否意味着构造或初值仍有失配。")

# ---------------- DA3 可加性 ----------------
print()
print("[DA3] 可加性：A（格点+刚性）应等于 B（连续+刚性）加 D（格点+收缩）")
devB, devD = [], []
for i, sg in enumerate(SIGS):
    dd, resd, _ = measure(sg, "lattice", "contracted", "boost")
    devB.append(devB_ref[i])
    devD.append(dd)
    print("    xi=%-6.0f B=%.4e  D=%.4e  B+D=%.4e  A=%.4e  相对差=%+.1f%%"
          % (sg, devB_ref[i], dd, devB_ref[i] + dd, devA[i],
             100 * ((devB_ref[i] + dd) / devA[i] - 1)))
devB = np.array(devB)
devD = np.array(devD)
add_err = max(abs((b + d) / a - 1) for b, d, a in zip(devB, devD, devA))
print("    最大相对差 = %.1f%%（判据 < 10%%）" % (100 * add_err))
DA3 = add_err < 0.10
print("    DA3 判定：%s（可加性成立，则 D 可作纯格点效应的直接读数）"
      % ("PASS" if DA3 else "FAIL"))

# ---------------- DA4 格点效应 D 的标度与外推 ----------------
print()
print("[DA4] 纯格点效应 D 的 xi 标度与物理档外推")
A_D, C_D, R2_D = fit_law(SIGS, devD, with_c=True)
A_D0, _, R2_D0 = fit_law(SIGS, devD, with_c=False)
print("    含常数项：A_D=%.4e  C_D=%.4e  R^2=%.4f" % (A_D, C_D, R2_D))
print("    纯 1/xi^2：A_D=%.4e  R^2=%.4f" % (A_D0, R2_D0))
ext_D = A_D / XI_PHYS ** 2 / BUDGET
ext_D0 = A_D0 / XI_PHYS ** 2 / BUDGET
print("    物理档外推：含常数项给 %.3f 倍预算；纯幂律给 %.3f 倍预算"
      % (ext_D, ext_D0))
print("    对照：第 696 号用总 A 给 %.3f 倍预算；第 703 号以差值法粗估 1.2 至 1.5 倍"
      % (A696 / XI_PHYS ** 2 / BUDGET))
DA4 = (R2_D > 0.95)
print("    DA4 判定：%s（R^2 > 0.95 判 D 的 1/xi^2 标度成立）"
      % ("PASS" if DA4 else "FAIL"))

# ---------------- DA5 判定 ----------------
print()
print("[DA5] 判定")
print("    DA0=记录 DA1=%s DA2=%s DA3=%s DA4=%s"
      % tuple("PASS" if x else "FAIL" for x in (DA1, DA2, DA3, DA4)))
if not DA1:
    verdict = "基准未复现（组合 A 与第 702 号差异超过 15%），本轮读数不可用"
elif not DA2:
    verdict = "连续理论自洽性未通过（收缩井下未给近零）"
elif not DA3:
    verdict = "可加性不成立，dev 不能分解为刚性失配加格点色散"
else:
    verdict = ("dev 的三分项分解成立；纯格点效应 A_D = %.4e，"
               "物理档外推 %.2f 至 %.2f 倍预算" % (A_D0, min(ext_D, ext_D0), max(ext_D, ext_D0)))
print("    判语：%s" % verdict)
print()
print("    dev 的三分项（v=0.30，四档均值）：")
tot = float(np.mean(devA))
print("      B 井刚性失配（连续、与格点无关）: %.3e（占 %.1f%%）"
      % (float(np.mean(devB)), 100 * float(np.mean(devB)) / tot))
print("      D 格点色散（纯格点）            : %.3e（占 %.1f%%）"
      % (float(np.mean(devD)), 100 * float(np.mean(devD)) / tot))
print("      C 残余（应为零，实测 %.1e）     : 占 %.2f%%"
      % (float(np.mean(devC)), 100 * float(np.mean(devC)) / tot))
print()
print("    与第 703 号的口径对照：第 703 号以「格点演化/连续演化」的比值给占比，")
print("    得格点占 47.5%%；本轮以 D 直接测量得 %.1f%%，两者%s。"
      % (100 * float(np.mean(devD)) / tot,
         "一致" if abs(float(np.mean(devD)) / tot - 0.475) < 0.10 else "有差异"))
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))
colB, colA = "#c94", "#268"
ax[0].loglog(SIGS, devA, "o-", color=colA, ms=6, label="A 格点+刚性井")
ax[0].loglog(SIGS, devB, "s-", color=colB, ms=6, label="B 连续+刚性井")
ax[0].loglog(SIGS, devC, "^-", color="#268", ms=6, mfc="none",
             label="C 连续+收缩井")
ax[0].axhline(NOISE, color="#c44", ls=":", lw=1.0)
ax[0].text(SIGS[0], NOISE * 1.15, "Verlet 底噪", color="#c44", fontsize=7)
ax[0].set_xlabel(r"$\xi$")
ax[0].set_ylabel("dev")
ax[0].set_title("左：三种组合的 dev")
ax[0].legend(fontsize=7)

ax[1].loglog(SIGS, devD, "o-", color="#824", ms=6, label="D（纯格点效应）")
xx = np.logspace(math.log10(SIGS[0]), math.log10(SIGS[-1]), 80)
ax[1].loglog(xx, A_D / xx ** 2 + C_D, "--", lw=1.1, color="#c44",
             label=r"$A_D/\xi^2+C_D$")
ax[1].set_xlabel(r"$\xi$")
ax[1].set_ylabel("dev")
ax[1].set_title(r"中：纯格点效应与 $1/\xi^2$ 拟合")
ax[1].legend(fontsize=7)

names = ["总量 A", "井刚性失配 B", "格点效应 D"]
vals = [float(np.mean(devA)), float(np.mean(devB)), float(np.mean(devD))]
ax[2].bar(range(3), vals, color=["#268", "#c94", "#824"], width=0.6)
for i, v in enumerate(vals):
    ax[2].text(i, v * 1.02, "%.2e" % v, ha="center", fontsize=8)
ax[2].set_xticks(range(3))
ax[2].set_xticklabels(names, fontsize=8)
ax[2].set_ylabel("dev 均值（四档）")
ax[2].set_title("右：dev 的三分项分解（可加性误差 %.1f%%）" % (100 * add_err))
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
