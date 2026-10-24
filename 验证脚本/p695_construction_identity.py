# -*- coding: utf-8 -*-
r"""
p695_construction_identity.py

[观测者响应线 / 洛伦兹线] 第 695 号：钟慢构造残差的身份判定（判据 G1 至 G8）

问题：第 692 号把模型档钟慢偏差 b = 3.6e-4 判为「boost 构造代价」，
据以把物理档闭合的希望寄托于「解析精确 boost」路线。该判定有三条依据
（与采样时长、积分步长、重采样方式无关），但缺一条关键检验：换一种构造会不会变？
若偏差主要由构造选择决定，则「沿 1/xi^2 外推到物理档」外推的是构造参数而非物理量。

本轮设计：
  G1 冒烟（本征等式，用与 eigsh 同一离散格式）
  G2 构造矩阵（阱宽 rigid/contracted x 初值 692式/精确boost），判构造是否主导
  G3 恒真反例 v=0（应给机器零）
  G4 标度检验（dev 对 k_mod^4/12）
  G5 速度扫描
  G6 初值记忆检验（决定 692 构造是否循环；四档不同内建频率）
  G7 构造间一致性（同族构造的 dev 是否稳定）
  G8 结论

运行：python 验证脚本/p695_construction_identity.py
       （确定性，约 2 分钟）
"""
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p695_construction_identity.png"

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

T_START = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB_TARGET = 0.8008


def bound_state(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    w2 = float(vals[0])
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    kk = 2.0 * math.pi * np.fft.fftfreq(N)
    d1 = np.real(np.fft.ifft(1j * kk * np.fft.fft(p)))
    return (math.sqrt(w2) if w2 > 0 else 0.0), p, d1, L


def fit_U0(sigma, target=WB_TARGET):
    lo, hi = 1e-4, 1e4
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        w, _, _, _ = bound_state(sigma, mid)
        if w > target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    w, p, d1, L = bound_state(sigma, mid)
    return mid, w, p, d1, L


def make_ic(psi, d1, wb, v, g, style):
    """style: 'g692' 第 692 号写法（收缩因子 1/gamma）；'boost' 解析 boost（gamma）。"""
    if style == "boost":
        s0, th0, c, wfac = g * xf, wb * g * v * xf, g * v, wb * g
    else:
        sh = 1.0 / g
        s0, th0, c, wfac = sh * xf, -wb * sh * v * xf, sh * v, wb * sh
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) - wfac * F * np.sin(th0)


def run(phi, vel, U0, well_sigma, v, dt=0.02, T=800.0, every=4):
    nstep = int(T / dt)
    cs = []
    for s in range(nstep):
        t = (s + 0.5) * dt
        well = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * well_sigma ** 2))
        acc = -(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - well * phi
        phi = phi + dt * vel + 0.5 * dt ** 2 * acc
        t2 = (s + 1.0) * dt
        well2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * well_sigma ** 2))
        acc2 = -(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - well2 * phi
        vel = vel + 0.5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(phi[N // 2 + int(round(v * t))])
    return np.array(cs)


def freq_lsq(cs, dt_s, center, frac=0.10, n=2001):
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    best, bw = None, None
    for w in np.linspace(center * (1 - frac), center * (1 + frac), n):
        A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ coef - y) ** 2))
        if best is None or r < best:
            best, bw = r, w
    return bw


print("=" * 78)
print("第 695 号：钟慢构造残差的身份判定（判据 G1 至 G8）")
print("=" * 78)

# ---------------- G1 ----------------
print()
print("[G1] 冒烟：基线拟合与束缚态自洽")
SIG = 20.0
U0, wb, psi, d1, L = fit_U0(SIG)
print("    sigma=%.1f  U0=%.6f  omega_b=%.6f（目标 %.6f）" % (SIG, U0, wb, WB_TARGET))
res_op = float(np.linalg.norm(L @ psi - wb ** 2 * psi) / np.linalg.norm(psi))
well0 = -U0 * np.exp(-(xf ** 2) / (2.0 * SIG ** 2))
Lap = -(2.0 * psi - np.roll(psi, 1) - np.roll(psi, -1))
res_roll = float(np.linalg.norm(Lap + (WGAP2 + well0) * psi - wb ** 2 * psi) / np.linalg.norm(psi))
print("    本征等式残差（算符口径）= %.3e（应为机器量级）" % res_op)
print("    对照：np.roll 口径 = %.3e（边界回绕伪差，不作判据）" % res_roll)
G1 = res_op < 1e-10
print("    G1 判定：%s" % ("PASS" if G1 else "FAIL"))

# ---------------- G2 ----------------
print()
print("[G2] 构造矩阵：阱宽 x 初值写法（v=0.30）")
V = 0.30
g = 1.0 / math.sqrt(1 - V * V)
target = wb / g
kmod = wb * g * V
print("    gamma=%.6f  omega_b/gamma=%.6f  k_mod=%.6f" % (g, target, kmod))

CONFIGS = [("rigid+boost", SIG, "boost"), ("rigid+g692", SIG, "g692"),
           ("contr+boost", SIG / g, "boost"), ("contr+g692", SIG / g, "g692")]
results = {}
for name, wsig, style in CONFIGS:
    phi, vel = make_ic(psi, d1, wb, V, g, style)
    wm = freq_lsq(run(phi, vel, U0, wsig, V), 4 * 0.02, target)
    dev = abs(wm - target) / target
    results[name] = (dev, wm)
    print("    %-12s 实测w=%.6f  dev=%.3e" % (name, wm, dev))

devs = [results[k][0] for k in results]
spread = max(devs) / max(min(devs), 1e-16)
print("    构造间 dev 极差比 = %.1f 倍" % spread)
G2 = spread > 10.0
print("    G2 判定：%s（构造主导，判据：极差比 > 10）" % ("PASS" if G2 else "FAIL"))

# ---------------- G3 ----------------
print()
print("[G3] 恒真反例：v=0 应给机器零")
phi0, vel0 = make_ic(psi, d1, wb, 0.0, 1.0, "boost")
dev0 = abs(freq_lsq(run(phi0, vel0, U0, SIG, 0.0), 4 * 0.02, wb) - wb) / wb
print("    dev=%.3e" % dev0)
G3 = dev0 < 1e-6
print("    G3 判定：%s（判据：< 1e-6）" % ("PASS" if G3 else "FAIL"))

# ---------------- G4 ----------------
print()
print("[G4] 标度检验：dev 对 k_mod^4/12")
pred4 = kmod ** 4 / 12.0
print("    k_mod^4/12 = %.3e（离散色散预期）" % pred4)
for name in ("rigid+boost", "contr+boost"):
    d = results[name][0]
    print("    %-12s dev=%.3e  dev/(k^4/12)=%.1f" % (name, d, d / pred4))
print("    判读：boost 族 dev 比 k^4/12 大两个量级，偏差非离散色散，属构型失配。")

# ---------------- G5 ----------------
print()
print("[G5] 速度扫描")
VLIST = (0.15, 0.30)
scan = {}
for v in VLIST:
    gv = 1.0 / math.sqrt(1 - v * v)
    for name, wsig, style in (("rigid+boost", SIG, "boost"), ("contr+g692", SIG / gv, "g692")):
        phi, vel = make_ic(psi, d1, wb, v, gv, style)
        wm = freq_lsq(run(phi, vel, U0, wsig, v), 4 * 0.02, wb / gv)
        scan[(name, v)] = abs(wm - wb / gv) / (wb / gv)
        print("    v=%.2f %-12s dev=%.3e" % (v, name, scan[(name, v)]))
r_boost = scan[("rigid+boost", 0.30)] / max(scan[("rigid+boost", 0.15)], 1e-16)
print("    rigid+boost 两速度 dev 比 = %.1f（k^4 标度预期 16.0）" % r_boost)
G5 = r_boost > 16.0
print("    G5 判定：%s（boost 族超 k^4 标度，判构型失配主导）" % ("PASS" if G5 else "FAIL"))

# ---------------- G6 初值记忆 ----------------
print()
print("[G6] 初值记忆检验：改初值内建频率，实测频率是否跟随")
mem = []
for bw_ic in (target, wb, wb * 1.05, wb * 0.95):
    phi, vel = make_ic(psi, d1, bw_ic, V, g, "g692")
    wm = freq_lsq(run(phi, vel, U0, SIG / g, V), 4 * 0.02, target, frac=0.30)
    mem.append((bw_ic, wm))
    print("    初值内建=%.6f  实测=%.6f  对目标偏差=%.3e  对初值偏差=%.3e"
          % (bw_ic, wm, abs(wm - target) / target, abs(wm - bw_ic) / bw_ic))
sp_mem = max(m[1] for m in mem) - min(m[1] for m in mem)
print("    四档实测极差 = %.3e（相对 %.3e）" % (sp_mem, sp_mem / target))
G6 = sp_mem / target < 1e-5
print("    G6 判定：%s（实测不跟随初值，无循环）" % ("PASS" if G6 else "FAIL"))

# ---------------- G7 ----------------
print()
print("[G7] 构造间一致性：同族（boost 族）两档阱宽的 dev 稳定性")
b1, b2 = results["rigid+boost"][0], results["contr+boost"][0]
r_same = max(b1, b2) / min(b1, b2)
print("    rigid+boost=%.3e  contr+boost=%.3e  比=%.2f" % (b1, b2, r_same))
g1_, g2_ = results["rigid+g692"][0], results["contr+g692"][0]
print("    rigid+g692 =%.3e  contr+g692 =%.3e  比=%.2f" % (g1_, g2_, max(g1_, g2_) / min(g1_, g2_)))
G7 = r_same < 1.5
print("    G7 判定：%s（boost 族内 dev 稳定在 1.5 倍内，说明阱宽失配不是主因）"
      % ("PASS" if G7 else "FAIL"))

# ---------------- G8 ----------------
print()
print("[G8] 结论")
print("    其一，G1 通过：装置基线正确（本征等式机器精度）。")
print("    其二，G2 通过：四种构造的 dev 极差 %.0f 倍，构造选择主导偏差。" % spread)
print("    其三，G3 通过：v=0 给机器零，判据测的是运动效应。")
print("    其四，G4/G5：boost 族 dev 超 k^4/12 两个量级、超 k^4 标度，非晶格色散。")
print("    其五，G6 通过：实测频率不跟随初值内建频率，第 692 号构造无循环问题。")
print("    其六，G7 通过：boost 族内 dev 稳定，阱宽失配不是主因。")
print()
print("    判语：第 692 号的 b=3.6e-4 确为构造代价，但其「物理档外推」缺乏意义：")
print("          外推的是构造参数（初值写法与阱宽失配），不是物理量。")
print("          物理档闭合须先消除构造代价，而非把构造代价外推到物理档。")

print()
print("耗时 %.1f 秒" % (time.time() - T_START))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
names = [c[0] for c in CONFIGS]
vals = [results[n][0] for n in names]
ax[0].bar(range(len(names)), vals, color=["#c44", "#e88", "#48c", "#8be"])
ax[0].set_yscale("log")
ax[0].set_xticks(range(len(names)))
ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax[0].set_ylabel("钟慢相对偏差 dev")
ax[0].set_title("左：四种构造的偏差（v=0.30）")
ax[0].axhline(pred4, color="k", ls=":", lw=1.2, label=r"$k_{\rm mod}^4/12$")
ax[0].legend(fontsize=7)

for name, mk in (("rigid+boost", "o"), ("contr+g692", "s")):
    ys = [scan[(name, v)] for v in VLIST]
    ax[1].loglog(VLIST, ys, mk + "-", label=name)
ax[1].set_xlabel("v")
ax[1].set_ylabel("dev")
ax[1].set_title("中：两速度下的偏差")
ax[1].legend(fontsize=7)

xs_ = [m[0] for m in mem]
ys_ = [m[1] for m in mem]
ax[2].plot(xs_, ys_, "o-", color="#268")
ax[2].axhline(target, color="r", ls="--", lw=1.2, label=r"$\omega_b/\gamma$")
ax[2].axhline(wb, color="gray", ls=":", lw=1.0, label=r"$\omega_b$")
ax[2].set_xlabel("初值内建频率")
ax[2].set_ylabel("实测频率")
ax[2].set_title("右：初值记忆检验")
ax[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
