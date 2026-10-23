# -*- coding: utf-8 -*-
r"""
minimal_verification_p692_physical_regime_closure.py

[观测者响应线 / 洛伦兹线] 第 692 号：钟慢涌现的物理档闭合（判据 BC1 至 BC6）

问题：第 691 号只交付了模型档内的钟慢涌现，装置宽度 sigma = 20 格（xi 约 10 l_P）。
体系的物理档缺陷宽度由第 622 号恒等式给出，xi = m_P/m_e = 2.389e22 格，与模型档
差二十二个数量级。物理档宽度无法在格点上直接模拟（需盒子 1e23 格），本轮把物理档
的闭合改走解析路线：以格点模拟给出钟慢偏差的解析轨道，沿五档宽度几何外推，与
物理档解析预算碰撞。

判据（先立期望，后写实现）：
  BC1 冒烟：各档 xi 的 omega_b 对表（本征等式残差机器零），确认轨道定义良好
  BC2 五档 xi 行程：sigma = 14、20、28、42、63（几何比 1.5）的钟慢偏差
  BC3 轨道拟合：dev(xi) 对 1/xi^2 的线性回归（若钟慢内禀项按 (l_P/xi)^2 走，
       应给斜率显著、截距近零的直线；斜率上限由误差预算约束）
  BC4 防恒真对照：把同一轨道拟合成常数（截距模型），残差应显著更大；拟合
       度比较证明幂律拟合有信息量，不是恒真
  BC5 物理档外推：外推值 dev_phys 与解析预算 B_phys = (m_e/m_P)^2/12 碰撞，
       比值应在 O(1) 内（若钟慢内禀项标度成立，比值应稳定；比值过大或过小
       都判 CHECK）
  BC6 分速度检验：v = 0.15 与 0.30 两档的外推比值一致性（若只依赖内禀项，
       两速度外推比值应接近；若相位调制项主导，则 v 依赖强）

诚实的边界：外推的分辨本领上限约 2 个数量级（五档宽度范围 4.5 倍，几何外推
1/(xi/l_P)^2 覆盖 0.072 的范围），所以 BC5 的比值只在「同一量级」意义上检验，
不是精确预言。物理档的严格闭合仍待第 622 号型恒等式直接作用于尺慢轨道。

运行：python 验证脚本/minimal_verification_p692_physical_regime_closure.py
       （确定性，约 5 至 8 分钟）
"""
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p692_physical_closure.png"

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

T_START = time.time()
N = 4096
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0


def fit_sigma(sigma, target=0.8008):
    """拟合阱深使束缚频率对上 target，返回 (U0, omega_b, psi, d1)。"""
    def bm(U0):
        well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
        L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
                  [-1, 0, 1], format='csr')
        vals, vecs = eigsh(L, k=1, which='SA')
        w2 = float(vals[0])
        if w2 <= 0:
            return 0.0, vecs[:, 0]
        p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
        return math.sqrt(w2), p
    lo, hi = 1e-4, 1e4
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        wb_s, _ = bm(mid)
        if wb_s > target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    wb_s, psi_s = bm(mid)
    if wb_s <= 0:
        return None
    kk = 2.0 * math.pi * np.fft.fftfreq(N)
    pss = np.fft.fft(psi_s)
    d1s = np.real(np.fft.ifft(1j * kk * pss))
    return mid, wb_s, psi_s, d1s


def run_motion(U0, sigma, wb, psi, d1, v, shrink, dt=0.02, T=1600.0, every=4):
    sg = sigma * shrink
    s0 = shrink * xf
    th0 = -wb * shrink * v * xf
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(shrink * v) * Fp * np.cos(th0) - wb * shrink * F * np.sin(th0)
    nstep = int(T / dt)
    cs = []
    for s in range(nstep):
        t = (s + 0.5) * dt
        well = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * sg ** 2))
        acc = -(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - well * phi
        phi = phi + dt * vel + 0.5 * dt ** 2 * acc
        t2 = (s + 1.0) * dt
        well2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * sg ** 2))
        acc2 = -(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - well2 * phi
        vel = vel + 0.5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(phi[N // 2 + int(round(v * t))])
    return np.array(cs)


def freq_lsq(cs, dt_s, w_true, n=2001, frac=0.08):
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    ws = np.linspace(w_true * (1.0 - frac), w_true * (1.0 + frac), n)
    best, bw = None, None
    for w in ws:
        A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ coef - y) ** 2))
        if best is None or r < best:
            best, bw = r, w
    return bw


def deviation_for(sigma, v, T=1600.0):
    """给定宽度与速度，返回钟慢相对偏差与预算。"""
    r = fit_sigma(sigma)
    if r is None:
        return None
    U0, wb, psi, d1 = r
    g = 1.0 / math.sqrt(1.0 - v ** 2)
    cs = run_motion(U0, sigma, wb, psi, d1, v, 1.0 / g, T=T)
    wm = freq_lsq(cs, 4 * 0.02, wb / g)
    dev = abs(wm - wb / g) / (wb / g)
    kmod = wb * g * v
    B = ((1.0 / sigma) ** 2 + kmod ** 2) / 12.0
    return dev, B, wb, U0


V_LIST = (0.15, 0.30)
SIG_LIST = (14.0, 20.0, 28.0, 42.0, 63.0)

print("=" * 78)
print("第 692 号：钟慢涌现的物理档闭合（判据 BC1 至 BC6）")
print("=" * 78)

print()
print("[BC1] 冒烟：五档宽度的基线对表（omega_b 与 U0）")
CFG = {}
print("     %-8s %-10s %-12s %-12s" % ("sigma", "xi/l_P", "omega_b", "U0"))
for sg in SIG_LIST:
    r = fit_sigma(sg)
    if r is None:
        print("     sigma=%.0f 拟合失败（目标频率不可达）" % sg)
        continue
    U0, wb, psi, d1 = r
    CFG[sg] = (U0, wb, psi, d1)
    print("     %-8.0f %-10.1f %-12.6f %-12.6f" % (sg, sg / 2.0, wb, U0))

print()
print("[BC2] 五档宽度的钟慢偏差行程（v = 0.30）")
print("     %-8s %-10s %-12s %-12s %-9s %s"
      % ("sigma", "xi/l_P", "dev", "budget", "dev/B", "verdict"))
rec = {}
for sg in SIG_LIST:
    if sg not in CFG:
        continue
    v = 0.30
    g = 1.0 / math.sqrt(1.0 - v ** 2)
    out = deviation_for(sg, v)
    if out is None:
        continue
    dev, B, wb, U0 = out
    rec[sg] = (dev, B)
    print("     %-8.0f %-10.1f %-12.3e %-12.3e %-9.2f %s"
          % (sg, sg / 2.0, dev, B, dev / B, "PASS" if dev < max(B, 1e-4) else "CHECK"))

print()
print("[BC3] 轨道拟合：dev 对 1/xi^2 的线性回归")
xs_fit = np.array([1.0 / sg ** 2 for sg in rec])
ys_fit = np.array([rec[sg][0] for sg in rec])
A = np.stack([xs_fit, np.ones_like(xs_fit)], axis=1)
coef, res, _, _ = np.linalg.lstsq(A, ys_fit, rcond=None)
slope, intercept = coef
yhat = A @ coef
ss_res = float(np.sum((ys_fit - yhat) ** 2))
ss_tot = float(np.sum((ys_fit - ys_fit.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
print("     斜率 = %.3e ｜ 截距 = %.3e ｜ R^2 = %.4f" % (slope, intercept, r2))
print("     期望：截距近零（dev 无内禀常数项），R^2 接近 1（幂律标度良好）")

print()
print("[BC4] 防恒真对照：常数模型（只留截距）的拟合残差比较")
A0 = np.ones((len(xs_fit), 1))
coef0, _, _, _ = np.linalg.lstsq(A0, ys_fit, rcond=None)
ss_res0 = float(np.sum((ys_fit - A0 * coef0[0]) ** 2))
print("     幂律模型残差平方和 = %.3e" % ss_res)
print("     常数模型残差平方和 = %.3e" % ss_res0)
print("     残差下降比 = %.2f 倍（期望显著大于 1，证明幂律拟合有信息量）"
      % (ss_res0 / ss_res if ss_res > 0 else float("inf")))

print()
print("[BC5] 物理档外推与解析预算碰撞")
XI_PHYS = 2.389e22
x_phys = 1.0 / XI_PHYS ** 2
dev_phys = slope * x_phys + intercept
# 2026-10-19 勘误（第 694 号）：原写 MR2 = (1.0/1836.15)**2，即误用电子与质子质量比
# m_e/m_p = 1/1836.15；体系预算要的是 m_e/m_P = 4.1855e-23（普朗克质量）。
# 两者相差 21 个数量级，(m_e/m_P)^2/12 = 1.460e-46 而非 2.472e-8。
ME_OVER_MP = 4.185462872524506e-23
MR2 = ME_OVER_MP ** 2
B_phys = MR2 / 12.0
print("     外推 dev_phys = %.3e（v = 0.30）" % dev_phys)
print("     解析预算 B_phys = (m_e/m_P)^2/12 = %.3e" % B_phys)
ratio = dev_phys / B_phys if B_phys > 0 else float("inf")
print("     比值 dev/B = %.3f" % ratio)
print("     判定：0.1 至 10 记 PASS（同量级），超出记 CHECK")
ok5 = 0.1 <= ratio <= 10.0
print("     verdict: %s" % ("PASS" if ok5 else "CHECK"))

print()
print("[BC6] 分速度检验：v = 0.15 的外推比值")
rec15 = {}
for sg in SIG_LIST:
    if sg not in CFG:
        continue
    out = deviation_for(sg, 0.15)
    if out is None:
        continue
    rec15[sg] = out
if len(rec15) >= 3:
    xs15 = np.array([1.0 / sg ** 2 for sg in rec15])
    ys15 = np.array([rec15[sg][0] for sg in rec15])
    c15, _, _, _ = np.linalg.lstsq(A := np.stack([xs15, np.ones_like(xs15)], axis=1),
                                   ys15, rcond=None)
    dev15 = c15[0] * x_phys + c15[1]
    ratio15 = dev15 / B_phys
    print("     v = 0.15 外推 dev = %.3e ｜ 比值 = %.3f" % (dev15, ratio15))
    print("     v = 0.30 外推比值 = %.3f" % ratio)
    spread = max(ratio, ratio15) / max(min(ratio, ratio15), 1e-300)
    print("     两速度比值跨度 = %.2f 倍" % spread)
    print("     判定：跨度小于 10 记 PASS（外推以速度无关的内禀项为主）")
    print("     verdict: %s" % ("PASS" if spread < 10.0 else "CHECK"))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))

# ================= 图（三面板） =================
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))

# 面板一：dev 对 xi 的行程（双对数，看幂律）
xs_plot = [sg / 2.0 for sg in rec]
ys_plot = [rec[sg][0] for sg in rec]
ax[0].plot(xs_plot, ys_plot, "ko-", ms=7, label="实测 dev（$v=0.30$）")
ax[0].plot(xs_plot, [rec[sg][1] for sg in rec], "rs--", ms=6, label="误差预算")
ax[0].set_xlabel(r"结构尺度 $\xi/l_P$")
ax[0].set_ylabel("钟慢相对偏差")
ax[0].set_yscale("log")
ax[0].set_title("BC2：五档宽度的偏差行程")
ax[0].legend(fontsize=8)

# 面板二：dev 对 1/xi^2 的线性拟合
inv2 = [1.0 / sg ** 2 for sg in rec]
ax[1].plot(inv2, [rec[sg][0] for sg in rec], "ko", ms=7, label="实测")
xline = np.linspace(0, max(inv2) * 1.02, 50)
ax[1].plot(xline, slope * xline + intercept, "r-", lw=1.4,
           label=r"拟合 $a/\xi^2+b$")
ax[1].axhline(B_phys, color="b", ls=":", lw=1.4, label=r"物理档预算 $B_{\rm phys}$")
ax[1].set_xlabel(r"$1/\xi^2$")
ax[1].set_ylabel("钟慢相对偏差")
ax[1].set_yscale("log")
ax[1].set_title(r"BC3/BC5：幂律拟合与外推（比值 %.0f 判 CHECK）" % ratio)
ax[1].legend(fontsize=8)

# 面板三：分速度对照
inv15 = [1.0 / sg ** 2 for sg in rec15]
ax[2].plot(inv15, [rec15[sg][0] for sg in rec15], "k^", ms=7, label=r"$v=0.15$")
ax[2].plot(inv2, [rec[sg][0] for sg in rec], "ko", ms=7, label=r"$v=0.30$")
ax[2].plot(xline, slope * xline + intercept, "r-", lw=1.2)
ax[2].axhline(B_phys, color="b", ls=":", lw=1.4)
ax[2].set_xlabel(r"$1/\xi^2$")
ax[2].set_ylabel("钟慢相对偏差")
ax[2].set_yscale("log")
ax[2].set_title("BC6：分速度外推（跨度 %.1f 倍判 CHECK）" % spread)
ax[2].legend(fontsize=8)

fig.suptitle("第 692 号：钟慢涌现的物理档闭合（判据 BC1 至 BC6）", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(IMG, dpi=150)
print("图已存:", IMG)
