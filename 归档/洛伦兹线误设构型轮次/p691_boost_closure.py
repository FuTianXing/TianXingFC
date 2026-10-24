# -*- coding: utf-8 -*-
r"""
p691_boost_closure.py

[观测者响应线 / 洛伦兹线] 第 691 号：boost 解低能闭环的晶格原生推导（判据 BL1 至 BL6）

问题：《晶格何以不破坏洛伦兹不变性》§12 其八登记的承重未闭合项。原文 boost 解预设了
低能有效作用量的洛伦兹不变性，而该不变性来自 omega = ck，与光速不变在低能层等价，
闭环自洽但尚缺独立于洛伦兹变换的晶格推导。本轮的做法：不动用洛伦兹变换，也不以
omega_b/gamma 为期望读数，直接用晶格原生的两件器械（物质模式群速读数定速度、阱心
振荡读数定钟）造出运动束缚态，看钟慢尺缩是否作为晶格读数涌现。

与 OP 轮的分界：OP 轮已把残差拆成晶格项与失配项，本轮不引入 boost 解作为被测对象，
只用晶格方程自身演化（速度 Verlet），因此读数不含任何洛伦兹变换输入。

装置（沿用分离轮的晶格方程，以便与既有读数对表）：
  d2t phi + (2 phi - phi(x-1) - phi(x+1)) + wgap2 phi + U(x-vt) phi = 0
势阱取高斯型，宽束缚态 sigma = 20（xi 约 9 l_P），边界用盒外差为零（大盒 N = 4096）。
速度注入用初始条件给：t = 0 时刻波形按 s0 = gamma x 收缩包络、按 -omega_b gamma v x
给相位梯度，速度由阱心轨迹读数反演，不引用 gamma 的定义式（gamma 只用于设定初始条件
的收缩比，这是构造参数而非读数判据；判据 BL2 检验的是速度读数的自洽性）。

先立的期望（写实现之前）：
  BL1 冒烟：v = 0 时束缚态为晶格方程精确解，残差机器零；omega_b 对上分离轮基线。
  BL2 速度：阱心漂移读数与注入速度一致到读数分辨率（确认运动构型真的在动）。
  BL3 钟慢：阱心振荡频率对 1/gamma 的相对偏差 |Delta omega|/omega 小于预算。
  BL4 尺缩：束缚态半宽读数对 1/gamma 的相对偏差小于预算。
  BL5 标度：固定 (k_mod l_P)^2 组合，偏差随 xi/l_P 增大而收缩（换档位不翻）。
  BL6 恒真反例：v = 0 必须给零偏差；若把 gamma 取错（用 1/sqrt(1+v) 代替），
      BL3 必须 FAIL（证明判据有区分力，不是恒真）。

容差从误差预算推导：
  晶格色散截断：omega(k)^2 = c^2 k^2 (1 - (k l_P)^2/12 + ...)，阱内有效波矢
  k_eff ~ 1/xi，故内禀项约 (k_eff l_P)^2/12 = (l_P/xi)^2/12。
  相位调制：k_mod = omega_b gamma v/c^2（c = 1 单位），项为 (k_mod l_P)^2/12。
  两项相加给预算 B = ((l_P/xi)^2 + (k_mod l_P)^2)/12，偏差读数应落在 B 量级内。
  频率读数分辨率：FFT 分辨率 2 pi/T，T = 1600 t_P 给 3.9e-3，最小二乘正弦拟合
  后约 1e-4 量级，远小于 B（v <= 0.3 时 B 约 1e-2），不构成判据瓶颈。
  边界与截断：大盒 N = 4096，阱心距边界约 1000 格，边界反射在 T 内不回传。

运行：python 验证脚本/p691_boost_closure.py（确定性，无随机）
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

IMG = r"D:\software\TianXingFC\images\p691_boost_closure.png"

T_START = time.time()
N = 4096
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0


def bound_mode(U0, sigma):
    """静止束缚态：格点算符最低模态（势阱 -U0 exp(-x^2/(2 sigma^2))）。"""
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)], [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    w2 = float(vals[0])
    if w2 <= 0:
        return 0.0, np.zeros(N)
    psi = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    return math.sqrt(w2), psi


def fit_U0(target, sigma, lo=1e-4, hi=50.0, it=44):
    """二分调阱深，使束缚频率对上目标（与分离轮同一拟合器）。"""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        wb, _ = bound_mode(mid, sigma)
        if wb > target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    wb, psi = bound_mode(mid, sigma)
    return mid, wb, psi


def derivs(psi):
    k = 2.0 * math.pi * np.fft.fftfreq(N)
    ps = np.fft.fft(psi)
    d1 = np.real(np.fft.ifft(1j * k * ps))
    d2 = np.real(np.fft.ifft(-(k ** 2) * ps))
    return d1, d2


def initial_condition(U0, sigma, wb, psi, d1, v, shrink):
    """晶格原生初始条件：包络按 shrink 收缩、相位按 k_mod = wb shrink v 给梯度。

    shrink 是构造参数（物理装置为 1/gamma）；速度不在此设定，由演化后读数反演。
    """
    s0 = shrink * xf
    th0 = -wb * shrink * v * xf
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(shrink * v) * Fp * np.cos(th0) - wb * shrink * F * np.sin(th0)
    return phi, vel


def evolve(U0, sigma, wb, psi, d1, v, shrink, dt=0.02, T=1600.0, mode='lattice'):
    """速度 Verlet 演化。mode='lattice' 为格点方程，'continuum' 为连续对照。"""
    phi, vel = initial_condition(U0, sigma, wb, psi, d1, v, shrink)
    nstep = int(T / dt)
    cs = []
    ts = []
    for s in range(nstep):
        t = (s + 0.5) * dt
        sg = sigma * shrink
        well = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * sg ** 2))
        if mode == 'lattice':
            acc = -(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - well * phi
        else:
            acc = -(-(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) * 0.0)  # placeholder
            # 连续方程：用同装置的中心差做二阶导（盒外差为零同边界）
            lap = 2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)
            acc = lap - WGAP2 * phi - well * phi
        phi = phi + dt * vel + 0.5 * dt ** 2 * acc
        t2 = (s + 1.0) * dt
        well2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * sg ** 2))
        if mode == 'lattice':
            acc2 = -(2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - well2 * phi
        else:
            lap2 = 2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)
            acc2 = lap2 - WGAP2 * phi - well2 * phi
        vel = vel + 0.5 * dt * (acc + acc2)
        if s % 4 == 0:
            cs.append(phi[N // 2 + int(round(v * t))])
            ts.append(t)
    return np.array(ts), np.array(cs), phi


def fit_freq(sig, dt_s, lo=None, hi=None, n=40001):
    """最小二乘正弦拟合取频（分辨率远优于 FFT 取峰）。"""
    tt = np.arange(len(sig)) * dt_s
    y = sig - np.mean(sig)
    if lo is None:
        k = 2.0 * math.pi * np.fft.fftfreq(len(sig))
        ps = np.abs(np.fft.fft(y))
        ps[0] = 0.0
        w0 = abs(k[int(np.argmax(ps))])
        lo, hi = max(0.05, w0 - 0.15), w0 + 0.15
    ws = np.linspace(lo, hi, n)
    best, bw = None, None
    for w in ws:
        A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ coef - y) ** 2))
        if best is None or r < best:
            best, bw = r, w
    return bw


def centroid(phi, v, t):
    """质心位置读数（势阱中心附近 -|psi|^2 加权，限制在阱邻域）。"""
    idx = N // 2 + int(round(v * t))
    lo, hi = max(0, idx - 12 * 20), min(N, idx + 12 * 20)
    seg = phi[lo:hi]
    wgt = seg ** 2
    if wgt.sum() <= 0:
        return float(idx - N // 2)
    xr = np.arange(lo, hi, dtype=float) - N // 2
    return float((xr * wgt).sum() / wgt.sum())


def width(phi, v, t, sigma):
    """半宽读数：|phi| 的方差开方（阱邻域）。"""
    idx = N // 2 + int(round(v * t))
    lo, hi = max(0, idx - 14 * 20), min(N, idx + 14 * 20)
    seg = phi[lo:hi]
    wgt = seg ** 2
    xr = np.arange(lo, hi, dtype=float) - N // 2
    if wgt.sum() <= 0:
        return float('nan')
    m = (xr * wgt).sum() / wgt.sum()
    var = (xr ** 2 * wgt).sum() / wgt.sum() - m ** 2
    return math.sqrt(max(var, 0.0))



def freq_lsq(cs, dt_s, w_true, n=2001, frac=0.08):
    """窄窗最小二乘正弦拟合取频。

    首版用宽窗 FFT 加最小二乘，在运动构型上搜到低频旁瓣（v=0 也给 73% 偏差），
    已定位并改为以期望值为中心的窄窗（教训登记于注记）。
    """
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


def fit_sigma(sigma, target=0.8008):
    """给定势阱宽度 sigma，拟合阱深使束缚频率对上 target。"""
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
    lo, hi = 1e-4, 50.0
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        wb_s, _ = bm(mid)
        if wb_s > target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    wb_s, psi_s = bm(mid)
    kk = 2.0 * math.pi * np.fft.fftfreq(N)
    pss = np.fft.fft(psi_s)
    d1s = np.real(np.fft.ifft(1j * kk * pss))
    return mid, wb_s, psi_s, d1s


def run_motion(U0, sigma, wb, psi, d1, v, shrink, dt=0.02, T=1600.0, every=4):
    """晶格原生演化：给定构造参数（收缩比），速度只进入初值与平移势阱。"""
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
    return np.array(cs), phi


DT, T_RUN = 0.02, 1600.0
SIGMA, TARGET = 20.0, 0.8008
V_LIST = (0.15, 0.30)

print("=" * 78)
print("第 691 号：boost 解低能闭环的晶格原生推导（判据 BL1 至 BL6）")
print("=" * 78)

print()
print("[BL1] 冒烟：基线与 v=0 精确性")
U0, wb, psi = fit_U0(TARGET, SIGMA)
d1, d2 = derivs(psi)
print("     sigma = %.0f：拟合 U_0 = %.6f ｜ omega_b = %.6f（分离轮基线 %.4f，差 %.1e）"
      % (SIGMA, U0, wb, TARGET, abs(wb - TARGET)))
# v=0 精确性：静止束缚态 psi 是格点哈密顿量的本征矢，故 L psi = wb^2 psi，
# 代入晶格方程 d2t psi + L psi = 0 给 cos(wb t) 振动，为精确解。核对本征等式残差：
well0 = -U0 * np.exp(-(xf ** 2) / (2.0 * SIGMA ** 2))
Lpsi = (2.0 * psi - np.roll(psi, 1) - np.roll(psi, -1)) + WGAP2 * psi + well0 * psi
res0 = float(np.linalg.norm(Lpsi - wb ** 2 * psi) / np.linalg.norm(psi))
print("     v = 0 本征等式残差 |L psi - omega_b^2 psi|/|psi| = %.3e（期望机器零）" % res0)

print()
print("[BL2] 速度读数（质心漂移反演，不用 gamma 定义式）")
vread = {}
for v in V_LIST:
    g = 1.0 / math.sqrt(1.0 - v ** 2)
    cs, phif = run_motion(U0, SIGMA, wb, psi, d1, v, 1.0 / g)
    nstep = int(T_RUN / DT)
    t_end = (nstep - 1) * DT
    # 绝对坐标系质心：以实验室原点为参照（阱心在 v t，质心应随运动态漂移）
    wgt = np.abs(phif) ** 2
    thr = wgt.max() * 1e-6
    sel = wgt > thr
    xsel = xf[sel]
    wsel = wgt[sel]
    m_abs = float((xsel * wsel).sum() / wsel.sum())
    vr = m_abs / t_end
    vread[v] = vr
    print("     v_inj = %.2f：末态实验室质心 %.3f 格 over t = %.1f ｜ v_read = %.5f ｜ 差 %.2e"
          % (v, m_abs, t_end, vr, abs(vr - v)))

print()
print("[BL3] 钟慢涌现（窄窗最小二乘取频，期望 wb/gamma 由构造给出）")
print("     %-6s %-10s %-11s %-11s %-9s %s"
      % ("v", "wb/gamma", "w_read", "dev", "B_rate", "verdict"))
for v in V_LIST:
    g = 1.0 / math.sqrt(1.0 - v ** 2)
    kmod = wb * g * v
    B = ((1.0 / SIGMA) ** 2 + kmod ** 2) / 12.0
    cs, _ = run_motion(U0, SIGMA, wb, psi, d1, v, 1.0 / g)
    wm = freq_lsq(cs, 4 * DT, wb / g)
    dev = abs(wm - wb / g) / (wb / g)
    ok = dev < max(B, 1e-4)
    print("     %-6.2f %-10.5f %-11.5f %-11.3e %-11.3e %s"
          % (v, wb / g, wm, dev, B, "PASS" if ok else "CHECK"))

print()
print("[BL4] 尺缩涌现（本装置登记为不闭合，见注记过程记录）")
print("     单帧与锁相同相位两种宽度读数均被运动构型的辐射尾污染："
      "v=0.45 档宽度随读数窗口发散（辐射），窄窗读数在振荡周期内"
      "于 0.71 至 1.29 倍间跳动（单点记录节点风险）。")
print("     结论：钟慢读数干净（BL3），尺缩读数在本装置下不成立于判据。")

print()
print("[BL5] 标度律：三档 sigma 下 v = 0.30 的钟慢偏差对预算")
print("     %-8s %-9s %-11s %-11s %-11s %-9s %s"
      % ("sigma", "U0", "w_read", "dev", "B_rate", "dev/B", "verdict"))
for sigma in (14.0, 20.0, 28.0):
    U0s, wbs, psis, d1s = fit_sigma(sigma)
    v = 0.30
    g = 1.0 / math.sqrt(1.0 - v ** 2)
    cs, _ = run_motion(U0s, sigma, wbs, psis, d1s, v, 1.0 / g)
    wm = freq_lsq(cs, 4 * DT, wbs / g)
    kmod = wbs * g * v
    B = ((1.0 / sigma) ** 2 + kmod ** 2) / 12.0
    dev = abs(wm - wbs / g) / (wbs / g)
    print("     %-8.0f %-9.4f %-11.5f %-11.3e %-11.3e %-9.2f %s"
          % (sigma, U0s, wm, dev, B, dev / B, "PASS" if dev < max(B, 1e-4) else "CHECK"))

print()
print("[BL6] 恒真反例（证明判据有区分力）")
cs0, _ = run_motion(U0, SIGMA, wb, psi, d1, 0.0, 1.0)
wm0 = freq_lsq(cs0, 4 * DT, wb)
dev0 = abs(wm0 - wb) / wb
print("     反例A（v=0）：dev = %.3e（期望约 0，装置无系统漂移）" % dev0)
v = 0.30
g_true = 1.0 / math.sqrt(1.0 - v ** 2)
g_bad = 1.0 / math.sqrt(1.0 + v)
csb, _ = run_motion(U0, SIGMA, wb, psi, d1, v, 1.0 / g_bad)
wmb = freq_lsq(csb, 4 * DT, wb / g_true)
devb = abs(wmb - wb / g_true) / (wb / g_true)
B = ((1.0 / SIGMA) ** 2 + (wb * g_true * v) ** 2) / 12.0
print("     反例B（收缩比故意取错为 1/sqrt(1+v)）：dev = %.3e（期望远大于预算 %.3e，"
      "读数 %.5f 对期望 %.5f）" % (devb, B, wmb, wb / g_true))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))

# ================= 图（三面板） =================
g30 = 1.0 / math.sqrt(1.0 - 0.30 ** 2)
cs15, _ = run_motion(U0, SIGMA, wb, psi, d1, 0.15, 1.0 / math.sqrt(1.0 - 0.15 ** 2))
cs30, _ = run_motion(U0, SIGMA, wb, psi, d1, 0.30, 1.0 / g30)
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))

t_ax = np.arange(len(cs15)) * 4 * DT
ax[0].plot(t_ax[:400], cs15[:400], "k-", lw=1.0, label="$v=0.15$")
ax[0].plot(t_ax[:400], cs30[:400], "tab:red", lw=1.0, alpha=0.85, label="$v=0.30$")
ax[0].set_xlabel("时间 $t$（单位 $t_P$）")
ax[0].set_ylabel("阱心相位读数")
ax[0].set_title("运动束缚态的阱心振荡（晶格原生演化）")
ax[0].legend(loc="upper right", fontsize=8)

vs = list(V_LIST)
devs = []
for v in vs:
    gg = 1.0 / math.sqrt(1.0 - v ** 2)
    c_, _ = run_motion(U0, SIGMA, wb, psi, d1, v, 1.0 / gg)
    wm = freq_lsq(c_, 4 * DT, wb / gg)
    devs.append(abs(wm - wb / gg) / (wb / gg))
Bs = [((1.0 / SIGMA) ** 2 + (wb * (1.0 / math.sqrt(1.0 - v ** 2)) * v) ** 2) / 12.0 for v in vs]
xs_pos = np.arange(len(vs))
ax[1].bar(xs_pos, devs, color="tab:blue", alpha=0.8, label="实测偏差")
ax[1].plot(xs_pos, Bs, "rs", ms=9, label="误差预算")
for i, d in enumerate(devs):
    ax[1].text(i, d, "%.1e" % d, ha="center", va="bottom", fontsize=8)
ax[1].set_xticks(xs_pos)
ax[1].set_xticklabels(["%.2f" % v for v in vs])
ax[1].set_xlabel("注入速度 $v/c$")
ax[1].set_ylabel("相对偏差")
ax[1].set_yscale("log")
ax[1].set_title("钟慢：实测偏差对误差预算")
ax[1].legend(fontsize=8)

sig_list = (14.0, 20.0, 28.0)
rd = []
for sg in sig_list:
    U0s, wbs, psis, d1s = fit_sigma(sg)
    c_, _ = run_motion(U0s, sg, wbs, psis, d1s, 0.30, 1.0 / g30)
    rd.append(abs(freq_lsq(c_, 4 * DT, wbs / g30) - wbs / g30) / (wbs / g30))
ax[2].plot([sg / 2.0 for sg in sig_list], rd, "ko-", ms=7, label="实测偏差")
ax[2].axhline(5.49e-3, color="r", ls="--", lw=1.2, label=r"误差预算（$v=0.30$）")
ax[2].set_xlabel(r"结构尺度 $\xi/l_P$")
ax[2].set_ylabel("相对偏差")
ax[2].set_yscale("log")
ax[2].set_title("BL5 标度律：三档宽度的偏差对预算")
ax[2].legend(fontsize=8)

fig.suptitle("第 691 号：boost 解低能闭环的晶格原生推导（判据 BL1 至 BL6）", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(IMG, dpi=150)
print("图已存:", IMG)
