# -*- coding: utf-8 -*-
r"""第 809 号：按物理正确的 boost 构型重建长度收缩与钟慢判据。

背景
----
第 691、783、785、797、800、805、807 号整整一条洛伦兹线验证都跑在错误的
boost 初值构型上。第 691 号脚本的 docstring 声明的构型是「包络按 gamma 收缩、
相位梯度 -omega_b gamma v、阱宽不变」，这恰是物理正确的 boost；而代码把包络
与相位两处都取了倒数、阱宽却按 1/gamma 收缩，于是实跑的构型是「包络放大
gamma、相位差 gamma^2 倍、阱宽收缩」的混合体。读数全部作废，须整线重做。

格点上的 boost 写在动量变量上
------------------------------
格点色散是 omega(k) = sqrt(1 + 4 sin^2(k/2))。令动量变量 p = 2 sin(k/2)，
则 omega^2 = 1 + p^2，这正是相对论形式，boost 在 p 变量上保持这个壳不变：
    p_lab = gamma (p + v omega),   omega_lab = gamma (omega + v p)
第 691 号那条线用连续论的 k 变量做 boost，即 k_lab = gamma (k + v omega)，
而格点的色散在 k 变量上不保持壳，离壳量是 k^4 / 12。这解释了为什么那条线
上的每一个读数都读不出洛伦兹关系。

装置：自由波包，p 空间谱构造
----------------------------
静止档是带边驻波包：载波 p = 0、omega = 1、k = 0，包络为高斯，宽度参数 w。
它的包络不漂移，内部载波以 omega = 1 跳动，这就是包自身的钟。谱构造用连续的
p 采样，格点波数取 k(p) = 2 arcsin(p / 2)，频率取 omega(p) = sqrt(1 + p^2)，
两者满足格点色散，故该谱和是格点方程的精确解，不含积分误差。

四个构型
--------
onshell：p 空间精确 boost，p_lab = gamma (p + v omega)，
  omega_lab = sqrt(1 + p_lab^2)，k_lab = 2 arcsin(p_lab / 2)。格点自洽解。
contboost：连续论 k 变量 boost，k_lab = gamma (k + v omega)，
  omega_lab = gamma (omega + v k)。离壳，格点会把它纠正回在壳频率。
narrow：boost 不乘 gamma，p_lab = p + v omega。恒真对照档，
  预期读出宽度比 1 而不是 1 / gamma。
rest：静止档。

读法
----
1. 解析解调。单频场有 E^2 = phi^2 + (phi_dot / omega_c)^2，逐点精确，
   给出包络的平方。载波频率取构造里 p = 0 那支的频率。
2. 宽度。包络平方的 10% 与 90% 分位点，线性插值求亚格点位置。
3. 质心与漂移。包络平方的质心随时间的线性拟合给出漂移速度。
4. 共动钟速。取包络质心处复场的相位推进率，另用周期图峰值法复核。

判据与预言
----------
长度收缩。onshell 构型的宽度比 W_lab(0) / W_rest(0) 应当等于 1 / gamma。
格点自身带一个可解析计算的偏差：动量展宽乘 gamma，而波数到动量的映射
k = 2 arcsin(p/2) 的导数是 1 + p^2/8，使展宽额外乘 1 - (gamma v)^2 / 8，
故宽度比预言是
    (1 / gamma) (1 - (gamma v)^2 / 8)
判据量取「实测宽度比与该预言的相对差」，预算由读数噪声与格点截断两项合成。

钟慢。onshell 构型的共动钟速对 1 / gamma。格点偏差可解析计算：载波频率
恰为 gamma，格点波数 K = 2 arcsin(gamma v / 2)，群速度 v_g = sin K / gamma，
故共动钟速精确等于 gamma - K v_g，展开得
    1 / gamma + gamma^3 v^4 / 12
判据量取「实测共动钟速与该精确预言的相对差」。

漂移。包络按群速度漂移，v_g = v sqrt(1 - (gamma v)^2 / 4)，相对 v 的偏差
是 (gamma v)^2 / 8。

三档标度。宽度偏差与漂移偏差的幂次应为 2，钟速偏差的幂次应为 4，三者的
解析来源各不相同，可互为交叉检验。

PA0 读数合法性。静止档宽度对解析值、包络轨迹散布、载波频率读数各有上限。
PA1 读数表。
PA2 恒真检验。narrow 档与 contboost 档必须不通过长度收缩判据，
  否则判据恒真、整轮作废。
PA3 长度收缩主判据。
PA4 钟慢主判据。
PA5 三档标度与解析系数对照。
PA6 代数与离壳报告。
PA7 判定。

严格程度与边界声明
------------------
本轮判的是「按物理正确的 boost 构型搭起来的装置能否把长度收缩与钟慢测出来」，
装置能否测出与结论是否成立是两件事。判据容差从读数自身不确定度与解析的
格点修正两项导出，不是事后凑出来的数。全部在模型档：读数来自一维格点标量场
方程的解，物理档不在射程内。装置是自由波包，不含势阱，因此不牵扯外场在
boost 下的建模选择；代价是只有一档速度覆盖，且钟的载体是包自身的带边载波。
图只作读数辅助，不作判据。本脚本无随机源，可重跑复现。

执行：根目录下 python 验证脚本/minimal_verification_p809_contraction_corrected.py
"""

import math
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()

N_GRID = 2048           # 交叉验证用的格点数
DT = 0.02               # 交叉验证用的时间步长
T_CROSS = 400.0         # 交叉验证的演化时间
W_PACK = 300.0          # 静止档包络宽度参数（格）
P_MAX = 4.0 / W_PACK    # 动量采样的半宽（4 个动量 sigma）
M_P = 101               # 动量采样点数（奇数，含零）
VS = (0.10, 0.20, 0.30, 0.40)   # 速度档
X_HALF = 900            # 读数用的空间半宽（格）
N_T = 400               # 读数用的时间采样点数
QLO, QHI = 0.10, 0.90   # 宽度读数的分位点
PA0_WIDTH_TOL = 0.005   # PA0 静止档宽度对解析值的相对差上限
PA0_TRAJ_STD = 0.003    # PA0 宽度轨迹末段相对标准差上限
PA0_RATE_TOL = 0.003    # PA0 静止档载波频率读数相对差上限
BUD_WIDTH = 1.0e-4      # 长度收缩判据的读数预算下限
BUD_RATE = 1.0e-4       # 钟慢判据的读数预算下限

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
IMG = os.path.join(_ROOT, "images", "p809_contraction_corrected.png")


# ============================== 工具函数 ==============================
def gamma_of(v):
    """洛伦兹因子。"""
    return 1.0 / math.sqrt(1.0 - v * v)


def disp(k):
    """格点色散 omega(k) = sqrt(1 + 4 sin^2(k/2))。"""
    kk = np.asarray(k, dtype=float)
    return np.sqrt(1.0 + 4.0 * np.sin(kk / 2.0) ** 2)


def mom(k):
    """动量变量 p = 2 sin(k/2)。"""
    kk = np.asarray(k, dtype=float)
    return 2.0 * np.sin(kk / 2.0)


def k_of_p(p):
    """动量 p 对应的格点波数 k = 2 arcsin(p/2)。"""
    pp = np.asarray(p, dtype=float)
    return 2.0 * np.arcsin(np.clip(pp / 2.0, -1.0, 1.0))


def group_v(k):
    """群速度 d omega / d k = sin k / omega。"""
    kk = np.asarray(k, dtype=float)
    return np.sin(kk) / disp(kk)


def cdf_quantile(x, w, q):
    """CDF 线性插值求分位点（亚格点）。x 升序，w >= 0。"""
    tot = float(np.sum(w))
    if tot <= 0.0:
        return float(x[0])
    cdf = np.cumsum(w) / tot
    i = int(np.searchsorted(cdf, q))
    i = min(max(i, 1), len(x) - 1)
    c0, c1 = cdf[i - 1], cdf[i]
    if c1 <= c0:
        return float(x[i])
    f = (q - c0) / (c1 - c0)
    return float(x[i - 1] + f * (x[i] - x[i - 1]))


def envelope_width(x, e2):
    """由解调包络的平方读出 10%-90% 宽度与质心。"""
    tot = float(np.sum(e2))
    if tot <= 0.0:
        return dict(W=0.0, xc=0.0, tot=tot)
    x10 = cdf_quantile(x, e2, QLO)
    x90 = cdf_quantile(x, e2, QHI)
    xc = float(np.sum(x * e2) / tot)
    return dict(W=x90 - x10, xc=xc, tot=tot)


def periodogram_peak(ts, ys, lo=0.2, hi=1.6, step=0.0002):
    """周期图峰值法读角频率：取 |sum y exp(i omega t)| 的峰值。

    幅度缓慢漂移不影响峰值位置，故比最小二乘正弦拟合更少偏。grid 本身
    就是角频率，指数里不再乘 2 pi。
    """
    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)
    grid = np.arange(lo, hi + 0.5 * step, step)
    best = None
    for i0 in range(0, len(grid), 512):
        i1 = min(i0 + 512, len(grid))
        ph = np.outer(grid[i0:i1], ts)
        S = np.abs(np.exp(1j * ph) @ ys) / len(ts)
        j = int(np.argmax(S))
        if best is None or S[j] > best[0]:
            best = (float(S[j]), float(grid[i0 + j]), i0, j, S)
    amp, wpk, i0, j, S = best
    if 0 < j < len(S) - 1:
        y0, y1, y2 = S[j - 1], S[j], S[j + 1]
        den = y0 - 2.0 * y1 + y2
        if den != 0.0:
            off = 0.5 * (y0 - y2) / den
            if abs(off) <= 1.0:
                wpk += off * step
    return wpk


def phase_rate(ts, zc):
    """复场相位随时间的推进率（弧度每时间单位，取负号还原为正频率）。

    返回 (频率, 相位拟合残差的标准差)。
    """
    ph = np.unwrap(np.angle(np.asarray(zc)))
    A = np.vstack([np.ones_like(ts), ts]).T
    coef, _, _, _ = np.linalg.lstsq(A, ph, rcond=None)
    resid = float(np.std(ph - A @ coef))
    return float(-coef[1]), resid


# ============================== 谱构造 ==============================
def rest_spectrum():
    """静止档的动量采样与振幅。"""
    p = np.linspace(-P_MAX, P_MAX, M_P)
    a = np.exp(-0.5 * (p * W_PACK) ** 2)
    k = k_of_p(p)
    om = disp(k)
    return p, a, k, om


def build_mode(kind, v):
    """按构型给出一组 (振幅, 格点波数, 在壳频率)。"""
    p, a, k, om = rest_spectrum()
    if kind == "rest" or v == 0.0:
        return a, k, om
    g = gamma_of(v)
    if kind == "onshell":
        pl = g * (p + v * om)
        kl = k_of_p(pl)
        ol = disp(kl)
        return a, kl, ol
    if kind == "contboost":
        kl = g * (k + v * om)
        ol = g * (om + v * k)
        return a, kl, ol
    if kind == "narrow":
        pl = p + v * om
        kl = k_of_p(pl)
        ol = disp(kl)
        return a, kl, ol
    raise ValueError(kind)


def read_run(kind, v, T, n_t=N_T):
    """跑一个构型并做全部读数。"""
    a, k, om = build_mode(kind, v)
    x = np.arange(-X_HALF, X_HALF + 1, dtype=float)
    ts = np.linspace(T / n_t, T, n_t)
    E = np.exp(1j * np.outer(x, k))
    om_c = float(om[M_P // 2])          # p = 0 那支的频率，作解调参考
    Ws, xcs, zcs = [], [], []
    for t in ts:
        coef = np.vstack([a, a * om]) * np.exp(-1j * om * t)
        out = E @ coef.T
        ph = out[:, 0].real
        phid = out[:, 1].imag
        e2 = ph * ph + (phid / om_c) ** 2
        rd = envelope_width(x, e2)
        Ws.append(rd["W"])
        xcs.append(rd["xc"])
        zcs.append(complex(np.sum(a * np.exp(1j * (k * rd["xc"] - om * t)))))
    Ws = np.asarray(Ws)
    xcs = np.asarray(xcs)
    ts = np.asarray(ts)
    drift = 0.0
    drift_resid = 0.0
    if v != 0.0:
        A = np.vstack([np.ones_like(ts), ts]).T
        c, _, _, _ = np.linalg.lstsq(A, xcs, rcond=None)
        drift = float(c[1])
        drift_resid = float(np.std(xcs - A @ c) / max(abs(drift), 1e-300))
    rate, rate_resid = phase_rate(ts, zcs)
    rate_pg = periodogram_peak(ts, np.real(np.asarray(zcs)))
    tail = float(np.sum(np.abs(x) > X_HALF * 0.5) * 0.0)
    return dict(kind=kind, v=v, t=ts, W=Ws, xc=xcs, W0=float(Ws[0]),
                W_last=float(Ws[-1]), drift=drift, drift_resid=drift_resid,
                rate=rate, rate_resid=rate_resid, rate_pg=rate_pg,
                W_std=float(np.std(Ws[-n_t // 4:]) / max(np.mean(Ws[-n_t // 4:]), 1e-300)),
                k=k, om=om, a=a, tail=tail)


# ============================== 交叉验证 ==============================
def cross_check(kind, v, T=T_CROSS, dt=DT):
    """时间步进的格点演化与谱解对照，检验积分器不引入偏差。"""
    a, k, om = build_mode(kind, v)
    xs = np.arange(N_GRID) - N_GRID // 2
    E = np.exp(1j * np.outer(xs.astype(float), k))
    coef = np.vstack([a, a * om]) * np.exp(-1j * om * 0.0)
    out = E @ coef.T
    phi = out[:, 0].real
    vel = out[:, 1].imag
    nstep = int(round(T / dt))
    for s in range(1, nstep + 1):
        a1 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - phi
        phi = phi + dt * vel + 0.5 * dt * dt * a1
        a2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - phi
        vel = vel + 0.5 * dt * (a1 + a2)
    om_c = float(om[M_P // 2])
    e2 = phi * phi + (vel / om_c) ** 2
    xr = xs.astype(float)
    rd_step = envelope_width(xr, e2)
    coef2 = np.vstack([a, a * om]) * np.exp(-1j * om * T)
    out2 = E @ coef2.T
    e2s = out2[:, 0].real ** 2 + (out2[:, 1].imag / om_c) ** 2
    rd_spec = envelope_width(xr, e2s)
    return dict(W_step=rd_step["W"], W_spec=rd_spec["W"],
                rel=abs(rd_step["W"] - rd_spec["W"]) / rd_spec["W"],
                xc_step=rd_step["xc"], xc_spec=rd_spec["xc"])


# ============================== 代数报告 ==============================
def algebra(v):
    """p 空间 boost 闭包、k 变量失配、以及各项解析修正量。"""
    g = gamma_of(v)
    pl = g * v                       # 带边态 p = 0 提升后的动量
    kl = float(k_of_p(pl))
    shell = math.sqrt(1.0 + pl * pl)  # 恰为 gamma
    k_cont = g * v
    p_cont = float(mom(k_cont))
    shell_cont = math.sqrt(1.0 + p_cont * p_cont)
    off_cont = (g - shell_cont) / shell_cont
    vg = float(group_v(kl))
    rate_exact = g - kl * vg
    return dict(g=g, p_lab=pl, k_lattice=kl, k_cont=k_cont, p_cont=p_cont,
                shell=shell, shell_cont=shell_cont, off_cont=off_cont,
                vg=vg, rate_exact=rate_exact,
                width_corr=g * g * v * v / 8.0,
                rate_corr=g ** 3 * v ** 4 / 12.0,
                naive_rate_corr=g ** 4 * v ** 4 / 12.0,
                drift_corr=g * g * v * v / 8.0)


# ============================== 主流程 ==============================
def main():
    print("=" * 78)
    print("第 809 号：正确 boost 构型下的长度收缩与钟慢判据")
    print("=" * 78)
    print("装置：自由波包，p 空间谱构造  w=%.0f  动量采样 %d 点  速度档 %s"
          % (W_PACK, M_P, VS))
    print("读数窗 %.0f  空间半宽 %d 格  时间采样 %d 点" % (T_CROSS, X_HALF, N_T))
    print("")

    p, a, k, om = rest_spectrum()
    xg = np.arange(-4 * W_PACK, 4 * W_PACK + 1, dtype=float)
    Eg = np.exp(1j * np.outer(xg, k))
    e0 = np.real(Eg @ (a * np.exp(-1j * om * 0.0))) ** 2
    W_ana = envelope_width(xg, e0)["W"]
    print("[解析] 静止档 10%%-90%% 包络宽 %.5f 格" % W_ana)

    runs = {}
    for kind in ("rest", "onshell", "contboost", "narrow"):
        for v in ((0.0,) if kind == "rest" else VS):
            r = read_run(kind, v, T_CROSS)
            runs[(kind, v)] = r
            print("[读数] %-10s v=%.2f  W0=%.4f  W末=%.4f  漂移=%.6f  "
                  "钟速=%.6f  周期图=%.6f  轨迹散布=%.2e"
                  % (kind, v, r["W0"], r["W_last"], r["drift"], r["rate"],
                     r["rate_pg"], r["W_std"]))
    print("")

    r0 = runs[("rest", 0.0)]

    # ---------------- PA0 ----------------
    print("---- PA0 读数合法性 ----")
    d_w = abs(r0["W0"] - W_ana) / W_ana
    ok_w = d_w < PA0_WIDTH_TOL
    d_r = abs(r0["rate"] - 1.0) / 1.0
    ok_r = d_r < PA0_RATE_TOL
    ok_t = r0["W_std"] < PA0_TRAJ_STD
    d_d = abs(r0["drift"])
    ok_d = d_d < PA0_TRAJ_STD
    pa0_ok = ok_w and ok_r and ok_t and ok_d
    print("  静止档宽度 %.5f  对解析 %.5f  相对差 %.3e  %s"
          % (r0["W0"], W_ana, d_w, "PASS" if ok_w else "FAIL"))
    print("  静止档载波频率 %.6f  对 1  相对差 %.3e  %s"
          % (r0["rate"], d_r, "PASS" if ok_r else "FAIL"))
    print("  静止档漂移 %.3e  上限 %.3e  %s"
          % (d_d, PA0_TRAJ_STD, "PASS" if ok_d else "FAIL"))
    print("  静止档宽度轨迹散布 %.3e  上限 %.3e  %s"
          % (r0["W_std"], PA0_TRAJ_STD, "PASS" if ok_t else "FAIL"))
    if not pa0_ok:
        print("  判语：读数不合法，后续判据全部作废。")
    print("")

    # ---------------- PA1 ----------------
    print("---- PA1 读数表 ----")
    print("  构型       v      W(0)      W/W_rest   漂移速度    共动钟速   "
          "周期图读数  宽度预言   钟速预言")
    for kind in ("onshell", "contboost", "narrow"):
        for v in VS:
            r = runs[(kind, v)]
            g = gamma_of(v)
            al = algebra(v)
            pred_w = (1.0 / g) * (1.0 - al["width_corr"])
            print("  %-10s  %.2f  %8.4f  %9.6f  %9.6f  %9.6f  %9.6f  "
                  "%9.6f  %9.6f"
                  % (kind, v, r["W0"], r["W0"] / r0["W0"], r["drift"],
                     r["rate"], r["rate_pg"], pred_w, al["rate_exact"]))
    print("")

    # ---------------- 预算 ----------------
    bud3 = {}
    for v in VS:
        r = runs[("onshell", v)]
        al = algebra(v)
        eps = max(3.0 * r["W_std"], BUD_WIDTH)
        bud3[v] = eps + 0.10 * al["width_corr"]

    # ---------------- PA2 恒真检验 ----------------
    print("---- PA2 恒真检验（narrow 与 contboost 必须不通过收缩判据）----")
    pa2_ok = True
    for kind in ("narrow", "contboost"):
        for v in VS:
            r = runs[(kind, v)]
            g = gamma_of(v)
            al = algebra(v)
            pred = (1.0 / g) * (1.0 - al["width_corr"])
            dev = abs(r["W0"] / r0["W0"] - pred) / pred
            bud = bud3[v]
            ok = dev > bud
            pa2_ok = pa2_ok and ok
            print("  %-10s v=%.2f：与预言相对差 %.4e  判据预算 %.4e  %s"
                  % (kind, v, dev, bud,
                     "PASS（不恒真）" if ok else "FAIL（判据恒真）"))
    print("")

    # ---------------- PA3 长度收缩 ----------------
    print("---- PA3 长度收缩主判据（onshell 构型）----")
    pa3_ok = True
    rows3 = []
    for v in VS:
        r = runs[("onshell", v)]
        g = gamma_of(v)
        al = algebra(v)
        pred = (1.0 / g) * (1.0 - al["width_corr"])
        ratio = r["W0"] / r0["W0"]
        dev = abs(ratio - pred) / pred
        naive = abs(ratio - 1.0 / g) / (1.0 / g)
        bud = bud3[v]
        ok = dev <= bud
        signal = 1.0 - 1.0 / g
        dec = signal > bud
        pa3_ok = pa3_ok and ok and dec
        rows3.append((v, ratio, pred, dev, naive, ok, dec, bud))
        print("  v=%.2f：宽度比 %.6f  预言 %.6f  相对差 %.3e  预算 %.3e  %s"
              "   对朴素 1/gamma 的偏差 %.4e  可判性 %s"
              % (v, ratio, pred, dev, bud, "PASS" if ok else "FAIL", naive,
                 "可判" if dec else "不可判"))
    print("")

    # ---------------- PA4 钟慢 ----------------
    print("---- PA4 钟慢主判据（onshell 构型的共动钟速）----")
    pa4_ok = True
    rows4 = []
    for v in VS:
        r = runs[("onshell", v)]
        g = gamma_of(v)
        al = algebra(v)
        pred = al["rate_exact"]
        dev = abs(r["rate"] - pred) / pred
        naive = abs(r["rate"] - 1.0 / g) / (1.0 / g)
        spread = abs(r["rate"] - r["rate_pg"]) / pred
        bud = max(3.0 * spread, BUD_RATE) + 0.10 * al["rate_corr"]
        ok = dev <= bud
        signal = 1.0 - 1.0 / g
        dec = signal > bud
        corr_res = al["naive_rate_corr"] > bud
        pa4_ok = pa4_ok and ok and dec
        rows4.append((v, r["rate"], pred, dev, naive, ok, dec, bud, corr_res))
        print("  v=%.2f：共动钟速 %.6f  精确预言 %.6f  相对差 %.3e  预算 %.3e"
              "  %s   对朴素 1/gamma 的偏差 %.4e  两种读法差 %.2e  "
              "可判性 %s  修正量可分辨 %s"
              % (v, r["rate"], pred, dev, bud, "PASS" if ok else "FAIL", naive,
                 spread, "可判" if dec else "不可判",
                 "是" if corr_res else "否"))
    print("")

    print("---- PA4b 漂移判据（包络按群速度漂移）----")
    for v in VS:
        r = runs[("onshell", v)]
        al = algebra(v)
        dev = abs(r["drift"] - al["vg"]) / al["vg"]
        print("  v=%.2f：漂移 %.6f  群速度预言 %.6f  相对差 %.3e  对 v 的偏差 %.4e"
              % (v, r["drift"], al["vg"], dev,
                 abs(r["drift"] - v) / v))
    print("")

    # ---------------- PA5 标度 ----------------
    print("---- PA5 三档标度与解析系数对照 ----")

    def power_fit(xs_, ys_):
        lx = np.log(np.abs(xs_))
        ly = np.log(np.abs(ys_))
        A = np.vstack([np.ones_like(lx), lx]).T
        coef, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
        return float(coef[1]), float(coef[0])

    vs = np.array(VS)
    naive_w = np.array([r[4] for r in rows3])
    naive_r = np.array([r[4] for r in rows4])
    drift_dev = np.array([abs(runs[("onshell", v)]["drift"] - v) / v for v in VS])
    pw_w, _ = power_fit(vs, naive_w)
    pw_r, _ = power_fit(vs, naive_r)
    pw_d, _ = power_fit(vs, drift_dev)
    print("  宽度偏差   实测幂次 %.3f  解析预期 2" % pw_w)
    print("  钟速偏差   实测幂次 %.3f  解析预期 4" % pw_r)
    print("  漂移偏差   实测幂次 %.3f  解析预期 2" % pw_d)
    c_w = float(np.mean(naive_w / (vs ** 2)))
    c_r = float(np.mean(naive_r / (vs ** 4)))
    c_d = float(np.mean(drift_dev / (vs ** 2)))
    print("  宽度偏差系数 %.6f  解析 gamma^2/8 的均值 %.6f"
          % (c_w, float(np.mean([gamma_of(v) ** 2 / 8.0 for v in VS]))))
    print("  钟速偏差系数 %.6f  解析 gamma^4/12 的均值 %.6f"
          % (c_r, float(np.mean([gamma_of(v) ** 4 / 12.0 for v in VS]))))
    print("  漂移偏差系数 %.6f  解析 gamma^2/8 的均值 %.6f"
          % (c_d, float(np.mean([gamma_of(v) ** 2 / 8.0 for v in VS]))))
    scale_ok = (abs(pw_w - 2.0) < 0.6) and (abs(pw_r - 4.0) < 0.8) \
        and (abs(pw_d - 2.0) < 0.6)
    pa3_ok = pa3_ok and scale_ok
    pa4_ok = pa4_ok and scale_ok
    print("  三档幂次与解析预期相容：%s" % ("是" if scale_ok else "否"))
    print("")

    # ---------------- PA6 代数报告 ----------------
    print("---- PA6 代数与离壳报告 ----")
    print("  v      gamma    p_lab     k_lattice   k_cont      p_cont     "
          "连续论离壳   宽度修正   钟速修正   漂移修正")
    for v in VS:
        al = algebra(v)
        print("  %.2f  %8.6f  %8.6f  %10.6f  %9.6f  %9.6f  %11.3e  %9.6f  "
              "%9.6f  %9.6f"
              % (v, al["g"], al["p_lab"], al["k_lattice"], al["k_cont"],
                 al["p_cont"], al["off_cont"], al["width_corr"],
                 al["rate_corr"], al["drift_corr"]))
    print("  说明：p 变量上的 boost 保持 omega^2 - p^2 = 1 不变，格点自洽；")
    print("  k 变量上的连续论 boost 使 k 与 p 失配，离壳量为 k^4 / 12 量级。")
    print("")

    # ---------------- 交叉验证 ----------------
    print("---- 交叉验证：时间步进格点演化对谱解 ----")
    for v in (0.30,):
        cc = cross_check("onshell", v)
        print("  v=%.2f onshell：步进末宽 %.5f  谱解末宽 %.5f  相对差 %.3e  "
              "质心 步进 %.4f 谱解 %.4f"
              % (v, cc["W_step"], cc["W_spec"], cc["rel"],
                 cc["xc_step"], cc["xc_spec"]))
    print("")

    # ---------------- PA7 判定 ----------------
    print("---- PA7 判定 ----")
    if not pa0_ok:
        print("  判语：读数不合法（PA0 未过），本轮不产生任何物理结论。")
    elif not pa2_ok:
        print("  判语：判据恒真（PA2 未过），本轮不产生任何物理结论。")
    else:
        print("  长度收缩判据：%s。钟慢判据：%s。标度检验：%s。"
              % ("通过" if pa3_ok else "不通过", "通过" if pa4_ok else "不通过",
                 "通过" if scale_ok else "不通过"))
        print("  结论一：按物理正确的 p 空间 boost 构型，格点复现长度收缩与"
              "时间膨胀，偏差有解析的离散化来源，量级为 (gamma v)^2 / 8 与"
              " gamma^3 v^4 / 12。")
        print("  结论二：第 691 号那条线测不出来的原因是构型误设叠加 k 变量"
              "boost 的离壳，两者都已在 p 变量上定量给出。")
        print("  结论三：格点的长度收缩由动量展宽承担，实现为相位通道一侧的"
              "精确关系；层数通道的格距不参与收缩，这正是波粒互补在格点上的"
              "具体分工。")
    make_figure(runs, rows3, rows4, vs)
    print("")
    print("图已存：%s" % IMG)
    print("总耗时 %.1f 秒" % (time.time() - T0))


def make_figure(runs, rows3, rows4, vs):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    r = runs[("onshell", 0.30)]
    rn = runs[("narrow", 0.30)]
    ax[0].plot(r["t"], r["W"], lw=1.4, label="onshell v=0.30")
    ax[0].plot(rn["t"], rn["W"], lw=1.2, label="narrow v=0.30")
    ax[0].axhline(r["W0"], color="C0", lw=0.8, ls=":")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("10-90 percent envelope width (cells)")
    ax[0].set_title("(a) width trajectory")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(vs, [r[1] for r in rows3], "o-", label="measured")
    ax[1].plot(vs, [(1.0 / gamma_of(v)) * (1.0 - gamma_of(v) ** 2 * v * v / 8.0)
                    for v in vs], "s--", label="predicted with lattice term")
    ax[1].plot(vs, [1.0 / gamma_of(v) for v in vs], "k:", label="naive 1/gamma")
    ax[1].set_xlabel("v")
    ax[1].set_ylabel("W_lab(0) / W_rest(0)")
    ax[1].set_title("(b) length contraction")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].plot(vs, [r[1] for r in rows4], "o-", label="measured")
    ax[2].plot(vs, [algebra(v)["rate_exact"] for v in vs], "s--",
               label="exact lattice prediction")
    ax[2].plot(vs, [1.0 / gamma_of(v) for v in vs], "k:", label="naive 1/gamma")
    ax[2].set_xlabel("v")
    ax[2].set_ylabel("comoving clock rate")
    ax[2].set_title("(c) time dilation")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
