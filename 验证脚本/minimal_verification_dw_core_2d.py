# -*- coding: utf-8 -*-
"""
第 618 号：二维芯结构模型立案轮（弹性本构主路线恢复入口）
方案甲：直壁错排带（环面，第 486 条机制纯平移基准档）
方案乙：壁段端点芯（开边界半无限壁，实 sigma 壁段回卷 = 乙1）
判据：W1-W5（甲 + 对照）、P1-P4（乙）
判据设计：D 立案轮（存在性与可读性）；E 审计轮另立。

模型（甲/乙同泛函）：
  E = sum[ kapp/2 |grad sigma|^2 + lam (sigma^2-1)^2 ]
      + (k/2) sum_{mu,a} (D_mu s^a)^2
      - J sum_{a<b} cos(2pi (v^a - v^b)),  v^a = s^a + Q_a * aa
  aa = (1 - sigma)/2（定义性联结：sigma 定义相位零点，单向 sigma -> aa -> v）
  Q = (1, 0, 0)（错排分配：分量 1 带单位错排，598 条 alpha 分配口径）

弛豫：梯度流（步长 eps 从 Hessian 上界推导），周期边界（甲）/ Neumann 开边界（乙）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images"
DPI = 150
TWO_PI = 2.0 * np.pi
Q = np.array([1.0, 0.0, 0.0])

kappa_def, lam, k_el, J = 1.0, 1.0, 1.0, 1.0


def lap_periodic(f):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4.0 * f)


def lap_open(f):
    if f.ndim == 2:
        fp = np.pad(f, 1, mode="edge")
        return (fp[2:, 1:-1] + fp[:-2, 1:-1]
                + fp[1:-1, 2:] + fp[1:-1, :-2] - 4.0 * f)
    fp = np.pad(f, [(0, 0), (1, 1), (1, 1)], mode="edge")
    return (fp[:, 2:, 1:-1] + fp[:, :-2, 1:-1]
            + fp[:, 1:-1, 2:] + fp[:, 1:-1, :-2] - 4.0 * f)


def energy(sigma, s, kappa, periodic, Qv=Q):
    if periodic:
        dsx = np.roll(s, -1, 2) - s
        dsy = np.roll(s, -1, 1) - s
        grad2_s = (dsx ** 2).sum(0) + (dsy ** 2).sum(0)
        dsig_x = np.roll(sigma, -1, 1) - sigma
        dsig_y = np.roll(sigma, -1, 0) - sigma
    else:
        dsx = np.zeros_like(s); dsy = np.zeros_like(s)
        dsx[:, :, :-1] = s[:, :, 1:] - s[:, :, :-1]
        dsy[:, :-1, :] = s[:, 1:, :] - s[:, :-1, :]
        grad2_s = (dsx ** 2).sum(0) + (dsy ** 2).sum(0)
        dsig_x = np.zeros_like(sigma); dsig_y = np.zeros_like(sigma)
        dsig_x[:, :-1] = sigma[:, 1:] - sigma[:, :-1]
        dsig_y[:-1, :] = sigma[1:, :] - sigma[:-1, :]
    aa = (1.0 - sigma) / 2.0
    v = s + Qv[:, None, None] * aa[None, :, :]
    dv12 = v[0] - v[1]; dv13 = v[0] - v[2]; dv23 = v[1] - v[2]
    E_wall = (kappa / 2.0) * (dsig_x ** 2 + dsig_y ** 2).sum() \
        + lam * ((sigma ** 2 - 1.0) ** 2).sum()
    E_s = (k_el / 2.0) * grad2_s.sum()
    E_J = -J * (np.cos(TWO_PI * dv12) + np.cos(TWO_PI * dv13)
                + np.cos(TWO_PI * dv23)).sum()
    return E_wall + E_s + E_J


def gradients(sigma, s, kappa, periodic, Qv=Q):
    lap = lap_periodic if periodic else lap_open
    aa = (1.0 - sigma) / 2.0
    v = s + Qv[:, None, None] * aa[None, :, :]
    dv12 = v[0] - v[1]; dv13 = v[0] - v[2]; dv23 = v[1] - v[2]
    g0 = TWO_PI * J * (np.sin(TWO_PI * dv12) + np.sin(TWO_PI * dv13))
    g1 = TWO_PI * J * (np.sin(TWO_PI * (v[1] - v[0]))
                       + np.sin(TWO_PI * (v[1] - v[2])))
    g2 = TWO_PI * J * (np.sin(TWO_PI * (v[2] - v[0]))
                       + np.sin(TWO_PI * (v[2] - v[1])))
    g = np.stack([g0, g1, g2])
    ds = -k_el * lap(s) + g
    dsigma = -kappa * lap(sigma) + 4.0 * lam * sigma * (sigma ** 2 - 1.0) \
        - 0.5 * (Qv[:, None, None] * g).sum(0)
    return dsigma, ds


def relax(sigma, s, kappa, steps, periodic, eps=None, track=False):
    if eps is None:
        eps_s = 0.25 / (8.0 * k_el + 2.0 * TWO_PI ** 2 * J * 2.0)
        eps_sigma = 0.25 / (8.0 * kappa + 8.0 * lam
                           + 0.5 * TWO_PI ** 2 * J * 2.0 * 0.25 * 2.0)
    else:
        eps_s = eps_sigma = eps
    E_prev = energy(sigma, s, kappa, periodic)
    E_track = [E_prev]
    monoton = True
    for t in range(steps):
        dsigma, ds = gradients(sigma, s, kappa, periodic)
        sigma = sigma - eps_sigma * dsigma
        s = s - eps_s * ds
        E_now = energy(sigma, s, kappa, periodic)
        if E_now > E_prev + 1e-7 * abs(E_prev):
            monoton = False
        E_prev = E_now
        if track and (t % max(1, steps // 200) == 0):
            E_track.append(E_prev)
    return sigma, s, E_prev, monoton, eps, np.array(E_track)


def readings(sigma, s, periodic, Qv=Q):
    aa = (1.0 - sigma) / 2.0
    v = s + Qv[:, None, None] * aa[None, :, :]
    delta12 = TWO_PI * (v[0] - v[1])
    frac = np.abs(((delta12 / TWO_PI) % 1.0 + 1.0) % 1.0)
    frac = np.minimum(frac, 1.0 - frac)
    nonint = float((frac > 0.05).mean())
    if periodic:
        dv1x = np.roll(v[0], -1, 1) - v[0]
        dv1y = np.roll(v[0], -1, 0) - v[0]
        dv2x = np.roll(v[1], -1, 1) - v[1]
        dv2y = np.roll(v[1], -1, 0) - v[1]
    else:
        dv1x = np.zeros_like(v[0]); dv1y = np.zeros_like(v[0])
        dv2x = np.zeros_like(v[1]); dv2y = np.zeros_like(v[1])
        dv1x[:, :-1] = v[0][:, 1:] - v[0][:, :-1]
        dv1y[:-1, :] = v[0][1:, :] - v[0][:-1, :]
        dv2x[:, :-1] = v[1][:, 1:] - v[1][:, :-1]
        dv2y[:-1, :] = v[1][1:, :] - v[1][:-1, :]
    w12 = 1.0 - np.cos(delta12)
    w12_x = 0.5 * (w12 + np.roll(w12, 1, 1))
    w12_y = 0.5 * (w12 + np.roll(w12, 1, 0))
    if not periodic:
        w12_x = np.zeros_like(w12)
        w12_x[:, :-1] = 0.5 * (w12[:, :-1] + w12[:, 1:])
        w12_y = np.zeros_like(w12)
        w12_y[:-1, :] = 0.5 * (w12[:-1, :] + w12[1:, :])
    X_beta = float(((dv1x ** 2 + dv2x ** 2) / 2.0 * w12_x).sum()
                   + ((dv1y ** 2 + dv2y ** 2) / 2.0 * w12_y).sum())
    return delta12, nonint, X_beta, v, aa


def wall_width(sigma, periodic):
    m = sigma.mean(axis=1)
    d = np.abs(np.diff(m))
    if d.max() < 1e-6:
        return 0.0, []
    peaks = []
    thr = d.max() / 2.0
    above = d > thr
    idx = np.where(above)[0]
    if len(idx) == 0:
        return 0.0, []
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    widths = []
    for g in groups:
        widths.append(len(g))
        peaks.append(int(g[np.argmax(d[g])]))
    return float(np.mean(widths)), peaks


# ---------------- 方案甲：直壁错排带（环面） ----------------

def init_wall_pair(Nx, Ny, w0, shift=0):
    """错配记账（份额 Q=(1,0,0) 的合法性要求）：s1 必须自带奇数条整数台阶。
    初值：台阶与上壁 y2 重合（s1 阶跃），a 两壁沿 y 分带。"""
    y = np.arange(Ny)
    y1 = (Ny // 4 + shift) % Ny
    y2 = (3 * Ny // 4 + shift) % Ny
    yy = y - y1
    yy2 = y - y2
    sigma_row = -np.tanh(yy / w0) * np.tanh(-yy2 / w0)
    sigma = np.tile(sigma_row[:, None], (1, Nx))
    s = np.zeros((3, Ny, Nx))
    s[0][y >= y2, :] = 1.0
    return sigma, s


def run_plan_A(kappa_list=(1.0, 4.0, 16.0, 64.0), Nx=64, Ny=32, T=20000):
    print("=" * 70)
    print("方案甲：直壁错排带（环面 %dx%d）判据 W1-W3" % (Nx, Ny))
    results = {}
    for kap in kappa_list:
        sigma, s = init_wall_pair(Nx, Ny, 2.0)
        sigma, s, E, mono, eps, _ = relax(sigma, s, kap, T, periodic=True)
        delta12, nonint, X_beta, v, aa = readings(sigma, s, periodic=True)
        w, peaks = wall_width(sigma, True)
        Npt = sigma.size
        Evac = -3.0 * J * Npt
        aa_dev = np.abs(aa - np.round(aa))
        aa_nonint = float((aa_dev > 0.05).mean())
        ms = (s[0] - s[1]).mean(axis=1)
        inb = sigma.mean(axis=1) < -0.5
        s_jump = float(abs(ms[inb].mean() - ms[~inb].mean())) if inb.any() and (~inb).any() else 0.0
        results[kap] = dict(E=E, Eexc=E - Evac, mono=mono, nonint=nonint,
                            X=X_beta, w=w, peaks=peaks, aa_nonint=aa_nonint,
                            s_jump=s_jump, sigma=sigma.copy(), s=s.copy())
        ds1y = np.abs(np.diff(s[0].mean(axis=1)))
        sw = float(np.sum(ds1y > 0.05))
        print("kappa=%5.1f  E-Evac=%10.4f  单调=%s  壁宽=%4.1f格  s1台阶键=%4.1f  "
              "a通道=%.3f  delta通道=%.3f  s跨壁差=%.3f  X_beta=%.4f"
              % (kap, E - Evac, mono, w, sw, aa_nonint, nonint, s_jump, X_beta))
    # W2 壁位平移零模（kappa=16）
    base = results[16.0]["E"]
    sigma, s = init_wall_pair(Nx, Ny, 2.0, shift=4)
    sigma, s, E_sh, _, _, _ = relax(sigma, s, 16.0, T, periodic=True)
    rel = abs(E_sh - base) / abs(base)
    print("W2 平移零模: E(shift+4)=%12.4f vs E(base)=%12.4f 相对差=%.3e"
          % (E_sh, base, rel))
    results["shift_rel"] = rel
    # W1 收敛性（kappa=16 档，3 倍步数）
    sigma, s = init_wall_pair(Nx, Ny, 2.0)
    sigma, s, E_long, mono_l, _, _ = relax(sigma, s, 16.0, 3 * T, periodic=True)
    conv = abs(E_long - base) / abs(base)
    print("W1 收敛性: E(%d步)=%12.4f vs E(%d步)=%12.4f 相对差=%.3e 单调=%s"
          % (T, base, 3 * T, E_long, conv, mono_l))
    results["conv_rel"] = conv
    results["mono"] = results[16.0]["mono"] and mono_l
    return results


def run_1d_ref(N=256, T=8000):
    """W5 对照齿二：第 598 条原版一维位错（无 a 场）：v1 单台阶初值，
    三分量键弹性 + 异类耦合，弛豫后台阶宽度由 k/J 竞争定。"""
    y = np.arange(N)
    v1 = 1.0 / (1.0 + np.exp(-(y - N // 2) / 1.0))
    s = np.stack([v1, np.zeros(N), np.zeros(N)])[None, :, :]
    kap = k_el
    E0 = energy(np.zeros((1, N)), s, 0.0, periodic=True)
    for t in range(T):
        _, ds = gradients(np.zeros((1, N)), s, 0.0, periodic=True)
        eps_s = 0.25 / (8.0 * k_el + 2.0 * TWO_PI ** 2 * J * 2.0)
        s = s - eps_s * ds
    v = s
    delta12 = TWO_PI * (v[0, 0] - v[0, 1])[None, :]
    frac = np.abs(((delta12 / TWO_PI) % 1.0 + 1.0) % 1.0)
    frac = np.minimum(frac, 1.0 - frac)
    nonint = float((frac > 0.05).mean())
    dv1 = np.roll(v[0, 0], -1) - v[0, 0]
    dv2 = np.roll(v[0, 1], -1) - v[0, 1]
    w12 = 1.0 - np.cos(delta12[0])
    w12x = 0.5 * (w12 + np.roll(w12, 1))
    X_beta = float(((dv1 ** 2 + dv2 ** 2) / 2.0 * w12x).sum())
    width = float(np.sum((np.abs(np.diff(v[0, 0])) > 0.1)))
    print("W5 一维对照(598原版): 非整delta占比=%.4f X_beta=%.5f 台阶键数=%.1f"
          % (nonint, X_beta, width))
    return nonint, X_beta


def run_vortex_ref(Nx=64, Ny=32, T=8000):
    """W4 对照齿一：无壁裸涡旋（aa=0，sigma 冻结为 1），错配无人承接。"""
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    cx, cy = Nx / 2.0, Ny / 2.0
    theta = np.arctan2(Y - cy, X - cx)
    v1_0 = theta / TWO_PI  # 主值场，绕芯一圈相位连续变化 1
    s = np.stack([v1_0, np.zeros_like(v1_0), np.zeros_like(v1_0)])
    sigma = np.ones((Ny, Nx))
    # sigma 冻结：能量与梯度只对 s（把 sigma 梯度关掉）
    E0 = energy(sigma, s, kappa_def, periodic=True)
    eps = 0.5 / (8.0 * lam + 8.0 * k_el + 2.0 * TWO_PI ** 2 * J * 3.0)
    for t in range(T):
        _, ds = gradients(sigma, s, kappa_def, periodic=True)
        s = s - eps * ds
    E1 = energy(sigma, s, kappa_def, periodic=True)
    # 摊平读数：wrap 键应变的最大值（初值切割线附近 ~0.5，摊平后 ~1/N）
    dsx = np.roll(s[0], -1, 1) - s[0]
    dsy = np.roll(s[0], -1, 0) - s[0]
    wrapmax = float(np.max(np.abs(np.concatenate(
        [((dsx + 0.5) % 1.0 - 0.5).ravel(), ((dsy + 0.5) % 1.0 - 0.5).ravel()]))))
    print("W4 裸涡旋对照: E0=%.2f -> E1=%.2f (比值 %.4f)  max|wrap ds1|=%.4f"
          % (E0, E1, E1 / E0, wrapmax))
    return E1 / E0, wrapmax


# ---------------- 方案乙：环面闭合壁环（错配岛，位错环 2D 对象） ----------------
# 实现层修正登记：立案材料方案乙原设计为环面割线有限壁段（端点芯），
# 拓扑审查发现环面上实 sigma 场的壁必闭合（pi_0 区域边界无端点），
# 开边界半无限壁冒烟显示端点被壁张力快速拉回边界（瞬态对象）。
# 故乙执行版取环面闭合壁环（错配岛）：岛内 aa=1、岛外 0，错配守恒撑住
# 岛面积（无缩短通道），环线张力使其趋圆，s 长波摊平通道动力学极慢
# （窗口内为亚稳存活，热力学摊平如实登记为边界声明）。

def init_island(Nx=64, Ny=64, R=10.0, w0=2.0):
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    dx = np.minimum(np.abs(X - Nx / 2.0), Nx - np.abs(X - Nx / 2.0))
    dy = np.minimum(np.abs(Y - Ny / 2.0), Ny - np.abs(Y - Ny / 2.0))
    r = np.sqrt(dx ** 2 + dy ** 2)
    sigma = np.tanh((r - R) / w0)
    s = np.zeros((3, Ny, Nx))
    s[0][X >= Nx / 2.0] = 1.0
    return sigma, s, r


def run_plan_B(kappa_list=(16.0, 64.0), Nx=64, Ny=64, T=20000):
    print("=" * 70)
    print("方案乙：环面闭合壁环（错配岛 %dx%d）判据 P1-P3" % (Nx, Ny))
    results = {}
    for kap in kappa_list:
        sigma, s, r = init_island(Nx, Ny)
        E0 = energy(sigma, s, kap, periodic=True)
        area0 = float((sigma < -0.5).sum())
        eps_b = (0.25 / (8.0 * k_el + 2.0 * TWO_PI ** 2 * J * 2.0),
                 0.25 / (8.0 * kap + 8.0 * lam
                         + 0.5 * TWO_PI ** 2 * J * 2.0 * 0.25 * 2.0))
        sigma, s, E1, mono, _, _ = relax(sigma, s, kap, T, periodic=True,
                                         eps=None)
        delta12, nonint, X_beta, v, aa = readings(sigma, s, periodic=True)
        area1 = float((sigma < -0.5).sum())
        ms = (s[0] - s[1]).mean()
        s_jump = float(abs(ms - (-float((sigma < -0.5).mean()))))
        # P3 环带罚集中：X_beta 密度按环带归属
        band = np.abs(r - 10.0) < 5.0
        w12 = 1.0 - np.cos(delta12)
        dv1x = np.roll(v[0], -1, 1) - v[0]
        dv1y = np.roll(v[0], -1, 0) - v[0]
        dv2x = np.roll(v[1], -1, 1) - v[1]
        dv2y = np.roll(v[1], -1, 0) - v[1]
        w12_x = 0.5 * (w12 + np.roll(w12, 1, 1))
        w12_y = 0.5 * (w12 + np.roll(w12, 1, 0))
        Xdens = ((dv1x ** 2 + dv2x ** 2) / 2.0 * w12_x
                 + (dv1y ** 2 + dv2y ** 2) / 2.0 * w12_y)
        Xband = float(Xdens[band].sum())
        frac_band = Xband / max(X_beta, 1e-12)
        aa_dev = np.abs(aa - np.round(aa))
        aa_nonint = float((aa_dev > 0.05).mean())
        results[kap] = dict(E0=E0, E1=E1, mono=mono, area0=area0, area1=area1,
                            nonint=nonint, X=X_beta, s_jump=s_jump,
                            frac_band=frac_band, aa_nonint=aa_nonint,
                            sigma=sigma.copy(), s=s.copy(), r=r, delta=delta12)
        print("kappa=%5.1f  E-Evac=%10.3f  单调=%s  岛面积 %d->%d (比值 %.3f)  "
              "a通道=%.3f  delta通道=%.3f  s跨岛差=%.3f  环带X份额=%.3f  X_beta=%.4f"
              % (kap, E1 - (-3.0 * J * sigma.size),
                 mono, area0, area1, area1 / area0, aa_nonint, nonint,
                 s_jump, frac_band, X_beta))
    return results


def energy_density(sigma, s, kappa, periodic):
    if periodic:
        dsx = np.roll(s, -1, 2) - s
        dsy = np.roll(s, -1, 1) - s
        dsig_x = np.roll(sigma, -1, 1) - sigma
        dsig_y = np.roll(sigma, -1, 0) - sigma
    else:
        dsx = np.zeros_like(s); dsy = np.zeros_like(s)
        dsx[:, :, :-1] = s[:, :, 1:] - s[:, :, :-1]
        dsy[:, :-1, :] = s[:, 1:, :] - s[:, :-1, :]
        dsig_x = np.zeros_like(sigma); dsig_y = np.zeros_like(sigma)
        dsig_x[:, :-1] = sigma[:, 1:] - sigma[:, :-1]
        dsig_y[:-1, :] = sigma[1:, :] - sigma[:-1, :]
    aa = (1.0 - sigma) / 2.0
    v = s + Q[:, None, None] * aa[None, :, :]
    dv12 = v[0] - v[1]; dv13 = v[0] - v[2]; dv23 = v[1] - v[2]
    Ew = (kappa / 2.0) * (dsig_x ** 2 + dsig_y ** 2)         + lam * (sigma ** 2 - 1.0) ** 2
    Es = (k_el / 2.0) * ((dsx ** 2).sum(0) + (dsy ** 2).sum(0))
    EJ = -J * (np.cos(TWO_PI * dv12) + np.cos(TWO_PI * dv13)
               + np.cos(TWO_PI * dv23))
    return Ew + Es + EJ


def run_Jscan(kappa=16.0, Jlist=(1.0, 0.25, 0.06, 0.015), Nx=64, Ny=32, T=20000):
    """J 档扫描：delta 势阱强度与畴壁 delta 可读性的竞争。"""
    global J
    print("=" * 70)
    print("J 档扫描（kappa=%g）: delta 势阱弱化是否打开延展非整区" % kappa)
    J0 = J
    out = []
    for Jv in Jlist:
        J = Jv
        sigma, s = init_wall_pair(Nx, Ny, 2.0)
        sigma, s, E, mono, _, _ = relax(sigma, s, kappa, T, periodic=True)
        delta12, nonint, X_beta, v, aa = readings(sigma, s, periodic=True)
        Npt = sigma.size
        Evac = -3.0 * Jv * Npt
        ds1y = np.abs(np.diff(s[0].mean(axis=1)))
        sw = float(np.sum(ds1y > 0.05))
        w, _ = wall_width(sigma, True)
        print("J=%6.3f  E-Evac=%10.3f  壁宽=%4.1f  s1台阶键=%4.1f  "
              "delta占比=%.3f  X_beta=%.4f  单调=%s"
              % (Jv, E - Evac, w, sw, nonint, X_beta, mono))
        out.append((Jv, E - Evac, nonint, X_beta))
    J = J0
    return out


def run_plot(T=20000):
    """出图：图 1 甲 J 档核心发现；图 2 对照与乙判否。"""
    global J
    kappa = 16.0
    J0 = J
    prof = {}
    sig = {}
    sl = {}
    for Jv in (1.0, 0.25, 0.06, 0.015):
        J = Jv
        sigma, s = init_wall_pair(64, 32, 2.0)
        sigma, s, E, mono, _, _ = relax(sigma, s, kappa, T, periodic=True)
        delta12, nonint, X_beta, v, aa = readings(sigma, s, periodic=True)
        prof[Jv] = (delta12.mean(axis=1) / TWO_PI, nonint, X_beta)
        sig[Jv] = sigma
        sl[Jv] = s[0].mean(axis=1)
        print("plot J=%.3f done nonint=%.3f X=%.3f" % (Jv, nonint, X_beta))
    J = J0
    Nx, Ny = 64, 32

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0][0]
    for Jv, (pr, ni, xb) in prof.items():
        ax.plot(pr, label="J=%g" % Jv)
    ax.set_xlabel("y（格点）"); ax.set_ylabel(r"$\delta^{12}/2\pi$（x 平均）")
    ax.set_title(r"(a) $\delta^{12}$ 剖面：J 档决定延展非整区")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    Js = sorted(prof)
    ax.plot(Js, [prof[j][1] for j in Js], "o-", label=r"非整 $\delta$ 占比")
    ax.set_xscale("log"); ax.set_xlabel("J"); ax.set_ylabel("占比")
    ax2 = ax.twinx()
    ax2.plot(Js, [prof[j][2] for j in Js], "s--", color="tab:red",
             label=r"$X_\beta$")
    ax2.set_ylabel(r"$X_\beta$", color="tab:red")
    ax.set_title("(b) 可读性随 J 单调打开")
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")

    ax = axes[1][0]
    im = ax.imshow(sig[0.06], aspect="auto", cmap="RdBu_r",
                   vmin=-1.2, vmax=1.2)
    ax.set_title(r"(c) $\sigma$ 场（J=0.06，壁带存活）")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[1][1]
    for Jv in Js:
        ax.plot(sl[Jv], label="J=%g" % Jv)
    ax.set_xlabel("y"); ax.set_ylabel(r"$\langle s_1 \rangle_x$")
    ax.set_title("(d) $s_1$ 台阶补偿形貌")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT + r"\dw618_Jscan.png", dpi=DPI)
    plt.close(fig)
    print("图1存盘")

    # 图 2
    sigma_v0 = np.ones((32, 64))
    X, Y = np.meshgrid(np.arange(64), np.arange(32))
    v1_0 = np.arctan2(Y - 16.0, X - 32.0) / TWO_PI
    s_v = np.stack([v1_0, np.zeros_like(v1_0), np.zeros_like(v1_0)])
    eps_v = 0.25 / (8.0 * k_el + 2.0 * TWO_PI ** 2 * J * 2.0)
    for t in range(8000):
        _, ds = gradients(sigma_v0, s_v, 0.0, periodic=True)
        s_v = s_v - eps_v * ds
    sig_i0, s_i0, r0 = init_island(64, 64)
    Jv_save = J; J = 1.0
    sig_i1, s_i1, E_i, mono_i, _, _ = relax(sig_i0.copy(), s_i0.copy(),
                                            16.0, 20000, periodic=True)
    J = Jv_save
    sig_w0, s_w0 = init_wall_pair(64, 32, 2.0)
    sig_w1, s_w1, E_w, _, _, _ = relax(sig_w0.copy(), s_w0.copy(),
                                       16.0, 20000, periodic=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0][0]
    ax.imshow(s_v[0], aspect="auto", cmap="twilight", vmin=-0.5, vmax=0.5)
    ax.set_title("(a) 裸涡旋初值 $v_1$（W4 对照）")
    ax = axes[0][1]
    ax.imshow(s_v[0] - np.round(s_v[0]), aspect="auto", cmap="twilight",
              vmin=-0.5, vmax=0.5)
    ax.set_title("(b) 裸涡旋终态：摊平（E 趋真空）")
    ax = axes[1][0]
    ax.imshow(sig_i1, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    ax.set_title("(c) 乙岛终态（J=1）：岛消失（P2 判否）")
    ax = axes[1][1]
    ax.imshow(sig_w1, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    ax.set_title("(d) 甲壁带终态（J=1）：带结构存活（亚稳）")
    for ax in axes.ravel():
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUT + r"\dw618_refs.png", dpi=DPI)
    plt.close(fig)
    print("图2存盘")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        print(">>> 冒烟：甲 kappa=16 小网格短步 + 乙短步")
        rA = run_plan_A(kappa_list=(16.0,), Nx=32, Ny=16, T=2000)
        rB = run_plan_B(kappa_list=(16.0,), Nx=32, Ny=32, T=2000)
    elif mode == "plot":
        run_plot()
    else:
        rA = run_plan_A()
        n1, x1 = run_1d_ref()
        vr, wm = run_vortex_ref()
        rB = run_plan_B()
        run_Jscan()
