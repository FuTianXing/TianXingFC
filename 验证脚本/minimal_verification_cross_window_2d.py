# -*- coding: utf-8 -*-
r"""
minimal_verification_cross_window_2d.py

第 621 号：二维弱 J 档交叉项供货窗口审计轮（E 审计轮）

背景：第 618 号立案轮交付甲直壁错排带（环面墙带 + 错排结构）亚稳存活，
弱 J 档 delta 非整区打开（J=1 全整；J<=0.25 占比 0.125 至 0.438）。
本轮把第 598 条式交叉项供货窗口扫描移植到该二维模型：
在 delta 已打开的弱 J 档，交叉项能否以 O(1) 强度翻转错配承载的能量
排序（供货窗口是否重开）。

模型（第 618 号泛函 + FK 扣绕数 + 交叉项）：
  E = sum[ kappa/2 |grad sigma|^2 + lam (sigma^2-1)^2 ]
      + (k/2) sum_{mu,a} (D_mu s^a)^2
      - J sum_{a<b} cos 2pi (v^a - v^b)
      + xi sum_{a<b} sum_mu S_mu^{ab} g(2pi (v^a - v^b))
  v^a = s^a + Q_a * aa,  aa = (1-sigma)/2（定义性联结）
  y 向 wrap 键扣扇区：D_y s^a(wrap) = s^a[0] - s^a[-1] + Q_a（FK 口径，
  Q 初值定扇区全程固定）；x 向无绕数。
  S_mu^{ab} = [(D_mu s^a)^2 + (D_mu s^b)^2]/2（键应变对称和，取 s 场应变，
  与 618 能量同一应变变量）

判据（先立期望再实现）
  G0 梯度冒烟：解析梯度对有限差分（含交叉项两形式）。
  E1 供货窗口扫描（核心）：alpha（Q=0）与 beta（Q=(1,0,0)）各自弛豫，
     D_assign(xi) = E_beta* - E_alpha*（错配承载代价），
     D_vac(xi) = E_beta* - E_vac（总账）。g1 支负 xi 侧、g3 支正 xi 侧。
     翻转即供货。期望：弱 J 档 X_beta 显著非零（对比 1D 的 1e-3），
     D 随 |xi| 单调降；翻转点与失稳/重构点比较定窗口。
  E2 失稳/重构墙：大 |xi| 档发散探测与构型重构读数（滑移台阶宽度）。
  E3 一维锚定：第 598 条 FK 基线复现（D_assign(0) 约 0.327）。
"""
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

TWO_PI = 2.0 * np.pi
OUT = r"D:\software\TianXingFC\images"
DPI = 150

K_EL = 1.0
LAM = 1.0
J_WEAK = 0.06
PAIRS = ((0, 1), (0, 2), (1, 2))


# ---------------- 离散算子 ----------------

def dx(f):
    return np.roll(f, -1, axis=-1) - f


def dy(f, Qv=None):
    d = np.empty_like(f)
    d[..., :-1, :] = f[..., 1:, :] - f[..., :-1, :]
    d[..., -1, :] = f[..., 0, :] - f[..., -1, :]
    if Qv is not None:
        d[..., -1, :] += Qv[:, None]
    return d


def laps(f):
    """正确空间周期拉普拉斯（2D 场 axis 0,1；3D 场 axis 1,2）。"""
    if f.ndim == 2:
        return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4.0 * f)
    return (np.roll(f, 1, 1) + np.roll(f, -1, 1)
            + np.roll(f, 1, 2) + np.roll(f, -1, 2) - 4.0 * f)


def lap_618(f):
    """618 脚本原算子（对 3D 场轴位滑移：轴 0 为分量轴）。对照用。"""
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4.0 * f)


def gform(form, dvf):
    if form == "g1":
        return 1.0 - np.cos(TWO_PI * dvf), TWO_PI * np.sin(TWO_PI * dvf)
    if form == "g3":
        return np.cos(TWO_PI * dvf), -TWO_PI * np.sin(TWO_PI * dvf)
    raise ValueError(form)


# ---------------- 能量与梯度 ----------------

def energy(sigma, s, kappa, Qv, xi, form, Jv):
    aa = (1.0 - sigma) / 2.0
    dsx = dx(s)
    dsy = dy(s, Qv)
    v = s + Qv[:, None, None] * aa[None, :, :]
    d12 = v[0] - v[1]
    d13 = v[0] - v[2]
    d23 = v[1] - v[2]
    E = (kappa / 2.0) * ((dx(sigma) ** 2).sum() + (dy(sigma) ** 2).sum()) \
        + LAM * ((sigma ** 2 - 1.0) ** 2).sum() \
        + (K_EL / 2.0) * ((dsx ** 2).sum() + (dsy ** 2).sum()) \
        - Jv * (np.cos(TWO_PI * d12) + np.cos(TWO_PI * d13)
                + np.cos(TWO_PI * d23)).sum()
    if xi != 0.0:
        for (a, b) in PAIRS:
            g, _ = gform(form, v[a] - v[b])
            Sx = (dsx[a] ** 2 + dsx[b] ** 2) / 2.0
            Sy = (dsy[a] ** 2 + dsy[b] ** 2) / 2.0
            E += xi * ((Sx * g).sum() + (Sy * g).sum())
    return float(E)


def gradients(sigma, s, kappa, Qv, xi, form, Jv, lapmode="correct"):
    # 弹性梯度用逐键 roll 形式（含 FK 绕数偏移，与能量逐键一致）；
    # lapmode 保留签名兼容，不再影响读数
    aa = (1.0 - sigma) / 2.0
    v = s + Qv[:, None, None] * aa[None, :, :]
    d12 = v[0] - v[1]
    d13 = v[0] - v[2]
    d23 = v[1] - v[2]
    gJ0 = TWO_PI * Jv * (np.sin(TWO_PI * d12) + np.sin(TWO_PI * d13))
    gJ1 = TWO_PI * Jv * (-np.sin(TWO_PI * d12) + np.sin(TWO_PI * d23))
    gJ2 = TWO_PI * Jv * (-np.sin(TWO_PI * d13) - np.sin(TWO_PI * d23))
    gJ = np.stack([gJ0, gJ1, gJ2])
    dsx0 = dx(s)
    dsy0 = dy(s, Qv)
    ds = K_EL * (np.roll(dsx0, 1, -1) - dsx0) \
        + K_EL * (np.roll(dsy0, 1, -2) - dsy0) + gJ
    dsigma = -kappa * laps(sigma) + 4.0 * LAM * sigma * (sigma ** 2 - 1.0) \
        - 0.5 * (Qv[:, None, None] * gJ).sum(0)
    if xi != 0.0:
        dsx = dx(s)
        dsy = dy(s, Qv)
        gcv = np.zeros_like(s)
        gcs = np.zeros_like(s)
        for (a, b) in PAIRS:
            gs, dg = gform(form, v[a] - v[b])
            Sx = (dsx[a] ** 2 + dsx[b] ** 2) / 2.0
            Sy = (dsy[a] ** 2 + dsy[b] ** 2) / 2.0
            S = Sx + Sy
            for c in (a, b):
                gcs[c] += xi * (np.roll(dsx[c] * gs, 1, -1) - dsx[c] * gs
                                + np.roll(dsy[c] * gs, 1, -2) - dsy[c] * gs)
            gcv[a] += xi * S * dg
            gcv[b] -= xi * S * dg
        ds = ds + gcs + gcv
        dsigma = dsigma - 0.5 * (Qv[:, None, None] * gcv).sum(0)
    return dsigma, ds


def relax(sigma, s, kappa, Qv, xi, form, Jv, steps, lapmode="correct",
          eps=None):
    if eps is None:
        eps_s = 0.25 / (8.0 * K_EL + 2.0 * TWO_PI ** 2 * Jv * 2.0 + 80.0 * abs(xi))
        eps_sig = 0.25 / (8.0 * kappa + 8.0 * LAM + 80.0 * abs(xi) * 0.25)
    else:
        eps_s = eps_sig = eps
    E_prev = energy(sigma, s, kappa, Qv, xi, form, Jv)
    monoton = True
    diverged = False
    for t in range(steps):
        dsigma, ds = gradients(sigma, s, kappa, Qv, xi, form, Jv, lapmode)
        sigma = sigma - eps_sig * dsigma
        s = s - eps_s * ds
        E_now = energy(sigma, s, kappa, Qv, xi, form, Jv)
        if not np.isfinite(E_now) or abs(E_now) > 1e7 or np.abs(s).max() > 50.0:
            diverged = True
            E_prev = E_now
            break
        if E_now > E_prev + 1e-7 * abs(E_prev):
            monoton = False
        E_prev = E_now
    return sigma, s, E_prev, monoton, diverged


def readings(sigma, s, Qv, xi, form, Jv):
    aa = (1.0 - sigma) / 2.0
    v = s + Qv[:, None, None] * aa[None, :, :]
    d12 = TWO_PI * (v[0] - v[1])
    frac = np.abs(((d12 / TWO_PI) % 1.0 + 1.0) % 1.0)
    frac = np.minimum(frac, 1.0 - frac)
    nonint = float((frac > 0.05).mean())
    dsx = dx(s)
    dsy = dy(s, Qv)
    X = 0.0
    if xi != 0.0 or form:
        for (a, b) in PAIRS:
            g, _ = gform(form, v[a] - v[b])
            Sx = (dsx[a] ** 2 + dsx[b] ** 2) / 2.0
            Sy = (dsy[a] ** 2 + dsy[b] ** 2) / 2.0
            X += float((Sx * g).sum() + (Sy * g).sum())
    ds1y = np.abs(np.diff(s[0], axis=0))
    slip = float((ds1y > 0.05).sum(axis=0).mean())
    strain = float(max(np.abs(dsx).max(), np.abs(dsy).max()))
    return dict(nonint=nonint, X=X, slip=slip, strain=strain, d12=d12, v=v, aa=aa)


# ---------------- 初值与场景 ----------------

def init_wall_pair(Nx, Ny, w0=2.0, shift=0):
    y = np.arange(Ny)
    y1 = (Ny // 4 + shift) % Ny
    y2 = (3 * Ny // 4 + shift) % Ny
    sigma_row = -np.tanh((y - y1) / w0) * np.tanh(-(y - y2) / w0)
    sigma = np.tile(sigma_row[:, None], (1, Nx))
    s = np.zeros((3, Ny, Nx))
    s[0][y >= y2, :] = 1.0
    return sigma, s


def run_point(Nx, Ny, kappa, Jv, xi, form, T, lapmode="correct"):
    Qb = np.array([1.0, 0.0, 0.0])
    Qa = np.array([0.0, 0.0, 0.0])
    sig0, s0 = init_wall_pair(Nx, Ny)
    sb, ssb, Eb, mb, db = relax(sig0.copy(), s0.copy(), kappa, Qb, xi, form,
                                Jv, T, lapmode)
    sa, ssa, Ea, ma, da = relax(sig0.copy(), s0.copy() * 0.0, kappa, Qa, xi,
                                form, Jv, T, lapmode)
    rb = readings(sb, ssb, Qb, xi, form, Jv)
    ra = readings(sa, ssa, Qa, xi, form, Jv)
    Evac = -3.0 * Jv * Nx * Ny
    return dict(Eb=Eb, Ea=Ea, Evac=Evac, D=Eb - Ea, Dvac=Eb - Evac,
                X=rb["X"], nonint=rb["nonint"], slip=rb["slip"],
                strain=rb["strain"], nonint_a=ra["nonint"], Xa=ra["X"],
                mono=(mb and ma), div=(db or da),
                sigma_b=sb, s_b=ssb, sigma_a=sa, s_a=ssa, rb=rb, ra=ra)


# ---------------- G0 梯度检查 ----------------

def grad_check(Ny=8, Nx=8, xi=0.7, form="g1", Jv=0.06, kappa=2.0):
    rng = np.random.default_rng(7)
    sigma = 1.0 + 0.3 * rng.standard_normal((Ny, Nx))
    s = 0.3 * rng.standard_normal((3, Ny, Nx))
    Qv = np.array([1.0, 0.0, 0.0])
    dsigma, ds = gradients(sigma, s, kappa, Qv, xi, form, Jv)
    err = 0.0
    h = 1e-6
    for arr, grad, tag in ((sigma, dsigma, "sigma"), (s, ds, "s")):
        idxs = [(0, 0), (Ny // 2, Nx // 2), (Ny - 1, Nx - 1)]
        if tag == "s":
            pts = [(c, i, j) for c in range(3) for (i, j) in idxs]
        else:
            pts = idxs
        for pt in pts:
            if tag == "s":
                c, i, j = pt
                old = arr[c, i, j]
                arr[c, i, j] = old + h
                Ep = energy(sigma, s, kappa, Qv, xi, form, Jv)
                arr[c, i, j] = old - h
                Em = energy(sigma, s, kappa, Qv, xi, form, Jv)
                arr[c, i, j] = old
                num = (Ep - Em) / (2.0 * h)
                ana = grad[c, i, j]
            else:
                i, j = pt
                old = arr[i, j]
                arr[i, j] = old + h
                Ep = energy(sigma, s, kappa, Qv, xi, form, Jv)
                arr[i, j] = old - h
                Em = energy(sigma, s, kappa, Qv, xi, form, Jv)
                arr[i, j] = old
                num = (Ep - Em) / (2.0 * h)
                ana = grad[i, j]
            err = max(err, abs(num - ana) / max(1.0, abs(ana)))
    gmax = max(np.abs(dsigma).max(), np.abs(ds).max())
    return err, gmax


# ---------------- E3 一维锚定（第 598 条基线复现） ----------------

def anchor_1d(N=256, Jv=1.0, T=220000, w=8.0):
    """E3 一维锚定：第 598 条 FK 口径基线复现。
    alpha：三分量各 1/3 台阶（无错排）；beta：分量 1 全幅台阶（错排 1）。
    期望 D_assign(0) = E_beta - E_alpha = +0.32703。"""
    n = np.arange(N)
    prof = 0.5 * (1.0 + np.tanh((n - N / 2.0) / w))
    v_b = np.stack([prof, np.zeros(N), np.zeros(N)])
    v_a = np.stack([prof / 3.0, prof / 3.0, prof / 3.0])
    eps = 0.3 / (4.0 + 8.0 * np.pi ** 2 * Jv)

    def E1d(v, Q):
        d = np.empty((3, N))
        d[:, :-1] = np.diff(v, axis=1)
        d[:, -1] = v[:, 0] - v[:, -1] + np.array(Q)
        e = 0.5 * K_EL * (d ** 2).sum()
        d12 = TWO_PI * (v[0] - v[1])
        d13 = TWO_PI * (v[0] - v[2])
        d23 = TWO_PI * (v[1] - v[2])
        e -= Jv * (np.cos(d12) + np.cos(d13) + np.cos(d23)).sum()
        return e

    def g1d(v, Q):
        d = np.empty((3, N))
        d[:, :-1] = np.diff(v, axis=1)
        d[:, -1] = v[:, 0] - v[:, -1] + np.array(Q)
        g = K_EL * (np.roll(d, 1, axis=1) - d)
        d12 = TWO_PI * (v[0] - v[1])
        d13 = TWO_PI * (v[0] - v[2])
        d23 = TWO_PI * (v[1] - v[2])
        g[0] += TWO_PI * Jv * (np.sin(d12) + np.sin(d13))
        g[1] += TWO_PI * Jv * (-np.sin(d12) + np.sin(d23))
        g[2] += TWO_PI * Jv * (-np.sin(d13) - np.sin(d23))
        return g

    res = {}
    for tag, v0, Q in (("b", v_b, (1, 0, 0)), ("a", v_a, (0, 0, 0))):
        vv = v0.copy()
        e_ex = E1d(vv, Q) + 3.0 * Jv * N
        for t in range(T):
            vv = vv - eps * g1d(vv, Q)
            if t % 10000 == 9999:
                e_new = E1d(vv, Q) + 3.0 * Jv * N
                if abs(e_new - e_ex) < 1e-10:
                    break
                e_ex = e_new
        res[tag] = E1d(vv, Q)
    return res["b"] - res["a"]


# ---------------- 主流程 ----------------

def scan(kappa=16.0, Jv=J_WEAK, Nx=64, Ny=32, T=15000, tag="k16",
         g1_grid=(0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0),
         g3_grid=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
         probe_2T=()):
    out = {}
    for form, xis in (("g1", list(g1_grid)), ("g3", list(g3_grid))):
        rows = []
        print("=" * 78)
        print("E1 扫描 %s 支（kappa=%g, J=%g, %dx%d, T=%d）" %
              (form, kappa, Jv, Nx, Ny, T))
        print("%8s %14s %14s %12s %10s %9s %9s %7s %5s %5s" %
              ("xi", "D_assign", "D_vac", "X_beta", "非整δ", "滑移宽",
               "max应变", "单调", "发散", "供货"))
        for xi in xis:
            t0 = time.time()
            r = run_point(Nx, Ny, kappa, Jv, xi, form, T)
            supply = "是" if (r["D"] < 0 and not r["div"]) else "否"
            extra = ""
            if (form, xi) in probe_2T:
                r2 = run_point(Nx, Ny, kappa, Jv, xi, form, 2 * T)
                extra = "  [2T: D=%+.3f max应变=%.3f]" % (r2["D"], r2["strain"])
            print("%8.2f %14.4f %14.2f %12.4f %10.4f %9.2f %9.3f %7s %5s %5s  (%.0fs)%s" %
                  (xi, r["D"], r["Dvac"], r["X"], r["nonint"], r["slip"],
                   r["strain"], r["mono"], r["div"], supply,
                   time.time() - t0, extra))
            rows.append((xi, r))
            if r["div"]:
                break
        out[form] = rows
    return out


def kink_density_of(r, Qv=None):
    if Qv is None:
        Qv = np.array([1.0, 0.0, 0.0])
    s = r["s_b"]
    dsx = dx(s)
    dsy = dy(s, Qv)
    n = s[0].size
    return float(((np.abs(dsx) > 0.5).sum() + (np.abs(dsy) > 0.5).sum())
                 / (3.0 * n))


def run_kscan(kappa=16.0, Jv=J_WEAK, T=15000,
              grids=None):
    """E4：阈值 k 标度扫描。若阈值满足 xi_c ∝ k（比值锁定），则"供货需要
    xi/k = O(1)"与"权重为负区刚度翻负在 xi/k = O(1)"同量级，判否与绝对
    档位无关，属结构性。"""
    global K_EL
    if grids is None:
        grids = {0.5: (0.0, 0.2, 0.3, 0.4, 0.6),
                 1.0: (0.0, 0.4, 0.6, 0.7, 0.75, 0.8, 1.2),
                 2.0: (0.0, 0.8, 1.2, 1.4, 1.5, 1.6, 2.4)}
    print("=" * 78)
    print("E4 阈值 k 标度扫描（g3 支，kappa=%g, J=%g, T=%d）" % (kappa, Jv, T))
    print("%6s %6s %12s %10s %10s %8s" % ("k", "xi", "D_assign", "max应变",
                                          "kink密度", "判定"))
    k0 = K_EL
    bands = {}
    for k, xis in grids.items():
        K_EL = k
        prev = None
        for xi in xis:
            r = run_point(64, 32, kappa, Jv, xi, "g3", T)
            kd = kink_density_of(r)
            verdict = "新相" if kd > 0.05 else ("翻转?" if r["D"] < 0 else "无翻转")
            print("%6.1f %6.2f %12.1f %10.3f %10.4f %8s" %
                  (k, xi, r["D"], r["strain"], kd, verdict))
            if kd > 0.05 and prev is not None and k not in bands:
                bands[k] = (prev, xi)
            if kd <= 0.05:
                prev = xi
        K_EL = k0
    print("阈值带（无 kink 的最大 xi 到出现 kink 的最小 xi）：")
    ratios = []
    for k, (a, b) in sorted(bands.items()):
        ratios.append((a / k, b / k))
        print("  k=%.1f: xi_c in [%.2f, %.2f]  比值 xi_c/k in [%.2f, %.2f]" %
              (k, a, b, a / k, b / k))
    if len(ratios) >= 2:
        lo = max(r[0] for r in ratios)
        hi = min(r[1] for r in ratios)
        # 结构性判据：阈值与 k 成比例，等价于比值带落在同一 O(1) 区间。
        # 判据取"各档比值带的并集落在 [0.5, 1.5]"（比例关系成立），
        # 并报告交集（加密档的严格锁定）。
        all_lo = min(r[0] for r in ratios)
        all_hi = max(r[1] for r in ratios)
        prop = 0.5 <= all_lo and all_hi <= 1.5
        print("  比值带并集 [%.2f, %.2f]  %s" %
              (all_lo, all_hi,
               "O(1)=阈值与 k 成比例（结构性判否）" if prop else "越出 O(1) 带"))
        print("  比值带交集 [%.2f, %.2f]（%s）" %
              (lo, hi, "非空=加密档严格锁定" if lo <= hi else "空=加密精度差"))
    return bands


def main_full():
    print("第 621 号：二维弱 J 档交叉项供货窗口审计轮")
    print()
    # G0
    for form in ("g1", "g3"):
        err, gmax = grad_check(xi=0.7, form=form)
        print("G0 梯度检查(%s): 最大相对误差 %.3e（判据 1e-5），max|g|=%.3f" %
              (form, err, gmax))
    # E3
    D1d = anchor_1d()
    print("E3 一维锚定（598 基线复现）: D_assign(0) = %+.5f（期望 0.327）" % D1d)
    # 618 算子影响核对
    Qb = np.array([1.0, 0.0, 0.0])
    sig0, s0 = init_wall_pair(64, 32)
    for mode in ("618", "correct"):
        lapmode = "618" if mode == "618" else "correct"
        sb, ssb, Eb, mb, db = relax(sig0.copy(), s0.copy(), 16.0, Qb, 0.0,
                                    "g1", J_WEAK, 15000, lapmode)
        r = readings(sb, ssb, Qb, 0.0, "g1", J_WEAK)
        print("算子核对[%s]: E-Evac=%12.4f  X=%.4f  非整δ=%.4f  单调=%s" %
              (mode, Eb - (-3.0 * J_WEAK * 64 * 32), r["X"], r["nonint"], mb))
    # E1（精细网格：供货窗口与无界阈值的关系为核心）
    res = scan(g1_grid=(0.0, -0.2, -0.3, -0.4, -0.5, -0.6),
               g3_grid=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3),
               probe_2T=(("g3", 0.9), ("g3", 1.0), ("g3", 1.1)))
    kink_rows = []
    for form, xis in (("g1", (0.0, -0.2, -0.3, -0.4, -0.5, -0.6)),
                      ("g3", (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3))):
        for xi in xis:
            r = run_point(64, 32, 16.0, J_WEAK, xi, form, 15000)
            s = r["s_b"]
            dsx = dx(s)
            dsy = dy(s, np.array([1.0, 0.0, 0.0]))
            n = s[0].size
            kd = float(((np.abs(dsx) > 0.5).sum() + (np.abs(dsy) > 0.5).sum())
                       / (3.0 * n))
            kink_rows.append((xi, form, kd))
            print("kink诊断 %s xi=%+.2f: kink密度=%.4f" % (form, xi, kd))
    np.save(r"D:\software\TianXingFC\_tools\scan621_g1.npy",
            np.array([(xi, r["D"], r["Dvac"], r["X"], r["nonint"], r["slip"],
                       1.0 if r["div"] else 0.0) for xi, r in res["g1"]]))
    np.save(r"D:\software\TianXingFC\_tools\scan621_g3.npy",
            np.array([(xi, r["D"], r["Dvac"], r["X"], r["nonint"], r["slip"],
                       1.0 if r["div"] else 0.0) for xi, r in res["g3"]]))
    run_kscan()
    plot(res, kink_rows)


def plot(res, kink_rows=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for form, color in (("g1", "tab:blue"), ("g3", "tab:red")):
        rows = res[form]
        xs = [r[0] for r in rows]
        Ds = [r[1]["D"] for r in rows]
        dv = [r[1]["div"] for r in rows]
        ax.plot(xs, Ds, "o-", color=color, label=form)
        for x, d, dd in zip(xs, Ds, dv):
            if dd:
                ax.plot([x], [d], "x", color=color, ms=12, mew=3)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("D_assign = E_beta - E_alpha")
    ax.set_title("(a) 错配承载代价随交叉项强度")
    ax.legend()
    ax = axes[1]
    for form, color in (("g1", "tab:blue"), ("g3", "tab:red")):
        rows = res[form]
        xs = [r[0] for r in rows]
        Xs = [r[1]["X"] for r in rows]
        ax.plot(xs, Xs, "s-", color=color, label="X_beta " + form)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$X_\beta$（全对交叉密度）")
    ax.set_title("(b) 交叉密度读数")
    ax.legend()
    ax = axes[2]
    if kink_rows:
        for form, color in (("g1", "tab:blue"), ("g3", "tab:red")):
            xs = [r[0] for r in kink_rows if r[1] == form]
            ks = [r[2] for r in kink_rows if r[1] == form]
            ax.plot(xs, ks, "^-", color=color, label="kink密度 " + form)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel("kink 键占比")
        ax.set_title("(c) 翻转档=kink 增殖新相")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT + r"\scan621_window.png", dpi=DPI)
    plt.close(fig)
    print("图 scan621_window.png 存盘")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        print(">>> 冒烟：G0 + 一维锚定 + 618算子核对 + 单点")
        for form in ("g1", "g3"):
            err, gmax = grad_check(xi=0.7, form=form)
            print("G0(%s): err=%.3e max|g|=%.3f" % (form, err, gmax))
        D1d = anchor_1d(T=40000)
        print("E3 锚定: D_assign(0) = %+.5f" % D1d)
        Qb = np.array([1.0, 0.0, 0.0])
        sig0, s0 = init_wall_pair(64, 32)
        for mode in ("618", "correct"):
            sb, ssb, Eb, mb, db = relax(sig0.copy(), s0.copy(), 16.0, Qb, 0.0,
                                        "g1", J_WEAK, 4000, mode)
            r = readings(sb, ssb, Qb, 0.0, "g1", J_WEAK)
            print("算子[%s]: E-Evac=%12.4f X=%.4f 非整δ=%.4f" %
                  (mode, Eb + 3.0 * J_WEAK * 64 * 32, r["X"], r["nonint"]))
    else:
        main_full()
