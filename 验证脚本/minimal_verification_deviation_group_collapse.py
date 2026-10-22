# -*- coding: utf-8 -*-
"""
minimal_verification_deviation_group_collapse.py
773 号：偏离度的无量纲单组塌缩与机制分辨

起因
  第 771 号把偏离度归给宽度差支，第 772 号否证该归属。随后二维扫描给出
  「dev 对 K 幂约 0.85 至 0.95、对 k 幂约 -2.1」。若两方向幂次同时成立，
  dev 依赖两群组合 K*k^(-2)。该扫描有三处协议缺陷，须先修。

协议缺陷一（速度档）
  k 轴扫描用与 k 无关的绝对速度档。k=4 档相对速度只到 0.2，k=0.25 档初值
  相对速度达 1.6，已超链信号速度。偏离度在窄速度窗内与宽速度窗内不是同
  一个量，跨 k 不可逐位比较。本轮速度档一律以相对速度 beta = v/c_s 定义。

协议缺陷二（探测窗口）
  窗口半径若固定 30 格，而 kink 半宽 ell = 1/(2*pi*sqrt(K/k)) 随参数变化，
  则窗口覆盖的物理内容随档而异，读数混入探测尺度效应。本轮窗口自相似：
  W = NW*ell，NW = 26.65。

协议缺陷三（初始构型未静定）
  第 767 号与第 771 号的静置阶段都无阻尼（767 号的 relax_kink 虽带
  gamma_d 参数，函数体内并未使用）。诊断显示无阻尼 Verlet 的收敛残差在
  1e-4 至 2.4e-1 之间振荡而不降（长寿命驻波模），K=0.02 档质心漂移 4 格，
  即静置后的构型并非静态 kink。本轮静置阶段加阻尼 gamma = 0.05，推进阶段
  保持无阻尼，使推进的物理内容与前轮同族。

理论期望（量纲分析，本轮判据锐度的来源）
  FK 链的参数只有位点势深 K、键刚度 k、格距 a。以 a 为长度单位后 K 与 k
  同为频率平方，无量纲组只有一个 x = K/k。
  kink 半宽以格距计 ell = 1/(2*pi*sqrt(x))，故 (a/ell)^2 = 4*pi^2*x。
  格点能隙（平面波代入给出 omega^2 = 4*k*sin^2(q/2) + 4*pi^2*K 在 q 取零
  处的极限）为 2*pi*sqrt(K)，链信号速度 c_s = sqrt(k)，故质量比
  omega_0/c_s = 2*pi*sqrt(x)，与 (a/ell)^2 只差常数因子，同属一个群。
  因此任何无量纲的偏离度必为 x 的单值函数 dev = F(x)。
  本轮的协议本身也只依赖 x（速度档以相对速度定义，窗口随半宽缩放，行程
  以格距计），故塌缩是量纲结构给出的硬预言。
  凡要求两群组合的机制候选都在固定 x 处预言随 k 变化，可由塌缩否证。
  两群候选取两个极端：
    候选 A（依赖 K 单一参数）：固定 x 时 K 随 k 变，预言偏离度按 k 档间距
    比例分开。
    候选 B（依赖 K*k^(-2)）：固定 x 时 dev 随 1/k 变，预言偏离度同样按 k
    档间距比例分开，但方向相反。
  两者的共同特征是在固定 x 处预言比值为档间距，单群形式预言比值为 1。

对象类型
  比例量与标度指数。可问幅值比、单调性、幂次、塌缩，不可问对易。

判据（先立理论期望，再实现）
  J1 塌缩（主判据）：相同 x 的不同 k 点对上，偏离度在容差内相同。
      若偏离度依赖两群组合，此处必炸开。
  J2 单群幂次：合并可用点拟合 dev ∝ x^p，p 落在 (0.5, 1.5)。
  J3 一次方预言：|p - 1| <= 0.30，即与离散化参数 (a/ell)^2 的一次方预言
      一致（四阶差分类误差的相对量按 (a/ell)^2 增长）。
  J4 两群候选的排除：每个候选至少在两个塌缩组上被测量排除，即候选预言
      比值与实测比值相差 3 倍以上。
  J5 恒真替身：纯相对论数据上偏离度应落在数值零。
  J6 协议稳定性：三组窗口协议（探测半径正负 10%）下，同一档的偏离度读数
      相对极差不超过 20%。协议只改探测方式，不改物理速度窗。
  J7 可用性门槛：可用点总数不少于 6，每个 k 档至少 2 个可用点，各可用点
      满足 v_max/c_s >= 0.30、dev < 0.30、M*c_s^2/E_rest 落在 0.5 至 2.0、
      松弛残差相对量（残差除以剖面跨度 N）不超过 1e-6。
  J8 初始构型变体敏感度：改用加长的静置长度重做，读数相差不超过 5%。

容差预算（从误差预算推导）
  J1 的容差取实测比值落在因子 2 以内（相对极差 100%）。理由：跨协议散布
  在第 767 号为 21.9%，在第 772 号最高 85.5%；取 100% 约为最差协议散布
  的 1.2 倍。
  J3 的容差 0.30 来自幂次拟合的预期标准误（约 0.2）的 1.5 倍。
  J4 的 3 倍取自「候选预言与实测相差 3 倍以上才算排除」这一惯例，与库内
  既有区分力判据同族。
  J6 的 20% 来自「窗口半径改动正负 10% 时探测范围随之改动，偏离度作为窗内
      均方根对其边界敏感度约线性」，故取 10% 的两倍。
  J5 的 1e-6 为纯相对论数据的残差量级。
  J8 的 5% 取自静置变体敏感度在健康档的实测上界（0.31%）的 16 倍余量。
  J7 的 v_max/c_s 与 dev 门槛沿用第 771、772 号；质量门槛改为无量纲的
      M*c_s^2/E_rest，区间取相对论质量参数与静止能量同阶的因子 2 倍余量；
      松弛残差门槛改用相对量 1e-6，因残差本为构型绝对偏离而剖面跨度为 N。

模型
  FK 链，E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos 2*pi*u)]，
  dt = 0.01，N = 1400，缠绕守恒边界。确定性，无随机源。

网格
  以无量纲组 x 与 k 参数化：K = x*k。x 取五档，k 取三档，共 15 点；每个
  x 档在三个 k 上各取一点，故每个塌缩组含三点，k 档间距 16 倍。

跨轮不可比声明
  偏离度读数与第 767、769、771、772 号不可直接比较，因窗口、速度档、静置
  阻尼与参数网格四处协议已改。跨轮只比较量纲结构与幂次形态。

时间尺度依据
  格点能隙 2*pi*sqrt(K) 是链的最低非零频率，与键刚度无关，静置步数按
  sqrt(1/K) 缩放；推进步数按 1/sqrt(k) 缩放，使各档行程同为 600*beta 格。
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\\software\\TianXingFC\\images\\deviation_group_collapse_773.png"

N = 1400
cN = 700
dt = 0.01

XSETS = (0.00125, 0.0025, 0.005, 0.01, 0.02)
KKS = (0.25, 1.0, 4.0)

# 相对速度档（beta = v/c_s）
BETA_SETS = {
    "基准五档": (0.20, 0.35, 0.50, 0.65, 0.80),
    "宽七档": (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95),
    "窄四档": (0.25, 0.40, 0.55, 0.70),
}

NW = 26.65          # 窗口自相似系数，W = NW * ell
GAMMA = 0.05        # 静置阶段阻尼系数

# 探测协议三档：只改窗口半径，速度档集合固定，用于 J6 协议稳定性
NW_PROTOS = (0.90 * NW, NW, 1.10 * NW)

# 静置收敛残差的无量纲门槛（残差除以剖面跨度 N）
CONV_REL_MAX = 1e-6

# 无量纲相对论质量参数 M*c_s^2/E_rest 的可接受区间
MCS2_LO, MCS2_HI = 0.5, 2.0

# 变体敏感度用的速度档（偏离度至少需三点）
SENS_BETAS = (0.35, 0.50, 0.65)

# 窗口协议之间偏离度读数的容许相对极差
NW_SPREAD_MAX = 0.20
REL_REF = 0.02      # 松弛步数的参考 K
NREL_REF = 12000    # 参考点松弛步数
NB0 = 60000         # k=1 的推进步数（行程 600*beta 格）

k = 1.0
K = 0.0
w2 = 0.0
c_s = 1.0
ell = 1.0
Wd = 30


def set_params(kv, Kv, nw=NW):
    """设参数并按半宽自相似地定窗口半径，nw 为窗口自相似系数。"""
    global k, K, w2, c_s, ell, Wd
    k = float(kv)
    K = float(Kv)
    w2 = 2 * np.pi * K
    c_s = np.sqrt(k)
    ell = 1.0 / (2 * np.pi * np.sqrt(K / k))
    Wd = max(10, int(round(nw * ell)))


def accel(u, v):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    return a


def kink_profile(x0):
    n = np.arange(N)
    el = 1.0 / (2 * np.pi * np.sqrt(K / k))
    return n + (2 / np.pi) * np.arctan(np.exp(np.clip((n - x0) / el, -700.0, 700.0)))


def center(u, half=8):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - half), min(len(w), ks + half + 1)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan


def window_energy(u, v, xc, Wd):
    n = np.arange(N)
    ms = np.abs(n - xc) <= Wd
    b = np.arange(N - 1)
    mb = np.abs(b + 0.5 - xc) <= Wd
    E = 0.5 * np.sum(v[ms] ** 2) + np.sum(K * (1 - np.cos(2 * np.pi * u[ms])))
    E += 0.5 * k * np.sum((u[b[mb] + 1] - u[b[mb]] - 1.0) ** 2)
    return E


def relax_kink(nsteps, snap_at=0, gamma=GAMMA):
    """Verlet 静置，返回 (u, v, 快照)。快照取自第 snap_at 步。

    gamma 为每步速度衰减系数，仅作用于静置阶段；推进阶段不加阻尼。
    """
    u = kink_profile(cN * 1.0)
    v = np.zeros(N)
    a = accel(u, v)
    snap = None
    for t in range(nsteps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        if gamma:
            v = v * (1.0 - gamma * dt)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if snap_at and t + 1 == snap_at:
            snap = u.copy()
    return u, np.zeros(N), snap


def run_boost(u_rest, v0, steps, Wd, sample=20):
    """无阻尼推进，返回采样时间、质心位置、窗能量。"""
    u = u_rest.copy()
    v = -v0 * np.gradient(u_rest - np.arange(N))
    v[0] = v[-1] = 0.0
    a = accel(u, v)
    ts, xs, Ews = [], [], []
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0:
            xc = center(u)
            ts.append(t * dt)
            xs.append(xc)
            Ews.append(window_energy(u, v, xc, Wd))
            if not np.isnan(xc) and xc > N - 100:
                break
    return np.array(ts), np.array(xs), np.array(Ews)


def collect_all(u_rest, E_rest_w, betas):
    """对给定 beta 集合各推进一次，返回以 beta 为键的行表。"""
    steps = max(6000, int(NB0 / np.sqrt(k)))
    out = {}
    for b in betas:
        v0 = b * c_s
        ts, xs, Ews = run_boost(u_rest, v0, steps, Wd)
        n1, n2 = len(ts) // 3, 2 * len(ts) // 3
        ok = ~np.isnan(xs[n1:n2])
        if ok.sum() < 5:
            continue
        v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
        Ew_avg = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
        out[b] = (b, v_meas, Ew_avg - E_rest_w)
    return out


def measure(u_rest, betas, nw):
    """按给定窗口协议与速度档测一行表，返回 (M, dev, v_max, E_rest_w)。

    窗口半径随协议改变，故静止窗口能量须按该窗口重算。
    """
    global Wd
    Wd = max(10, int(round(nw * ell)))
    x0 = center(u_rest)
    E_rest_w = window_energy(u_rest, np.zeros(N), x0, Wd)
    rows = collect_all(u_rest, E_rest_w, betas)
    sel = [rows[b] for b in betas if b in rows]
    if len(sel) < 3:
        return float("nan"), float("nan"), float("nan"), E_rest_w
    arr = np.array(sel)
    M, dev, res = deviation(arr[:, 1], arr[:, 2])
    return M, dev, float(arr[:, 1].max()), E_rest_w


def deviation(v, dE):
    """返回 (M, 偏离度 rms 相对量, 相对论拟合相对残差)。

    口径与第 767、771、772 号逐字一致，便于比较判据量本身。
    """
    g1 = c_s ** 2 * (1.0 / np.sqrt(1.0 - (v / c_s) ** 2) - 1.0)
    M = float(np.sum(g1 * dE) / np.sum(g1 ** 2))
    R = dE - M * g1
    dev = float(np.sqrt(np.mean(R ** 2)) / np.mean(np.abs(dE)))
    res = float(np.max(np.abs(R)) / np.max(np.abs(dE)))
    return M, dev, res


def loglin(x, y):
    """对数线性拟合，返回 (斜率, 截距, 残差平方和, 斜率标准误)。"""
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    rss = float(np.sum((ly - pred) ** 2))
    n = len(lx)
    if n > 2:
        s2 = rss / (n - 2)
        sxx = float(np.sum((lx - lx.mean()) ** 2))
        se = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("inf")
    else:
        se = float("inf")
    return float(slope), float(intercept), rss, se


def Mvals_ratio(M, E):
    return M / E if E > 0 else float("inf")


def relax_steps():
    """静置步数：按格点能隙 2*pi*sqrt(K) 的时间尺度 sqrt(1/K) 缩放。"""
    return max(3000, int(NREL_REF * np.sqrt(REL_REF / K)))


def protocol_devs(u_rest, E_rest_w, rows_by_beta, betas):
    """按一组速度档算偏离度，返回 (M, dev, v_max)。"""
    sel = [rows_by_beta[b] for b in betas if b in rows_by_beta]
    if len(sel) < 3:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(sel)
    M, dev, res = deviation(arr[:, 1], arr[:, 2])
    return M, dev, float(arr[:, 1].max())


def main():
    if "--smoke" in sys.argv:
        print("== 冒烟检查 ==")
        set_params(1.0, 0.005)
        nrel = relax_steps()
        u_rest, v_rest, snap = relax_kink(2 * nrel, snap_at=nrel)
        conv = float(np.max(np.abs(u_rest - snap)))
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, Wd)
        rows = collect_all(u_rest, E_rest_w, (0.35, 0.65))
        arr = np.array([rows[b] for b in (0.35, 0.65) if b in rows])
        M, dev, res = deviation(arr[:, 1], arr[:, 2])
        print(f"  静置步数 {nrel}，窗口 W={Wd} 格，半宽 ell={ell:.4f} 格，"
              f"收敛残差 {conv:.3e}")
        print(f"  M={M:.5f} 偏离度={dev * 100:.4f}%")
        for r in arr:
            print(f"    beta={r[0]:.2f} v_meas={r[1]:.4f} dE={r[2]:.6f}")
        v_rel = np.linspace(0.01, 0.60, 5) * c_s
        dE_rel = 0.0935 * c_s ** 2 * (1.0 / np.sqrt(1.0 - (v_rel / c_s) ** 2) - 1.0)
        M2, dev2, res2 = deviation(v_rel, dE_rel)
        print(f"  替身：M={M2:.5f} 偏离度={dev2:.3e}")
        if dev > 1e-5 and dev2 < 1e-6 and conv < 1e-5:
            print("  冒烟通过：数据偏离非零、替身偏离为零、静置已收敛")
        else:
            print("  冒烟未通过")
        return

    print("=" * 78)
    print("773 号：偏离度的无量纲单组塌缩与机制分辨")
    print("=" * 78)
    print(f"x 档 {XSETS}；k 档 {KKS}；相对速度档 {list(BETA_SETS)}")
    print(f"窗口自相似系数 NW={NW}（W = NW * ell）；静置阻尼 gamma={GAMMA}")

    all_betas = sorted({b for bs in BETA_SETS.values() for b in bs})
    base_betas = BETA_SETS["基准五档"]
    table = []
    for kv in KKS:
        for xv in XSETS:
            Kv = xv * kv
            set_params(kv, Kv, NW)
            nrel = relax_steps()
            u_rest, v_rest, snap = relax_kink(2 * nrel, snap_at=nrel)
            conv = float(np.max(np.abs(u_rest - snap)))
            conv_rel = conv / float(N)
            x0 = center(u_rest)
            E_rest_w = window_energy(u_rest, v_rest, x0, Wd)
            rows_by_beta = collect_all(u_rest, E_rest_w, all_betas)
            devs, mvals, vmaxs = {}, {}, {}
            for name, betas in BETA_SETS.items():
                M, dev, vmax = protocol_devs(u_rest, E_rest_w, rows_by_beta, betas)
                devs[name] = dev
                mvals[name] = M
                vmaxs[name] = vmax
            base = devs["基准五档"]
            # 窗口协议三档：同一速度档集合，只改探测窗口半径
            devs_win = []
            for nw in NW_PROTOS:
                if abs(nw - NW) < 1e-12:
                    devs_win.append(base)
                    continue
                _, dv, _, _ = measure(u_rest, base_betas, nw)
                devs_win.append(dv)
            set_params(kv, Kv, NW)
            # 变体敏感度：加长静置长度后重做同一速度档集合
            sens = float("nan")
            nrel2 = int(1.5 * nrel)
            u2, v2, _ = relax_kink(2 * nrel2, snap_at=nrel2)
            _, dev2, _, _ = measure(u2, SENS_BETAS, NW)
            _, devc, _ = protocol_devs(u_rest, E_rest_w, rows_by_beta,
                                       SENS_BETAS)
            if devc > 0 and np.isfinite(dev2):
                sens = float(abs(dev2 - devc) / devc)
            Mcs2 = (mvals["基准五档"] * c_s ** 2 / E_rest_w
                    if E_rest_w > 0 else float("inf"))
            rec = dict(k=kv, K=Kv, x=xv, ell=ell, W=Wd,
                       aoe2=4 * np.pi ** 2 * xv,
                       dev=base, devs=devs, devs_win=devs_win,
                       M=mvals["基准五档"], Mcs2=Mcs2,
                       vmax_over_cs=vmaxs["基准五档"] / c_s,
                       nrel=nrel, conv=conv, conv_rel=conv_rel, sens=sens)
            rec["usable"] = (rec["vmax_over_cs"] >= 0.30 and base < 0.30
                             and MCS2_LO <= Mcs2 <= MCS2_HI
                             and conv_rel <= CONV_REL_MAX)
            table.append(rec)
            print(f"\n  k={kv} K={Kv:.7g}  x=K/k={xv:.6g}  ell={ell:.3f}格  "
                  f"W={Wd}格  (a/ell)^2={rec['aoe2']:.6g}")
            print(f"    偏离度 {base * 100:.4f}%  M*c_s^2/E_rest={rec['Mcs2']:.3f}  "
                  f"v_max/c_s={rec['vmax_over_cs']:.3f}  "
                  f"{'可用' if rec['usable'] else '不可用'}")
            print(f"    静置步数 {nrel}，收敛残差 {conv:.3e}"
                  f"（相对 {conv_rel:.3e}），变体敏感度 {rec['sens'] * 100:.3f}%")
            print(f"    窗口协议读数 {[f'{d * 100:.4f}%' for d in devs_win]}")
            for name in BETA_SETS:
                print(f"      {name}: {devs[name] * 100:.4f}%")

    usable = [r for r in table if r["usable"]]
    print(f"\n可用点：{len(usable)} / {len(table)}")

    # ---------- J7 可用性门槛 ----------
    per_k = {}
    for r in usable:
        per_k.setdefault(r["k"], []).append(r)
    conv_rel_max = max(r["conv_rel"] for r in table)
    j7 = (len(usable) >= 6 and all(len(v) >= 2 for v in per_k.values())
          and conv_rel_max <= CONV_REL_MAX)
    print("\n[J7] 可用性门槛")
    for kv in KKS:
        print(f"  k={kv}: 可用点 {len(per_k.get(kv, []))}")
    print(f"  静置收敛残差相对量最大 {conv_rel_max:.3e}"
          f"（门槛 {CONV_REL_MAX:.0e}）")
    print(f"  可用点总数 {len(usable)}（需 >= 6），每 k 档 >= 2："
          f"{'是' if len(usable) >= 6 and all(len(v) >= 2 for v in per_k.values()) else '否'}")
    print(f"  J7: {'PASS' if j7 else 'FAIL'}")

    # ---------- J1 塌缩 ----------
    print("\n[J1] 单组塌缩：相同 x 的不同 k 点对")
    groups = {}
    for r in usable:
        groups.setdefault(round(r["x"], 10), []).append(r)
    collapse = []
    for xv, rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        vals = np.array([r["dev"] for r in rs])
        spread = float((np.max(vals) - np.min(vals)) / np.median(vals))
        entry = dict(x=xv, rs=rs, spread=spread,
                     ratio=float(np.max(vals) / np.min(vals)))
        collapse.append(entry)
        ks = ", ".join(f"k={r['k']}(K={r['K']:.4g})" for r in rs)
        print(f"  x={xv:.6g}: {ks}")
        print(f"    偏离度 {[f'{r['dev'] * 100:.4f}%' for r in rs]}  "
              f"相对极差 {spread * 100:.3f}%  比值 {entry['ratio']:.4f}")
    j1 = len(collapse) >= 2 and all(c["spread"] < 1.0 for c in collapse)
    print(f"  塌缩组数 {len(collapse)}（需 >= 2），全部相对极差 < 100%："
          f"{'是' if j1 else '否'}")
    print(f"  J1: {'PASS' if j1 else 'FAIL'}")

    # ---------- J2 / J3 单群幂次与一次方预言 ----------
    print("\n[J2][J3] 单群幂次与 (a/ell)^2 一次方预言")
    xs = np.array([r["x"] for r in usable])
    ys = np.array([r["dev"] for r in usable])
    slope, intercept, rss, se = loglin(xs, ys)
    print(f"  合并拟合 dev ∝ x^{slope:+.4f}（标准误 {se:.4f}，残差平方和 {rss:.4g}）")
    j2 = 0.5 < slope < 1.5
    j3 = abs(slope - 1.0) <= 0.30
    print(f"  J2 幂次落在 (0.5, 1.5)：{'是' if j2 else '否'}")
    print(f"  J3 |p-1| <= 0.30（当前 |p-1|={abs(slope - 1.0):.4f}）："
          f"{'是' if j3 else '否'}")
    print("  分档读数（固定 k 对 K 的幂次，单群预言为 +1）：")
    for kv in KKS:
        rs = [r for r in usable if r["k"] == kv]
        if len(rs) >= 3:
            s, ic, r2, se2 = loglin([r["K"] for r in rs], [r["dev"] for r in rs])
            print(f"    k={kv}: K 幂 {s:+.4f}（点 {len(rs)}）")
    print("  分档读数（固定 x 对 k 的幂次，单群预言为 -1）：")
    for c in collapse:
        rs = c["rs"]
        if len(rs) >= 3:
            s, ic, r2, se2 = loglin([r["k"] for r in rs], [r["dev"] for r in rs])
            print(f"    x={c['x']:.6g}: k 幂 {s:+.4f}（点 {len(rs)}）")

    # ---------- J4 两群候选的排除 ----------
    print("\n[J4] 两群候选的排除检验")
    excl = {"候选A(依赖K)": 0, "候选B(依赖K*k^-2)": 0}
    j4 = bool(collapse)
    for c in collapse:
        Ks = [r["K"] for r in c["rs"]]
        ks_ = [r["k"] for r in c["rs"]]
        predA = max(Ks) / min(Ks)
        pairB = []
        for i in range(len(c["rs"])):
            for j in range(len(c["rs"])):
                if i != j:
                    pairB.append((Ks[i] / Ks[j]) * (ks_[i] / ks_[j]) ** (-2))
        predB = max(abs(p) for p in pairB) if pairB else 1.0
        meas = c["ratio"]
        facA = max(predA / meas, meas / predA)
        facB = max(predB / meas, meas / predB)
        if facA >= 3.0:
            excl["候选A(依赖K)"] += 1
        if facB >= 3.0:
            excl["候选B(依赖K*k^-2)"] += 1
        c["predA"], c["predB"], c["facA"], c["facB"] = predA, predB, facA, facB
        print(f"  x={c['x']:.6g}: 实测比值 {meas:.4f}，候选A 预言 {predA:.3f}"
              f"（相差 {facA:.2f} 倍），候选B 预言 {predB:.4f}（相差 {facB:.2f} 倍）")
    for nm, cnt in excl.items():
        print(f"  {nm} 被排除的组数 {cnt}（需 >= 2）")
        if cnt < 2:
            j4 = False
    print(f"  J4: {'PASS' if j4 else 'FAIL'}")

    # ---------- J5 恒真替身 ----------
    print("\n[J5] 恒真替身")
    v_rel = np.linspace(0.01, 0.60, 5) * c_s
    dE_rel = 0.0935 * c_s ** 2 * (1.0 / np.sqrt(1.0 - (v_rel / c_s) ** 2) - 1.0)
    M2, dev2, res2 = deviation(v_rel, dE_rel)
    print(f"  替身：M={M2:.5f} 偏离度={dev2:.3e} 相对残差={res2:.3e}")
    j5 = dev2 < 1e-6
    print(f"  替身偏离度 < 1e-6：{'是' if j5 else '否'}")
    print(f"  J5: {'PASS' if j5 else 'FAIL'}")

    # ---------- J6 协议稳定性 ----------
    print("\n[J6] 协议稳定性（窗口协议三档，同一速度档集合）")
    spreads = []
    for r in usable:
        d = np.array(r["devs_win"], dtype=float)
        if np.all(np.isfinite(d)) and np.median(d) > 0:
            sp = float((np.max(d) - np.min(d)) / np.median(d))
            spreads.append(sp)
            print(f"  k={r['k']} x={r['x']:.6g}: 窗口协议读数 "
                  f"{[f'{v * 100:.4f}%' for v in d]} 相对极差 {sp * 100:.3f}%")
    if spreads:
        spmax = float(np.max(spreads))
        j6 = spmax <= NW_SPREAD_MAX
        print(f"  最大相对极差 {spmax * 100:.3f}%"
              f"（门槛 {NW_SPREAD_MAX * 100:.0f}%）：{'是' if j6 else '否'}")
    else:
        j6 = False
        print("  无窗口协议读数，判否")
    print(f"  J6: {'PASS' if j6 else 'FAIL'}")

    # ---------- J8 变体敏感度 ----------
    print("\n[J8] 初始构型变体敏感度")
    sens = [r["sens"] for r in usable if np.isfinite(r["sens"])]
    if sens:
        smax = float(np.max(sens))
        j8 = smax <= 0.05
        print(f"  可用档 {len(sens)} 个，最大敏感度 {smax * 100:.3f}%"
              f"（门槛 5%）：{'是' if j8 else '否'}")
    else:
        j8 = False
        print("  无变体读数，判否")
    print(f"  J8: {'PASS' if j8 else 'FAIL'}")

    # ---------- 汇总 ----------
    print("\n[判据汇总]")
    verdicts = [("J1 单组塌缩", j1), ("J2 单群幂次", j2),
                ("J3 一次方预言", j3), ("J4 两群候选排除", j4),
                ("J5 恒真替身", j5), ("J6 协议稳定性", j6),
                ("J7 可用性门槛", j7), ("J8 变体敏感度", j8)]
    for name, ok in verdicts:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  总判据: {sum(1 for _, ok in verdicts if ok)}/{len(verdicts)}")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    colors = {0.25: "#1f77b4", 1.0: "#d62728", 4.0: "#2ca02c"}
    ax = axes[0]
    for kv in KKS:
        rs = [r for r in table if r["k"] == kv]
        ax.plot([r["x"] for r in rs], [r["dev"] * 100 for r in rs],
                "o-", color=colors[kv], label=f"k={kv}")
        ax.plot([r["x"] for r in rs],
                [r["devs"]["窄四档"] * 100 for r in rs],
                ".", color=colors[kv], alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x=K/k$")
    ax.set_ylabel("偏离度 rms（%）")
    ax.set_title("(a) 偏离度对无量纲组 x")
    ax.legend(fontsize=8)

    ax = axes[1]
    for kv in KKS:
        rs = [r for r in usable if r["k"] == kv]
        ax.plot([r["x"] for r in rs], [r["dev"] * 100 for r in rs],
                "o", color=colors[kv], label=f"k={kv}")
    for r in usable:
        for d in r["devs_win"]:
            ax.plot([r["x"]], [d * 100], "x", color="0.45", markersize=4)
    if len(usable) >= 3:
        lx = np.log(xs)
        xx = np.linspace(lx.min(), lx.max(), 40)
        ax.plot(np.exp(xx), np.exp(intercept + slope * xx) * 100, "k--",
                label=f"单群拟合 幂次 {slope:+.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x=K/k$")
    ax.set_ylabel("偏离度 rms（%）")
    ax.set_title("(b) 单群塌缩与幂次")
    ax.legend(fontsize=8)

    ax = axes[2]
    if collapse:
        labels = [f"{c['x']:.4g}" for c in collapse]
        meas = [c["ratio"] for c in collapse]
        predA = [c["predA"] for c in collapse]
        predB = [c["predB"] for c in collapse]
        xi = np.arange(len(labels))
        ax.bar(xi - 0.26, meas, width=0.25, label="实测比值")
        ax.bar(xi, predA, width=0.25, label="候选A 预言（依赖K）")
        ax.bar(xi + 0.26, predB, width=0.25, label="候选B 预言（依赖K*k^-2）")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xticks(xi)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("塌缩组（横轴为 x=K/k）")
        ax.set_ylabel("偏离度比值（最大/最小）")
        ax.set_title("(c) 塌缩与两群形式的分辨")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG, dpi=150)
    print(f"\n图已存: {IMG}")


if __name__ == "__main__":
    main()
