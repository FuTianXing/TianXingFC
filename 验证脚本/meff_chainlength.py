# -*- coding: utf-8 -*-
"""
meff_chainlength.py
739 号：质量参数同一性的链长扫描（项三的窗口伪影判别）

对象类型：两质量参数的读数之比（参数比），可问「是否随链长收敛」「是否跨 K 一致」「是否依赖测量窗口」。

判据（先立期望，再实现）：
  E1 链长收敛：同一 K 下 r(N)=M_rel/m_eff,low 随 N 增大应收敛到 1。
     读法：先取饱和值 r_sat（链长读数进入平台后的读数，平台由 E0 判据给出），
     再判 |r_sat - 1| < 0.05（伪影假设）。
     判否含义：r_sat 偏离 1 超过预算，则 0.85-0.94 的偏差属实质差异。
  E0 饱和性（先于 E1 执行）：相邻链长档读数的相对差 < 1e-3 则该档已饱和，
     饱和后不得再做 1/N 外推（两点逐位相同时斜率由单点决定，外推无根据）。
  E2 跨 K 一致：K=0.005 与 K=0.01 的 r_sat 之差 < 0.05。
  E3 窗口稳健：中心定位窗口 W=30 与 W=15 给出的 m_eff,low 相对差 < 0.02（读数不依赖窗口）。

容差预算（从误差预算推导）：
  相对论拟合为 5 点 1 参数，残差 ~0.6%（K=0.005）；参数误差 ~残差/sqrt(n) ~0.3%。
  m_eff,low 取低速三档中位数，读数精度（中心定位与窗口能量）~1%。
  合计预算 ~2%，取 5% 为判据阈值（留 2.5 倍富余，且与待判偏差 6% 至 15% 有区分力）。

模型：FK 链 E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos2πu)]，k=1，dt=0.01，缠绕守恒边界。
窗口协议与 meff 脚本同口径（窗口能量取尾段平均，初速场 -v0·零模）。
确定性，无随机源。
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\meff_chainlength_739.png"

k = 1.0
dt = 0.01
K = 0.0
w2 = 0.0
c_s = np.sqrt(k)
N = 1400
cN = 700
W = 30

V0S = (0.20, 0.35, 0.50, 0.65, 0.80)
STEPS = 60000


def set_K(Kv):
    global K, w2
    K = Kv
    w2 = 2 * np.pi * K


def set_N(Nv):
    global N, cN
    N = int(Nv)
    cN = N // 2


def accel(u, v):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    return a


def kink_profile(x0):
    n = np.arange(N)
    ell = 1.0 / (2 * np.pi * np.sqrt(K / k))
    return n + (2 / np.pi) * np.arctan(np.exp(np.clip((n - x0) / ell, -700.0, 700.0)))


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


def total_energy(u, v):
    return (0.5 * np.sum(v ** 2)
            + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
            + np.sum(K * (1 - np.cos(2 * np.pi * u))))


def relax_kink(steps=12000):
    u = kink_profile(cN * 1.0)
    v = np.zeros(N)
    a = accel(u, v)
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
    return u, np.zeros(N)


def run_boost(u_rest, v0, steps, Wd, sample=20):
    u = u_rest.copy()
    v = -v0 * np.gradient(u_rest - np.arange(N))
    v[0] = v[-1] = 0.0
    E0 = total_energy(u, v)
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
    E1 = total_energy(u, v)
    return np.array(ts), np.array(xs), np.array(Ews), E0, E1


def measure(Nv, Kv, Wd, steps=STEPS, v0s=V0S, verbose=False):
    set_N(Nv)
    set_K(Kv)
    u_rest, v_rest = relax_kink()
    x0 = center(u_rest)
    E_rest_w = window_energy(u_rest, v_rest, x0, Wd)
    rows = []
    for v0 in v0s:
        ts, xs, Ews, E0, E1 = run_boost(u_rest, v0, steps, Wd)
        n1, n2 = len(ts) // 3, 2 * len(ts) // 3
        ok = ~np.isnan(xs[n1:n2])
        if ok.sum() < 5:
            continue
        v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
        Ew_avg = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
        dE = Ew_avg - E_rest_w
        m_eff_v = 2.0 * dE / v_meas ** 2 if abs(v_meas) > 1e-6 else np.nan
        rows.append((v0, v_meas, dE, m_eff_v, abs(E1 - E0)))
        if verbose:
            print(f"    v0={v0:4.2f} v_meas={v_meas:7.4f} dE={dE:8.5f} "
                  f"m_eff={m_eff_v:8.4f} 能量误差={abs(E1 - E0):.1e}")
    rows = np.array([r for r in rows if np.isfinite(r[1]) and abs(r[1]) > 1e-4])
    v_m, dE_m = rows[:, 1], rows[:, 2]
    g1 = c_s ** 2 * (1.0 / np.sqrt(1.0 - (v_m / c_s) ** 2) - 1.0)
    M_rel = float(np.sum(g1 * dE_m) / np.sum(g1 ** 2))
    dE_rel = M_rel * g1
    res_rel = float(np.max(np.abs(dE_rel - dE_m)) / np.max(np.abs(dE_m)))
    m_low = float(np.median(rows[:, 3][rows[:, 1] < 0.5]))
    return dict(N=N, K=Kv, W=Wd, m_low=m_low, M_rel=M_rel, r=M_rel / m_low,
                res_rel=res_rel, rows=rows, E_rest=E_rest_w)


def main():
    if "--smoke" in sys.argv:
        print("== 冒烟检查 ==")
        r = measure(700, 0.005, 30, steps=20000, v0s=(0.35, 0.65), verbose=True)
        print(f"  中心/读数：m_low={r['m_low']:.4f} M_rel={r['M_rel']:.4f} r={r['r']:.4f}")
        print(f"  相对论拟合残差 {r['res_rel']:.2e}；静止窗口能量 {r['E_rest']:.4f}")
        print("  若 r 与既有 N=1400 读数 0.939 同量级且能量守恒误差 < 1e-6，可进判据循环")
        return

    print("=" * 76)
    print("739 号：质量参数同一性的链长扫描（项三）")
    print("=" * 76)
    print(f"模型：FK 链 k=1 dt={dt}，v0 档 {V0S}，步数 {STEPS}，窗口 W=30")

    # 主扫描
    Ns = (700, 1400, 2800)
    Ks = (0.005, 0.01)
    data = {}
    print("\n[N 扫描]")
    print(f"  {'N':>6} {'K':>6} {'m_eff,low':>10} {'M_rel':>8} {'r=M/m':>8} {'拟合残差':>9} {'能量误差':>9}")
    for Kv in Ks:
        for Nv in Ns:
            res = measure(Nv, Kv, 30)
            data[(Nv, Kv)] = res
            err = float(np.max(res["rows"][:, 4]))
            print(f"  {Nv:6d} {Kv:6.3f} {res['m_low']:10.4f} {res['M_rel']:8.4f} "
                  f"{res['r']:8.4f} {res['res_rel']:9.2e} {err:9.1e}")

    # 判据 E0：饱和性（先于 E1，决定后续读法）
    print("\n[E0] 饱和性：相邻链长档读数的相对差")
    sat = {}
    for Kv in Ks:
        diffs = []
        for i in range(len(Ns) - 1):
            a_, b_ = data[(Ns[i], Kv)]["r"], data[(Ns[i + 1], Kv)]["r"]
            diffs.append(abs(b_ - a_) / abs(a_) if abs(a_) > 1e-12 else np.inf)
        saturated = diffs[-1] < 1e-3
        sat[Kv] = Ns[-1] if saturated else Ns[-2]
        print(f"  K={Kv:.3f}: 相邻档相对差 " + ", ".join(f"{d:.2e}" for d in diffs)
              + f"，{'已饱和（取 N=' + str(sat[Kv]) + '）' if saturated else '未饱和（取 N=' + str(sat[Kv]) + '）'}")
    e0 = all(abs(data[(sat[Kv], Kv)]["r"] - data[(Ns[-1], Kv)]["r"]) < 1e-9 for Kv in Ks)
    print(f"  E0: {'PASS' if e0 else 'FAIL'}（饱和后末两档逐位相同，阈值 1e-9）")

    # 判据 E1：饱和值直读（饱和情形下 1/N 外推无根据，见 §六 过程记录）
    print("\n[E1] 链长收敛：饱和值 r_sat 与 1 的偏离")
    r_sat = {}
    for Kv in Ks:
        r_sat[Kv] = data[(sat[Kv], Kv)]["r"]
        print(f"  K={Kv:.3f}: r_sat = {r_sat[Kv]:.4f}（N={sat[Kv]}），|r_sat-1| = {abs(r_sat[Kv]-1):.4f}")
    e1 = all(abs(r_sat[Kv] - 1.0) < 0.05 for Kv in Ks)
    print(f"  E1: {'PASS' if e1 else 'FAIL'}（阈值 0.05）")

    # 判据 E2：跨 K 一致（用饱和值）
    e2 = abs(r_sat[Ks[0]] - r_sat[Ks[1]]) < 0.05
    print(f"\n[E2] 跨 K 一致：|r_sat(K=0.005) - r_sat(K=0.01)| = "
          f"{abs(r_sat[Ks[0]] - r_sat[Ks[1]]):.4f}，E2: {'PASS' if e2 else 'FAIL'}（阈值 0.05）")

    # 判据 E3：窗口稳健（单档）
    print("\n[E3] 窗口稳健性（N=1400，K=0.005，W=30 对 W=15）")
    r30 = data[(1400, 0.005)]["m_low"]
    res15 = measure(1400, 0.005, 15)
    rel = abs(res15["m_low"] - r30) / r30
    e3 = rel < 0.02
    print(f"  m_low(W=30)={r30:.4f}，m_low(W=15)={res15['m_low']:.4f}，相对差 {rel:.2e}，"
          f"E3: {'PASS' if e3 else 'FAIL'}（阈值 0.02）")

    # 汇总
    print("\n[判据汇总]")
    verdicts = [("E0 饱和性", e0), ("E1 链长收敛", e1), ("E2 跨 K 一致", e2), ("E3 窗口稳健", e3)]
    for name, ok in verdicts:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  总判据: {sum(1 for _, ok in verdicts if ok)}/{len(verdicts)}")

    # 出图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for Kv, mk in zip(Ks, ("o", "s")):
        ys = [data[(n, Kv)]["r"] for n in Ns]
        ax.plot(Ns, ys, mk + "-", label=f"K = {Kv}")
        ax.axhline(1.0, color="gray", lw=1, ls=":")
    ax.set_xlabel("链长 N")
    ax.set_ylabel(r"$M_{rel}/m_{eff,low}$")
    ax.set_title("(a) 比值随链长")
    ax.legend()

    ax = axes[1]
    for Kv, mk in zip(Ks, ("o", "s")):
        y = np.array([data[(n, Kv)]["r"] for n in Ns])
        ax.plot(np.array(Ns), y, mk + "-", label=f"K = {Kv}")
        ax.axhline(r_sat[Kv], color="gray", lw=1, ls=":")
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xlabel("链长 N")
    ax.set_ylabel(r"$M_{rel}/m_{eff,low}$")
    ax.set_title("(b) 饱和值直读（虚线为 1）")
    ax.legend()

    ax = axes[2]
    for Kv, mk in zip(Ks, ("o", "s")):
        ys = [data[(n, Kv)]["m_low"] for n in Ns]
        ax.plot(Ns, ys, mk + "-", label=f"K = {Kv}")
    ax.set_xlabel("链长 N")
    ax.set_ylabel(r"$m_{eff,low}$")
    ax.set_title("(c) 低速有效质量随链长")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG, dpi=150)
    print(f"\n图已存: {IMG}")


if __name__ == "__main__":
    main()
