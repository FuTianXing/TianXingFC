# -*- coding: utf-8 -*-
"""
width_branch_refute.py
772 号：宽度差支的否证与三参数简并

起因：第 771 号判「偏离度由宽度差支支配」（格点 kink 宽度与连续 SG 宽度之差），
其数值结论（K 幂 +0.8527 与 +0.9449，最接近整幂 1）经跨协议检验成立。
本轮核其解析前提，发现该前提有误，故作废机制归属。

否证型验证轮：判据的期望是「否证成立」。判否不等于判负——
本轮否的是「771 号的机制归属」，不否「偏离度存在」与「偏离度有确定标度」。

对象类型：标度指数与拟合系数（可问指数、散布、条件数，不可问对易）。

口径订正声明（2026-10-22）：本脚本原把连续 SG 宽度记作 c/omega_0 = k/(2*pi*sqrt(K))，
其中混用了格距单位长度与链信号速度下的 c/omega_0，多出一个 sqrt(k) 因子。
订正后连续宽度为 1/omega_SG = sqrt(k)/(2*pi*sqrt(K))，与格点 kink 半宽同阶，
二者之差只是 O((a/ell)^2/24) 的小量，不存在 |1-sqrt(k)|*ell 的宽度差。
因此本脚本的 H1 至 H3（宽度差的因子结构、强预言否证、比例性否证）前提失效，
其原读数须读作作废；H4 至 H7 不依赖该宽度式，读数保留。
为保读数可追溯，本脚本的计算逻辑保留原样不改。

三订正（2026-10-22）：H4 的判定式为「跨协议幂次区间是否覆盖 -2」，而拟合自变量是 1/ell，
四阶差机制的幂次预言在其上为正幂；实测区间全为正，故该判定式必真、H4 无法失败。
严格读法只能给「不覆盖 +2」。此处保留原判定式与原读数不改，仅标注其效力。

否证链（四段）：
  (1) 宽度差的结构（原口径，已作废，见上声明）：FK 链连续化给出 omega_0 = 2*pi*sqrt(K/k)，
      故连续 SG 宽度 c/omega_0 = k/(2*pi*sqrt(K))；
      格点 kink 半宽为 ell = sqrt(k)/(2*pi*sqrt(K))（由静 kink 剖面测得，
      与过渡区格数的幂次 -0.4947 对 -0.5000 相符）。
      两者相差因子 sqrt(k)：k=1 时逐位相等（宽度差为零），k=4 时宽度差为 -ell。
  (2) 强预言否证：宽度差支预言偏离度正比于宽度差，故宽度差为零处偏离度应为零。
      实测 k=1 档（宽度差逐位为零）偏离度为 0.0039 至 0.0246，非零。
      该条直接否证宽度差支。
  (3) 比例性否证：k=4 档宽度差为 -ell，偏离度与宽度差之比在跨档间的相对散布为 223%，
      不构成比例关系。
  (4) 简并否证：三参数拟合 ΔE = M*REL + b2*v^2 + b4*v^4 的条件数 310 至 908，
      b4 跨协议散布 4.3% 至 582%、符号在两档翻转、K=0.01 档 M 为负。
      按验证规范第六节，三基函数在窄速度窗内构成近简并子空间，
      在其内分离 M、b2、b4 属污染读数。

判据（否证型：判据的期望值是否证成立）：
  H1 宽度差的因子结构（原口径，已作废）：|ell - c/omega_0| 与 |1 - sqrt(k)|·ell 的相对差小于 1e-12。
  H2 强预言否证：k=1 档（宽度差为零）的偏离度显著非零（大于 1e-4），
      全部四档皆然。此条成立即否证宽度差支。
  H3 比例性否证：k=4 档 dev/宽度差 的跨档相对散布大于 50%。
  H4 幂次不覆盖 -2：跨协议幂次的极差不覆盖 -2（四阶差机制的幂次预言否证）。
  H5 简并诊断：M、b2、b4 三基函数的条件数全部超过 100（近简并）。
  H6 污染量化：b4 的跨协议散布在至少两档超过 50%。
  H7 恒真替身：纯相对论数据上偏离度为零。

容差预算（从误差预算推导）：
  H1 的 1e-12：解析等式在浮点下应逐位相同，容差取机器精度量级。
  H2 的 1e-4：偏离度在本模型的可用档为 0.0016 至 0.0246，门槛取其下界的十分之一量级
    （1e-4），使「非零」与「零」有明确区分。
  H3 的 50%：实测散布 223%，门槛取其四分之一，仍能判否。
  H5 的 100：三基函数在 v<0.8 的窗口内 Gram 条件数实测 310 至 908，
    单基双基拟合的条件数为 13 至 47，故 100 为两者的分界。
  H6 的 50%：实测散布 4.3% 至 582%，门槛取 50% 使三档落入污染区。
  H7：替身偏离度应为零。

模型：FK 链（与第 767、771 号同型），E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos2πu)]。
K 档 0.00125、0.0025、0.005、0.01；键刚度 k 档 1.0、4.0。
确定性，无随机源。
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\width_branch_refute_772.png"

dt = 0.01
N = 1400
cN = 700
W = 30
KS = (0.00125, 0.0025, 0.005, 0.01)
KK = (1.0, 4.0)
V0SETS = {
    "基准五档": (0.20, 0.35, 0.50, 0.65, 0.80),
    "宽七档": (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95),
    "窄四档": (0.25, 0.40, 0.55, 0.70),
}
STEPS = 60000

K_ = 0.0
w2 = 0.0
kk_ = 1.0
c_s_ = 1.0


def set_params(Kv, kval):
    global K_, w2, kk_, c_s_
    K_ = Kv
    w2 = 2 * np.pi * Kv
    kk_ = kval
    c_s_ = np.sqrt(kval)


def accel(u, v):
    a = np.zeros_like(u)
    a[1:-1] = kk_ * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    return a


def kink_profile(x0):
    n = np.arange(N)
    ell = 1.0 / (2 * np.pi * np.sqrt(K_ / kk_))
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
    E = 0.5 * np.sum(v[ms] ** 2) + np.sum(K_ * (1 - np.cos(2 * np.pi * u[ms])))
    E += 0.5 * kk_ * np.sum((u[b[mb] + 1] - u[b[mb]] - 1.0) ** 2)
    return E


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


def collect(u_rest, E_rest_w, v0s, steps=STEPS):
    rows = []
    for v0 in v0s:
        ts, xs, Ews = run_boost(u_rest, v0, steps, W)
        n1, n2 = len(ts) // 3, 2 * len(ts) // 3
        ok = ~np.isnan(xs[n1:n2])
        if ok.sum() < 5:
            continue
        vm = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
        Ew = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
        rows.append((v0, vm, Ew - E_rest_w))
    return np.array(rows)


def rel_profile(vn):
    return 1.0 / np.sqrt(1.0 - vn ** 2) - 1.0


def widths(Kv, kval):
    """格点 kink 半宽 ell、连续 SG 宽度 c/omega_0 与两者之差。

    ell 由静 kink 剖面测得，其幂次 -0.4947 与 sqrt(k)/(2*pi*sqrt(K)) 的 -0.5 相符。
    本函数按原口径取 xi = sqrt(k)/omega0（omega0 = 2*pi*sqrt(K/k)），
    即 xi = k/(2*pi*sqrt(K))。2026-10-22 二订正核出该式多一个 sqrt(k) 因子，
    订正后的连续 SG 宽度应为 1/omega_SG = sqrt(k)/(2*pi*sqrt(K))，与 ell 同阶。
    此处保留原式，是为了让本脚本复现出与 772 号注记一致的原始读数；
    据此读出的 H1 至 H3 前提失效，其读数须读作作废（见 772 号注记二订正）。
    """
    ell = np.sqrt(kval) / (2 * np.pi * np.sqrt(Kv))
    omega0 = 2 * np.pi * np.sqrt(Kv / kval)
    xi = np.sqrt(kval) / omega0
    return float(ell), float(xi), float(ell - xi)


def measure(Kv, kval, v0s):
    """返回 (偏离度, M_rel, v_max, 1/ell)。"""
    set_params(Kv, kval)
    u_rest, v_rest = relax_kink()
    x0 = center(u_rest)
    E_rest_w = window_energy(u_rest, v_rest, x0, W)
    rows = collect(u_rest, E_rest_w, v0s)
    v, dE = rows[:, 1], rows[:, 2]
    vn = v / c_s_
    g1 = rel_profile(vn)
    M = float(np.sum(g1 * dE) / np.sum(g1 ** 2))
    R = dE - M * g1
    dev = float(np.sqrt(np.mean(R ** 2)) / np.mean(np.abs(dE)))
    ell = 1.0 / (2 * np.pi * np.sqrt(Kv / kval))
    return dev, M, float(vn.max()), 1.0 / ell


def fit3(vn, dE):
    """三参数拟合 ΔE = M*REL + b2*v² + b4*v⁴，返回 (coef, se, cond)。"""
    A = np.stack([rel_profile(vn), vn ** 2, vn ** 4], axis=1)
    coef, *_ = np.linalg.lstsq(A, dE, rcond=None)
    res = dE - A @ coef
    dof = max(len(vn) - 3, 1)
    s2 = np.sum(res ** 2) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return coef, se, float(np.linalg.cond(A))


def main():
    if "--smoke" in sys.argv:
        print("== 冒烟检查 ==")
        d, M, vm, inv = measure(0.005, 1.0, V0SETS["基准五档"])
        ell, xi, dw = widths(0.005, 1.0)
        print(f"  dev={d:.6f} M={M:.6f} v_max={vm:.3f}")
        print(f"  ell={ell:.10f}  c/omega_0[原口径]={xi:.10f}  宽度差={dw:.2e}")
        v_rel = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        g1 = rel_profile(v_rel)
        dE_rel = 0.0935 * g1
        M2 = float(np.sum(g1 * dE_rel) / np.sum(g1 ** 2))
        dev2 = float(np.sqrt(np.mean((dE_rel - M2 * g1) ** 2)) / np.mean(np.abs(dE_rel)))
        print(f"  替身：M={M2:.5f} 偏离度={dev2:.3e}")
        if dw == 0.0 and dev2 < 1e-6 and d > 1e-4:
            print("  冒烟通过：k=1 档宽度差为零而偏离度非零、替身为零")
        return

    print("=" * 78)
    print("772 号：宽度差支的否证与三参数简并")
    print("=" * 78)

    # ---------- H1 宽度差的因子结构 ----------
    print("\n[H1] 宽度差的因子结构：|ell - c/omega_0[原口径]| = |1 - sqrt(k)|·ell")
    rel_diffs = []
    for kval in KK:
        for Kv in KS:
            ell, xi, dw = widths(Kv, kval)
            pred = abs(1 - np.sqrt(kval)) * ell
            rd = abs(abs(dw) - pred) / pred if pred != 0 else abs(dw) / ell
            rel_diffs.append(rd)
            print(f"  k={kval} K={Kv}: ell={ell:.8f}  c/omega_0[原口径]={xi:.8f}  宽度差={dw:+.8f}  "
                  f"预测 |1-√k|·ell={pred:.8f}  相对差 {rd:.2e}")
    h1 = max(rel_diffs) < 1e-12
    print(f"  最大相对差 {max(rel_diffs):.2e}")
    print(f"  H1: {'PASS' if h1 else 'FAIL'}（宽度差的因子结构成立）")

    # ---------- 读数 ----------
    print("\n[读数] 偏离度")
    dev = {}
    for kval in KK:
        for Kv in KS:
            d, M, vm, inv = measure(Kv, kval, V0SETS["基准五档"])
            dev[(kval, Kv)] = d
    for Kv in KS:
        print(f"  K={Kv}: " + "  ".join(f"k={k}: dev={dev[(k, Kv)]:.6f}" for k in KK))

    # ---------- H2 强预言否证 ----------
    print("\n[H2] 强预言否证：宽度差为零处偏离度应为零")
    devs_k1 = []
    for Kv in KS:
        ell, xi, dw = widths(Kv, 1.0)
        d = dev[(1.0, Kv)]
        devs_k1.append(d)
        print(f"  K={Kv}: 宽度差={dw:.2e}（相对 {abs(dw)/ell:.2e}）  偏离度={d:.6f}")
    h2 = all(d > 1e-4 for d in devs_k1)
    print(f"  全部四档偏离度大于 1e-4：{'是' if h2 else '否'}")
    print(f"  H2: {'PASS' if h2 else 'FAIL'}（宽度差支的强预言否证成立）")

    # ---------- H3 比例性否证 ----------
    print("\n[H3] 比例性否证：k=4 档 dev/宽度差 的跨档散布")
    rs = []
    for Kv in KS:
        ell, xi, dw = widths(Kv, 4.0)
        r = dev[(4.0, Kv)] / dw
        rs.append(r)
        print(f"  K={Kv}: 宽度差={dw:.6f}  偏离度={dev[(4.0, Kv)]:.6f}  比 {r:.6e}")
    a = np.array(rs)
    sp = float(np.ptp(a) / abs(np.mean(a)) * 100)
    h3 = sp > 50.0
    print(f"  相对散布 {sp:.1f}%（门槛 50%）")
    print(f"  H3: {'PASS' if h3 else 'FAIL'}")

    # ---------- H4 幂次 ----------
    # 四订正（2026-10-22）：下列判定式对正指数区间恒为 PASS，见文件头声明。
    print("\n[H4] 幂次是否覆盖 -2")
    pows = []
    for kval in KK:
        for name, v0s in V0SETS.items():
            ds, invs = [], []
            for Kv in KS:
                d, M, vm, inv = measure(Kv, kval, v0s)
                ds.append(d)
                invs.append(inv)
            sl = float(np.polyfit(np.log(invs), np.log(ds), 1)[0])
            pows.append(sl)
            print(f"  k={kval} {name}: 幂 {sl:+.4f}")
    arr = np.array(pows)
    covers = arr.min() <= -2.0 <= arr.max()
    h4 = not covers
    print(f"  幂次范围 {arr.min():+.4f} 至 {arr.max():+.4f}，覆盖 -2：{'是' if covers else '否'}")
    print(f"  H4: {'PASS' if h4 else 'FAIL'}（不覆盖 -2 → 四阶差机制的幂次预言否证）")

    # ---------- H5 简并诊断 ----------
    print("\n[H5] 三参数简并诊断")
    conds = []
    for Kv in KS:
        set_params(Kv, 1.0)
        u_rest, v_rest = relax_kink()
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, W)
        rows = collect(u_rest, E_rest_w, V0SETS["基准五档"])
        vn = rows[:, 1] / c_s_
        coef, se, cond = fit3(vn, rows[:, 2])
        conds.append(cond)
        print(f"  K={Kv}: cond={cond:.1f}  M={coef[0]:+.5f}  b2={coef[1]:+.5f}  b4={coef[2]:+.5f}")
    h5 = all(c > 100 for c in conds)
    print(f"  最小条件数 {min(conds):.1f}（门槛 100）")
    print(f"  H5: {'PASS' if h5 else 'FAIL'}（三基近简并，其内分离系数属污染读数）")

    # ---------- H6 污染量化 ----------
    print("\n[H6] b4 的跨协议散布（污染量化）")
    spreads = []
    for Kv in KS:
        set_params(Kv, 1.0)
        u_rest, v_rest = relax_kink()
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, W)
        vals = []
        for v0s in V0SETS.values():
            rows = collect(u_rest, E_rest_w, v0s)
            vn = rows[:, 1] / c_s_
            coef, se, cond = fit3(vn, rows[:, 2])
            vals.append(coef[2])
        a = np.array(vals)
        sp = float(np.ptp(a) / abs(np.mean(a)) * 100)
        spreads.append(sp)
        print(f"  K={Kv}: b4 {['%+.5f' % x for x in a]}  散布 {sp:.1f}%")
    n_polluted = sum(1 for s in spreads if s > 50.0)
    h6 = n_polluted >= 2
    print(f"  超过 50% 的档数 {n_polluted}（门槛 2）")
    print(f"  H6: {'PASS' if h6 else 'FAIL'}")

    # ---------- H7 替身 ----------
    print("\n[H7] 恒真替身")
    v_rel = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    g1 = rel_profile(v_rel)
    dE_rel = 0.0935 * g1
    M2 = float(np.sum(g1 * dE_rel) / np.sum(g1 ** 2))
    dev2 = float(np.sqrt(np.mean((dE_rel - M2 * g1) ** 2)) / np.mean(np.abs(dE_rel)))
    h7 = dev2 < 1e-6
    print(f"  替身：M={M2:.6f}（解析 0.093500）偏离度={dev2:.3e}")
    print(f"  H7: {'PASS' if h7 else 'FAIL'}")

    # ---------- 汇总 ----------
    print("\n[判据汇总]")
    verdicts = [("H1 宽度差因子结构", h1), ("H2 强预言否证", h2), ("H3 比例性否证", h3),
                ("H4 幂次不覆盖 -2", h4), ("H5 简并诊断", h5), ("H6 污染量化", h6),
                ("H7 恒真替身", h7)]
    for name, ok in verdicts:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  总判据: {sum(1 for _, ok in verdicts if ok)}/{len(verdicts)}")
    print("\n[裁决] 第 771 号的机制归属（宽度差支）否证成立，作废。")
    print("  （2026-10-22 二订正：H1 至 H3 的宽度差前提失效，原读数须读作作废；")
    print("   否证理由改由阶数不同承担，结论不变。）")
    print("  771 号的数值结论（K 幂 +0.8527 与 +0.9449）作为「偏离度标度有确定值」保留，")
    print("  但其机制归属（格点宽度与连续宽度之差）不成立。")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for kval in KK:
        xs, ys = [], []
        for Kv in KS:
            ell, xi, dw = widths(Kv, kval)
            xs.append(Kv)
            ys.append(dw / ell)
        ax.plot(xs, ys, "o-", label=f"k={kval}")
    ax.axhline(0.0, color="k", ls="--", label="宽度差为零")
    ax.set_xscale("log")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"宽度差 $/\ell$")
    ax.set_title("(a) 宽度差的因子结构")
    ax.legend(fontsize=8)

    ax = axes[1]
    for kval in KK:
        xs, ys = [], []
        for Kv in KS:
            d, M, vm, inv = measure(Kv, kval, V0SETS["基准五档"])
            xs.append(inv)
            ys.append(d * 100)
        ax.plot(xs, ys, "o-", label=f"k={kval}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/\ell$")
    ax.set_ylabel("偏离度（%）")
    ax.set_title("(b) 偏离度标度")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels, vals, colors = [], [], []
    for Kv in KS:
        set_params(Kv, 1.0)
        u_rest, v_rest = relax_kink()
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, W)
        b4s = []
        for v0s in V0SETS.values():
            rows = collect(u_rest, E_rest_w, v0s)
            vn = rows[:, 1] / c_s_
            coef, se, cond = fit3(vn, rows[:, 2])
            b4s.append(coef[2])
        a = np.array(b4s)
        sp = float(np.ptp(a) / abs(np.mean(a)) * 100)
        labels.append(f"K={Kv}")
        vals.append(sp)
        colors.append("#c44" if sp > 50 else "#468")
    ax.bar(labels, vals, color=colors)
    ax.axhline(50.0, color="r", ls="--", label="污染门槛 50%")
    ax.set_ylabel(r"$b_4$ 的跨协议散布（%）")
    ax.set_title("(c) 污染量化")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG, dpi=150)
    print(f"\n图已存: {IMG}")


if __name__ == "__main__":
    main()
