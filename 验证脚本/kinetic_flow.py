# -*- coding: utf-8 -*-
r"""
kinetic_flow.py

[张量桥 d / 动能流项] 动量密度的相位梯度形式判据（判据级验证）

定位
----
上游：(d) 的静质量密度形式经作者核定（二次型，W4 通路）。本脚本判定 (d) 的
第二项：动能流（动量密度 T^{0i}）的场侧形式。体系语境的候选为「相位梯度形式」，
本脚本把它精确化为可检验陈述、给判据并作对照判定。本脚本不改动基准文件。

判定的精确陈述
----
记 θ = u - n 为超额相位（n 是均匀缠绕背景，每键一层），θ_x 为中心差分。
候选一（相位梯度形式，待判）：
    T^{0i}_n = u̇_n · θ_x,n，   P = -Σ u̇_n θ_x,n（右移为正）
刚体运动读数：T^{0i}(n) = v · θ_x,n²（相位梯度平方乘速度），全域局域于芯。
积分系数 M_zero = Σ θ_x²；连续极限位力关系 M_zero = E_rest（静息能量），
由此 P = M v 与 γ 对应同时成立。
对照（判负候选）：
    原坡形形式 T = u̇ u_x：P_raw = P + v·Σθ_x（加法多出台阶项，超量因子约
    1 + 1/(γ M_zero)，本区约 12 倍）；
    刚性拖动形式 T = v u_x²（首轮脚本的 I2 = Σ u_x² 候选）：积分约 N·v，
    超量 10^4 量级。

协议（对齐 2026-09-06 相对论色散验证的锚定协议）
----
FK 链：E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos 2πu_n)]，k = 1，c_s = 1，
N = 1400，dt = 0.01，窗口半径 W = 30，缠绕守恒边界（两端每步钉回），
静止剖面 kink_profile + relax_kink，boost v = -v0 ∂_nθ（超额相位梯度方向），
中段线性拟合测 v，尾段窗口平均测能量与动量（辐射已离开窗口）。
主判据区 K = 0.005（ell = 2.25 格距）；K = 0.02（ell = 1.13）作窄 kink 退化对照。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
K0 对接锚（复现既有读数）：E_rest_W 对连续值 4√K/π 偏差 <= 0.5%；位力比
   E_rest/M_zero <= 1.05；全族能量守恒漂移 <= 1e-5。
K1 动量-质量关系：低速档 v0 = 0.05/0.10：|P/(M_zero v) - 1| <= 3%；
   全档：|P/(γ M_zero v) - 1| <= 5%（γ = 1/√(1 - v²/c_s²)）；静能口径
   |P/(γ E_rest v) - 1| 作读数。
K2 密度形式判别（核心）：
   a. 轮廓：尾段动量密度 T^{0i}(n) = u̇_n θ_x,n 与 v θ_x² 的加权偏差 <= 5%、
      余弦重叠 >= 0.99（v0 = 0.10 主判据档；其余档位作读数）。
   b. 原坡形超量核对：（P_raw - P）= v Σ_winθ_x（台阶滑移项，±2%）；超量因子
      1 + 1/(γ M_zero) 报告。
   c. 刚性拖动对照（判负记录）：P_I2/P = Σu_x²/M_zero 报告，要求 >= 100。
K3 gamma 对应（(d) 总判据接口）：E_W = s γ E_rest 的塌缩幅度 |s - 1| <= 2%。
K4 朴素 boost 判负记录（首轮失败根因）：ΔE_naive/ΔE_θ = Σu_x²/M_zero >= 100，
   并记录该初条件下的核心读数与动量记账。

运行
----
python 验证脚本/kinetic_flow.py
确定性（无随机数），运行约 30 秒。图存 D:/software/TianXingFC/images/kinetic_flow.png。
"""
import time

import numpy as np

T_START = time.time()

k = 1.0
dt = 0.01
N = 1400
cN = N // 2
W = 30
STEPS = 60000
SAMPLE = 20

n_arr = np.arange(N, dtype=float)

K = None
w2 = None

CHECKS = []


def set_K(Kv):
    global K, w2
    K = float(Kv)
    w2 = 2.0 * np.pi * K


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def accel(u, v):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2.0 * u[1:-1]) - w2 * np.sin(2.0 * np.pi * u[1:-1])
    return a


def kink_profile(x0):
    ell = 1.0 / (2.0 * np.pi * np.sqrt(K / k))
    return n_arr + (2.0 / np.pi) * np.arctan(np.exp((n_arr - x0) / ell))


def center(u):
    wgt = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(wgt))
    lo, hi = max(0, ks - 8), min(len(wgt), ks + 9)
    ww = np.clip(wgt[lo:hi], 0.0, None)
    if ww.sum() <= 1e-12:
        return float("nan")
    return float((np.arange(lo, hi) * ww).sum() / ww.sum())


def thx_central(u):
    """窗口内 θ_x 中心差分（θ = u - n）。"""
    return (np.roll(u, -1) - np.roll(u, 1) - 2.0) / 2.0


def ux_all(u):
    """全域安全 u_x（两端一边差；无 roll 环绕伪值）。"""
    return np.gradient(u)


def thx_all(u):
    """全域安全 θ_x。"""
    return np.gradient(u) - 1.0


def window_energy(u, v, xc):
    ms = np.abs(n_arr - xc) <= W
    b = np.arange(N - 1)
    mb = np.abs(b + 0.5 - xc) <= W
    E = 0.5 * np.sum(v[ms] ** 2) + np.sum(K * (1.0 - np.cos(2.0 * np.pi * u[ms])))
    E += 0.5 * k * np.sum((u[b[mb] + 1] - u[b[mb]] - 1.0) ** 2)
    return float(E)


def total_energy(u, v):
    return float(0.5 * np.sum(v ** 2)
                 + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
                 + np.sum(K * (1.0 - np.cos(2.0 * np.pi * u))))


def p_field(u, v, xc):
    """相位梯度形式动量（窗口）：P = -Σ u̇_n θ_x,n（右移为正）。"""
    ms = np.abs(n_arr - xc) <= W
    return float(-np.sum(v[ms] * thx_central(u)[ms]))


def p_raw_win(u, v, xc):
    """原坡形形式动量（窗口）：P_raw = -Σ u̇_n u_x,n。"""
    ms = np.abs(n_arr - xc) <= W
    return float(-np.sum(v[ms] * ux_all(u)[ms]))


def relax_kink(steps=12000):
    u = kink_profile(cN * 1.0)
    v = np.zeros(N)
    a = accel(u, v)
    for _ in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
    return u


def run_boost(u_rest, v0, boost):
    if boost == "theta":
        v = -v0 * np.gradient(u_rest - n_arr)
    elif boost == "raw":
        v = -v0 * np.gradient(u_rest)
    else:
        raise ValueError(boost)
    u = u_rest.copy()
    v[0] = v[-1] = 0.0
    E0 = total_energy(u, v)
    a = accel(u, v)
    ts, xs, Ews, Pws = [], [], [], []
    uf, vf, xcf = u.copy(), v.copy(), float("nan")
    for t in range(STEPS):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % SAMPLE == 0 or t == STEPS - 1:
            xc = center(u)
            ts.append(t * dt)
            xs.append(xc)
            if np.isnan(xc):
                Ews.append(float("nan"))
                Pws.append(float("nan"))
            else:
                Ews.append(window_energy(u, v, xc))
                Pws.append(p_field(u, v, xc))
            uf, vf, xcf = u.copy(), v.copy(), xc
            if (xc == xc) and xc > N - 120:
                break
    E1 = total_energy(u, v)
    return (np.array(ts), np.array(xs), np.array(Ews), np.array(Pws),
            E0, E1, uf, vf, xcf)


def measure(ts, xs, Ews, Pws):
    n1, n2 = len(ts) // 3, 2 * len(ts) // 3
    ok = ~np.isnan(xs[n1:n2])
    v_meas = float(np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0])
    okE = ~np.isnan(Ews[n2:])
    E_mean = float(np.mean(Ews[n2:][okE]))
    E_std = float(np.std(Ews[n2:][okE]))
    P_mean = float(np.mean(Pws[n2:][okE]))
    P_std = float(np.std(Pws[n2:][okE]))
    return v_meas, E_mean, E_std, P_mean, P_std


def profile_readout(u, v, xc, v_meas):
    """窗口化轮廓判定：T^{0i}(n) = u̇_n θ_x,n 对 v θ_x²（边界安全）。"""
    win = np.abs(n_arr - xc) <= W
    thx = thx_all(u)
    t2 = thx ** 2
    peak = float(t2[win].max())
    core = win & (t2 >= 0.05 * peak)
    g = -v * thx / max(abs(v_meas), 1e-12)
    dev = np.abs(g[core] / np.maximum(t2[core], 1e-30) - 1.0)
    wsum = float(t2[core].sum())
    wdev = float((t2[core] * dev).sum() / wsum)
    maxdev = float(dev.max())
    ov = float((g[core] * t2[core]).sum()
               / np.sqrt((g[core] ** 2).sum() * (t2[core] ** 2).sum()))
    g_raw = -v * ux_all(u) / max(abs(v_meas), 1e-12)
    ov_raw = float((g_raw[win] * t2[win]).sum()
                   / np.sqrt((g_raw[win] ** 2).sum() * (t2[win] ** 2).sum()))
    g_drag = ux_all(u) ** 2
    ov_drag = float((g_drag[win] * t2[win]).sum()
                    / np.sqrt((g_drag[win] ** 2).sum() * (t2[win] ** 2).sum()))
    e_dens = 0.5 * thx ** 2 + K * (1.0 - np.cos(2.0 * np.pi * u))
    ov_Edens = float((e_dens[win] * t2[win]).sum()
                     / np.sqrt((e_dens[win] ** 2).sum() * (t2[win] ** 2).sum()))
    v_dens = K * (1.0 - np.cos(2.0 * np.pi * u))
    ov_Vdens = float((v_dens[win] * t2[win]).sum()
                     / np.sqrt((v_dens[win] ** 2).sum() * (t2[win] ** 2).sum()))
    ell = 1.0 / (2.0 * np.pi * np.sqrt(K / k))
    bg_drag = float(np.median(g_drag[np.abs(n_arr - xc) > 5.0 * ell]))
    return dict(wdev=wdev, maxdev=maxdev, overlap=ov, overlap_raw=ov_raw,
                overlap_drag=ov_drag, bg_drag=bg_drag,
                overlap_Edens=ov_Edens, overlap_Vdens=ov_Vdens)


# ----------------------------------------------------------------------
print("=" * 72)
print("[张量桥 d / 动能流项] 动量密度的相位梯度形式判据")
print("=" * 72)

SUMMARY = {}

for Kv in (0.005, 0.02):
    set_K(Kv)
    ell = 1.0 / (2.0 * np.pi * np.sqrt(K / k))
    u_rest = relax_kink()
    x0 = center(u_rest)
    th = u_rest - n_arr
    M_zero = float(np.sum(np.gradient(th) ** 2))
    E_rest_W = window_energy(u_rest, np.zeros(N), x0)
    E_cont = 4.0 * np.sqrt(K) / np.pi
    uxn_rest = ux_all(u_rest)
    Sigma_ux2 = float(np.sum(uxn_rest ** 2))
    print(f"\n--- K = {K}（ell = {ell:.2f} 格距）---")
    print(f"    静止 kink：x0 = {x0:.2f}｜E_rest_W = {E_rest_W:.5f}"
          f"（连续 4√K/pi = {E_cont:.5f}）｜M_zero = {M_zero:.4f}"
          f"｜位力比 = {E_rest_W / M_zero:.4f}")
    print(f"    结构对照：原坡形超量因子 1 + 1/M_zero = {1.0 + 1.0 / M_zero:.2f}；"
          f"整链拖动因子 Σu_x²/M_zero = {Sigma_ux2 / M_zero:.0f}")
    if Kv == 0.005:
        check("K0a", abs(E_rest_W / E_cont - 1.0) <= 0.005,
              f"E_rest/连续值 - 1 = {E_rest_W / E_cont - 1.0:+.3%}（复现既有读数）")
        check("K0b", E_rest_W / M_zero <= 1.05,
              f"位力比 {E_rest_W / M_zero:.4f} <= 1.05（连续 = 1）")

    v0_list = (0.05, 0.10, 0.20, 0.40) if Kv == 0.005 else (0.10, 0.20)
    rows = []
    emax = 0.0
    print("    [boost 沿 θ_x（超额相位梯度方向）]")
    for v0 in v0_list:
        ts, xs, Ews, Pws, E0, E1, uf, vf, xcf = run_boost(u_rest, v0, "theta")
        emax = max(emax, abs(E1 - E0))
        vm, Em, Es, Pm, Ps = measure(ts, xs, Ews, Pws)
        va = abs(vm)
        gam = 1.0 / np.sqrt(max(1.0 - va ** 2, 1e-12))
        Pmag = abs(Pm)
        ratio_M = Pmag / (M_zero * va)
        ratio_g = Pmag / (gam * M_zero * va)
        ratio_Mr = Pmag / (E_rest_W * va)
        ratio_gr = Pmag / (gam * E_rest_W * va)
        prof = profile_readout(uf, vf, xcf, va)
        P_raw = abs(p_raw_win(uf, vf, xcf))
        ms = np.abs(n_arr - xcf) <= W
        sum_thx_win = float(np.sum(thx_all(uf)[ms]))
        slide_charge = va * sum_thx_win
        excess_meas = P_raw - Pmag
        add_err = abs(excess_meas - slide_charge) / max(abs(slide_charge), 1e-30)
        ratio_raw = P_raw / max(Pmag, 1e-30)
        pred_raw = 1.0 + 1.0 / (gam * M_zero)
        I2_win = float(np.sum(ux_all(uf)[ms] ** 2))
        ratio_I2_win = va * I2_win / max(Pmag, 1e-30)
        ratio_I2_full = va * Sigma_ux2 / max(Pmag, 1e-30)
        rows.append(dict(v0=v0, v=va, E=Em, E_std=Es, P=Pmag, P_std=Ps,
                         ratio_M=ratio_M, ratio_g=ratio_g, ratio_Mr=ratio_Mr,
                         ratio_gr=ratio_gr, prof=prof, ratio_raw=ratio_raw,
                         pred_raw=pred_raw, add_err=add_err,
                         ratio_I2_win=ratio_I2_win, ratio_I2_full=ratio_I2_full,
                         gam=gam, state=(uf, vf, xcf)))
        print(f"      v0 = {v0:4.2f}｜v = {va:6.4f}｜max|u̇| = {np.abs(vf).max():.4f}"
              f"｜E_W = {Em:8.5f} ± {Es:.1e}｜P_W = {Pmag:8.5f} ± {Ps:.1e}"
              f"｜能量漂移 {abs(E1 - E0):.1e}")
        print(f"            P/(M v) = {ratio_M:.4f}｜P/(γ M v) = {ratio_g:.4f}"
              f"｜P/(E_rest v) = {ratio_Mr:.4f}｜P/(γ E_rest v) = {ratio_gr:.4f}")
        print(f"            轮廓：加权偏差 {prof['wdev'] * 100:.2f}%，最大 {prof['maxdev'] * 100:.2f}%，"
              f"重叠 {prof['overlap']:.5f}（θ² 基准；原坡形 {prof['overlap_raw']:.5f}，"
              f"拖动 {prof['overlap_drag']:.5f}，拖动背景 {prof['bg_drag']:.3f}，"
              f"静能密度 {prof['overlap_Edens']:.5f}，势能密度 {prof['overlap_Vdens']:.5f}）")
        print(f"            对照：原坡形 {ratio_raw:.3f}（预期 {pred_raw:.3f}）｜超量核对 {add_err:.2%}"
              f"（Σ_winθ_x = {sum_thx_win:.3f}）｜窗口拖动 {ratio_I2_win:.0f}｜整链拖动 {ratio_I2_full:.0f}")

    if Kv == 0.005:
        check("K0c", emax <= 1e-5, f"全族能量守恒最大漂移 {emax:.1e} <= 1e-5")
        k1a = all(abs(r["ratio_M"] - 1.0) <= 0.03 for r in rows if r["v0"] <= 0.10)
        check("K1a", k1a, "低速档 |P/(M_zero v) - 1| <= 3%："
              + "；".join(f"v0={r['v0']:.2f}: {r['ratio_M'] - 1:+.3%}" for r in rows if r["v0"] <= 0.10))
        k1b = all(abs(r["ratio_g"] - 1.0) <= 0.05 for r in rows)
        check("K1b", k1b, "全档 |P/(γ M_zero v) - 1| <= 5%："
              + "；".join(f"v0={r['v0']:.2f}: {r['ratio_g'] - 1:+.3%}" for r in rows))
        main = next(r for r in rows if r["v0"] == 0.10)
        check("K2a", main["prof"]["wdev"] <= 0.05 and main["prof"]["overlap"] >= 0.99,
              f"v0=0.10 轮廓：加权偏差 {main['prof']['wdev'] * 100:.2f}% <= 5%，"
              f"重叠 {main['prof']['overlap']:.5f} >= 0.99")
        k2b = all(r["add_err"] <= 0.02 for r in rows)
        check("K2b", k2b, "原坡形超量核对：超量（P_raw - P）= v Σ_winθ_x（台阶滑移项，±2%）："
              + "；".join(f"v0={r['v0']:.2f}: {r['add_err']:.2%}" for r in rows))
        k2c = all(r["ratio_I2_full"] >= 100 for r in rows)
        check("K2c", k2c, "刚性拖动对照（判负记录）整链因子："
              + "／".join(f"{r['ratio_I2_full']:.0f}" for r in rows) + "（全部 >= 100）")
        v_arr = np.array([r["v"] for r in rows])
        E_arr = np.array([r["E"] for r in rows])
        P_arr = np.array([r["P"] for r in rows])
        gam_arr = 1.0 / np.sqrt(1.0 - v_arr ** 2)
        gr = gam_arr * E_rest_W
        s_coll = float(np.sum(gr * E_arr) / np.sum(gr ** 2))
        check("K3", abs(s_coll - 1.0) <= 0.02,
              f"E_W = s γ E_rest 塌缩幅度 s = {s_coll:.4f}（|s-1| <= 2%）")
        R = (E_arr ** 2 - P_arr ** 2) / E_rest_W ** 2
        print("    不变量 R = (E² - P²)/E_rest² 全档 = "
              + "，".join(f"{x:.4f}" for x in R))
        SUMMARY["K005"] = dict(rows=rows, M_zero=M_zero, E_rest=E_rest_W, N=N)
    else:
        SUMMARY["K02"] = dict(rows=rows, M_zero=M_zero, E_rest=E_rest_W, N=N)

    # ---- 朴素 boost 对照（首轮脚本方向：v = -v0 ∂_n u）----
    vB0 = 0.10
    vB = -vB0 * np.gradient(u_rest)
    dE_pred_B = 0.5 * vB0 ** 2 * float(np.sum(np.gradient(u_rest) ** 2))
    dE_pred_A = 0.5 * vB0 ** 2 * M_zero
    E_rest_tot = total_energy(u_rest, np.zeros(N))
    dE_meas_B = total_energy(u_rest, vB) - E_rest_tot
    print(f"    [朴素 boost 对照（v0 = {vB0}，boost 沿 u_x）]")
    print(f"      注入能量：预测 {dE_pred_B:.4e}（= ½v²Σu_x²）对 θ 方向 {dE_pred_A:.4e}"
          f"（= ½v²M_zero），比值 {dE_pred_B / dE_pred_A:.0f}；实测 {dE_meas_B:.4e}")
    ts, xs, Ews, Pws, E0, E1, uf, vf, xcf = run_boost(u_rest, vB0, "raw")
    vm, Em, Es, Pm, Ps = measure(ts, xs, Ews, Pws)
    va = abs(vm)
    P_rawB = abs(p_raw_win(uf, vf, xcf))
    print(f"      读数：v = {va:.4f}｜E_W = {Em:.5f} ± {Es:.1e}｜P_θ = {abs(Pm):.5f}｜"
          f"P_θ/(M v) = {abs(Pm) / (M_zero * max(va, 1e-12)):.2f}"
          f"｜P_raw = {P_rawB:.5f}｜能量漂移 {abs(E1 - E0):.1e}")
    if Kv == 0.005:
        check("K4", dE_pred_B / dE_pred_A >= 100
              and abs(dE_meas_B - dE_pred_B) <= 0.02 * dE_pred_B,
              f"朴素 boost 注入能量比 {dE_pred_B / dE_pred_A:.0f} >= 100"
              f"（把整链当运动质量），实测对预测偏差 {abs(dE_meas_B - dE_pred_B) / dE_pred_B:.3%}")

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))

# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    S = SUMMARY["K005"]
    rows = S["rows"]
    M0 = S["M_zero"]
    E0r = S["E_rest"]
    vv = np.array([r["v"] for r in rows])
    PP = np.array([r["P"] for r in rows])
    EE = np.array([r["E"] for r in rows])

    fig, axs = plt.subplots(1, 3, figsize=(16.8, 4.5))
    vg = np.linspace(0.02, 0.45, 60)
    axs[0].semilogy(vg, M0 * vg, "-", label="M_zero v（%.3f v）" % M0)
    axs[0].semilogy(vg, (1.0 + M0) * vg, "--", label="原坡形 (1+M_zero) v（%.2f v）" % (1 + M0))
    axs[0].semilogy(vg, E0r * vg, ":", label="静能形式 E_rest v（%.3f v）" % E0r)
    axs[0].semilogy(vg, (S["N"] + M0) * vg, "-.",
                    label="整链拖动 (N+M_zero) v（%.0f v）" % (S["N"] + M0))
    axs[0].semilogy(vv, PP, "o", color="k", label="实测 P_W（θ_x boost）")
    axs[0].set_xlabel("v（c_s = 1）")
    axs[0].set_ylabel("|P|")
    axs[0].set_title("(a) 动量-速度：候选阶梯")
    axs[0].legend(fontsize=7)

    main = next(r for r in rows if r["v0"] == 0.20)
    uf, vf, xcf = main["state"]
    nlo, nhi = int(xcf) - 25, int(xcf) + 26
    idx = np.arange(nlo, nhi)
    off = idx - xcf
    thx = thx_all(uf)[idx]
    uxc = ux_all(uf)[idx]
    profs = {
        "实测 T0i（-u̇θ_x）": -vf[idx] * thx,
        "相位梯度平方 θ_x²": thx ** 2,
        "原坡形 u̇u_x": -vf[idx] * uxc,
        "刚性拖动 v u_x²": main["v"] * uxc ** 2,
        "静能密度（½θ_x² + V）": 0.5 * thx ** 2 + K * (1.0 - np.cos(2.0 * np.pi * uf[idx])),
    }
    for lbl, f in profs.items():
        fn = f / max(np.abs(f).sum(), 1e-30)
        axs[1].semilogy(off, np.maximum(fn, 1e-7), label=lbl, lw=1.2)
    axs[1].set_xlabel("n - x_c（格）")
    axs[1].set_ylabel("归一化密度")
    axs[1].set_title("(b) 动量密度轮廓（v = 0.20 尾段）")
    axs[1].legend(fontsize=7)

    axs[2].plot(vv, (EE / E0r) / (1.0 / np.sqrt(1.0 - vv ** 2)), "o-", label="(E_W/E_rest)/γ")
    axs[2].plot(vv, (EE ** 2 - PP ** 2) / E0r ** 2, "s--", label="(E² - P²)/E_rest²")
    axs[2].axhline(1.0, color="gray", lw=0.8)
    axs[2].set_xlabel("v")
    axs[2].set_ylim(0.9, 1.15)
    axs[2].set_title("(c) γ 对应与不变量")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/kinetic_flow.png", dpi=150)
    print("图已保存：images/kinetic_flow.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print(f"\n[动能流] 完成；总运行 {time.time() - T_START:.0f} s。")