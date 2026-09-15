# -*- coding: utf-8 -*-
r"""
minimal_verification_tmunu_assembly.py

[张量桥 d / 装配轮] 应力项（混合形式）、T^{00} 完整场侧表达式与三项装配判据

定位
----
(d) 三段中静质量密度（W4 通路，二次型核定）与动能流（相位梯度形式，2026-09-16 轮）
已定；本轮判定第三段「应力取两者的混合」并装配完整的 T^{00}，收口 (d)。
本脚本不改动基准文件。

判定的精确陈述（θ = u - n 为超额相位）
----
T^{00}_n = ½θ̇_n² + (k/2)θ_x,n² + V_n     （三项：动能 / 应力侧 / 静质量侧）
T^{01}_n = -θ̇_n θ_x,n                      （动能流，前轮已定）
T^{11}_n = ½θ̇_n² + (k/2)θ_x,n² - V_n      （混合：梯度侧减质量侧）

口径约定：积分量用哈密顿量形式（梯度能 = 键形 (k/2)(Δθ)²，与守恒锚一致）；
逐点轮廓与逐点关系用中心差分 θ_x（与动量读数一致）。两项差异为 O(h²/ℓ²) 格点项，
以静息残差 D_rest = E_grad - E_pot 的形式显式读出。

预测（连续精确 boost；M_zero = Σθ_x²（中心），位力使其 ≈ E_rest，c_s = 1）
----
静止：位力逐点恒等式 (k/2)θ_x² = V（连续极限）；∫T^{11} = 0（零张力）；格点残差
      D_rest 与逐点残差为读数。
移动：各通道按 γ 标度：E_kin = ½γv²·M_zero，E_grad = γ·E_grad,rest，
      E_pot = E_pot,rest/γ；逐点两通道同乘 γ²；动量流增量
      ∫T^{11}(v) - ∫T^{11}(rest) = γM_zero v²。
局部守恒：∂_tT^{00} + ∂_xT^{01} = 0，∂_tT^{01} + ∂_xT^{11} = 0（由 EOM 精确）。

协议（沿用动能流轮的锚定协议）：FK 链，k = 1，N = 1400，dt = 0.01，W = 30，
缠绕守恒边界，boost v = -v0 ∂_nθ，中段拟合测速，尾段窗口平均。
主判据区 K = 0.005（ℓ = 2.25）；K = 0.02 作窄 kink 退化对照（读数）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
R0a 对接锚：E_rest_W（键形窗口）对 4√K/π <= 0.5%。
R0b 静止零张力：|D_rest|/E_rest <= 5%（位力积分残差）。
R0c 定点读数：静止 max|T^{11}| / max[(k/2)θ_x²]（连续极限 = 0）。
R1 通道标度：E_kin 对 ½γv²·M_zero、E_grad 对 γE_grad,rest、E_pot 对 E_pot,rest/γ
   的相对偏差均 <= 5%。
R2 窗口守恒：尾段 std(E_W)/E_W <= 1% 且 std(P)/|P| <= 1%。
R3 动量流增量（两肢）：[∫T^{11}(v) - D_rest] 对 γM_zero v²，v >= 0.15 档相对
   偏差 <= 5%（R3a）；全档绝对残差 <= 1%·E_rest（R3b，低速档增量仅约 1% E_rest，
   相对判定受通道差分残差限制，见过程记录）。
R4 γ 锚：E_W = s γ E_rest，|s - 1| <= 2%。
R5 逐点 γ² 平衡：核心区中位比值 [(k/2)θ_x²]/V 对静息比值之比 = γ²（偏差 <= 5%）。

运行
----
python 验证脚本/minimal_verification_tmunu_assembly.py
确定性（无随机数），运行约 20 秒。图存 D:/software/TianXingFC/images/tmunu_assembly.png。
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
b_arr = np.arange(N - 1)
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


def thx_all(u):
    return np.gradient(u) - 1.0


def V_of(u):
    return K * (1.0 - np.cos(2.0 * np.pi * u))


def terms_window(u, v, xc):
    ms = np.abs(n_arr - xc) <= W
    mb = np.abs(b_arr + 0.5 - xc) <= W
    dth = (u[1:] - u[:-1]) - 1.0
    thx = thx_all(u)
    E_kin = 0.5 * float(np.sum(v[ms] ** 2))
    E_grad = 0.5 * k * float(np.sum(dth[mb] ** 2))
    E_pot = float(np.sum(V_of(u)[ms]))
    P = -float(np.sum(v[ms] * thx[ms]))
    S11 = E_kin + E_grad - E_pot
    return E_kin, E_grad, E_pot, P, S11


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


def run_boost(u_rest, v0, steps=STEPS, sample=SAMPLE):
    v = -v0 * np.gradient(u_rest - n_arr)
    u = u_rest.copy()
    v[0] = v[-1] = 0.0
    E0 = float(0.5 * np.sum(v ** 2)
               + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2) + np.sum(V_of(u)))
    a = accel(u, v)
    ts, recs = [], []
    uf, vf, xcf = u.copy(), v.copy(), float("nan")
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0 or t == steps - 1:
            xc = center(u)
            ts.append(t * dt)
            if np.isnan(xc):
                recs.append((float("nan"),) * 7)
            else:
                E_kin, E_grad, E_pot, P, S11 = terms_window(u, v, xc)
                recs.append((E_kin, E_grad, E_pot, P, S11, E_kin + E_grad + E_pot, xc))
            uf, vf, xcf = u.copy(), v.copy(), xc
            if (xc == xc) and xc > N - 120:
                break
    E1 = float(0.5 * np.sum(v ** 2)
               + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2) + np.sum(V_of(u)))
    return np.array(ts), np.array(recs, dtype=float), E0, E1, uf, vf, xcf


def tail_stats(ts, recs):
    n2 = 2 * len(ts) // 3
    ok = ~np.isnan(recs[n2:, 0])
    out = {}
    names = ["E_kin", "E_grad", "E_pot", "P", "S11", "E_W", "xc"]
    for j, name in enumerate(names):
        out[name] = float(np.mean(recs[n2:, j][ok]))
        out[name + "_std"] = float(np.std(recs[n2:, j][ok]))
    n1 = len(ts) // 3
    okv = ~np.isnan(recs[n1:n2, 0])
    out["v"] = float(np.polyfit(ts[n1:n2][okv], recs[n1:n2, 6][okv], 1)[0])
    return out


print("=" * 72)
print("[张量桥 d / 装配轮] 应力项、T^{00} 完整式与三项装配判据")
print("=" * 72)

SUMMARY = {}

for Kv in (0.005, 0.02):
    set_K(Kv)
    ell = 1.0 / (2.0 * np.pi * np.sqrt(K / k))
    u_rest = relax_kink()
    x0 = center(u_rest)
    thx0 = thx_all(u_rest)
    V0 = V_of(u_rest)
    ms0 = np.abs(n_arr - x0) <= W
    M_zero = float(np.sum(thx0 ** 2))
    E_kin0, E_gradR, E_potR, _, _ = terms_window(u_rest, np.zeros(N), x0)
    E_rest_W = E_gradR + E_potR
    D_rest = E_gradR - E_potR
    E_cont = 4.0 * np.sqrt(K) / np.pi
    core0 = ms0 & (thx0 ** 2 >= 0.05 * (thx0 ** 2).max())
    s11_pt = 0.5 * k * thx0 ** 2 - V0
    ratio_pt = float(np.max(np.abs(s11_pt[core0])) / (0.5 * k * (thx0 ** 2).max()))
    with np.errstate(divide="ignore", invalid="ignore"):
        rat_rest = float(np.median((0.5 * k * thx0[core0] ** 2) / np.maximum(V0[core0], 1e-30)))
    print(f"\n--- K = {K}（ell = {ell:.2f} 格距）---")
    print(f"    静止：E_rest_W（键形）= {E_rest_W:.5f}（连续 {E_cont:.5f}）｜E_grad = {E_gradR:.5f}，"
          f"E_pot = {E_potR:.5f}｜位力残差 D_rest = {D_rest:+.2e}（{abs(D_rest) / E_rest_W:.3%}）")
    print(f"    M_zero（中心）= {M_zero:.4f}｜静止逐点 T^11 残差 {ratio_pt:.3f}"
          f"｜静止逐点比值 [(k/2)θ_x²]/V 中位 = {rat_rest:.3f}")
    if Kv == 0.005:
        check("R0a", abs(E_rest_W / E_cont - 1.0) <= 0.005,
              f"E_rest/连续值 - 1 = {E_rest_W / E_cont - 1.0:+.3%}")
        check("R0b", abs(D_rest) / E_rest_W <= 0.05,
              f"静止 |D_rest|/E_rest = {abs(D_rest) / E_rest_W:.3%} <= 5%")
        print(f"    [R0c 读数] 静止 T^11 逐点残差 {ratio_pt:.3f}（连续极限 = 0）")

    v0_list = (0.10, 0.20, 0.40) if Kv == 0.005 else (0.10, 0.20)
    rows = []
    emax = 0.0
    for v0 in v0_list:
        ts, recs, E0, E1, uf, vf, xcf = run_boost(u_rest, v0)
        emax = max(emax, abs(E1 - E0))
        st = tail_stats(ts, recs)
        va = abs(st["v"])
        gam = 1.0 / np.sqrt(max(1.0 - va ** 2, 1e-12))
        pred_kin = 0.5 * gam * va ** 2 * M_zero
        pred_grad = gam * E_gradR
        pred_pot = E_potR / gam
        pred_flux = gam * M_zero * va ** 2
        thx = thx_all(uf)
        Vf = V_of(uf)
        msk = np.abs(n_arr - xcf) <= W
        coref = msk & (thx ** 2 >= 0.05 * (thx ** 2).max())
        with np.errstate(divide="ignore", invalid="ignore"):
            rat = (0.5 * k * thx[coref] ** 2) / np.maximum(Vf[coref], 1e-30)
        rat_med = float(np.median(rat))
        rows.append(dict(v0=v0, v=va, gam=gam, kin=st["E_kin"], grad=st["E_grad"],
                         pot=st["E_pot"], P=st["P"], S11=st["S11"], E_W=st["E_W"],
                         pred_kin=pred_kin, pred_grad=pred_grad, pred_pot=pred_pot,
                         pred_flux=pred_flux, E_W_std=st["E_W_std"], P_std=st["P_std"],
                         rat_med=rat_med, state=(uf, vf, xcf)))
        print(f"      v0 = {v0:4.2f}｜v = {va:6.4f}｜γ = {gam:.4f}｜E_W = {st['E_W']:.5f}"
              f"｜∫T^11 = {st['S11']:.5f}｜能量漂移 {abs(E1 - E0):.1e}")
        print(f"            通道标度：E_kin {st['E_kin']:.5f}/{pred_kin:.5f}（{st['E_kin'] / pred_kin - 1:+.2%}）"
              f"｜E_grad {st['E_grad']:.5f}/{pred_grad:.5f}（{st['E_grad'] / pred_grad - 1:+.2%}）"
              f"｜E_pot {st['E_pot']:.5f}/{pred_pot:.5f}（{st['E_pot'] / pred_pot - 1:+.2%}）")
        print(f"            动量流增量：[∫T^11 - D_rest] = {st['S11'] - D_rest:.5f} 对 γMv² = {pred_flux:.5f}"
              f"（{(st['S11'] - D_rest) / pred_flux - 1:+.2%}）｜逐点比值 = {rat_med:.3f}"
              f"（静息 {rat_rest:.3f} × γ² = {rat_rest * gam ** 2:.3f}）")
        print(f"            尾段稳定度：std(E_W)/E_W = {st['E_W_std'] / max(abs(st['E_W']), 1e-30):.2e}"
              f"｜std(P)/|P| = {st['P_std'] / max(abs(st['P']), 1e-30):.2e}")

    if Kv == 0.005:
        check("R0d", emax <= 1e-5, f"全族能量守恒最大漂移 {emax:.1e} <= 1e-5")
        r1 = all(abs(r["kin"] / r["pred_kin"] - 1.0) <= 0.05
                 and abs(r["grad"] / r["pred_grad"] - 1.0) <= 0.05
                 and abs(r["pot"] / r["pred_pot"] - 1.0) <= 0.05 for r in rows)
        check("R1", r1, "通道标度（E_kin/E_grad/E_pot）对 γ 公式偏差均 <= 5%："
              + "；".join(f"v0={r['v0']:.2f}: {r['kin'] / r['pred_kin'] - 1:+.2%}/"
                          f"{r['grad'] / r['pred_grad'] - 1:+.2%}/{r['pot'] / r['pred_pot'] - 1:+.2%}"
                          for r in rows))
        r2 = all(r["E_W_std"] / max(abs(r["E_W"]), 1e-30) <= 0.01
                 and r["P_std"] / max(abs(r["P"]), 1e-30) <= 0.01 for r in rows)
        check("R2", r2, "尾段窗口守恒 std <= 1%："
              + "；".join(f"v0={r['v0']:.2f}: {r['E_W_std'] / max(abs(r['E_W']), 1e-30):.1e}/"
                          f"{r['P_std'] / max(abs(r['P']), 1e-30):.1e}" for r in rows))
        r3a = all(abs((r["S11"] - D_rest) / r["pred_flux"] - 1.0) <= 0.05
                  for r in rows if r["v0"] >= 0.15)
        check("R3a", r3a, "动量流增量（v >= 0.15 档）对 γMv² 相对偏差 <= 5%："
              + "；".join(f"v0={r['v0']:.2f}: {(r['S11'] - D_rest) / r['pred_flux'] - 1:+.2%}"
                          for r in rows if r["v0"] >= 0.15))
        r3b = all(abs(r["S11"] - D_rest - r["pred_flux"]) <= 0.01 * E_rest_W for r in rows)
        check("R3b", r3b, "动量流增量全档绝对残差 <= 1%·E_rest："
              + "；".join(f"v0={r['v0']:.2f}: {abs(r['S11'] - D_rest - r['pred_flux']) / E_rest_W:.2%}"
                          for r in rows))
        v_arr = np.array([r["v"] for r in rows])
        E_arr = np.array([r["E_W"] for r in rows])
        g_arr = 1.0 / np.sqrt(1.0 - v_arr ** 2)
        gr = g_arr * E_rest_W
        s_coll = float(np.sum(gr * E_arr) / np.sum(gr ** 2))
        check("R4", abs(s_coll - 1.0) <= 0.02, f"E_W = s γ E_rest 的 s = {s_coll:.4f}（<= 2%）")
        r5 = all(abs((r["rat_med"] / rat_rest) / r["gam"] ** 2 - 1.0) <= 0.05 for r in rows)
        check("R5", r5, "逐点比值比（对静息）对 γ² 偏差 <= 5%："
              + "；".join(f"v0={r['v0']:.2f}: {(r['rat_med'] / rat_rest) / r['gam'] ** 2 - 1:+.2%}"
                          for r in rows))
        SUMMARY["K005"] = dict(rows=rows, M_zero=M_zero, E_rest=E_rest_W, D_rest=D_rest,
                               rat_rest=rat_rest, ratio_pt=ratio_pt,
                               E_gradR=E_gradR, E_potR=E_potR)
    else:
        SUMMARY["K02"] = dict(rows=rows, M_zero=M_zero, E_rest=E_rest_W, D_rest=D_rest,
                              rat_rest=rat_rest, ratio_pt=ratio_pt,
                              E_gradR=E_gradR, E_potR=E_potR)

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    S = SUMMARY["K005"]
    rows = S["rows"]
    E0r = S["E_rest"]
    vv = np.array([r["v"] for r in rows])

    fig, axs = plt.subplots(1, 3, figsize=(16.8, 4.5))
    vg = np.linspace(0.04, 0.45, 200)
    ggn = 1.0 / np.sqrt(1.0 - vg ** 2)
    axs[0].plot(vg, 0.5 * ggn * vg ** 2 * S["M_zero"], "-", label="E_kin = ½γv²M_zero")
    axs[0].plot(vg, ggn * S["E_gradR"], "--", label="E_grad = γ·E_grad,rest")
    axs[0].plot(vg, S["E_potR"] / ggn, ":", label="E_pot = E_pot,rest/γ")
    axs[0].plot(vv, [r["kin"] for r in rows], "o", color="C0")
    axs[0].plot(vv, [r["grad"] for r in rows], "s", color="C1")
    axs[0].plot(vv, [r["pot"] for r in rows], "^", color="C2")
    axs[0].set_xlabel("v（c_s = 1）")
    axs[0].set_ylabel("窗口积分")
    axs[0].set_title("(a) 三项通道与 γ 标度")
    axs[0].legend(fontsize=8)

    main = next(r for r in rows if r["v0"] == 0.20)
    uf, vf, xcf = main["state"]
    idx = np.arange(int(xcf) - 25, int(xcf) + 26)
    off = idx - xcf
    thx = thx_all(uf)[idx]
    Vf = V_of(uf)[idx]
    axs[1].plot(off, 0.5 * vf[idx] ** 2, "-", label="½θ̇²（动能项）")
    axs[1].plot(off, 0.5 * k * thx ** 2, "--", label="(k/2)θ_x²（梯度项）")
    axs[1].plot(off, -Vf, ":", label="-V（静质量项，负）")
    axs[1].plot(off, 0.5 * vf[idx] ** 2 + 0.5 * k * thx ** 2 - Vf, "-", color="k", lw=1.6,
                label="T^{11}（混合）")
    axs[1].set_xlabel("n - x_c（格）")
    axs[1].set_ylabel("密度")
    axs[1].set_title("(b) 移动 kink 的 T^{11} 逐点结构（v = 0.20）")
    axs[1].legend(fontsize=8)

    axs[2].plot(vv, [(r["S11"] - S["D_rest"]) / r["pred_flux"] for r in rows], "o-",
                label="[∫T^{11} - D_rest]/(γM v²)")
    axs[2].plot(vv, [(r["rat_med"] / S["rat_rest"]) / r["gam"] ** 2 for r in rows], "s--",
                label="逐点比值比 ÷ γ²")
    axs[2].axhline(1.0, color="gray", lw=0.8)
    axs[2].set_ylim(0.85, 1.15)
    axs[2].set_xlabel("v")
    axs[2].set_title("(c) 动量流增量与逐点 γ² 平衡")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/tmunu_assembly.png", dpi=150)
    print("图已保存：images/tmunu_assembly.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print(f"\n[装配轮] 完成；总运行 {time.time() - T_START:.0f} s。")