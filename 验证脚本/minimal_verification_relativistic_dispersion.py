# -*- coding: utf-8 -*-
"""
相对论色散验证（动量扇区与不变量）：转换波 E-p 关系的完整检验

定位：minimal_verification_meff.py（P1-33）已确认 E(v) 涌现相对论形状
（SG 拟合残差 0.006 vs 牛顿 0.616，K=0.005；v^4 系数为正）。本脚本补齐色散
闭环的缺口：
  1. 动量扇区：场动量 P = Σ v_n (θ_{n+1} - θ_{n-1})/2，θ = u - n，测 P(v)；
  2. 洛伦兹型不变量：E^2 - c_s^2 P^2 = E_rest^2（不变量组合的速度平坦性）；
  3. 无自由参数 gamma 塌缩：E = gamma · E_rest（斜率 s 对 1，不用拟合质量）；
  4. 宽度趋势（双侧）：K = 0.0005/0.001/0.005/0.01/0.02，SG 偏差随 kink 宽度收敛；
  5. 初条件协议独立性：简单 boost（v = -v0·零模）vs 洛伦兹 boost（gamma 收缩
     形状 + gamma·v 速度场，连续 SG 精确 boosted 解作初条件）。

理论靶标（连续 sine-Gordon kink，精确、无自由参数）：
  E(v) = E_rest · gamma，P(v) = (E_rest/c_s^2) · gamma · v，
  E^2 - c_s^2 P^2 = E_rest^2；
  低速动量斜率 p1 = M_zero = E_rest/c_s^2（位力关系），p2/p1 = 1/2。

模型：FK 链（同 m_eff 脚本）：E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos 2πu_n)]，
  k = 1，c_s = 1，w2 = 2πK（2π 因子已修正），缠绕守恒边界 u_0 = 0, u_{N-1} = u_0 + N。
  确定性演化，无随机数。
"""
import numpy as np

print("=" * 72)
print("相对论色散验证：转换波的 E-p 关系、不变量与宽度趋势")
print("=" * 72)

k = 1.0
dt = 0.01
N = 1400
cN = N // 2
W = 30          # kink 能量/动量窗口半径（同 m_eff 脚本）


def set_K(Kv):
    global K, w2, c_s
    K = Kv
    w2 = 2 * np.pi * K
    c_s = np.sqrt(k)


def accel(u, v, f=0.0):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    a[1:-1] += f
    return a


def kink_profile(x0):
    n = np.arange(N)
    ell = 1.0 / (2 * np.pi * np.sqrt(K / k))
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ell))


def center(u):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - 8), min(len(w), ks + 9)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan


def window_energy(u, v, xc):
    n = np.arange(N)
    ms = np.abs(n - xc) <= W
    b = np.arange(N - 1)
    mb = np.abs(b + 0.5 - xc) <= W
    E = 0.5 * np.sum(v[ms] ** 2) + np.sum(K * (1 - np.cos(2 * np.pi * u[ms])))
    E += 0.5 * k * np.sum((u[b[mb] + 1] - u[b[mb]] - 1.0) ** 2)
    return E


def p_window(u, v, xc):
    """场动量（窗口）：P = Σ v_n (θ_{n+1} - θ_{n-1}) / 2，θ = u - n。

    连续极限 P = ∫ θ̇ θ_x dx；缓慢移动 kink 给 |P| = |Ẋ|·M_zero，
    洛伦兹 boosted 解给 |P| = gamma·β·E_rest/c_s。"""
    n = np.arange(N)
    ms = np.abs(n - xc) <= W
    th_c = (np.roll(u, -1) - np.roll(u, 1) - 2.0) / 2.0
    return float(np.sum(v[ms] * th_c[ms]))


def total_energy(u, v):
    return (0.5 * np.sum(v ** 2)
            + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
            + np.sum(K * (1 - np.cos(2 * np.pi * u))))


def relax_kink(gamma_d=0.5, steps=12000):
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


def lorentz_ic(v0):
    """连续 SG 精确 boosted 解作初条件：形状 gamma 收缩，速度场 gamma·v·θ'。"""
    gam = 1.0 / np.sqrt(1.0 - v0 ** 2)
    n = np.arange(N)
    th = u_rest - n
    dth = np.gradient(th)
    xi = gam * (n - cN)
    th_i = np.interp(xi, n - cN, th)
    dth_i = np.interp(xi, n - cN, dth)
    return n + th_i, -gam * v0 * dth_i


def run_boost(v0, steps=60000, sample=20, lorentz=False):
    """初速度场演化并采样：中心、窗口能量、窗口动量。kink 向 +n 运动。"""
    if lorentz:
        u, v = lorentz_ic(v0)
    else:
        u = u_rest.copy()
        v = -v0 * np.gradient(u_rest - np.arange(N))
    v[0] = v[-1] = 0.0
    E0 = total_energy(u, v)
    a = accel(u, v)
    ts, xs, Ews, Pws = [], [], [], []
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
            if np.isnan(xc):
                Ews.append(np.nan)
                Pws.append(np.nan)
            else:
                Ews.append(window_energy(u, v, xc))
                Pws.append(p_window(u, v, xc))
            if not np.isnan(xc) and xc > N - 100:
                break
    E1 = total_energy(u, v)
    return np.array(ts), np.array(xs), np.array(Ews), np.array(Pws), E0, E1


def measure(ts, xs, Ews, Pws):
    """中段线性拟合测 v；尾段平均测窗口能量/动量（辐射已离开）。"""
    n1, n2 = len(ts) // 3, 2 * len(ts) // 3
    ok = ~np.isnan(xs[n1:n2])
    v_meas = float(np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0])
    okE = ~np.isnan(Ews[n2:])
    E_mean = float(np.mean(Ews[n2:][okE]))
    E_std = float(np.std(Ews[n2:][okE]))
    P_mean = float(np.mean(Pws[n2:][okE]))
    P_std = float(np.std(Pws[n2:][okE]))
    return v_meas, E_mean, E_std, P_mean, P_std


# ----------------------------------------------------------------------
results = {}
for Kv in (0.0005, 0.001, 0.005, 0.01, 0.02):
    set_K(Kv)
    ell = 1.0 / (2 * np.pi * np.sqrt(K / k))
    print(f"\n--- K = {K}（kink 宽度 ell = {ell:.2f} 格距，真空阈值 2πK = {2 * np.pi * K:.4f}）---")
    u_rest, v_rest = relax_kink()
    x0 = center(u_rest)
    th = u_rest - np.arange(N)
    M_zero = float(np.sum(np.gradient(th) ** 2))
    M_shape = float(np.sum((u_rest[1:] - u_rest[:-1] - 1.0) ** 2))
    E_rest_w = window_energy(u_rest, v_rest, x0)
    E_vac_w = window_energy(np.arange(N).astype(float), np.zeros(N), xN := x0)
    E_cont = 4.0 * np.sqrt(K) / np.pi
    print(f"  静止 kink：E_rest_W = {E_rest_w:.5f}（连续 SG 预期 {E_cont:.5f}），"
          f"M_zero = {M_zero:.4f}，M_shape = {M_shape:.4f}，"
          f"位力比 E_rest/(c_s²·M_zero) = {E_rest_w / M_zero:.3f}（连续 = 1）")

    v0_list = (0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90) if K == 0.005 \
        else (0.20, 0.35, 0.50, 0.65, 0.80, 0.90)
    rows = []
    e_drifts = []
    for v0 in v0_list:
        ts, xs, Ews, Pws, E0, E1 = run_boost(v0)
        e_drifts.append(abs(E1 - E0))
        vm, Em, Es, Pm, Ps = measure(ts, xs, Ews, Pws)
        rows.append(dict(v0=v0, v=abs(vm), E=Em, E_std=Es, P=abs(Pm), P_std=Ps,
                         tag="boost"))
        print(f"  v0 = {v0:4.2f} | v = {abs(vm):6.4f} | E_W = {Em:9.5f} ± {Es:.1e} | "
              f"P_W = {abs(Pm):8.5f} ± {Ps:.1e} | 能量误差 {abs(E1 - E0):.1e}")
    lorentz_rows = []
    if K == 0.005:
        print("  [协议 B：洛伦兹 boost 初条件（gamma 收缩 + gamma·v 速度场）]")
        for v0 in v0_list:
            ts, xs, Ews, Pws, E0, E1 = run_boost(v0, lorentz=True)
            e_drifts.append(abs(E1 - E0))
            vm, Em, Es, Pm, Ps = measure(ts, xs, Ews, Pws)
            lorentz_rows.append(dict(v0=v0, v=abs(vm), E=Em, E_std=Es,
                                     P=abs(Pm), P_std=Ps, tag="lorentz"))
            print(f"  v0 = {v0:4.2f} | v = {abs(vm):6.4f} | E_W = {Em:9.5f} ± {Es:.1e} | "
                  f"P_W = {abs(Pm):8.5f} ± {Ps:.1e} | 能量误差 {abs(E1 - E0):.1e}")
    print(f"  全族能量守恒最大漂移 = {max(e_drifts):.1e}")

    # ---------------- 色散分析（简单 boost 族） ----------------
    v_arr = np.array([r["v"] for r in rows])
    E_arr = np.array([r["E"] for r in rows])
    P_arr = np.array([r["P"] for r in rows])
    dE = E_arr - E_rest_w
    gam = 1.0 / np.sqrt(1.0 - (v_arr / c_s) ** 2)

    # [形状] 牛顿 vs 相对论拟合（同 m_eff 脚本口径）
    lo = v_arr < 0.5
    m_newt = np.sum(v_arr[lo] ** 2 * dE[lo]) / np.sum(v_arr[lo] ** 4) if lo.sum() else np.nan
    res_newt = np.max(np.abs(0.5 * m_newt * v_arr ** 2 - dE)) / np.max(np.abs(dE))
    g1 = c_s ** 2 * (gam - 1.0)
    M_rel = np.sum(g1 * dE) / np.sum(g1 ** 2)
    res_rel = np.max(np.abs(M_rel * g1 - dE)) / np.max(np.abs(dE))
    A = np.stack([v_arr ** 2, v_arr ** 4], axis=1)
    (a4, b4), *_ = np.linalg.lstsq(A, dE, rcond=None)

    # [γ 塌缩] 无自由参数：E = s · gamma · E_rest
    gr = gam * E_rest_w
    s_coll = float(np.sum(gr * E_arr) / np.sum(gr ** 2))

    # [动量] P = p1 v + p2 v^3；动量侧 gamma 塌缩幅度 M_P
    Ap = np.stack([v_arr, v_arr ** 3], axis=1)
    (p1, p2), *_ = np.linalg.lstsq(Ap, P_arr, rcond=None)
    gP = gam * v_arr / c_s
    M_P = float(np.sum(gP * P_arr) / np.sum(gP ** 2))

    # [不变量] R = (E² - c_s² P²)/E_rest²
    R = (E_arr ** 2 - (c_s * P_arr) ** 2) / E_rest_w ** 2
    R_spread = float((R.max() - R.min()) / 2)
    E2_spread = float((E_arr ** 2).max() / E_rest_w ** 2 - 1.0)
    R_at02 = float(np.interp(0.20, v_arr, R))

    results[Kv] = dict(ell=ell, E_rest=E_rest_w, M_zero=M_zero, M_shape=M_shape,
                       m_newt=m_newt, M_rel=M_rel, res_newt=res_newt, res_rel=res_rel,
                       a4=a4, b4=b4, s_coll=s_coll, p1=p1, p2=p2, M_P=M_P,
                       R=R, R_spread=R_spread, E2_spread=E2_spread, R_at02=R_at02,
                       rows=rows, lorentz_rows=lorentz_rows, gam=gam, v=v_arr,
                       E=E_arr, P=P_arr, dE=dE)

    print(f"  [形状] 牛顿残差 {res_newt:.3f} vs SG 残差 {res_rel:.3f}"
          f"（{'SG 优' if res_rel < res_newt else '异常'}）；"
          f"M_rel = {M_rel:.4f}；v⁴ 系数 b = {b4:.5f} 对 SG 预期 3M/8 = {3 * M_rel / 8:.5f}")
    print(f"  [γ 塌缩] s = E/(gamma·E_rest) = {s_coll:.4f}（无自由参数，期望 ≈ 1）")
    print(f"  [动量] p1 = {p1:.4f}（M_zero = {M_zero:.4f}，比值 {p1 / M_zero:.3f}）；"
          f"p2/p1 = {p2 / p1:.3f}（相对论预期 0.5，牛顿 0）；"
          f"M_P = {M_P:.4f}（比值 M_P/M_zero = {M_P / M_zero:.3f}）")
    print(f"  [不变量] R = (E² - c_s²P²)/E_rest² = "
          f"{np.array2string(R, precision=3, separator=', ')}")
    print(f"          R 半展宽 = {R_spread:.4f}；对照：E²/E_rest² 本身的相对变化 = {E2_spread:.3f}")

# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("汇总：SG 偏差与不变量质量对 kink 宽度的依赖（宽度趋势，双侧）")
print("=" * 72)
print(f"{'K':>8} {'ell':>7} {'res_newt':>9} {'res_rel':>8} {'s':>7} {'M_P/M0':>7} "
      f"{'R@v=.2':>7} {'R_均值':>7} {'R_半展宽':>8}")
for Kv, r in results.items():
    print(f"{Kv:8.4f} {r['ell']:7.2f} {r['res_newt']:9.3f} {r['res_rel']:8.3f} "
          f"{r['s_coll']:7.3f} {r['M_P'] / r['M_zero']:7.3f} "
          f"{r['R_at02']:7.4f} {r['R'].mean():7.3f} {r['R_spread']:8.4f}")

# 协议独立性（K = 0.005：洛伦兹 boost 点对简单 boost 族的 SG 拟合曲线；
# 注意洛伦兹 IC 在高 v0 段到达简单族未覆盖的速度（0.71 > 0.64），故对照
# SG 拟合曲线而非逐点插值，插值在高段会截断为端点值造成假偏差）
r5 = results[0.005]
if r5["lorentz_rows"]:
    M_rel_s, M_P_s, Erest5 = r5["M_rel"], r5["M_P"], r5["E_rest"]
    print("\n[协议独立性] 洛伦兹 boost 初条件（B）对简单 boost 族（A）SG 拟合曲线（K = 0.005）")
    print(f"    A 族拟合质量：M_rel = {M_rel_s:.4f}，动量质量 M_P = {M_P_s:.4f}")
    devs = []
    for r in r5["lorentz_rows"]:
        gamL = 1.0 / np.sqrt(1.0 - r["v"] ** 2)
        dE_pred = M_rel_s * (gamL - 1.0)
        P_pred = M_P_s * gamL * r["v"]
        dev_E = (r["E"] - Erest5 - dE_pred) / dE_pred
        dev_P = (r["P"] - P_pred) / P_pred
        R_L = (r["E"] ** 2 - r["P"] ** 2) / Erest5 ** 2
        devs.append(abs(dev_E))
        print(f"    v0 = {r['v0']:4.2f}: v = {r['v']:.4f}，ΔE 偏差 = {dev_E:+.3f}，"
              f"P 偏差 = {dev_P:+.3f}，R = {R_L:.4f}")
    print(f"    最大 |ΔE 偏差| = {np.nanmax(devs):.3f}"
          f"（{'两族初条件落在同一色散曲线上' if np.nanmax(devs) < 0.15 else '超出阈值，需检查'}）")

print("\n验证完成。")

# ----------------------------------------------------------------------
# 图：四联版，归一化量按 E_rest 无量纲化后各 K 应当塌缩到同一条 SG 曲线
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    markers = {0.0005: "o", 0.001: "s", 0.005: "^", 0.01: "D", 0.02: "v"}
    colors = {0.0005: "#1f77b4", 0.001: "#2ca02c", 0.005: "#d62728",
              0.01: "#9467bd", 0.02: "#ff7f0e"}
    vv = np.linspace(0.01, 0.92, 300)
    gg = 1.0 / np.sqrt(1.0 - vv ** 2)
    r5 = results[0.005]
    if r5["lorentz_rows"]:
        vL = np.array([q["v"] for q in r5["lorentz_rows"]])
        EL = np.array([q["E"] for q in r5["lorentz_rows"]])
        PL = np.array([q["P"] for q in r5["lorentz_rows"]])
        dEL = (EL - r5["E_rest"]) / r5["E_rest"]
        PLn = PL * c_s / r5["E_rest"]
        RL = (EL ** 2 - (c_s * PL) ** 2) / r5["E_rest"] ** 2
    else:
        vL = None

    a = ax[0, 0]
    a.plot(vv, gg - 1, "k-", lw=1.2, label=r"SG: $\gamma-1$")
    a.plot(vv, 0.5 * vv ** 2, "gray", ls="--", lw=1.2, label=r"牛顿: $v^2/2$")
    for Kv, r in results.items():
        a.plot(r["v"], r["dE"] / r["E_rest"], markers[Kv], color=colors[Kv],
               ms=5, mfc="none", label=f"K={Kv}")
    if vL is not None:
        a.plot(vL, dEL, "x", color=colors[0.005], ms=7, label="K=0.005 洛伦兹IC")
    a.set_xlabel(r"$v/c_s$"); a.set_ylabel(r"$\Delta E / E_{rest}$")
    a.set_title("(a) 能量-速度关系（归一化塌缩检验）")
    a.legend(fontsize=7); a.set_ylim(-0.05, 1.1)

    a = ax[0, 1]
    a.plot(vv, gg * vv, "k-", lw=1.2, label=r"SG: $\gamma v$")
    a.plot(vv, vv, "gray", ls="--", lw=1.2, label=r"牛顿: $v$")
    for Kv, r in results.items():
        a.plot(r["v"], r["P"] * c_s / r["E_rest"], markers[Kv], color=colors[Kv],
               ms=5, mfc="none", label=f"K={Kv}")
    if vL is not None:
        a.plot(vL, PLn, "x", color=colors[0.005], ms=7)
    a.set_xlabel(r"$v/c_s$"); a.set_ylabel(r"$|P|c_s / E_{rest}$")
    a.set_title("(b) 动量-速度关系（补齐 m_eff 未测的扇区）")
    a.legend(fontsize=7)

    a = ax[1, 0]
    a.axhline(1.0, color="k", lw=1.0, label="不变量 = 1")
    a.plot(vv, gg ** 2, "gray", ls="--", lw=1.2, label=r"$E^2/E_{rest}^2=\gamma^2$")
    for Kv, r in results.items():
        a.plot(r["v"], r["R"], markers[Kv], color=colors[Kv], ms=5, mfc="none",
               label=f"K={Kv}")
    if vL is not None:
        a.plot(vL, RL, "x", color=colors[0.005], ms=7)
    a.set_xlabel(r"$v/c_s$"); a.set_ylabel(r"$R=(E^2-c_s^2P^2)/E_{rest}^2$")
    a.set_title("(c) 洛伦兹型不变量的速度平坦性")
    a.legend(fontsize=7); a.set_ylim(0.5, 2.4)

    a = ax[1, 1]
    Ks = sorted(results.keys())
    ells = [results[Kv]["ell"] for Kv in Ks]
    a.plot(ells, [results[Kv]["res_rel"] for Kv in Ks], "o-", color="#d62728",
           label="SG 拟合残差")
    a.plot(ells, [results[Kv]["res_newt"] for Kv in Ks], "s--", color="gray",
           label="牛顿拟合残差")
    for Kv in Ks:
        a.annotate(f"K={Kv}", (results[Kv]["ell"], results[Kv]["res_rel"]),
                   textcoords="offset points", xytext=(4, 4), fontsize=7)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"kink 宽度 $\ell$（格距）"); a.set_ylabel("拟合相对残差")
    a.set_title("(d) 宽度趋势：SG 形状随 kink 变宽收敛")
    a.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/minimal_verification_dispersion.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存：{out}")
except Exception as e:
    print(f"绘图跳过（matplotlib 异常：{e}），不影响数值结果")
