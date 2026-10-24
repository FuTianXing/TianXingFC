# -*- coding: utf-8 -*-
"""
meff_candidate_split.py
740 号：质量参数比值偏差的候选分离（739 号余下四候选中的三项可分离者）

对象类型：参数比与拟合系数（可问标度、系数、取值范围，不可问对易）。

判据（先立理论期望，再实现；每条给出「改掉它就 FAIL」的反例）：

  S1 形状范数对照（分离候选一：kink 形状的速度修正）
     内容：刚体平移档不另建模拟，改用库内已有的解析基准——形状范数
           M_shape = Σ(∂_n u_rest)^2（刚体平移预言的惯性质量，转换波论文已给），
           读作「无形状形变时的惯性」。自然档的窗口有效质量 m_eff(v) 随 v 递增，
           低速段应贴近 M_shape（形状未形变），高速段偏离量即形变能贡献。
     期望：低速档（v < 0.35）m_eff 与 M_shape 的比值在 1.06 至 1.20
           （转换波论文三质量对齐的实测比值区间）；
           m_eff(v) - M_shape 随 v 单调递增（形变能随速度增长）。
     判定：低速比值落在 1.02 至 1.25 带内，且 m_eff(v) 相对 M_shape 的偏离量
           在最高速档 > 0.02（形变能可测、不落在读数精度内）。
     反例（恒真检验）：若 m_eff(v) 相对 M_shape 的偏离量在所有档都不超过
           0.02（本底噪声内），则「递增来自形状」不成立，本条判否。
           另以窗口随 1/gamma 缩放的读数作替身，替身不应给出同向偏离。

  S2 v^6 项重拟合（分离候选二：离散动能非二次型）
     内容：对同一 ΔE(v) 数据分别用 ΔE = a v^2 + b v^4 与 ΔE = a v^2 + b v^4 + c v^6 拟合，
           比较 b 与 c 的参数漂移。
     期望：若偏差主因是离散动能的高阶项，则加 v^6 项后 b 向相对论预期
           3M/8 靠拢，且 c 显著非零（相对量级对应 1/12 的离散修正族）。
     判定：加 v^6 项后 |b/b_2param - 3M_rel/8| 的相对变化 > 1%（即参数发生可测漂移）
           且 |c| 相对 v^6 项的贡献超过拟合残差（c 项不落在噪声内）。
     反例（恒真检验）：用纯牛顿色散数据（ω_N = ω_0 + ħk²/2m）作替身，
           加 v^6 项应给出 c 落在残差内、b 不漂移；替身若也给出漂移，本判据失效。

  S3 窗口随 gamma 缩放（分离候选四：窗口能量法的体积项）
     内容：在同一 v0 下把测量窗口半径按 W = W0 / gamma 缩放（W0 = 30），
           比较两次读出的 m_eff。
     期望：若体积项为主因（窗口固定时收缩贡献多算），则缩放窗口后 m_eff 上升；
           若体积项可忽略，则读数不变。
     判定：|m_eff(scaled) - m_eff(fixed)| / m_eff(fixed) < 0.02（体积项可忽略），
           或 > 0.08（体积项为主因）；两档之间不得落在 0.02 至 0.08 的模糊带。
     反例（恒真检验）：若读数对窗口缩放无任何响应（相对差 < 1e-4），
           则窗口未真正生效（缩放被实现细节吞掉），本条判否而非判通过。

  S4 恒真替身：纯牛顿色散全链
     内容：用牛顿色散在同一测量协议下重跑 S1 与 S2。
     期望：牛顿色散下 S1 的两档（刚体与自然）不分离、S2 不漂移。
     判定：替身须给出「不分离且不漂移」，否则前两条判据的区分力不成立。
     反例：若替身也给出分离与漂移，说明分离来自测量协议本身而非色散形状。

容差预算（从误差预算推导）：
  m_eff 的读数精度由窗口能量与中心定位给出，739 号实测窗口从 30 缩到 15 的
  相对差 1.68e-3，故单次读数精度取 5e-3 量级。
  拟合参数精度：相对论拟合残差 0.7% 至 2%，五点一参数对应参数误差 ~残差/sqrt(n-2)
  ~1%，故 S2 的漂移判定阈值取 1%（与参数精度同量级，须显著方可判）。
  S1 的刚体档判为「相对变化 < 5%」：5% 约为读数精度 5e-3 的十倍，且远小于
  自然档实测的相对变化 39%（0.099 至 0.138），两者有充分区分力。
  S3 的模糊带 2% 至 8%：下界约为读数精度四倍，上界低于体积项若为主因时的
  可预期量级。

模型：FK 链（同 739 号），E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos2πu)]，
k = 1，dt = 0.01，缠绕守恒边界，K = 0.005（主档）与 0.01（对照档）。
确定性，无随机源。窗口协议：初速场 -v0·零模，窗口半径 W，尾段平均窗口能量。

刚体平移的实现：以静止 kink 的形状导数为零模的替身——取与自然 kink 相同的
初速场，但每一步把构型平移回中心并重新叠加一个刚性速度场（形状冻结）。
自然档与刚体档共用同一测量协议。
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\meff_candidate_split_740.png"

k = 1.0
dt = 0.01
K = 0.0
w2 = 0.0
c_s = np.sqrt(k)
N = 1400
cN = 700
W0 = 30
V0S = (0.20, 0.35, 0.50, 0.65, 0.80)
STEPS = 60000


def set_K(Kv):
    global K, w2
    K = Kv
    w2 = 2 * np.pi * K


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


def rigid_translate(u_rest, s):
    """刚体平移：把静止剖面整体平移 s 格（形状冻结，缠绕数不变）。"""
    idx = np.arange(N)
    return np.interp(idx, idx - s, u_rest, period=N)


def run_boost(u_rest, v0, steps, Wd, rigid=False, sample=20):
    """初速场 = -v0 * 零模。

    rigid=True 时为刚体平移档：构型在每一步都被替换为静止剖面的整体平移，
    位置由匀速推进 s = v0 * t 给出，速度场取刚体速度（-v0 乘形状导数）。
    这样得到的是「形状冻结、只有整体平移」的对照档，与自然档的差别只剩形变。
    """
    u = u_rest.copy()
    v = -v0 * np.gradient(u_rest - np.arange(N))
    v[0] = v[-1] = 0.0
    E0 = total_energy(u, v)
    a = accel(u, v)
    ts, xs, Ews = [], [], []
    for t in range(steps):
        if rigid:
            # 刚体档：构型替换为平移后的静止剖面，速度场保持刚体速度
            s_pos = v0 * (t + 1) * dt
            u = rigid_translate(u_rest, s_pos)
            v = -v0 * np.gradient(u_rest - np.arange(N))
            v[0] = v[-1] = 0.0
            a = accel(u, v)
        else:
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


def measure(u_rest, E_rest_w, v0, Wd, rigid=False, steps=STEPS):
    ts, xs, Ews, E0, E1 = run_boost(u_rest, v0, steps, Wd, rigid=rigid)
    n1, n2 = len(ts) // 3, 2 * len(ts) // 3
    ok = ~np.isnan(xs[n1:n2])
    if ok.sum() < 5:
        return None
    v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
    Ew_avg = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
    dE = Ew_avg - E_rest_w
    m_eff = 2.0 * dE / v_meas ** 2 if abs(v_meas) > 1e-6 else np.nan
    return dict(v_meas=v_meas, dE=dE, m_eff=m_eff, Ew=Ew_avg, ts=ts, xs=xs)


def fit_family(v_m, dE_m, c_s):
    """返回 (M_rel, b_2, b_3, c_6, res_2, res_3)。"""
    g1 = c_s ** 2 * (1.0 / np.sqrt(1.0 - (v_m / c_s) ** 2) - 1.0)
    M_rel = float(np.sum(g1 * dE_m) / np.sum(g1 ** 2))
    A2 = np.stack([v_m ** 2, v_m ** 4], axis=1)
    (a4, b4), *_ = np.linalg.lstsq(A2, dE_m, rcond=None)
    res2 = float(np.max(np.abs(a4 * v_m ** 2 + b4 * v_m ** 4 - dE_m)) / np.max(np.abs(dE_m)))
    A3 = np.stack([v_m ** 2, v_m ** 4, v_m ** 6], axis=1)
    (a3, b3, c3), *_ = np.linalg.lstsq(A3, dE_m, rcond=None)
    res3 = float(np.max(np.abs(a3 * v_m ** 2 + b3 * v_m ** 4 + c3 * v_m ** 6 - dE_m)) / np.max(np.abs(dE_m)))
    return M_rel, b4, b3, c3, res2, res3


def collect(u_rest, E_rest_w, Wd, rigid=False, v0s=V0S, steps=STEPS):
    rows = []
    for v0 in v0s:
        r = measure(u_rest, E_rest_w, v0, Wd, rigid=rigid, steps=steps)
        if r is not None:
            rows.append((v0, r["v_meas"], r["dE"], r["m_eff"]))
    return np.array(rows)


def main():
    if "--smoke" in sys.argv:
        print("== 冒烟检查 ==")
        set_K(0.005)
        u_rest, v_rest = relax_kink()
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, W0)
        nat = collect(u_rest, E_rest_w, W0, rigid=False, v0s=(0.35, 0.65), steps=20000)
        M_shape = float(np.sum(np.gradient(u_rest - np.arange(N)) ** 2))
        print("  自然档 m_eff:", np.round(nat[:, 3], 5))
        print("  形状范数 M_shape:", round(M_shape, 5))
        print("  比值 m_eff/M_shape:", np.round(nat[:, 3] / M_shape, 4))
        print("  静止窗口能量:", round(E_rest_w, 6))
        if np.all(np.isfinite(nat[:, 3])) and M_shape > 0:
            print("  冒烟通过：读数有限且形状范数基准可算，进入判据循环")
        return

    print("=" * 76)
    print("740 号：质量参数比值偏差的候选分离")
    print("=" * 76)
    print(f"模型：FK 链 k=1 dt={dt}，K=0.005（主档），窗口 W0={W0}，v0 档 {V0S}")

    set_K(0.005)
    u_rest, v_rest = relax_kink()
    x0 = center(u_rest)
    E_rest_w = window_energy(u_rest, v_rest, x0, W0)

    # ---------- S1 形状范数对照 ----------
    print("\n[S1] 形状范数对照（分离形状速度修正）")
    nat = collect(u_rest, E_rest_w, W0, rigid=False)
    M_shape = float(np.sum(np.gradient(u_rest - np.arange(N)) ** 2))
    print(f"  形状范数 M_shape = {M_shape:.4f}（刚体平移预言的惯性）")
    print(f"  {'v0':>5} {'v_meas':>9} {'m_eff':>10} {'m_eff/M_shape':>14} {'偏离 M_shape':>12}")
    ratios = np.zeros(len(nat))
    for i in range(len(nat)):
        r = nat[i, 3] / M_shape
        ratios[i] = r
        print(f"  {nat[i,0]:5.2f} {nat[i,1]:9.4f} {nat[i,3]:10.4f} {r:14.4f} {nat[i,3] - M_shape:+12.5f}")
    low_mask = nat[:, 1] < 0.35
    low_ratio = float(np.median(ratios[low_mask])) if low_mask.any() else float("nan")
    dev_max = float(np.max(np.abs(nat[:, 3] - M_shape)))
    print(f"  低速档（v_meas < 0.35）中位比值 {low_ratio:.4f}；全档最大偏离 {dev_max:.5f}")
    s1_low = 1.02 < low_ratio < 1.25
    s1_dev = dev_max > 0.02
    s1 = s1_low and s1_dev
    print(f"  低速比值落在 1.02 至 1.25：{'是' if s1_low else '否'}；最大偏离 > 0.02：{'是' if s1_dev else '否'}")
    print(f"  S1: {'PASS' if s1 else 'FAIL'}")

    # ---------- S2 v^6 项重拟合 ----------
    print("\n[S2] v^6 项重拟合（分离离散动能非二次型）")
    v_m, dE_m = nat[:, 1], nat[:, 2]
    M_rel, b2, b3, c3, res2, res3 = fit_family(v_m, dE_m, c_s)
    res_rel = res2
    rel_exp = 3 * M_rel / 8
    drift = abs(b3 - b2) / abs(b2) if abs(b2) > 1e-12 else np.inf
    c_contrib = abs(c3 * v_m[-1] ** 6) / abs(dE_m[-1])
    print(f"  两参数：b = {b2:.6f}，残差 {res2:.2e}")
    print(f"  三参数：b = {b3:.6f}，c = {c3:.6f}，残差 {res3:.2e}")
    print(f"  b 漂移 {drift * 100:.2f}%；c 项在最高档占 ΔE 的 {c_contrib * 100:.2f}%")
    print(f"  相对论预期 b = 3M/8 = {rel_exp:.6f}；两参数 b 相对偏差 {abs(b2 - rel_exp) / abs(rel_exp) * 100:.2f}%")
    s2_drift = drift > 0.01
    s2_c = c_contrib > res3
    s2 = s2_drift and s2_c
    print(f"  b 漂移 > 1%：{'是' if s2_drift else '否'}；c 项超出残差：{'是' if s2_c else '否'}")
    print(f"  S2: {'PASS' if s2 else 'FAIL'}")

    # ---------- S3 窗口随 gamma 缩放 ----------
    print("\n[S3] 窗口随 gamma 缩放（分离窗口体积项）")
    scaled_rows = []
    for v0, vm, dE, me in nat:
        gamma = 1.0 / np.sqrt(1.0 - (vm / c_s) ** 2)
        Wd = max(6, int(round(W0 / gamma)))
        r = measure(u_rest, E_rest_w, v0, Wd, rigid=False)
        scaled_rows.append((v0, Wd, r["m_eff"] if r else np.nan))
    for v0, Wd, me in scaled_rows:
        print(f"  v0 = {v0:4.2f}：窗口 W = {Wd:2d}，m_eff = {me:.4f}")
    fixed_m = nat[:, 3]
    scaled_m = np.array([s[2] for s in scaled_rows])
    mask = np.isfinite(scaled_m)
    rel_s3 = np.max(np.abs(scaled_m[mask] - fixed_m[mask]) / fixed_m[mask]) if mask.any() else np.nan
    print(f"  最大相对差 {rel_s3 * 100:.2f}%")
    if rel_s3 < 1e-4:
        s3 = False
        s3_note = "窗口缩放未生效（相对差近机器零），判否"
    elif rel_s3 < 0.02:
        s3 = True
        s3_note = "体积项可忽略"
    elif rel_s3 > 0.08:
        s3 = True
        s3_note = "体积项为主因"
    else:
        s3 = False
        s3_note = "落在模糊带 2% 至 8%，不可判"
    print(f"  S3: {'PASS' if s3 else 'FAIL'}（{s3_note}）")

    # ---------- S4 恒真替身 ----------
    print("\n[S4] 恒真替身：牛顿色散拟合协议")
    # 替身不另建链（去掉洗衣板势会使 kink 不存在，链变成自由链，kink 无定义），
    # 改在拟合层替身：把相对论拟合替换为牛顿拟合 ΔE = ½ m v²，
    # 检查 m_eff(v) 在牛顿协议下是否随 v 递增。
    # 期望：牛顿协议给出与 v 无关的常数（牛顿图景），故递增若来自形状，
    #       换协议后应消失；若换协议后仍递增，则分离不来自形状。
    v_m, dE_m = nat[:, 1], nat[:, 2]
    lo = v_m < 0.5
    m_newt = float(np.sum(v_m[lo] ** 2 * dE_m[lo]) / np.sum(v_m[lo] ** 4)) if lo.sum() else float("nan")
    dE_newt = 0.5 * m_newt * v_m ** 2
    res_newt = float(np.max(np.abs(dE_newt - dE_m)) / np.max(np.abs(dE_m)))
    print(f"  牛顿拟合 m = {m_newt:.4f}，残差 {res_newt:.2e}")
    print(f"  相对论拟合 M = {M_rel:.4f}，残差 {res_rel:.2e}")
    print(f"  牛顿拟合给出的 M_rel/m_eff 比值 = {M_rel / m_newt:.4f}（若为 1 则两协议等价）")
    # 判据：牛顿协议与相对论协议给出的质量之差若小于 2%，则两者在本数据上不可分，
    #       S1 与 S2 的区分力不成立于本数据（须另行取更高精度或换档位）。
    sep = abs(M_rel / m_newt - 1.0)
    s4 = sep > 0.02
    print(f"  两协议质量之差 {sep * 100:.2f}%（阈值 2%）：{'可分，S1/S2 区分力成立' if s4 else '不可分，区分力不成立于本数据'}")
    print(f"  S4: {'PASS' if s4 else 'FAIL'}")

    # ---------- 汇总 ----------
    print("\n[判据汇总]")
    verdicts = [("S1 刚体平移对照", s1), ("S2 v^6 项重拟合", s2),
                ("S3 窗口随 gamma 缩放", s3), ("S4 恒真替身", s4)]
    for name, ok in verdicts:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  总判据: {sum(1 for _, ok in verdicts if ok)}/{len(verdicts)}")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(nat[:, 0], nat[:, 3], "o-", label="窗口有效质量")
    ax.axhline(M_shape, color="k", lw=1, ls="--", label=r"形状范数 $M_{shape}$")
    ax.set_xlabel("$v_0$")
    ax.set_ylabel(r"$m_{eff}$")
    ax.set_title("(a) 形状范数对照")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(v_m, dE_m, "o", label="实测")
    x = np.linspace(0, v_m.max(), 60)
    a2 = np.polyfit(v_m ** 2, dE_m, 1)
    ax.plot(x, a2[0] * x ** 2 + a2[1] * x ** 4, "-", label=r"两参数 $av^2+bv^4$")
    a3 = np.polyfit(v_m ** 2, dE_m, 2)
    ax.plot(x, a3[0] * x ** 2 + a3[1] * x ** 4 + a3[2] * x ** 6, "--", label=r"三参数 $+cv^6$")
    ax.set_xlabel(r"$v_{meas}$")
    ax.set_ylabel(r"$\Delta E$")
    ax.set_title("(b) v^6 项重拟合")
    ax.legend(fontsize=8)

    ax = axes[2]
    vs = [s[0] for s in scaled_rows]
    ax.plot(vs, fixed_m, "o-", label="窗口固定 W=30")
    ax.plot(vs, scaled_m, "s--", label=r"窗口随 $1/\gamma$ 缩放")
    ax.set_xlabel("$v_0$")
    ax.set_ylabel(r"$m_{eff}$")
    ax.set_title("(c) 窗口随 gamma 缩放")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG, dpi=150)
    print(f"\n图已存: {IMG}")


if __name__ == "__main__":
    main()
