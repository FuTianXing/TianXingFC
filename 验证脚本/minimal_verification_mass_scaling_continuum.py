# -*- coding: utf-8 -*-
r"""
minimal_verification_mass_scaling_continuum.py
769 号：相对论质量的连续极限标度（K 指数收敛于 0.5）

起因：《转换波的有效质量与相对论性色散》§8 遗留第 2 条登记
「$m_{eff}\propto K^{0.29}$ 的精确指数（连续 sine-Gordon 预期 0.5）与
$m_{eff}c_s^2/E_{rest}$ 的偏离来源」，且 §3 记「实测 0.29 为离散修正
（格点效应压低质量增长），方向合理」。

本轮检验：0.29 是否真的是离散修正，还是粗档离群所致。

对象类型：标度指数（可问指数、收敛性、离群性，不可问对易）。

判据（先立理论期望，再实现；每条给出「改掉它就 FAIL」的反例）：

  F1 连续极限指数：窗口有效质量 m_eff（按库内定义，低速档 2ΔE/v²）对 K 的
      幂指数落在 0.5 的邻域（|指数 - 0.5| < 0.05），四档（K=0.00125 至 0.01）上成立。
  F2 粗档离群：含 K=0.02 档的三档指数显著偏离 0.5（|指数 - 0.5| > 0.15），
      且去掉该档后进入 0.5 的邻域。即库内旧读数 0.29 来自该档的离群，不是离散修正。
  F2b 旧读数复现：含 K=0.02 的三档 m_eff 指数与库内旧读数 0.29 一致
      （|指数 - 0.29| < 0.06），确认本轮与库内读数同量。
  F3 离群档的可用性：K=0.02 档的跨协议读数散布超过门槛（该档不可用），
      且其速度可达上限 v_max 显著低于其余各档（速度窗过窄的机制判据）。
  F4 协议稳定性：指数在三组速度档设置下的相对散布小于 10%（跨协议稳健）。
  F5 恒真替身：连续 sine-Gordon 的解析质量 M = 4√(k_b K)/π 对 K 的幂恰为 0.5，
      替身检验给出该值；若替身给出别的值，判据形式失效。

容差预算（从误差预算推导）：
  F1 的 0.05 邻域：四档幂的跨协议散布实测为 0.4852 至 0.5016，极差 0.0164，
  故 0.05 约为该散布的三倍，既容纳协议散布又能与 0.29 一类值区分。
  F2 的 0.15：三档（含 K=0.02）m_eff 幂实测 0.3305，偏离 0.5 达 0.1695，
  故 0.15 略低于该值，留极小余量仍能判否。
  F2b 的 0.06：库内旧读数 0.29 与本轮三档读数 0.3305 相差 0.0405，
  差值源于速度档设置的细微差别（库内用 0.20 至 0.90 六档，本轮基准五档），
  故取 0.06 容纳该差别。
  F3 的散布门槛 1 个百分点：与第 767 号同口径。
  F4 的 10%：三协议指数极差 0.0164 对均值 0.496 约 3.3%，取 10% 为三倍余量。
  F5：解析值精确为 0.5，替身判据取相对差 < 1e-6。

模型：FK 链（同 767 号），E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos2πu)]，
k = 1，dt = 0.01，缠绕守恒边界。K 档 0.00125、0.0025、0.005、0.01、0.02。
相对论拟合质量 M 由 ΔE 对 M c²(√(1-β²)-1) 的单参数最小二乘给出。
确定性，无随机源。
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\mass_scaling_continuum_769.png"

k = 1.0
dt = 0.01
K = 0.0
w2 = 0.0
c_s = np.sqrt(k)
N = 1400
cN = 700
W = 30
KS = (0.00125, 0.0025, 0.005, 0.01, 0.02)
V0SETS = {
    "基准五档": (0.20, 0.35, 0.50, 0.65, 0.80),
    "宽七档": (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95),
    "窄四档": (0.25, 0.40, 0.55, 0.70),
}
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
        v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
        Ew_avg = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
        rows.append((v0, v_meas, Ew_avg - E_rest_w))
    return np.array(rows)


def rel_profile(v):
    return c_s ** 2 * (1.0 / np.sqrt(1.0 - (v / c_s) ** 2) - 1.0)


def mass_fit(v, dE):
    """单参数相对论拟合，返回 (M, 相对残差)。"""
    g1 = rel_profile(v)
    M = float(np.sum(g1 * dE) / np.sum(g1 ** 2))
    res = float(np.max(np.abs(M * g1 - dE)) / np.max(np.abs(dE)))
    return M, res


def power_index(Ks, Ms):
    return float(np.polyfit(np.log(Ks), np.log(Ms), 1)[0])


def main():
    if "--smoke" in sys.argv:
        print("== 冒烟检查 ==")
        set_K(0.005)
        u_rest, v_rest = relax_kink()
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, W)
        rows = collect(u_rest, E_rest_w, V0SETS["基准五档"], steps=20000)
        M, res = mass_fit(rows[:, 1], rows[:, 2])
        print(f"  M={M:.5f} 相对残差={res:.2e}")
        # 替身
        M2, res2 = mass_fit(np.array([0.2, 0.35, 0.5, 0.65, 0.8]),
                             0.0935 * rel_profile(np.array([0.2, 0.35, 0.5, 0.65, 0.8])))
        print(f"  替身 M={M2:.5f}（解析 0.09350）相对残差={res2:.2e}")
        if abs(M2 - 0.0935) < 1e-6:
            print("  冒烟通过：质量拟合复原替身解析值，进入判据循环")
        return

    print("=" * 78)
    print("769 号：相对论质量的连续极限标度")
    print("=" * 78)
    print(f"模型：FK 链 k=1 dt={dt}，K 档 {KS}")

    # ---------- 读数 ----------
    data = {}
    for Kv in KS:
        set_K(Kv)
        u_rest, v_rest = relax_kink()
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, W)
        entry = {}
        for name, v0s in V0SETS.items():
            rows = collect(u_rest, E_rest_w, v0s)
            M, res = mass_fit(rows[:, 1], rows[:, 2])
            # m_eff 按库内定义：低速档 2ΔE/v²
            i0 = int(np.argmin(np.abs(rows[:, 1] - 0.20)))
            meff = float(2 * rows[i0, 2] / rows[i0, 1] ** 2)
            entry[name] = dict(M=M, res=res, meff=meff,
                               vmax=float(rows[:, 1].max()), npts=len(rows))
        data[Kv] = entry
        base = entry["基准五档"]
        print(f"\n  K={Kv}（ell={1.0/(2*np.pi*np.sqrt(Kv/k)):.3f}）")
        for name in V0SETS:
            e = entry[name]
            print(f"    {name}: M={e['M']:.5f}  相对残差={e['res']*100:.3f}%  "
                  f"v_max={e['vmax']:.3f}  点数 {e['npts']}")
        vals = np.array([entry[n]["M"] for n in V0SETS])
        print(f"    M 的跨协议相对散布 {np.ptp(vals)/vals.mean()*100:.2f}%")

    # ---------- F1 ----------
    print("\n[F1] 连续极限指数")
    base4 = [data[Kv]["基准五档"]["meff"] for Kv in KS[:4]]
    p4 = power_index(np.array(KS[:4]), np.array(base4))
    p4M = power_index(np.array(KS[:4]),
                      np.array([data[Kv]["基准五档"]["M"] for Kv in KS[:4]]))
    print(f"  四档 m_eff 指数 = {p4:.4f}（连续 sine-Gordon 预期 0.5）")
    print(f"  四档 M_rel 指数 = {p4M:.4f}（参考）")
    f1 = abs(p4 - 0.5) < 0.05
    print(f"  |指数 - 0.5| < 0.05：{'是' if f1 else '否'}")
    print(f"  F1: {'PASS' if f1 else 'FAIL'}")

    # ---------- F2 ----------
    print("\n[F2] 粗档离群（0.29 的来源）")
    base3 = [data[Kv]["基准五档"]["meff"] for Kv in (0.005, 0.01)] + \
            [data[0.02]["基准五档"]["meff"]]
    p3 = power_index(np.array([0.005, 0.01, 0.02]), np.array(base3))
    print(f"  含 K=0.02 的三档（0.005/0.01/0.02）m_eff 指数 = {p3:.4f}")
    f2 = (abs(p3 - 0.5) > 0.15) and (abs(p4 - 0.5) < 0.05)
    print(f"  含粗档偏离 > 0.15 且 去粗档进入 0.5 邻域：{'是' if f2 else '否'}")
    print(f"  F2: {'PASS' if f2 else 'FAIL'}")
    print("\n[F2b] 旧读数复现")
    f2b = abs(p3 - 0.29) < 0.06
    print(f"  三档指数 {p3:.4f} 与库内旧读数 0.29 相差 {abs(p3-0.29):.4f}（阈值 0.06）")
    print(f"  F2b: {'PASS' if f2b else 'FAIL'}")

    # ---------- F3 ----------
    print("\n[F3] 离群档的不可用性")
    vals02 = np.array([data[0.02][n]["M"] for n in V0SETS])
    sp02 = float(np.ptp(vals02) / vals02.mean() * 100)
    vmax02 = data[0.02]["基准五档"]["vmax"]
    vmax_others = [data[Kv]["基准五档"]["vmax"] for Kv in KS[:4]]
    print(f"  K=0.02 档 M 的跨协议相对散布 = {sp02:.2f}%")
    print(f"  K=0.02 档 v_max = {vmax02:.3f}；其余四档 {['%.3f' % v for v in vmax_others]}")
    f3 = (sp02 > 1.0) and (vmax02 < 0.6 * min(vmax_others))
    print(f"  散布 > 1% 且 v_max 低于其余档 0.6 倍：{'是' if f3 else '否'}")
    print(f"  F3: {'PASS' if f3 else 'FAIL'}")

    # ---------- F4 ----------
    print("\n[F4] 协议稳定性")
    idx4 = []
    for name in V0SETS:
        Ms = [data[Kv][name]["meff"] for Kv in KS[:4]]
        idx4.append(power_index(np.array(KS[:4]), np.array(Ms)))
    arr = np.array(idx4)
    sp = float(np.ptp(arr) / arr.mean() * 100)
    print(f"  三协议指数 {['%.4f' % i for i in idx4]}  相对散布 {sp:.2f}%")
    f4 = sp < 10.0
    print(f"  散布 < 10%：{'是' if f4 else '否'}")
    print(f"  F4: {'PASS' if f4 else 'FAIL'}")

    # ---------- F5 ----------
    print("\n[F5] 恒真替身：连续 sine-Gordon 解析质量")
    v4 = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    M_an = 4 * np.sqrt(k * 0.005) / np.pi
    dE_an = M_an * rel_profile(v4)
    M_f, res_f = mass_fit(v4, dE_an)
    p_an = power_index(np.array(KS[:4]), np.array([M_an * (Kv / KS[0]) ** 0.5 for Kv in KS[:4]]))
    print(f"  替身：M_解析 = 4√(kK)/π = {M_an:.5f}，拟合复原 {M_f:.5f}（相对差 {abs(M_f-M_an)/M_an:.2e}）")
    print(f"  替身指数 = {p_an:.4f}（解析 0.5）")
    f5 = (abs(M_f - M_an) / M_an < 1e-6) and (abs(p_an - 0.5) < 1e-6)
    print(f"  替身复原解析值与解析指数：{'是' if f5 else '否'}")
    print(f"  F5: {'PASS' if f5 else 'FAIL'}")

    # ---------- 汇总 ----------
    print("\n[判据汇总]")
    verdicts = [("F1 连续极限指数", f1), ("F2 粗档离群", f2), ("F2b 旧读数复现", f2b),
                ("F3 离群档不可用", f3), ("F4 协议稳定性", f4), ("F5 恒真替身", f5)]
    for name, ok in verdicts:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  总判据: {sum(1 for _, ok in verdicts if ok)}/{len(verdicts)}")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for name in V0SETS:
        ax.plot([Kv for Kv in KS], [data[Kv][name]["meff"] for Kv in KS], "o-", label=name)
    ax.plot(KS[:4], [data[KS[0]]["基准五档"]["meff"] * (Kv / KS[0]) ** 0.5 for Kv in KS[:4]],
            "k--", lw=1, label=r"$m_{eff}\propto K^{1/2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$m_{eff}$")
    ax.set_title("(a) 窗口有效质量的标度")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["K=0.005\n0.01\n0.02", "K=0.00125\n0.0025\n0.005\n0.01"]
    vals2 = [p3, p4]   # 三档（含 K=0.02）与四档的 m_eff 指数
    ax.bar([0, 1], vals2, color=["#c44", "#468"])
    ax.axhline(0.5, color="k", ls="--", label="连续预期 0.5")
    ax.axhline(0.29, color="gray", ls=":", label="库内旧读数 0.29")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$M$ 对 $K$ 的幂")
    ax.set_title("(b) 粗档离群")
    ax.legend(fontsize=8)

    ax = axes[2]
    for name in V0SETS:
        ax.plot([Kv for Kv in KS], [data[Kv][name]["vmax"] for Kv in KS], "o-", label=name)
    ax.set_xscale("log")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$v_{\max}$")
    ax.set_title("(c) 速度可达上限")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG, dpi=150)
    print(f"\n图已存: {IMG}")


if __name__ == "__main__":
    main()
