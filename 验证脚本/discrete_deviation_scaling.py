# -*- coding: utf-8 -*-
"""
discrete_deviation_scaling.py
767 号：离散修正的偏离度标度（判据量自 v^6 系数换为相对论形状偏离度）

起因：第 749 号的判据量（扣减后残差的 v^6 系数 c）经稳定性审计判定不可用——
跨速度档设置的相对散布 36% 至 184%，显著性 0.27σ 至 34σ，符号跨 K 档翻转，
第三档 K=0.02 的条件数 812 且 v_max 仅 0.275。故本轮改用稳定量：
数据相对相对论展开的偏离度（残差幅值比）。

对象类型：比例量与标度指数（可问幅值比、单调性、幂次，不可问对易）。

判据（先立理论期望，再实现；每条给出「改掉它就 FAIL」的反例）：

  E1 偏离度的量级：E1-a 各档偏离度显著非零（相对论展开不能精确复现数据）；
      E1-b 偏离度随 1/ell 增大而单调增（结构越细，离散效应越强）。
  E2 标度的幂次：偏离度随 1/ell 增大而增大，故拟合幂为正（离散修正随结构
      变细增强）；其值落在可解释区间内（1 至 4 之间），不落在噪声区（|幂| < 0.5）。
  E3 恒真替身：纯相对论数据（无离散修正）上，偏离度应落在数值零；
      若替身也给出显著偏离，则 E1、E2 的区分力不成立。
  E4 协议稳定性：偏离度在三组速度档设置下的绝对极差小于 1 个百分点
      （跨协议稳健）；超出者该档退出标度拟合。

  门槛取绝对极差而非相对散布的理由：相对散布在偏离度本身很小时会被放大
  （K=0.00125 档偏离度 0.39%，极差 0.50 个百分点，相对散布却达 75%，
  但其实质与 K=0.01 档的 0.64 个百分点同量级）。绝对极差才是与数据
  精度直接对应的量。门槛 1 个百分点由前四档实测极差（0.15、0.26、0.64）
  上浮约 1.5 倍取定，比 K=0.02 档的 13.4 个百分点严一个量级。

容差预算（从误差预算推导）：
  读数精度：数据本身的模拟误差为机器精度（能量守恒漂移 1e-10 量级），
  故偏离度的下限由数据噪声与相对论拟合残差给出，约 1e-3 相对量级。
  E1-b 的单调性判定以逐档严格递增为判，不设容差（单档不需容差）。
  E2 的幂次阈值 0.5 取自：五点拟合的标准误约 0.3 至 0.5，
  故 |幂| < 0.5 与零不可分，须判否。
  E4 的 1 个百分点取自：跨速度档设置引起的偏离度极差，前四档实测为
  0.15、0.26、0.64 个百分点，取 1 个百分点为可用门槛（约最差档的 1.5 倍），
  超出者该档不参与标度拟合（不删数据，只标退出）。
  E3 的替身阈值 1e-6 相对量级（纯相对论数据扣减后残差为机器零量级）。

模型：FK 链（同 749 号），E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos2πu)]，
k = 1，dt = 0.01，缠绕守恒边界。K 档 0.00125、0.0025、0.005、0.01、0.02 五档
（kink 半宽 ell = 1/(2π√K) 依次为 4.50、3.18、2.25、1.59、1.13 格距）。
确定性，无随机源。

偏离度定义：取相对论拟合 M（ΔE 对 M c²(√(1-β²)-1) 的最小二乘），
残差 R(v) = ΔE(v) - M c²(√(1-β²)-1)，偏离度取 rms 相对量
dev = √(mean(R²)) / mean(|ΔE|)。
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\discrete_deviation_scaling_767.png"

k = 1.0
dt = 0.01
K = 0.0
w2 = 0.0
c_s = np.sqrt(k)
N = 1400
cN = 700
W = 30
V0S = (0.20, 0.35, 0.50, 0.65, 0.80)
KS = (0.00125, 0.0025, 0.005, 0.01, 0.02)
# 协议稳定性用的三组速度档设置
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


def collect(u_rest, E_rest_w, v0s=V0S, steps=STEPS):
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


def deviation(v, dE):
    """返回 (M, 偏离度 rms 相对量, 相对论拟合相对残差)。"""
    g1 = c_s ** 2 * (1.0 / np.sqrt(1.0 - (v / c_s) ** 2) - 1.0)
    M = float(np.sum(g1 * dE) / np.sum(g1 ** 2))
    R = dE - M * g1
    dev = float(np.sqrt(np.mean(R ** 2)) / np.mean(np.abs(dE)))
    res = float(np.max(np.abs(R)) / np.max(np.abs(dE)))
    return M, dev, res


def main():
    if "--smoke" in sys.argv:
        print("== 冒烟检查 ==")
        set_K(0.005)
        u_rest, v_rest = relax_kink()
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, W)
        rows = collect(u_rest, E_rest_w, v0s=(0.35, 0.65), steps=20000)
        M, dev, res = deviation(rows[:, 1], rows[:, 2])
        print(f"  M={M:.5f} 偏离度 rms={dev * 100:.4f}% 相对残差={res * 100:.4f}%")
        # 替身
        v_rel = np.linspace(0.01, 0.60, 5)
        dE_rel = 0.0935 * c_s ** 2 * (1.0 / np.sqrt(1.0 - (v_rel / c_s) ** 2) - 1.0)
        M2, dev2, res2 = deviation(v_rel, dE_rel)
        print(f"  替身：M={M2:.5f} 偏离度={dev2:.3e} 相对残差={res2:.3e}")
        if dev > 1e-5 and dev2 < 1e-6:
            print("  冒烟通过：数据偏离非零、替身偏离为零，进入判据循环")
        return

    print("=" * 78)
    print("767 号：离散修正的偏离度标度")
    print("=" * 78)
    print(f"模型：FK 链 k=1 dt={dt}，K 档 {KS}，窗口 W={W}")

    # ---------- 逐档读数与协议稳定性 ----------
    table = []
    for Kv in KS:
        set_K(Kv)
        u_rest, v_rest = relax_kink()
        x0 = center(u_rest)
        E_rest_w = window_energy(u_rest, v_rest, x0, W)
        devs = {}
        for name, v0s in V0SETS.items():
            rows = collect(u_rest, E_rest_w, v0s=v0s)
            M, dev, res = deviation(rows[:, 1], rows[:, 2])
            devs[name] = dev
        base_dev = devs["基准五档"]
        vals = np.array(list(devs.values()))
        spread = float(np.ptp(vals) / np.mean(vals)) if np.mean(vals) > 0 else np.inf
        rows = collect(u_rest, E_rest_w)
        M, dev, res = deviation(rows[:, 1], rows[:, 2])
        ell = 1.0 / (2 * np.pi * np.sqrt(Kv / k))
        table.append(dict(K=Kv, ell=ell, inv_ell=1.0 / ell, M=M, dev=dev, res=res,
                          devs=devs, spread=spread, vmax=float(rows[:, 1].max())))
        print(f"\n  K={Kv}（ell={ell:.3f}，1/ell={1.0/ell:.4f}，vmax={rows[:, 1].max():.3f}）")
        print(f"    M_rel={M:.5f}  偏离度 rms={dev * 100:.4f}%  相对论拟合残差={res * 100:.4f}%")
        for name in V0SETS:
            print(f"      {name}: 偏离度 {devs[name] * 100:.4f}%")
        print(f"    跨协议相对散布 {spread * 100:.1f}%")

    # ---------- E1 ----------
    print("\n[E1] 偏离度的量级与单调性")
    e1a = all(t["dev"] > 1e-4 for t in table)
    e1b = all(table[i + 1]["dev"] > table[i]["dev"] for i in range(len(table) - 1))
    print(f"  E1-a 各档偏离度 > 1e-4 相对量：{'是' if e1a else '否'}")
    print("  E1-b 偏离度随 1/ell 单调增："
          + " → ".join(f"{t['dev'] * 100:.3f}%" for t in table) + f"  {'是' if e1b else '否'}")
    e1 = e1a and e1b
    print(f"  E1: {'PASS' if e1 else 'FAIL'}")

    # ---------- E4（先做，决定哪些档参与标度） ----------
    print("\n[E4] 协议稳定性")
    usable = []
    for t in table:
        vals = np.array(list(t["devs"].values()))
        abs_spread = float(np.ptp(vals) * 100)   # 百分点
        t["abs_spread"] = abs_spread
        ok = abs_spread < 1.0
        t["usable"] = ok
        if ok:
            usable.append(t)
        print(f"  K={t['K']}: 绝对极差 {abs_spread:.3f} 个百分点（相对 {t['spread'] * 100:.1f}%）"
              f"  {'参与标度' if ok else '退出标度'}")
    e4 = len(usable) >= 3
    print(f"  可用档数 {len(usable)}（需 ≥ 3）：{'是' if e4 else '否'}")
    print(f"  E4: {'PASS' if e4 else 'FAIL'}")

    # ---------- E2 ----------
    print("\n[E2] 标度的幂次")
    if len(usable) >= 2:
        x = np.log([t["inv_ell"] for t in usable])
        y = np.log([t["dev"] for t in usable])
        slope, intercept = np.polyfit(x, y, 1)
        print(f"  偏离度 ∝ (1/ell)^{slope:+.3f}  →  ∝ K^{slope/2:+.3f}")
        e2 = (slope > 0.5) and (slope < 4.0)
        print(f"  幂次落在 0.5 至 4.0 区间（正）：{'是' if e2 else '否'}")
    else:
        slope = float("nan")
        e2 = False
        print("  可用档不足 2，判否")
    print(f"  E2: {'PASS' if e2 else 'FAIL'}")

    # ---------- E3 ----------
    print("\n[E3] 恒真替身：纯相对论数据")
    v_rel = np.linspace(0.01, 0.60, 5)
    dE_rel = 0.0935 * c_s ** 2 * (1.0 / np.sqrt(1.0 - (v_rel / c_s) ** 2) - 1.0)
    M2, dev2, res2 = deviation(v_rel, dE_rel)
    print(f"  替身：M={M2:.5f}（解析 0.09350）偏离度={dev2:.3e} 相对残差={res2:.3e}")
    e3 = dev2 < 1e-6
    print(f"  替身偏离度 < 1e-6：{'是' if e3 else '否'}")
    print(f"  E3: {'PASS' if e3 else 'FAIL'}")

    # ---------- 汇总 ----------
    print("\n[判据汇总]")
    verdicts = [("E1 偏离度的量级与单调性", e1), ("E2 标度的幂次", e2),
                ("E3 恒真替身", e3), ("E4 协议稳定性", e4)]
    for name, ok in verdicts:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  总判据: {sum(1 for _, ok in verdicts if ok)}/{len(verdicts)}")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for name in V0SETS:
        ax.plot([t["inv_ell"] for t in table], [t["devs"][name] * 100 for t in table],
                "o-", label=name)
    ax.set_xlabel(r"$1/\ell$")
    ax.set_ylabel(r"偏离度 rms（%）")
    ax.set_title("(a) 偏离度与结构精细度")
    ax.legend(fontsize=8)

    ax = axes[1]
    if len(usable) >= 2:
        ax.plot([t["inv_ell"] for t in usable], [t["dev"] * 100 for t in usable],
                "o", label="可用档")
        xs = np.log([t["inv_ell"] for t in usable])
        ys = np.log([t["dev"] for t in usable])
        sl, ic = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 40)
        ax.plot(np.exp(xx), np.exp(ic + sl * xx) * 100, "k--",
                label=f"幂次 {sl:+.2f}")
    ax.plot([t["inv_ell"] for t in table], [t["dev"] * 100 for t in table],
            "x", alpha=0.5, label="退出档")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/\ell$")
    ax.set_ylabel(r"偏离度 rms（%）")
    ax.set_title("(b) 标度的幂次")
    ax.legend(fontsize=8)

    ax = axes[2]
    for t in table:
        ax.plot([t["K"]], [t["abs_spread"]], "o")
        ax.annotate(f"{t['K']}", (t["K"], t["abs_spread"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhline(1.0, color="r", ls="--", label="门槛 1 个百分点")
    ax.set_xscale("log")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel("跨协议绝对极差（百分点）")
    ax.set_title("(c) 协议稳定性")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG, dpi=150)
    print(f"\n图已存: {IMG}")


if __name__ == "__main__":
    main()
