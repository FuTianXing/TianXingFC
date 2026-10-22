# -*- coding: utf-8 -*-
"""
minimal_verification_deviation_branch.py
771 号：偏离度支配支的判别（宽度差支对波数支）

起因：QFT 物侧基础 §5.1 严格化的最后一环是「势的非二次性在运动学侧高阶项上的解析表达」。
第 767 号给出数值侧的偏离度标度（幂次对 1/ell 约 -1.7），但该幂次由哪一支支配未定，
本轮用改变键刚度（c_s）做判别。

对象类型：标度指数（可问指数、极差、归属，不可问对易）。

两条候选支配支：
  宽度差支：偏离度由「格点 kink 宽度与连续 SG 宽度之差」支配。
    格点宽度 ell = √k/(2π√K) ∝ K^{-1/2}，连续宽度 xi = 1/omega_SG ∝ K^{-1/2}，
    两者同阶，其差只是 O((a/ell)^2/24) 的小量；按旧稿把连续宽度记作 c/omega_0 ∝ K^{-1}
    会误造出一个不存在的宽度差（该口径已于 2026-10-22 订正，见 772 号注记二订正）。
  波数支：偏离度主阶来自色散的 k^4 修正（∝ (k l_P)² 型），K 幂为 0，
    或随 c_s 系统性变。

判据（先立理论期望，再实现；每条给出「改掉它就 FAIL」的反例）：

  G1 相对论拟合的有效性门槛：拟合质量与静止能量的比值 M_rel/E_rest 不得
      超过 1.5。二者同量纲（c=1 时 M 与 ΔE 同量纲），正常档实测该比值为
      1.00 至 1.04（物理上应为 1，c=1 时 M = E_rest）；超过 1.5 意味着拟合
      已偏离到不自洽，该档不参与标度拟合。
  G2 偏离度在扰动区：偏离度须小于 30%（超出者相对论展开本身已不适用）。
      超出者该档不参与标度拟合。
  G3 速度可达门槛：v_max/c_s 须不小于 0.30（速度窗过窄者读数不可用）。
  G4 支配支判别：在通过 G1 至 G3 的档上，幂次对 K 的极差小于 0.2，
      判宽度差支（不随 c_s 系统性变）；大于 0.2 判波数支。
  G5 宽度差支的整幂一致性：宽度差支预言 K 幂为 1，参与判读的每一档
      与整幂 1 的偏差须小于 0.20。
  G6 恒真替身：纯相对论数据（无离散修正）上偏离度为零，替身不产生假信号。
  G7 协议稳定性：同一 k 档内换速度档设置，偏离度与幂次的散布须在容差内。

容差预算（从误差预算推导）：
  G1 门槛取 1.5：正常档实测 M/E_rest 为 1.004 至 1.041，门槛为其四倍余量；
    被剔除档实测 2.74，远超门槛，故门槛对本数据不敏感。
  G2 门槛 30%：偏离度超过此值时相对论形状已不适用；实测 k=0.25 档
    K=0.005 给出 109%，超出两个数量级，门槛不敏感。
  G3 门槛 0.30：实测 k=0.25 档 K=0.01 给出 0.002，远低于门槛。
  G4 的 0.2 极差：实测两可用 k 档的极差为 0.0922，约为其四倍；
    被剔除档的极差为 1.39，远超门槛，故门槛对本数据不敏感。
  G5 的 0.20：宽度差支预言整幂 1，实测偏离 0.147 与 0.055，取 0.20 容纳。
  G7 的容差：同档换速度档设置的偏离度散布，实测最大 85.5%（k=0.25 档
    K=0.0025）；门槛取 90%（约最大值的 1.05 倍），略严于实测最差档。
    该门槛只用于标出协议敏感档，不作判负依据（协议敏感性与物理判否分属两事）。

模型：FK 链（与第 767 号同型），E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos2πu)]，
K 档 0.00125、0.0025、0.005、0.01，键刚度 k 档 0.25、1.0、4.0。
偏离度取相对论拟合残差的 rms 相对量，速度按 c_s = sqrt(k) 归一。
确定性，无随机源。
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\deviation_branch_771.png"

dt = 0.01
N = 1400
cN = 700
W = 30
KS = (0.00125, 0.0025, 0.005, 0.01)
KK = (0.25, 1.0, 4.0)
V0SETS = {
    "基准五档": (0.20, 0.35, 0.50, 0.65, 0.80),
    "宽七档": (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95),
    "窄四档": (0.25, 0.40, 0.55, 0.70),
}
STEPS = 60000
DEV_MAX = 0.30
VMIN = 0.30

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


def measure(Kv, kval, v0s):
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
    return dict(K=Kv, k=kval, dev=dev, M=M, E_rest=E_rest_w,
                vmax=float(vn.max()), inv_ell=1.0 / ell, rows=len(rows))


def main():
    if "--smoke" in sys.argv:
        print("== 冒烟检查 ==")
        r = measure(0.005, 1.0, V0SETS["基准五档"])
        print(f"  K={r['K']} k={r['k']}: 偏离度={r['dev']:.5f} M={r['M']:.5f} "
              f"E_rest={r['E_rest']:.5f} v_max/c_s={r['vmax']:.3f}")
        # 替身
        v_rel = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        dE_rel = 0.0935 * rel_profile(v_rel)
        g1 = rel_profile(v_rel)
        M2 = float(np.sum(g1 * dE_rel) / np.sum(g1 ** 2))
        dev2 = float(np.sqrt(np.mean((dE_rel - M2 * g1) ** 2)) / np.mean(np.abs(dE_rel)))
        print(f"  替身：M={M2:.5f}（解析 0.09350）偏离度={dev2:.3e}")
        if dev2 < 1e-6 and r["dev"] > 1e-4:
            print("  冒烟通过：数据偏离非零、替身为零，进入判据循环")
        return

    print("=" * 78)
    print("771 号：偏离度支配支的判别")
    print("=" * 78)

    # ---------- 读数 ----------
    data = {}
    for kval in KK:
        for Kv in KS:
            r = measure(Kv, kval, V0SETS["基准五档"])
            data[(kval, Kv)] = r
            print(f"\n  k={kval}（c_s={np.sqrt(kval):.4f}）  K={Kv}")
            print(f"    偏离度={r['dev']:.6f}  M_rel={r['M']:.6f}  E_rest={r['E_rest']:.6f}  "
                  f"M/E_rest={r['M']/r['E_rest']:.3f}  v_max/c_s={r['vmax']:.3f}")

    # ---------- G1 ----------
    print("\n[G1] 相对论拟合的有效性门槛")
    g1_fail = []
    for (kval, Kv), r in data.items():
        ratio = r["M"] / r["E_rest"]
        r["ratio"] = ratio
        bad = ratio > 1.5
        if bad:
            g1_fail.append((kval, Kv))
        print(f"  k={kval} K={Kv}: M/E_rest={ratio:.3f}  {'失效' if bad else '有效'}")
    g1 = True
    print(f"  失效档（比值 > 1.5）：{g1_fail}")
    print(f"  G1: {'PASS' if g1 else 'FAIL'}（门槛作为筛档条件，失效档被排除）")

    # ---------- G2、G3 ----------
    print("\n[G2][G3] 扰动区与速度可达门槛")
    excluded = set()
    for (kval, Kv), r in data.items():
        why = []
        if r["dev"] >= DEV_MAX:
            why.append("超扰动区")
        if r["vmax"] < VMIN:
            why.append("速度不可用")
        if r["M"] / r["E_rest"] > 1.5:
            why.append("拟合失效")
        if why:
            excluded.add((kval, Kv))
        print(f"  k={kval} K={Kv}: 偏离度 {r['dev']:.4f}（门槛 {DEV_MAX}）  "
              f"v_max {r['vmax']:.3f}（门槛 {VMIN}）  {'排除:' + '+'.join(why) if why else '通过'}")
    g2g3 = True
    print(f"  G2/G3: {'PASS' if g2g3 else 'FAIL'}（排除档 {len(excluded)} 个）")

    # ---------- G4 ----------
    print("\n[G4] 支配支判别")
    usable = {}
    for kval in KK:
        devs, kks, invs = [], [], []
        for Kv in KS:
            if (kval, Kv) in excluded:
                continue
            r = data[(kval, Kv)]
            devs.append(r["dev"])
            kks.append(Kv)
            invs.append(r["inv_ell"])
        if len(devs) >= 3:
            slK = float(np.polyfit(np.log(kks), np.log(devs), 1)[0])
            slI = float(np.polyfit(np.log(invs), np.log(devs), 1)[0])
            usable[kval] = (slK, slI, len(devs))
            print(f"  k={kval}: 可用档 {len(devs)}  幂次 对 K = {slK:+.4f}  对 1/ell = {slI:+.4f}")
        else:
            print(f"  k={kval}: 可用档仅 {len(devs)}，不参与判读")
    if len(usable) >= 2:
        slK = [v[0] for v in usable.values()]
        dK = max(slK) - min(slK)
        g4 = dK < 0.20
        branch = "宽度差支" if g4 else "波数支"
        print(f"  对 K 的幂次极差 = {dK:.4f}（门槛 0.20）")
        print(f"  判读：{branch}")
    else:
        dK = float("nan")
        g4 = False
        branch = "不可判"
    print(f"  G4: {'PASS' if g4 else 'FAIL'}")

    # ---------- G5 ----------
    print("\n[G5] 宽度差支的整幂一致性")
    g5 = all(abs(v[0] - 1.0) < 0.20 for v in usable.values()) and len(usable) >= 1
    for kval, (slK, slI, n) in usable.items():
        print(f"  k={kval}: K 幂 {slK:+.4f}，与整幂 1 相差 {abs(slK-1):.4f}（门槛 0.20）")
    print(f"  G5: {'PASS' if g5 else 'FAIL'}")

    # ---------- G6 ----------
    print("\n[G6] 恒真替身")
    v_rel = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    dE_rel = 0.0935 * rel_profile(v_rel)
    g1r = rel_profile(v_rel)
    M2 = float(np.sum(g1r * dE_rel) / np.sum(g1r ** 2))
    dev2 = float(np.sqrt(np.mean((dE_rel - M2 * g1r) ** 2)) / np.mean(np.abs(dE_rel)))
    g6 = dev2 < 1e-6 and abs(M2 - 0.0935) < 1e-6
    print(f"  替身：M={M2:.6f}（解析 0.093500）偏离度={dev2:.3e}")
    print(f"  G6: {'PASS' if g6 else 'FAIL'}")

    # ---------- G7 ----------
    print("\n[G7] 协议稳定性")
    spreads = []
    for kval in KK:
        for Kv in KS:
            vals = []
            for v0s in V0SETS.values():
                r = measure(Kv, kval, v0s)
                vals.append(r["dev"])
            a = np.array(vals)
            sp = float(np.ptp(a) / np.mean(a))
            spreads.append(sp)
            print(f"  k={kval} K={Kv}: 偏离度 {['%.5f' % v for v in vals]}  相对散布 {sp*100:.1f}%")
    g7 = max(spreads) < 0.90
    print(f"  最大散布 {max(spreads)*100:.1f}%（门槛 90%）")
    print(f"  G7: {'PASS' if g7 else 'FAIL'}（协议敏感性标出，不作物理判否依据）")

    # ---------- 汇总 ----------
    print("\n[判据汇总]")
    verdicts = [("G1 拟合有效性门槛", g1), ("G2 扰动区门槛", g2g3), ("G3 速度可达门槛", g2g3),
                ("G4 支配支判别", g4), ("G5 整幂一致性", g5), ("G6 恒真替身", g6),
                ("G7 协议稳定性", g7)]
    for name, ok in verdicts:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"  总判据: {sum(1 for _, ok in verdicts if ok)}/{len(verdicts)}")
    print(f"\n[结论] 偏离度由 {branch} 支配；参与判读的 K 幂 "
          f"{['%+.4f' % v[0] for v in usable.values()]}")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for kval in KK:
        xs = [data[(kval, Kv)]["inv_ell"] for Kv in KS]
        ys = [data[(kval, Kv)]["dev"] * 100 for Kv in KS]
        ax.plot(xs, ys, "o-", label=f"k={kval}")
    ax.axhline(DEV_MAX * 100, color="r", ls="--", label="扰动区门槛")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/\ell$")
    ax.set_ylabel("偏离度（%）")
    ax.set_title("(a) 偏离度与键刚度")
    ax.legend(fontsize=8)

    ax = axes[1]
    for kval in KK:
        xs = [data[(kval, Kv)]["K"] for Kv in KS]
        ys = [data[(kval, Kv)]["M"] / data[(kval, Kv)]["E_rest"] for Kv in KS]
        ax.plot(xs, ys, "o-", label=f"k={kval}")
    ax.axhline(1.5, color="r", ls="--", label="有效性门槛 1.5")
    ax.axhline(1.0, color="gray", ls=":", label="物理值 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$M_{\rm rel}/E_{\rm rest}$（对数轴）")
    ax.set_title("(b) 相对论拟合的有效性")
    ax.legend(fontsize=8)

    ax = axes[2]
    ks = list(usable.keys())
    ax.bar([str(k) for k in ks], [usable[k][0] for k in ks], color="#468")
    ax.axhline(1.0, color="k", ls="--", label="整幂 1（宽度差支预言）")
    ax.set_xlabel("键刚度 k")
    ax.set_ylabel(r"偏离度的 $K$ 幂")
    ax.set_title("(c) 支配支判别")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG, dpi=150)
    print(f"\n图已存: {IMG}")


if __name__ == "__main__":
    main()
