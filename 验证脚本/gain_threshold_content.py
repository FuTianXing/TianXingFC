# -*- coding: utf-8 -*-
r"""增益阈值的物理内容：读数门限、指针分辨与记录寿命三分（判据 T1 至 T8）

沿用 M2 的 FK 链模型与传感器阵口径：
  FK 链 k=1、K=0.005、N=1000、dt=0.01、kink 半宽 ell；
  甲-3 = 零模踢 + 阻尼 gamma=0.5（源锁定，供泄漏寿命测量）；
  甲-2 = 宽脉冲（源持续遍历，供保留对照）；
  传感器 y <- alpha y + G Theta(w - theta)(1 - alpha)，alpha = exp(-dt_s/tau)；
  D_sensor = 0.5 sum_j (yA_j - yB_j)^2，V = exp(-D/2)。
新增读数：记录相对"当前构型决定的平衡值" y_eq = G Theta(w(t) - theta) 的偏离
  Delta(t) = || y(t) - y_eq(t) ||_2，以及无泄漏持有档（latch：越阈即置 G、不衰减）。
确定性：无随机源。图存 D:/software/TianXingFC/images/gain_threshold_content.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/gain_threshold_content.png"
k, K, dt, N = 1.0, 0.005, 0.01, 1000
cN = N // 2
w2 = 2 * np.pi * K
ELL = 1.0 / (2 * np.pi * np.sqrt(K / k))
GAMMA_C = 0.5


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def accel(u, v, gamma=0.0):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    if gamma > 0.0:
        a -= gamma * v
    return a


def kink_profile(x0):
    n = np.arange(N)
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ELL))


def zero_mode(x0):
    n = np.arange(N)
    f = 1.0 / np.cosh((n - x0) / ELL)
    return f / np.sqrt((f ** 2).sum())


def bonds(u):
    return u[1:] - u[:-1] - 1.0


def relax(u0, steps=8000):
    u = u0.copy(); v = np.zeros(N)
    a = accel(u, v)
    for _ in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
    return u, np.zeros(N)


def run_source(kind, T=200.0, sample=10):
    uA, vA = relax(kink_profile(float(cN)))
    uB, vB = relax(kink_profile(float(cN + 1)))
    if kind == "A3":
        vA = vA + 0.01 * zero_mode(float(cN))
        vB = vB + 0.01 * zero_mode(float(cN + 1))
    elif kind == "A2":
        j = np.arange(N)
        vp = 0.03 * np.exp(-((j - (cN + 0.5)) ** 2) / (2 * 25.0 ** 2))
        vA = vA + vp
        vB = vB + vp
    gamma = GAMMA_C if kind == "A3" else 0.0
    aA, aB = accel(uA, vA, gamma), accel(uB, vB, gamma)
    ts, wA, wB = [], [], []
    steps = int(T / dt)
    for t in range(steps):
        uA = uA + dt * vA + 0.5 * dt ** 2 * aA
        uB = uB + dt * vB + 0.5 * dt ** 2 * aB
        aAn = accel(uA, vA + dt * aA, gamma)
        aBn = accel(uB, vB + dt * aB, gamma)
        vA = vA + 0.5 * dt * (aA + aAn)
        vB = vB + 0.5 * dt * (aB + aBn)
        uA[0], uA[-1] = 0.0, float(N)
        uB[0], uB[-1] = 0.0, float(N)
        vA[0] = vA[-1] = 0.0
        vB[0] = vB[-1] = 0.0
        aA, aB = aAn, aBn
        if t % sample == 0:
            ts.append(len(ts) * dt * sample)
            wA.append(bonds(uA)); wB.append(bonds(uB))
    return dict(ts=np.array(ts), wA=np.array(wA), wB=np.array(wB), dt_s=dt * sample)


def holder(w_s, G, theta, dt_s, tau=None):
    """返回 (y 序列, y_eq 序列)。tau=None 为 latch（无泄漏）。"""
    n_s, nB = w_s.shape
    y = np.zeros(nB)
    ys, eqs = [], []
    if tau is None:
        for s in range(n_s):
            y = np.where(w_s[s] > theta, G, y)
            ys.append(y.copy()); eqs.append(G * (w_s[s] > theta))
    else:
        alpha = np.exp(-dt_s / tau)
        for s in range(n_s):
            y = alpha * y + G * (w_s[s] > theta) * (1 - alpha)
            ys.append(y.copy()); eqs.append(G * (w_s[s] > theta))
    return np.array(ys), np.array(eqs)


def g_star(vgate, ndiff):
    # V = exp(-D/2)，D = n_diff G^2/2  =>  G* = sqrt(4 ln(1/V) / n_diff)
    return float(np.sqrt(4.0 * np.log(1.0 / vgate) / ndiff))


def part1():
    print("=" * 100)
    print("T1 读法甲：G* 的约定依赖 G*(V_gate, n_diff) = sqrt(4 ln(1/V)/n_diff)")
    print("=" * 100)
    vgs = [0.9, 0.5, 0.1, 0.01]
    nds = [1, 2, 4, 8]
    print("V_gate | " + " | ".join("n=%d" % q for q in nds))
    vals = []
    for vg in vgs:
        row = [g_star(vg, q) for q in nds]
        vals += row
        print("%-7.2f| " % vg + " | ".join("%.3f" % x for x in row))
    hit = g_star(0.1, 2)
    report("T1", "G*(0.1, 2) 对论文值 2.15 的相对偏差", abs(hit - 2.146) / 2.146, 0.01,
           "该格复现", "通过" if abs(hit - 2.146) / 2.146 < 0.01 else "未过")
    report("T1", "表内 G* 的极差（约定依赖的幅度）", float(max(vals) - min(vals)), "-",
           "同一模型内因门限与指针数变动", "读数")
    return vgs, nds


def main():
    vgs, nds_axis = part1()
    src3 = run_source("A3")
    wmax = float(max(src3["wA"].max(), src3["wB"].max()))
    print()
    print("甲-3 源：w_max = %.4f（键滑峰值，判据草案读法乙的阈值上限）" % wmax)

    th0 = 0.5 * wmax
    patA = src3["wA"][-1] > th0
    patB = src3["wB"][-1] > th0
    ndiff = int((patA != patB).sum())
    report("T3", "theta = w_max/2 时的差异键位数 n_diff", ndiff, 2, "复现 M2 的 2 键位事实",
           "通过" if ndiff == 2 else "未过")

    print()
    print("T2 深度律：D_sensor 末值 / G^2（期望 n_diff/2 = %.1f）" % (ndiff / 2.0))
    rows = []
    for G in (2.0, 5.0, 10.0):
        ys, _ = holder(src3["wA"], G, th0, src3["dt_s"], tau=1.0)
        ys2, _ = holder(src3["wB"], G, th0, src3["dt_s"], tau=1.0)
        D = 0.5 * ((ys[-1] - ys2[-1]) ** 2).sum()
        rows.append(D / G ** 2)
        print("  G=%4.1f：D_sensor 末 %8.3f，D/G^2 = %.6f" % (G, D, D / G ** 2))
    dev = max(abs(q - ndiff / 2.0) / (ndiff / 2.0) for q in rows)
    report("T2", "D_sensor/G^2 对 n_diff/2 的最大相对偏差", dev, 1e-9, "三档 G",
           "通过" if dev < 1e-9 else "未过")

    print()
    print("T4 读法乙：n_diff 对 theta/w_max")
    print("-" * 100)
    ths = [0.1, 0.25, 0.5, 0.75, 0.95, 1.0, 1.01]
    nds = []
    for q in ths:
        pa = src3["wA"][-1] > q * wmax
        pb = src3["wB"][-1] > q * wmax
        nd = int((pa != pb).sum())
        nds.append(nd)
        print("  theta/w_max = %.2f：n_diff = %d，G*(0.1) = %s"
              % (q, nd, ("%.3f" % g_star(0.1, nd)) if nd > 0 else "无定义"))
    report("T4", "theta >= w_max 档的 n_diff", max(nds[-2:]), 0, "记录不存在的物理阈值",
           "通过" if max(nds[-2:]) == 0 else "未过")
    report("T4", "theta < w_max 档的 n_diff", nds[0], 2, "平台",
           "通过" if nds[0] == ndiff else "未过")

    print()
    print("T5 读法丙：持有档的记录保留时间（受控单次越阈：源在 t = 2 移开）")
    print("-" * 100)
    G = 5.0
    nB = src3["wA"].shape[1]
    ts_ctl = np.arange(0.0, 40.0, 0.1)
    th_c = 0.5 * wmax
    Wh = np.zeros((len(ts_ctl), nB))
    for i, q in enumerate(ts_ctl):
        if 1.0 <= q < 2.0:
            Wh[i, 200] = 0.9 * wmax
    rates = {}
    for tau in (0.25, 1.0, 4.0):
        ys, _ = holder(Wh, G, th_c, 0.1, tau=tau)
        Ret = np.linalg.norm(ys, axis=1)
        m = ts_ctl > 2.0
        p = np.polyfit(ts_ctl[m], np.log(np.clip(Ret[m], 1e-300, None)), 1)
        rates[tau] = float(p[0])
        print("  tau=%4.2f：保留量衰减率 %+.6f（期望 %+.6f）" % (tau, p[0], -1.0 / tau))
    dev = max(abs(rates[t] + 1.0 / t) * t for t in rates)
    report("T5", "保留量衰减率对 -1/tau 的最大相对偏差", dev, 0.05, "三档 tau",
           "通过" if dev < 0.05 else "未过")
    rel = {}
    for tau in (1.0, 4.0):
        ys, _ = holder(Wh, G, th_c, 0.1, tau=tau)
        Ret = np.linalg.norm(ys, axis=1)
        rel[tau] = float(Ret[-1])
    yslat, _ = holder(Wh, G, th_c, 0.1, tau=None)
    report("T5", "t = 40 时保留量：无泄漏 / tau=1 / tau=4",
           "%.3f / %.3e / %.3e" % (np.linalg.norm(yslat[-1]), rel[1.0], rel[4.0]), "-",
           "泄漏档已归零，无泄漏档保留", "读数")

    print()
    print("T6 持久判据：tau_rec >> N_拍 t_P 的示例阈值")
    print("-" * 100)
    for Np in (10, 100, 1000):
        print("  N_拍 = %-5d：要求 tau_rec >> %g t_P（取 10 倍余量即 %g t_P）" % (Np, Np * 10, Np * 10))
    report("T6", "持久判据的形态", "-", "-", "tau_rec >> N_拍 t_P", "读数")

    src2 = run_source("A2")
    print()
    print("T7 保留对照（甲-2 源：记录窗口在前段，源其后移开）")
    print("-" * 100)
    keep = {}
    for lab, tau in (("无泄漏 latch", None), ("泄漏 tau=1", 1.0), ("泄漏 tau=0.25", 0.25)):
        ys, eqs = holder(src2["wA"], G, th0, src2["dt_s"], tau=tau)
        Dev = np.linalg.norm(ys - eqs, axis=1)
        late = src2["ts"] > 100.0
        keep[lab] = float(np.median(Dev[late]))
        print("  %-14s：t>100 的 Delta 中位数 %.4f（t=0 处 Delta %.4f）" % (lab, keep[lab], Dev[0]))
    report("T7", "无泄漏对泄漏(tau=1) 的 Delta 终值比",
           keep["无泄漏 latch"] / max(keep["泄漏 tau=1"], 1e-12), "-", "保留与丢失的分离",
           "读数" if keep["无泄漏 latch"] > keep["泄漏 tau=1"] else "未过")

    print()
    print("T8 增益解耦：无泄漏持有的 Delta 中位数 / G")
    print("-" * 100)
    ratios = []
    for Gq in (1.0, 2.0, 5.0, 10.0):
        ys, eqs = holder(src2["wA"], Gq, th0, src2["dt_s"], tau=None)
        Dev = np.linalg.norm(ys - eqs, axis=1)
        med = float(np.median(Dev[src2["ts"] > 100.0]))
        ratios.append(med / Gq)
        print("  G=%5.1f：Delta 中位数 %8.4f，Delta/G = %.4f" % (Gq, med, med / Gq))
    spread = (max(ratios) - min(ratios)) / max(np.mean(ratios), 1e-12)
    report("T8", "Delta/G 的相对散布", float(spread), 0.05, "四档 G；G 只定深度",
           "通过" if spread < 0.05 else "未过")

    print()
    print("=" * 100)
    print("判据 | 判据量 | 实测 | 期望 | 备注 | 结论")
    print("=" * 100)
    for tag, name, meas, exp, note, grade in RES:
        m = ("%.6g" % meas) if isinstance(meas, float) else str(meas)
        e = ("%.6g" % exp) if isinstance(exp, float) else str(exp)
        print("%-5s | %s | %s | %s | %s | %s" % (tag, name, m, e, note, grade))
    print("=" * 100)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))

    vg = np.logspace(-2, 0, 200)[::-1]
    for nd in (1, 2, 4, 8):
        ax[0, 0].semilogy(vg, [g_star(q, nd) for q in vg], label="n_diff=%d" % nd)
    ax[0, 0].semilogy([0.1], [g_star(0.1, 2)], "r*", ms=14, label="论文格 (0.1, 2) = 2.146")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("判定门限 V_gate"); ax[0, 0].set_ylabel("G*")
    ax[0, 0].set_title("T1：G* 是约定量（门限与指针数决定）"); ax[0, 0].legend(fontsize=8)

    ax2 = ax[0, 1].twinx()
    ax2.plot(ths, nds, "o-", color="#c44e52", label="n_diff(theta)")
    ax2.set_ylabel("n_diff", color="#c44e52")
    gg = [g_star(0.1, q) if q > 0 else np.nan for q in nds]
    ax[0, 1].step(ths, gg, where="mid", color="#4c72b0", label="G*(0.1)（n_diff=0 时无定义）")
    ax[0, 1].set_xlabel("theta / w_max"); ax[0, 1].set_ylabel("G*(0.1)")
    ax[0, 1].set_title("T3/T4：物理阈值在 theta 不在 G"); ax[0, 1].legend(fontsize=8)

    for tau, c in ((None, "#55a868"), (1.0, "#4c72b0"), (0.25, "#dd8452")):
        ys, _ = holder(Wh, G, th_c, 0.1, tau=tau)
        Ret = np.linalg.norm(ys, axis=1)
        ax[1, 0].semilogy(ts_ctl, np.clip(Ret, 1e-16, None), "-", color=c,
                          label=("无泄漏 latch" if tau is None else "tau=%.2f" % tau))
    ax[1, 0].axvline(2.0, color="gray", ls=":", lw=1.0)
    ax[1, 0].set_xlabel("时间（模型单位，虚线处源移开）"); ax[1, 0].set_ylabel("记录保留量 ||y||")
    ax[1, 0].set_title("T5：持有档的记录保留时间"); ax[1, 0].legend(fontsize=8)

    for lab, tau, c in (("无泄漏 latch", None, "#55a868"), ("泄漏 tau=1", 1.0, "#4c72b0"),
                        ("泄漏 tau=0.25", 0.25, "#dd8452")):
        ys, eqs = holder(src2["wA"], G, th0, src2["dt_s"], tau=tau)
        Dev = np.linalg.norm(ys - eqs, axis=1)
        ax[1, 1].semilogy(src2["ts"], np.clip(Dev, 1e-16, None), "-", label=lab, color=c)
    ax[1, 1].set_xlabel("时间（模型单位）"); ax[1, 1].set_ylabel("Delta(t)")
    ax[1, 1].set_title("T7：源移开后记录的保留与丢失"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
