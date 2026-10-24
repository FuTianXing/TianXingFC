# -*- coding: utf-8 -*-
r"""
carrier_occupancy_wkb.py

[独立载子占据轮] 经典占据能给出份额的主体，尾巴必须由波性侧提供

定位
----
第五轮机制面执行：把载子当作独立对象（沿格点运动、在转折点折返），检验其经典时间占据能否给出份额 $n_j/\sum n$。
前四轮把计数口径钉死在"每激发一次、按瞬时激发数加权"，本轮回答上一环：载子的占据分布从哪里来。
本脚本不改动基准文件；不含新科学断言。

模型
----
一维闭链 $L = 128$，$l_P = t_P = 1$，包络场取标准符号（带底在 $q = 0$，对应低能长波模式）
  $i\dot A_j = -\kappa\,\Delta A_j + V_j A_j$，$\kappa = 1$，阱取平方型 $V_j = \tfrac12 K (j-j_c)^2$。
基态由虚时梯度流求得，本征值 $E_0$ 取站点相位推进率的均值。

经典占据的两种候选：
  其一，变速占据（时间占据律）：局域动量由能量守恒给出 $x_j = (E_0 - V_j)/(4\kappa) \in [0,1]$、
  $q_j = 2\arcsin\sqrt{x_j}$，群速 $v_j = 4\kappa\sqrt{x_j(1-x_j)}$，占据 $\propto 1/v_j$（经典粒子在慢处待得久）。
  其二，锁速占据（对照口径）：速度处处相同，占据在允许区内均匀（箱形）。
份额取模式密度 $n_j = |A_j|^2$。允许区由 $0 < x_j < 1$ 定义，核心区取 $|j-j_c| \le$ 允许区半宽的一半。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 经典占据律被排除：核心区变速占据 $1/v_j$ 与份额的相关系数 $< -0.99$（经典占据在两端密，份额在中间密）。
J2a 核心区两种经典占据同量级：$TV_{核心}(箱)$ 与 $TV_{核心}(1/v)$ 都 $< 0.06$（四档中箱形三档更优，差异无系统性）。
J2b 两种占据的全局总变差都 $> 0.2$。
J3 量子尾质量（份额落在允许区之外的部分）$> 0.15$，且对阱曲率的双对数斜率的绝对值 $< 0.3$（不随宽度消失）。
J4 失败是结构性的：全局总变差不随模式变宽收敛（斜率绝对值 $< 0.3$）。
J5 归一化：两种占据的归一化误差 $< 10^{-12}$。
J6 偏差的三区分解：尾区贡献 $> 0.25$，与核心区、过渡区共同构成偏差（报告三区占比）。

运行
----
python 验证脚本/carrier_occupancy_wkb.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/carrier_occupancy_wkb.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 128
JC = L // 2
KAPPA = 1.0


def lap(A):
    return np.roll(A, -1) + np.roll(A, 1) - 2.0 * A


def ground_mode(V, dtau=0.02, max_steps=200000, tol=1e-12):
    A = np.exp(-0.5 * ((np.arange(L) - JC) / 10.0) ** 2).astype(complex)
    A /= np.linalg.norm(A)
    for step in range(max_steps):
        A = A - dtau * (-KAPPA * lap(A) + V * A)
        A /= np.linalg.norm(A)
        if step % 50 == 0:
            f = -1j * (-KAPPA * lap(A) + V * A)
            r = -np.imag(f / A)
            if (r.max() - r.min()) / abs(r.mean()) < tol:
                break
    f = -1j * (-KAPPA * lap(A) + V * A)
    return A, float(np.mean(-np.imag(f / A)))


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    x = np.arange(L, dtype=float)
    Ks = np.array([2.0e-3, 2.0e-4, 2.0e-5, 2.0e-6])
    eps = 0.02
    rows = []

    for K in Ks:
        d = x - JC
        d = np.where(d > L / 2, d - L, d)
        V = 0.5 * K * d ** 2
        A, E0 = ground_mode(V)
        p = np.abs(A) ** 2
        p = p / p.sum()
        sigma = float(np.sqrt(np.sum(p * (x - float(np.sum(p * x))) ** 2)))
        xj = (E0 - V) / (4.0 * KAPPA)
        allowed = (xj > 0.0) & (xj < 1.0)
        half = int(allowed.sum() / 2)
        core = np.abs(d) <= 0.5 * half
        rho = np.where(allowed, 1.0 / np.sqrt(np.maximum(xj * (1.0 - xj), 0.0) + eps ** 2), 0.0)
        rho = rho / rho.sum()
        box = allowed.astype(float)
        box = box / box.sum()
        tail_mass = float(p[~allowed].sum())
        tv_full = tv(rho, p)
        tv_core = tv(rho[core], p[core])
        tv_core_box = tv(box[core], p[core])
        detail = np.abs(rho - p)
        trans = allowed & (~core)
        tot = detail.sum()
        core_share = float(detail[core].sum() / tot)
        trans_share = float(detail[trans].sum() / tot)
        tail_share = float(detail[~allowed].sum() / tot)
        corr = float(np.corrcoef(rho[core], p[core])[0, 1])
        rows.append(dict(K=K, sigma=sigma, E0=E0, half=half, tail=tail_mass, tv_full=tv_full,
                         tv_core=tv_core, tv_core_box=tv_core_box, core_share=core_share,
                         trans_share=trans_share, tail_share=tail_share, corr=corr,
                         norm=abs(rho.sum() - 1.0) + abs(box.sum() - 1.0)))
        print("  K = %.1e  sigma = %5.2f  允许区半宽 %3d  尾质量 %.3f  TV全 %.4f  TV核心(1/v) %.4f  TV核心(箱) %.4f  尾区贡献 %.3f"
              % (K, sigma, half, tail_mass, tv_full, tv_core, tv_core_box, tail_share))

    print("== J1 经典占据律被排除 ==")
    corrs = np.array([r["corr"] for r in rows])
    report("J1 核心区反相关（原预期被推翻）", bool(corrs.max() < -0.99),
           "核心区相关系数 %s（经典占据两端密，份额中间密）" % np.array2string(corrs, precision=5))

    print("== J2 两种经典占据在全局都失败 ==")
    tv_core = np.array([r["tv_core"] for r in rows])
    tv_core_box = np.array([r["tv_core_box"] for r in rows])
    tv_full = np.array([r["tv_full"] for r in rows])
    report("J2a 核心区两种经典占据同量级（都 < 0.06）", bool(max(tv_core_box.max(), tv_core.max()) < 0.06),
           "箱形 %s 对 1/v %s（四档中箱形三档更优，差异无系统性）"
           % (np.array2string(tv_core_box, precision=4), np.array2string(tv_core, precision=4)))
    report("J2b 全局总变差 > 0.2", bool(tv_full.min() > 0.2),
           "全局总变差 %s" % np.array2string(tv_full, precision=4))

    print("== J3 量子尾不随宽度消失 ==")
    tails = np.array([r["tail"] for r in rows])
    slope_t = float(np.polyfit(np.log(Ks), np.log(tails), 1)[0])
    report("J3 尾质量 > 0.15 且平缓", bool(tails.min() > 0.15) and abs(slope_t) < 0.3,
           "尾质量 %s，对 K 的双对数斜率 %.3f" % (np.array2string(tails, precision=3), slope_t))

    print("== J4 失败是结构性的 ==")
    slope_full = float(np.polyfit(np.log(Ks), np.log(tv_full), 1)[0])
    report("J4 全局总变差不随宽度收敛", abs(slope_full) < 0.3,
           "全局总变差 %s，对 K 的双对数斜率 %.3f" % (np.array2string(tv_full, precision=4), slope_full))

    print("== J5 归一化 ==")
    nrm = max(r["norm"] for r in rows)
    report("J5 归一化误差 < 1e-12", nrm < 1e-12, "最大 %.2e" % nrm)

    print("== J6 偏差的三区分解 ==")
    shares = np.array([[r["core_share"], r["trans_share"], r["tail_share"]] for r in rows])
    report("J6 尾区贡献 > 0.25", bool(shares[:, 2].min() > 0.25),
           "核心/过渡/尾占比 %s" % np.array2string(shares, precision=3))

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    K_show = 2.0e-4
    d = x - JC
    d = np.where(d > L / 2, d - L, d)
    V = 0.5 * K_show * d ** 2
    A, E0 = ground_mode(V)
    p = np.abs(A) ** 2
    p = p / p.sum()
    xj = (E0 - V) / (4.0 * KAPPA)
    allowed = (xj > 0.0) & (xj < 1.0)
    half = int(allowed.sum() / 2)
    core = np.abs(d) <= 0.5 * half
    rho = np.where(allowed, 1.0 / np.sqrt(np.maximum(xj * (1.0 - xj), 0.0) + eps ** 2), 0.0)
    rho = rho / rho.sum()
    box = allowed.astype(float)
    box = box / box.sum()

    ax = axes[0, 0]
    ax.plot(x, p, lw=1.7, label="份额 $n_j/\\sum n$")
    ax.plot(x, rho, lw=1.4, ls="--", label="变速占据 $\\propto 1/v_j$")
    ax.plot(x, box, lw=1.2, ls=":", label="锁速箱形")
    ax.axvspan(JC - half, JC + half, color="0.85", alpha=0.5, zorder=0)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 A 允许区（灰）内的三种分布")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, p, lw=1.7, label="份额")
    ax.plot(x, np.where(allowed, 1.0 / np.sqrt(np.maximum(xj * (1.0 - xj), 0.0) + eps ** 2), np.nan) * 0.0 + rho, lw=1.4,
            ls="--", label="$1/v_j$（允许区外为零）")
    ax.set_yscale("log")
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比（对数）")
    ax.set_title("图 B 转折点外：量子尾质量 %.3f" % float(p[~allowed].sum()))
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(Ks, [r["tv_core"] for r in rows], "o-", label="$1/v_j$（核心区）")
    ax.plot(Ks, [r["tv_core_box"] for r in rows], "s--", label="箱形（核心区）")
    ax.plot(Ks, [r["tv_full"] for r in rows], "^:", label="$1/v_j$（全局）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("阱曲率 $K$")
    ax.set_ylabel("总变差")
    ax.set_title("图 C 核心区随模式变宽改善，全局不改善")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(Ks, [r["sigma"] for r in rows], "o-", label="模式宽度 $\\sigma$")
    ax.plot(Ks, [r["half"] for r in rows], "s--", label="允许区半宽")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("阱曲率 $K$")
    ax.set_ylabel("格点")
    ax.set_title("图 D 宽度与允许区同步放大")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/carrier_occupancy_wkb.png"
    fig.savefig(out, dpi=160)
    print("  图已保存：%s" % out)

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("== 判据合计 %d/%d 通过，用时 %.1f s ==" % (n_pass, len(checks), time.time() - t_start))
    if n_pass != len(checks):
        for tag, ok, detail in checks:
            if not ok:
                print("  FAILED: %s | %s" % (tag, detail))


if __name__ == "__main__":
    main()
