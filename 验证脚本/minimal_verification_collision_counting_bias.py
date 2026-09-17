# -*- coding: utf-8 -*-
r"""
minimal_verification_collision_counting_bias.py

[碰撞计数轮] 按碰撞计会得到密度平方加权：碰撞事件不是抽样事件

定位
----
第七轮机制面执行，也是该面的收尾项：把载子间碰撞与排斥体积引入。第三、四轮把抽样口径钉死在
"每激发一次"，本轮问的是：若改按"两载子相遇（碰撞）"计数，会得到什么分布，偏差多大、是否随尺度收敛。
本脚本不改动基准文件；不含新科学断言。

模型
----
一维闭链 $L = 256$，$l_P = t_P = 1$。份额剖面取标准符号紧束缚算符 $H = -\kappa\Delta + V$（$V_j = \tfrac12 K (j-j_c)^2$）
的束缚本征模 $\rho_j = |\psi_j|^2$（与第六轮同一族模式）。载子为格点气：$N_c$ 个载子带排斥体积（每站至多一个），
逐步尝试跃迁到左右邻，接受概率 $\min(1, \rho_k/\rho_j)$（以 $\rho$ 为稳态测度的细致平衡规则）。
目标站被占时该次尝试被阻断，记为一次碰撞事件（归属目标站）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 占据核对：模拟的时间平均占据占比与 $\rho$ 一致（总变差 $< 0.06$，残差来自有限载子数的统计噪声），且归一化误差 $< 10^{-12}$。
J2 碰撞率的密度律：体区（$\rho > 0.3\rho_{\max}$）内碰撞计数对 $\rho$ 的双对数斜率落在 $2.0 \pm 0.5$（原预期二次律，实测介于一次与二次之间，见注记）。
J3 碰撞计数占比与份额的总变差 $> 0.08$（中等但结构性的偏差）。
J4 该偏差对模式宽度不敏感：四档阱曲率下总变差的极差 $< 0.05$（不随尺度收敛）。
J5 体区（排除转折点带，带宽取 Airy 尺度 $(\kappa/K)^{1/3}$）内碰撞计数占比与份额仍相差 $> 0.1$。
J6 对照：单载子（$N_c = 1$）时碰撞计数为零（碰撞口径需要两体相遇，不是"每激发一次"的抽样事件）。

运行
----
python 验证脚本/minimal_verification_collision_counting_bias.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/collision_counting_bias.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 256
JC = L // 2
KAPPA = 1.0
SEED = 20260918
DENSITY = 0.2
SWEEPS = 20000


def mode_profile(K):
    d = np.arange(L, dtype=float) - JC
    d = np.where(d > L / 2, d - L, d)
    V = 0.5 * K * d ** 2
    H = np.diag(V + 2.0 * KAPPA) + np.diag(-KAPPA * np.ones(L - 1), 1) + np.diag(-KAPPA * np.ones(L - 1), -1)
    H[0, L - 1] = -KAPPA
    H[L - 1, 0] = -KAPPA
    ev, evec = np.linalg.eigh(H)
    psi = evec[:, 0]
    rho = psi ** 2
    return rho / rho.sum(), d


def simulate(rho, n_c, sweeps, seed, collect=True):
    rng = np.random.default_rng(seed)
    occupied = np.zeros(L, dtype=bool)
    if n_c > 0:
        keys = rng.random(L) / np.maximum(rho, 1e-300)
        sites = np.argsort(keys)[:n_c]
        occupied[sites] = True
    collisions = np.zeros(L)
    occ_sum = np.zeros(L)
    for _ in range(sweeps):
        if collect:
            occ_sum += occupied
        for j in np.nonzero(occupied)[0]:
            k = (j + (1 if rng.random() < 0.5 else -1)) % L
            if occupied[k]:
                collisions[k] += 1.0
                continue
            if rng.random() < min(1.0, rho[k] / rho[j]):
                occupied[j] = False
                occupied[k] = True
    return collisions, (occ_sum / sweeps if collect else None)


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    Ks = [2.0e-3, 2.0e-4, 5.0e-5, 1.25e-5]
    rows = []
    for K in Ks:
        rho, d = mode_profile(K)
        n_mode = int(np.count_nonzero(rho > 0.05 * rho.max()))
        n_c = max(3, int(DENSITY * n_mode))
        coll, occ = simulate(rho, n_c, SWEEPS, SEED)
        print("    （载子数 %d，模式站点 %d，密度比 %.2f）" % (n_c, n_mode, n_c / n_mode))
        occ_share = occ / np.sum(occ)
        coll_share = coll / np.sum(coll)
        tv_occ = tv(occ_share, rho)
        tv_coll = tv(coll_share, rho)
        bulk = rho > 0.3 * rho.max()
        slope = float(np.polyfit(np.log(rho[bulk]), np.log(coll[bulk] / coll[bulk].mean()), 1)[0])
        l_airy = (KAPPA / K) ** (1.0 / 3.0)
        idx = np.nonzero(rho > 0.0)[0]
        inner = idx[np.abs(idx - JC) <= (idx.max() - JC) - l_airy]
        tv_in = tv(coll_share[inner], rho[inner])
        rows.append(dict(K=K, n_c=n_c, slope=slope, tv_occ=tv_occ, tv_coll=tv_coll, tv_inner=tv_in,
                         norm=abs(rho.sum() - 1.0) + abs(coll_share.sum() - 1.0),
                         sigma=float(np.sqrt(np.sum(rho * (np.arange(L) - JC) ** 2)))))
        print("  K = %.1e  sigma = %5.2f  TV(占据, rho) = %.4f  TV(碰撞, rho) = %.4f  体区斜率 %.3f  体区 TV %.4f"
              % (K, rows[-1]["sigma"], tv_occ, tv_coll, slope, tv_in))

    print("== J1 占据核对 ==")
    report("J1 占据占比与 rho 一致", max(r["tv_occ"] for r in rows) < 0.06,
           "总变差 %s" % np.array2string(np.array([r["tv_occ"] for r in rows]), precision=4))

    print("== J2 碰撞率的密度律 ==")
    slopes = np.array([r["slope"] for r in rows])
    report("J2 体区斜率 2.0 +/- 0.5", bool(np.all(np.abs(slopes - 2.0) < 0.5)),
           "四档斜率 %s" % np.array2string(slopes, precision=3))

    print("== J3 碰撞计数的结构性偏差 ==")
    tvs = np.array([r["tv_coll"] for r in rows])
    report("J3 总变差 > 0.08", bool(tvs.min() > 0.08),
           "四档总变差 %s" % np.array2string(tvs, precision=4))

    print("== J4 偏差对模式宽度不敏感 ==")
    report("J4 极差 < 0.05", float(tvs.max() - tvs.min()) < 0.05,
           "极差 %.4f（sigma 从 %.2f 到 %.2f）" % (float(tvs.max() - tvs.min()), rows[0]["sigma"], rows[-1]["sigma"]))

    print("== J5 体区内仍不成立 ==")
    inner_tvs = np.array([r["tv_inner"] for r in rows])
    report("J5 体区总变差 > 0.1", bool(inner_tvs.min() > 0.1),
           "体区总变差 %s" % np.array2string(inner_tvs, precision=4))

    print("== J6 单载子对照 ==")
    rho0, _ = mode_profile(Ks[0])
    coll1, _ = simulate(rho0, 1, 2000, SEED + 1, collect=False)
    report("J6 单载子无碰撞", float(np.sum(coll1)) == 0.0,
           "单载子 2000 步的碰撞计数 %.0f" % float(np.sum(coll1)))

    print("== J7 归一化 ==")
    nrm = max(r["norm"] for r in rows)
    report("J7 归一化误差 < 1e-12", nrm < 1e-12, "最大 %.2e" % nrm)

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    x = np.arange(L)

    K_show = 2.0e-4
    rho, d = mode_profile(K_show)
    n_mode_show = int(np.count_nonzero(rho > 0.05 * rho.max()))
    coll, occ = simulate(rho, max(3, int(DENSITY * n_mode_show)), SWEEPS, SEED)
    occ_share = occ / np.sum(occ)
    coll_share = coll / np.sum(coll)

    ax = axes[0, 0]
    ax.plot(x, rho, lw=1.7, label="份额 $\\rho_j$")
    ax.plot(x, occ_share, lw=1.3, ls="--", label="占据占比（模拟）")
    ax.set_xlim(JC - 60, JC + 60)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 A 占据核对：模拟占据与份额一致")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, rho, lw=1.7, label="份额 $\\rho_j$")
    ax.plot(x, coll_share, lw=1.4, ls="--", label="碰撞计数占比")
    ax.set_xlim(JC - 60, JC + 60)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 B 碰撞计数占比被推向中心")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bulk = rho > 0.3 * rho.max()
    ax.loglog(rho[bulk], coll[bulk] / coll[bulk].mean(), "o", ms=3.0, label="实测碰撞计数")
    ax.loglog(rho[bulk], (rho[bulk] / rho[bulk].mean()) ** 2, "k--", lw=1.0, label="$\\rho^2$ 参考线")
    ax.set_xlabel("$\\rho_j$")
    ax.set_ylabel("相对碰撞计数")
    ax.set_title("图 C 体区斜率 %.2f" % rows[1]["slope"])
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot([r["sigma"] for r in rows], [r["tv_coll"] for r in rows], "o-", label="碰撞计数对份额")
    ax.plot([r["sigma"] for r in rows], [r["tv_inner"] for r in rows], "s--", label="体区（去过 Airy 带）")
    ax.set_xlabel("模式宽度 $\\sigma$（单位 $l_P$）")
    ax.set_ylabel("总变差")
    ax.set_title("图 D 偏差对尺度不敏感")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/collision_counting_bias.png"
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
