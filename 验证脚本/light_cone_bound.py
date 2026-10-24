# -*- coding: utf-8 -*-
"""光锥不等式最小验证（光锥-波场双层方案，README 第 522 条）。

判据
----
J1（采纳路线全域不超光速）：连续色散 omega(k) = (2/t_P) sqrt(S)，
   S = sum_a sin^2(k_a l_P / 2)，在四档布里渊区网格上 max |grad omega| < c。
J2（上确界结构）：网格上 max |grad omega| / c = cos(pi/N) 精确成立（解析预言），
   且随 N 增大单调趋于 1，即上确界 c 在 k -> 0 取得。
J3（恒真检验，EP5 精神"改掉它就 FAIL"）：第 441 条判弃的离散递推路线
   sin^2(omega t_P / 2) = S 在同一检验程序下斜向群速超光速（对角线扫描，
   期望 max > 3，q l_P = 0.3 处约 1.0237 为手算对照）；同一点集上采纳路线
   保持 |grad omega| < 1。

模型设定：l_P = t_P = 1，故 c = 1。梯度用解析公式，无截断误差；
容差取浮点舍入量级 TOL = 1e-12（约四个数量级裕度）。
判据可逆性：判据稳健（解析恒等式 + 精确网格极值公式，换实现换档不翻）。
"""

import numpy as np

TOL = 1e-12
NS = [16, 32, 64, 128]
Q_SCAN = np.arange(0.10, 1.201, 0.01)


def make_grids(axis):
    """用广播代替 meshgrid，省内存：返回 S 与 sum sin^2(k_a) 的三维数组。"""
    h = np.sin(axis / 2.0) ** 2
    s = np.sin(axis) ** 2
    S = h[:, None, None] + h[None, :, None] + h[None, None, :]
    ssum = s[:, None, None] + s[None, :, None] + s[None, None, :]
    return S, ssum


def grad2_cont(S, ssum):
    """采纳路线 |grad omega|^2（解析）：(1/4) sum sin^2(k_a) / S。"""
    return 0.25 * ssum / S


def grad2_rec(S, ssum):
    """递推路线 |grad omega|^2（解析）：(1/4) sum sin^2(k_a) / (S (1 - S))。"""
    return 0.25 * ssum / (S * (1.0 - S))


def main():
    print("=== 冒烟检查 ===")
    q = 1.0
    S_probe = 3.0 * np.sin(q / 2.0) ** 2
    print(f"探针 S(对角 q=1.0) = {S_probe:.6f}（须 < 1，递推路线有定义）")
    print(f"连续路线谱上限 omega_BZ = 2*sqrt(3) = {2.0 * np.sqrt(3.0):.6f}"
          "（对照时空观 §五）")

    print("\n=== J1 + J2：采纳路线，四档布里渊区网格 ===")
    j1_pass = True
    j2_pass = True
    prev_max = 0.0
    maxes = []
    for N in NS:
        axis = -np.pi + 2.0 * np.pi * np.arange(N) / N
        S, ssum = make_grids(axis)
        mask = S > 1e-300  # 排除 k = 0（梯度 0/0，极限方向依赖）
        g2 = grad2_cont(S[mask], ssum[mask])
        gmax = float(np.sqrt(g2.max()))
        maxes.append((N, gmax))
        expect = float(np.cos(np.pi / N))
        ok1 = gmax < 1.0 - 1e-9
        ok2 = abs(gmax - expect) < TOL
        j1_pass &= ok1
        j2_pass &= ok2 and (gmax > prev_max)
        prev_max = max(prev_max, gmax)
        print(f"N={N:4d}: max|grad|/c 实测 = {gmax:.12f} | "
              f"期望 cos(pi/N) = {expect:.12f} | "
              f"<1 判据 {'PASS' if ok1 else 'FAIL'} | "
              f"精确判据 {'PASS' if ok2 else 'FAIL'}")
    print(f"J1（全域不超光速）: {'PASS' if j1_pass else 'FAIL'}")
    print(f"J2（上确界结构 max/c = cos(pi/N)，单调趋 1）: "
          f"{'PASS' if j2_pass else 'FAIL'}")

    print("\n=== J3：恒真检验，递推路线反例（改掉它就 FAIL）===")
    kx, ky, kz = Q_SCAN, Q_SCAN, Q_SCAN
    S_d = (np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2 + np.sin(kz / 2) ** 2)
    s_d = (np.sin(kx) ** 2 + np.sin(ky) ** 2 + np.sin(kz) ** 2)
    g_rec = np.sqrt(grad2_rec(S_d, s_d))
    g_con = np.sqrt(grad2_cont(S_d, s_d))
    r30 = float(np.interp(0.3, Q_SCAN, g_rec))
    imax = int(np.argmax(g_rec))
    print(f"递推路线对角 q l_P = 0.3: |grad|/c 实测 = {r30:.6f} | "
          f"期望约 1.0237（手算对照）")
    print(f"递推路线对角扫描 max|grad|/c = {g_rec[imax]:.6f}"
          f"（q l_P = {Q_SCAN[imax]:.2f}）| 期望 > 3（临近无解区发散）")
    print(f"采纳路线同一扫描 max|grad|/c = {g_con.max():.6f} | 期望 < 1（对照）")
    ok3 = (g_rec.max() > 3.0) and (abs(r30 - 1.0237) < 1e-3) \
        and (g_con.max() < 1.0)
    print(f"J3（反例有齿：递推超光速、采纳路线不超光速）: "
          f"{'PASS' if ok3 else 'FAIL'}")

    n_pass = sum([bool(j1_pass), bool(j2_pass), bool(ok3)])
    print(f"\n判据 {n_pass}/3（J1、J2、J3 加 EP5 恒真检验）")
    print("边界声明：本验证只检验色散函数的群速上限数学结构，")
    print("不检验耦合标度锁定（尺度锁定属条件性命题，待与色散论文配对清单核对），")
    print("不检验光传播的完整动力学，也不构成对真实光速测量的重新计算。")

    # 绘图（图存 images/，注记与论文嵌入）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    kk = np.linspace(0.02, np.pi, 400)
    axes[0].plot(kk, np.cos(kk / 2.0), "b-", label="adopted [100] axis")
    qq = np.linspace(0.05, 1.2, 200)
    axes[0].plot(qq, np.cos(qq / 2.0) / np.sqrt(np.maximum(
        1.0 - 3.0 * np.sin(qq / 2.0) ** 2, 1e-9)), "r--",
        label="rejected recursion [111]")
    axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("$q\\,l_P$"); axes[0].set_ylabel("$|\\nabla\\omega|/c$")
    axes[0].set_ylim(0, 4.2); axes[0].legend(fontsize=8)
    axes[0].set_title("group velocity: adopted vs rejected")
    Ns = [m[0] for m in maxes]; gs = [m[1] for m in maxes]
    axes[1].plot(Ns, gs, "bo", label="grid max (measured)")
    axes[1].plot(Ns, [float(np.cos(np.pi / n)) for n in Ns], "g--",
                 label="$\\cos(\\pi/N)$ (analytic)")
    axes[1].set_xlabel("$N$"); axes[1].set_ylabel("$\\max|\\nabla\\omega|/c$")
    axes[1].set_xscale("log"); axes[1].legend(fontsize=8)
    axes[1].set_title("supremum structure: $c$ at $k\\to 0$")
    axes[2].plot(Q_SCAN, g_con, "b-", label="adopted (subluminal)")
    axes[2].plot(Q_SCAN, g_rec, "r--", label="rejected (divergent)")
    axes[2].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[2].set_xlabel("$q\\,l_P$ along [111]")
    axes[2].set_ylabel("$|\\nabla\\omega|/c$")
    axes[2].legend(fontsize=8); axes[2].set_title(
        "EP5 counterexample: recursion diverges")
    fig.tight_layout()
    fig.savefig(r"D:\software\TianXingFC\images\light_cone_bound_522.png",
                dpi=150)
    print("图已存 images/light_cone_bound_522.png")


if __name__ == "__main__":
    main()
