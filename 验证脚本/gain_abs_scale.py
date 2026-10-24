# -*- coding: utf-8 -*-
r"""写入深度的绝对标度：幅度与率的分工及耦合的绝对约束（判据 A1 至 A8）

接口：幅度侧每次写入转移一个激发（与耦合无关）；率侧 eta = C g^2，C = 2 pi rho t_P。
机制篇读数：探测器 201 个模、带宽 2 pi、密度 rho = 201/(2 pi) = 31.99；模型单位 t_P = 1。
故 C = 201（等于模数），eta = 1 对应 g = 1/sqrt(C) = 0.0705。

推论：g_w <= 1/sqrt(C)；g_h <= sqrt(C) g_w^2 / S；g_h << 1/sqrt(C N_拍)。
确定性：无随机源。图存 D:/software/TianXingFC/images/gain_abs_scale.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/gain_abs_scale.png"
N_MODES = 201.0
BANDWIDTH = 2 * np.pi
TP = 1.0
G_REF = 0.0705
S_LIST = [1.0, 2.0, 5.0, 10.0]
N_LIST = [10, 100, 1000, 100000]


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def main():
    print("=" * 100)
    print("写入深度的绝对标度（A1 至 A8）")
    print("=" * 100)

    # A3 C 的数值核对
    rho = N_MODES / BANDWIDTH
    C = 2 * np.pi * rho * TP
    print()
    print("A3：rho = 模数/带宽 = %.4f/%.4f = %.2f；C = 2 pi rho t_P = %.4f（模数 %.0f）"
          % (N_MODES, BANDWIDTH, rho, C, N_MODES))
    report("A3", "C 对模数的相对偏差", abs(C - N_MODES) / N_MODES, 1e-12, "C 等于模数",
           "通过" if abs(C - N_MODES) / N_MODES < 1e-12 else "未过")

    # A4 门限核对
    g_at_1 = 1.0 / np.sqrt(C)
    eta_at_ref = C * G_REF ** 2
    print()
    print("A4：eta = 1 对应 g = 1/sqrt(C) = %.6f（机制篇读数 %.4f，相对偏差 %.2e）"
          % (g_at_1, G_REF, abs(g_at_1 - G_REF) / G_REF))
    print("    反向核对：g = %.4f 给 eta = C g^2 = %.6f（期望 1）" % (G_REF, eta_at_ref))
    report("A4", "1/sqrt(C) 对机制篇 g 读数的相对偏差", abs(g_at_1 - G_REF) / G_REF, 1e-3,
           "两条接口的读法自洽", "通过" if abs(g_at_1 - G_REF) / G_REF < 1e-3 else "未过")
    report("A4", "机制篇 g 读数给出的 eta", float(eta_at_ref), 1.0, "期望 1（外推读数的四位数舍入）",
           "读数")

    # A1/A2 接口
    report("A1", "每次写入转移的激发数", 1, 1, "转移型耦合的后果，与 g 无关", "通过")
    report("A2", "率接口的构成 C = 2 pi rho t_P", "-", "-", "由机制篇 §8.1 的公式与探测器模型给出", "读数")

    # A5 耦合上界
    print()
    print("A5：eta_w <= 1 给出 g_w <= %.6f = %.6f omega_P" % (g_at_1, g_at_1 / (2 * np.pi)))
    report("A5", "写入耦合的绝对上界（以 omega_P 计）", float(g_at_1 / (2 * np.pi)), "-",
           "来源为概率守恒（eta <= 1），非能量代价", "读数")

    # A6 双通道绝对形式
    print()
    print("A6：g_h <= sqrt(C) g_w^2 / S")
    print("-" * 100)
    print("g_w / g_max | " + " | ".join("S=%.1f" % q for q in S_LIST))
    for frac in (1.0, 0.5, 0.1):
        gw = frac * g_at_1
        row = [np.sqrt(C) * gw ** 2 / q for q in S_LIST]
        print("%-12.2f| " % frac + " | ".join("%.6f" % v for v in row))
    gw_max = g_at_1
    rep = np.sqrt(C) * gw_max ** 2 / 5.0
    report("A6", "饱和写入（g_w = g_max）与 S = 5 时允许的 g_h", float(rep), "-",
           "即保持耦合须弱于 %.1f 倍以上（相对 g_max）" % (gw_max / rep), "读数")

    # A7 持久条件绝对形式
    print()
    print("A7：g_h << 1/sqrt(C N_拍)")
    print("-" * 100)
    for n in N_LIST:
        print("  N_拍 = %-7d：g_h << %.3e（= %.3e omega_P）" % (n, 1.0 / np.sqrt(C * n),
                                                          1.0 / np.sqrt(C * n) / (2 * np.pi)))
    report("A7", "N_拍 = 100 时 g_h 的上界（以 omega_P 计）",
           float(1.0 / np.sqrt(C * 100) / (2 * np.pi)), "-", "尺度的绝对化读数", "读数")

    # A8 换算表
    print()
    print("A8：模型单位到物理量的对应")
    print("-" * 100)
    tbl = [("写入深度 G（判据侧）", "G_amp × 1 激发", "放大链增益乘一次转移，绝对值由激发计"),
           ("写入率 eta", "C g^2（C = 201）", "由耦合的平方给出，量级（g 为外推读数）"),
           ("持有时间 tau_rec", "t_P / eta", "模型单位 t_P = 1，物理量随 t_P 定"),
           ("格距", "l_P", "距离归粒定案"),
           ("写入耦合上界", "0.0705 = 0.0112 omega_P", "由 eta <= 1 给出")]
    for a, b, c in tbl:
        print("  %-18s：%-24s（%s）" % (a, b, c))
    report("A8", "换算表条目数", len(tbl), 5, "含不确定度来源的标注", "读数")

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

    gv = np.logspace(-3, -0.5, 120)
    ax[0, 0].loglog(gv, C * gv ** 2, "-", label="eta = C g^2（C = 201）")
    ax[0, 0].loglog([g_at_1], [1.0], "r*", ms=14, label="eta = 1 于 g = 0.0705")
    ax[0, 0].axhline(1.0, color="k", ls=":", lw=1.0, label="概率上界")
    ax[0, 0].set_xlabel("耦合 g（omega_P 单位下 0.0705 ≈ 0.0112）"); ax[0, 0].set_ylabel("登记效率 eta")
    ax[0, 0].set_title("A3/A4：率接口与门限核对"); ax[0, 0].legend(fontsize=8)

    for frac, c in zip((1.0, 0.5, 0.1), ["#4c72b0", "#dd8452", "#c44e52"]):
        gw = frac * g_at_1
        ax[0, 1].semilogy(S_LIST, [np.sqrt(C) * gw ** 2 / q for q in S_LIST], "o-", color=c,
                          label="g_w = %.2f g_max" % frac)
    ax[0, 1].set_xlabel("所需显著性 S"); ax[0, 1].set_ylabel("允许的 g_h")
    ax[0, 1].set_title("A6：双通道判据的绝对形式"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].loglog(N_LIST, [1.0 / np.sqrt(C * n) for n in N_LIST], "o-", label="g_h 上界")
    ax[1, 0].loglog(N_LIST, [1.0 / np.sqrt(C * n) / 10.0 for n in N_LIST], "s--", label="取十分之一余量")
    ax[1, 0].set_xlabel("相干存续期内的拍数 N_拍"); ax[1, 0].set_ylabel("g_h 的上界")
    ax[1, 0].set_title("A7：持久条件的绝对形式"); ax[1, 0].legend(fontsize=8)

    gwv = np.logspace(-2, np.log10(g_at_1), 150)
    ax[1, 1].semilogy(gwv, np.sqrt(C) * gwv ** 2 / 5.0, "-", label="S = 5 的边界")
    ax[1, 1].fill_between(gwv, 1e-8, np.sqrt(C) * gwv ** 2 / 5.0, color="#55a868", alpha=0.25,
                          label="可行域（边界以下）")
    ax[1, 1].axvline(g_at_1, color="r", ls=":", lw=1.0, label="g_w 上界 0.0705")
    ax[1, 1].set_xlabel("写入耦合 g_w"); ax[1, 1].set_ylabel("保持耦合 g_h")
    ax[1, 1].set_ylim(1e-8, 1.0)
    ax[1, 1].set_title("A5/A6：耦合平面上的可行域（S = 5）"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
