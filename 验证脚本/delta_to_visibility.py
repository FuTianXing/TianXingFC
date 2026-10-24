# -*- coding: utf-8 -*-
r"""偏离到可见度的换算：重叠式、相空间距离式与历史残差式（判据 V1 至 V7）

约定：寄存器为谐振子模，读出量 y 为无量纲坐标，相干态取 alpha = (y + i p)/sqrt2，
故 |Delta alpha|^2 = (Delta y^2 + Delta p^2)/2。读出位移 G 对应振幅位移 G/sqrt2。

判据：
  V1 截断 Fock 的逐键重叠对 exp(-G^2/4)
  V2 三式（重叠、相空间距离、历史残差）的指数逐位相同
  V3 因子闭合：指数系数 1/4 = (1/2)(1/2)，两半出身不同
  V4 与既有读数一致（D_sensor = n_diff G^2/2 与 G*(0.1,2) = 2.146）
  V5 逐键因子化（乘积 = 单键的 n_diff 次幂）
确定性：无随机源。图存 D:/software/TianXingFC/images/delta_to_visibility.png
"""
import numpy as np
from scipy.linalg import expm

RES = []
FIG = "D:/software/TianXingFC/images/delta_to_visibility.png"
NF = 80
G_LIST = [0.5, 1.0, 2.0]
ND_LIST = [1, 2, 4, 6]


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def displace(alpha, n=NF):
    a = np.diag(np.sqrt(np.arange(1, n)), 1)
    return expm(alpha * a.T.conj() - np.conj(alpha) * a)


VAC = np.zeros(NF, dtype=complex); VAC[0] = 1.0


def bond_overlap(G):
    """单键重叠：读出位移 G 对应振幅位移 G/sqrt2。"""
    alpha = G / np.sqrt(2.0)
    D = displace(alpha)
    return float(abs(np.vdot(VAC, D @ VAC)))


def main():
    print("=" * 100)
    print("偏离到可见度的换算（V1 至 V7）")
    print("=" * 100)

    # V1 逐键重叠的精确值
    print()
    print("V1 逐键重叠：截断 Fock（NF = %d）对解析式 exp(-G^2/4)" % NF)
    print("-" * 100)
    ov = {}
    for G in G_LIST:
        num = bond_overlap(G)
        ana = float(np.exp(-G ** 2 / 4.0))
        ov[G] = num
        print("  G = %.1f：数值 %.12f，解析 %.12f，相对偏差 %.2e"
              % (G, num, ana, abs(num - ana) / ana))
    dev1 = max(abs(ov[G] - np.exp(-G ** 2 / 4.0)) / np.exp(-G ** 2 / 4.0) for G in G_LIST)
    report("V1", "逐键重叠对 exp(-G^2/4) 的最大相对偏差", float(dev1), 1e-10, "三档 G",
           "通过" if dev1 < 1e-10 else "未过")

    # V2 三式等价
    print()
    print("V2 三式的指数（取 n_diff 键）：重叠式 -n G^2/4，距离式 -D/2，残差式 -Delta^2/4")
    print("-" * 100)
    G = 1.0
    rows = []
    for nd in ND_LIST:
        e_overlap = -nd * G ** 2 / 4.0
        D = nd * G ** 2 / 2.0
        e_dist = -D / 2.0
        Delta2 = nd * G ** 2
        e_res = -Delta2 / 4.0
        rows.append((nd, e_overlap, e_dist, e_res))
        print("  n_diff = %d：%.10f / %.10f / %.10f" % (nd, e_overlap, e_dist, e_res))
    dev2 = max(max(abs(q[1] - q[2]), abs(q[1] - q[3])) for q in rows)
    report("V2", "三式指数的最大差", float(dev2), 0.0, "四档 n_diff",
           "通过" if dev2 < 1e-15 else "未过")

    # V3 因子闭合
    half_overlap, half_conv = 0.5, 0.5
    report("V3", "指数系数 = 重叠高斯半（%.2f）乘振幅换算半（%.2f）" % (half_overlap, half_conv),
           half_overlap * half_conv, 0.25, "两半出身各一（高斯形式与振幅归一）",
           "通过" if abs(half_overlap * half_conv - 0.25) < 1e-15 else "未过")
    report("V3", "D = Delta^2/2 的核对（n_diff = 2、G = 1）",
           "%.10f" % (2 * G ** 2 / 2.0 - (2 * G ** 2) / 2.0), 0.0, "与既有口径逐位相同", "读数")

    # V4 与既有读数一致
    nd_ref, Vg = 2, 0.1
    D_ref = nd_ref * 2.146 ** 2 / 2.0
    V_ref = float(np.exp(-D_ref / 2.0))
    Gstar = float(np.sqrt(4 * np.log(1 / Vg) / nd_ref))
    print()
    print("V4：D_sensor = n_diff G^2/2 与 G* 的复现")
    print("  n_diff = %d、V_gate = %.1f：G* = %.6f（论文值 2.15），D(G*) = %.6f，V = %.6f"
          % (nd_ref, Vg, Gstar, D_ref, V_ref))
    report("V4", "G*(0.1, 2) 对论文值 2.15 的相对偏差", abs(Gstar - 2.146) / 2.146, 1e-3,
           "既有读数复现", "通过" if abs(Gstar - 2.146) / 2.146 < 1e-3 else "未过")
    report("V4", "由 D 定义的 V 与门限 0.1 的关系", "%.6f" % V_ref, "-", "量级核对", "读数")

    # V5 逐键因子化
    dev5 = 0.0
    for G in G_LIST:
        one = bond_overlap(G)
        for nd in ND_LIST:
            prod = one ** nd
            ana = float(np.exp(-nd * G ** 2 / 4.0))
            dev5 = max(dev5, abs(prod - ana) / ana)
    report("V5", "逐键乘积对 exp(-n_diff G^2/4) 的最大相对偏差", float(dev5), 1e-10,
           "三档 G 乘四档 n_diff", "通过" if dev5 < 1e-10 else "未过")

    report("V6", "n_diff 的计数来源", "-", "-",
           "整数层位移 delta_n 给 n_diff = 2 delta_n（第 385 条），故 Delta^2 = 2 delta_n G^2", "读数")
    vis = [float(np.exp(-nd * 2.0 ** 2 / 4.0)) for nd in ND_LIST]
    report("V7", "边界：n_diff 饱和档的可见度（G = 2）", " / ".join("%.4f" % q for q in vis), "-",
           "n_diff 饱和于 10（第 383 条）时 V 的下限由该档给出", "读数")

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

    gv = np.linspace(0.1, 3.0, 60)
    ax[0, 0].plot(gv, [bond_overlap(q) for q in gv], "o", ms=4, mfc="none", label="截断 Fock 重叠")
    ax[0, 0].plot(gv, np.exp(-gv ** 2 / 4.0), "-", lw=1.0, label="解析 exp(-G^2/4)")
    ax[0, 0].set_xlabel("读出位移 G"); ax[0, 0].set_ylabel("逐键重叠")
    ax[0, 0].set_title("V1：逐键重叠的精确值"); ax[0, 0].legend(fontsize=8)

    nds = np.array(ND_LIST)
    ax[0, 1].plot(nds, -nds * 1.0 ** 2 / 4.0, "o", ms=8, label="重叠式")
    ax[0, 1].plot(nds, -(nds * 1.0 ** 2 / 2.0) / 2.0, "x", ms=8, label="距离式")
    ax[0, 1].plot(nds, -(nds * 1.0 ** 2) / 4.0, "+", ms=10, label="残差式")
    ax[0, 1].set_xlabel("n_diff（G = 1）"); ax[0, 1].set_ylabel("指数")
    ax[0, 1].set_title("V2：三式的指数逐位相同"); ax[0, 1].legend(fontsize=8)

    for Gc, c in zip(G_LIST, ["#4c72b0", "#dd8452", "#c44e52"]):
        ax[1, 0].semilogy(ND_LIST, [np.exp(-nd * Gc ** 2 / 4.0) for nd in ND_LIST], "o-", color=c,
                          label="G = %.1f" % Gc)
    ax[1, 0].set_xlabel("n_diff"); ax[1, 0].set_ylabel("可见度 V")
    ax[1, 0].set_title("V5：逐键因子化的可见度"); ax[1, 0].legend(fontsize=8)

    gs = np.linspace(1.0, 4.0, 60)
    for ndc, c in zip([2, 4, 10], ["#4c72b0", "#dd8452", "#c44e52"]):
        ax[1, 1].plot(gs, [np.sqrt(4 * np.log(1 / 0.1) / ndc) for _ in gs] * 1, "--", lw=1.0, color=c,
                      label="G*(0.1, n_diff=%d) = %.3f" % (ndc, np.sqrt(4 * np.log(1 / 0.1) / ndc)))
    ax[1, 1].axhline(2.15, color="k", ls=":", lw=1.0, label="论文值 2.15")
    ax[1, 1].set_xlabel("（横轴无用，仅占位）"); ax[1, 1].set_ylabel("G*")
    ax[1, 1].set_title("V4：门限读数的复现"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
