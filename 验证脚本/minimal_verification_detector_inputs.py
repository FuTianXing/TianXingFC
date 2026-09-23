# -*- coding: utf-8 -*-
r"""探测器物侧构造：读出量、阈值与持有时间的三件输入指认（判据 D1 至 D8）

本脚本不新造模型，只做复用与代数：
  D2 阈值指认的核对：解析吻键滑剖面 + 阈值 theta = w_max/2 给出 n_diff = 2
  D3 持有时间指认：tau_rec / t_P = 1 / eta 的换算表
  D4 单通道反证：S^2 <= eta <= 1 的可解域
  D5 双通道边界：eta_h <= eta_w^2 / S^2
  D6~D8 衔接读数：逐键传感器阵的双参数形态、与两处剩余项和限制一的接口
引用数值一律取自机制篇既有读数（§8.1 的 eta 公式与统计要求、§8.2 的稀释），
不重算其斜率。
确定性：无随机源。图存 D:/software/TianXingFC/images/detector_inputs.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/detector_inputs.png"
NB = 400
K, k = 0.005, 1.0
ELL = 1.0 / (2 * np.pi * np.sqrt(K / k))
WMAX = 1.0 / (np.pi * ELL)
ETA_LIST = [1e-3, 1e-2, 0.1, 0.5, 1.0]
S_LIST = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
ETA_W_LIST = [0.1, 0.5, 1.0]


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def w_profile(p):
    j = np.arange(NB) + 0.5
    return (1.0 / (np.pi * ELL)) / np.cosh((j - p) / ELL)


def ndiff(p, delta, theta):
    a = w_profile(p) > theta
    b = w_profile(p + delta) > theta
    return int(np.count_nonzero(a != b))


def main():
    print("=" * 100)
    print("探测器物侧构造：三件输入指认（D1 至 D8）")
    print("=" * 100)

    # D2 阈值指认
    theta = 0.5 * WMAX
    grid = np.linspace(100.0, 120.0, 801)
    nd2 = min(ndiff(p, 1.0, theta) for p in grid)
    print("D2 阈值指认：theta = w_max/2 = %.4f，delta_n = 1 的 min_p n_diff = %d" % (theta, nd2))
    report("D2", "阈值取 w_max/2 时的差异键位数", nd2, 2, "与判据篇 §2.3 的读数一致",
           "通过" if nd2 == 2 else "未过")

    # D3 持有时间指认
    print()
    print("D3 持有时间指认：tau_rec / t_P = 1 / eta")
    print("-" * 100)
    print("eta   | 登记率 Gamma t_P = eta | tau_rec / t_P = 1/eta | 该量级下的读出")
    for e in ETA_LIST:
        print("%-6.3f| %-22.3e | %-20.3e | %s" % (e, e, 1.0 / e,
              "周期结构显形" if e > 0.3 else ("量级估计" if e > 1e-2 else "与散粒噪声不可分辨")))
    report("D3", "eta = 1e-2 时的持有时间", 1.0 / 1e-2, 100.0, "换算表读数",
           "通过" if abs(1.0 / 1e-2 - 100.0) < 1e-9 else "未过")

    # 统计要求的一致性核对（机制篇 §8.1 的示例）
    S_ref = 5.0
    eta_ref = 1e-2
    N_ref = (S_ref / eta_ref) ** 2
    print()
    print("统计要求的一致性：S = %.0f、eta = %.2f 给 N = %.3e（机制篇 §8.1 的示例值 %.1e）"
          % (S_ref, eta_ref, N_ref, 2.5e5))
    report("D3", "统计要求示例值对机制篇原值的相对偏差", abs(N_ref - 2.5e5) / 2.5e5, 1e-9,
           "示例值一致性", "通过" if abs(N_ref - 2.5e5) / 2.5e5 < 1e-9 else "未过")

    # D4 单通道可解域
    print()
    print("D4 单通道反证：S^2 <= eta <= 1 的可解域")
    print("-" * 100)
    feas = {}
    for S in S_LIST:
        ok = (S ** 2) <= 1.0
        feas[S] = ok
        print("  S = %-5.1f：需 eta >= %.2f，与 eta <= 1 的合取 %s" % (S, S ** 2, "可解" if ok else "无解"))
    okc = sum(1 for q in feas if feas[q])
    report("D4", "单通道下可解的显著性档数", okc, 2,
           "仅 S <= 1 可解（0.5 与 1.0 两档）；机制篇示例取 S = 5，属无解档",
           "通过" if okc == 2 else "未过")

    # D5 双通道边界
    print()
    print("D5 双通道判据：eta_h <= eta_w^2 / S^2")
    print("-" * 100)
    print("eta_w | " + " | ".join("S=%.1f" % q for q in S_LIST))
    tbl = {}
    for ew in ETA_W_LIST:
        row = [ew ** 2 / (q ** 2) for q in S_LIST]
        tbl[ew] = row
        print("%-5.1f | " % ew + " | ".join("%.4f" % v for v in row))
    ratio = ETA_W_LIST[1] ** 2 / (5.0 ** 2)
    report("D5", "eta_w = 0.5 与 S = 5 时允许的 eta_h 上界", ratio, 0.01,
           "即保持通道须弱于写入通道 %.0f 倍以上" % (0.5 / ratio), "读数")
    report("D5", "双通道边界的单调性", "随 S 递减、随 eta_w 平方增长", "-", "形态读数", "读数")

    # D6~D8 衔接
    print()
    report("D1", "读出量指认（寄存器模 c 与逐键传感器 y_j 的关系）", "-", "-",
           "后者为前者的准经典极限；写入项 G Theta(w - theta) 为准古典化形态", "读数")
    report("D6", "结构相容（逐键传感器阵含写入深度与弛豫两参数）", "-", "-",
           "M2 的 G 与 tau 分居写入与保持两侧，形态与 D5 的双通道要求相容", "读数")
    report("D7", "与两处剩余项的接口", "-", "-",
           "指认后判据篇 §8（1）的场侧实现有对象、§8（3）的换算有形式（tau = t_P/eta）", "读数")
    report("D8", "与限制一的接口", "-", "-",
           "双通道分离给出放大链是构成条件的第二条理由（通道不对称），与深度的第一条并列", "读数")

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

    et = np.logspace(-4, 0, 200)
    ax[0, 0].loglog(et, 1.0 / et, "-")
    ax[0, 0].loglog([1e-2], [100.0], "r*", ms=14, label="机制篇示例 eta = 1e-2 -> 100 t_P")
    ax[0, 0].set_xlabel("登记效率 eta"); ax[0, 0].set_ylabel("持有时间 tau_rec / t_P")
    ax[0, 0].set_title("D3：持有时间与登记效率同源"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].semilogy(S_LIST, [q ** 2 for q in S_LIST], "o-", label="单通道下限 eta >= S^2")
    ax[0, 1].axhline(1.0, color="k", ls="--", lw=1.0, label="eta 的上界 1")
    ax[0, 1].axvline(1.0, color="gray", ls=":", lw=1.0, label="可解边界 S = 1")
    ax[0, 1].set_xlabel("所需显著性 S"); ax[0, 1].set_ylabel("所需 eta 下界")
    ax[0, 1].set_title("D4：单通道只能达到约 1 sigma"); ax[0, 1].legend(fontsize=8)

    for ew, c in zip(ETA_W_LIST, ["#4c72b0", "#dd8452", "#c44e52"]):
        ax[1, 0].semilogy(S_LIST, [ew ** 2 / (q ** 2) for q in S_LIST], "o-", color=c, label="eta_w = %.1f" % ew)
    ax[1, 0].set_xlabel("所需显著性 S"); ax[1, 0].set_ylabel("允许的 eta_h 上界")
    ax[1, 0].set_title("D5：双通道的保持耦合上界"); ax[1, 0].legend(fontsize=8)

    ewv = np.logspace(-2, 0, 200)
    ax[1, 1].semilogy(ewv, ewv ** 2 / 25.0, "-", label="S = 5 的边界")
    ax[1, 1].fill_between(ewv, 1e-8, ewv ** 2 / 25.0, color="#55a868", alpha=0.25, label="可行域（边界以下）")
    ax[1, 1].set_xlabel("写入通道 eta_w"); ax[1, 1].set_ylabel("保持通道 eta_h")
    ax[1, 1].set_ylim(1e-8, 1.0)
    ax[1, 1].set_title("D5：通道分离的可行域（S = 5）"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
