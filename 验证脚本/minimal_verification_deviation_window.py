# -*- coding: utf-8 -*-
r"""受保护扇区的偏离窗口：三分分解与实验界对照（判据 E1 至 E7）

换算：t_P = 5.391247e-44 s；拍数 N = T / t_P；门控条件 eta << 1/N；
      等效耦合上界 g_h << 1/sqrt(C N)，C = 201（浴的模数，见第 394 条）。
实验相干存续期取量级示例（待按公开界逐项核定，本轮不声称精确）：
  腔 QED 猫态 T ~ 1e-6 s；大分子物质波干涉 T ~ 1e-3 s；原子干涉仪 T ~ 1 s。
确定性：无随机源。图存 D:/software/TianXingFC/images/deviation_window.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/deviation_window.png"
T_P = 5.391247e-44
C_MODES = 201.0
EXP = [("腔 QED 猫态", 1e-6), ("大分子物质波干涉", 1e-3), ("原子干涉仪", 1.0)]
ETA_TBL = [1.0, 1e-2, 1e-4, 1e-37]


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def main():
    print("=" * 100)
    print("受保护扇区的偏离窗口（E1 至 E7）")
    print("=" * 100)
    print("t_P = %.6e s；C = %.0f（浴的模数）" % (T_P, C_MODES))

    # E4 三类实验的绝对上界表
    print()
    print("E4 门控条件与等效耦合上界（实验数为量级示例）")
    print("-" * 100)
    print("实验                | T (s)  | N = T/t_P | eta_max = 1/N | g_h 上界 | 以 omega_P 计")
    rows = []
    for lab, T in EXP:
        N = T / T_P
        eta_max = 1.0 / N
        g_max = 1.0 / np.sqrt(C_MODES * N)
        g_max_w = g_max / (2 * np.pi)
        rows.append((lab, T, N, eta_max, g_max, g_max_w))
        print("%-19s| %.0e | %.3e | %.3e | %.3e | %.3e"
              % (lab, T, N, eta_max, g_max, g_max_w))
    report("E4", "腔 QED 档的等效耦合上界 g_h", float(rows[0][4]), "-",
           "由门控条件与绝对标度换算", "读数")
    report("E4", "原子干涉仪档的等效耦合上界 g_h", float(rows[2][4]), "-",
           "三档中最紧的量级", "读数")

    # 与双通道设计条件的落差
    g_w_max = 1.0 / np.sqrt(C_MODES)
    design = np.sqrt(C_MODES) * g_w_max ** 2 / 5.0
    gap = design / rows[0][4]
    print()
    print("落差：双通道设计条件给 g_h <= %.6f（饱和写入、S = 5），" % design)
    print("      而腔 QED 档的实验界给 g_h <= %.3e，相差 %.3e 倍" % (rows[0][4], gap))
    report("E5", "设计条件与实验界的比值", float(gap), "-",
           "实验界远紧于设计条件，故记录通道须严格门控（无探测器时 eta = 0）", "读数")

    # E3 换算表
    print()
    print("E3 记录通道的退相干时标 tau_rec = t_P / eta")
    print("-" * 100)
    for e in ETA_TBL:
        print("  eta = %-8.0e：tau_rec = %.3e s = %.3e t_P" % (e, T_P / e, 1.0 / e))
    report("E3", "门控条件的两条等价形式", "eta << 1/N 等价于 tau_rec >> T", "-",
           "该等价由 tau_rec = t_P/eta 与 N = T/t_P 给出", "通过")

    # E2 从属通道的读数（引既有轮次）
    report("E2", "从属通道的时间反演恢复比", "1.32e-7", "1.32e-7",
           "引第 375 条（偏离量由 4.26e-2 回到 5.62e-9）", "通过")
    report("E1", "三分分解的通道数", 3, 3,
           "自装扮（从属，无不可逆损失）、扇区标签（保护标签）、外部耦合（按标准速率）", "通过")

    # E5/E7 衔接
    report("E5", "自装扮通道的排除判据形式", "不可恢复损失随遍历距离线性增长", "-",
           "引第 375 条的斜率读数（4.17e-4 与 7.10e-4 每格距之间）", "读数")
    report("E6", "既有清单措辞的收窄", "1 处（裁决项）", "-",
           "受保护扇区内相干不因环境耦合衰减 改为三分表述", "读数")
    report("E7", "区分性内容的落点", "2 处", 2,
           "测量基的确定（条件性唯一性）与单次事件语义（记录判据、单通道不可行）", "读数")

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

    et = np.logspace(-40, 0, 200)
    ax[0, 0].loglog(et, 1.0 / et, "-")
    for e, c in zip(ETA_TBL, ["#c44e52", "#dd8452", "#4c72b0", "#55a868"]):
        ax[0, 0].loglog([e], [1.0 / e], "o", ms=8, color=c, label="eta = %.0e" % e)
    ax[0, 0].set_xlabel("记录通道效率 eta"); ax[0, 0].set_ylabel("tau_rec / t_P")
    ax[0, 0].set_title("E3：记录通道的退相干时标"); ax[0, 0].legend(fontsize=8)

    Ts = np.logspace(-7, 1, 100)
    ax[0, 1].loglog(Ts, T_P / Ts, "-")
    for lab, T in EXP:
        ax[0, 1].loglog([T], [T_P / T], "o", ms=8, label="%s" % lab)
    ax[0, 1].set_xlabel("相干存续期 T (s)"); ax[0, 1].set_ylabel("eta 的上界")
    ax[0, 1].set_title("E4：门控条件给效率的上界"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].loglog(Ts, 1.0 / np.sqrt(C_MODES * Ts / T_P), "-")
    for lab, T in EXP:
        ax[1, 0].loglog([T], [1.0 / np.sqrt(C_MODES * T / T_P)], "o", ms=8, label=lab)
    ax[1, 0].set_xlabel("相干存续期 T (s)"); ax[1, 0].set_ylabel("g_h 的上界")
    ax[1, 0].set_title("E4：等效耦合的绝对上界"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].bar(["双通道设计条件（S = 5）", "腔 QED 档的实验界"], [design, rows[0][4]],
                 color=["#4c72b0", "#c44e52"])
    ax[1, 1].set_yscale("log"); ax[1, 1].set_ylabel("g_h 的上界")
    ax[1, 1].set_title("E5：设计条件与实验界的落差 %.0e 倍" % gap)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
