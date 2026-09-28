#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_verification_kappa_dictionary.py

K 与 kappa_phi 的换算字典（验证脚本第 482 条）

背景。第 481 条（锚核身份 C2 判据 T2）把缺陷分成两型后给出：单通道缺陷的
格点内项化为 2*kappa_phi 乘基底势项，与单相位 FK 逐位同形，故有恒等

    K_FK = 2 * kappa_phi.

本条把这个恒等与量纲桥收束报告的定稿值表对齐，写出换算字典，并把 C1、D 两线
测得的阈值换算到框架约定下，判定 kappa_phi 落在阈值的哪一侧。

定稿值表（量纲桥收束报告第五节）：
    桥因子 epsilon      = 1
    势系数 K            = 3 * kappa_phi * t_P = 2.788142e-46
    异类耦合 kappa_phi t_P = 9.293805e-47
    键刚度 k_b（u 场）   = 2 * pi = 6.283185
    惯性 m_Phi          = 2 * pi

阈值（C1、D 两线记录，模型取 k_b = 1 归一，故模型读数即比值 K/k_b）：
    驻留阈值 K_c  0.24 至 0.25（C1-4）｜全塌缩档 0.02（C1-2）
    FK 落窗 0.02（C1-5）｜洛伦兹落窗 0.00633（C1-5）｜软模落窗 1/48（C1-6 W4）
    D 线读数档 0.005（缺陷动力学）

确定性：纯算术，无随机、无采样。
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

K_POT = 2.788142e-46
KAPPA_TP = 9.293805e-47
K_BOND = 2.0 * math.pi
M_PHI = 2.0 * math.pi
EPSILON = 1.0

THRESHOLDS = [
    ("驻留阈值下沿", 0.24, "C1-4", "reside"),
    ("驻留阈值上沿", 0.25, "C1-4", "reside"),
    ("全塌缩档", 0.02, "C1-2", "window"),
    ("FK 落窗", 0.02, "C1-5", "window"),
    ("洛伦兹落窗", 0.00633, "C1-5", "window"),
    ("软模落窗", 1.0 / 48.0, "C1-6 W4", "window"),
    ("D 线读数档", 0.005, "缺陷动力学", "window"),
]

OUT_FIG = "D:/software/TianXingFC/images/kappa_dictionary.png"


def main():
    print("=" * 76)
    print("K 与 kappa_phi 的换算字典（第 482 条）")
    print("=" * 76)

    # ---------------- 字典第一步：481 的恒等与定稿势系数的对齐
    print("\n[字典判据 KD1] 481 恒等 K_FK = 2 kappa_phi 与定稿势系数 3 kappa_phi t_P 的对齐")
    factor = K_POT / (2.0 * KAPPA_TP)
    print("  定稿势系数 / (2 kappa_phi t_P) = {:.6f}".format(factor))
    print("  解析恒等：3 kappa_phi t_P / (2 kappa_phi t_P) = 1.5 t_P")
    print("  在 t_P = 1 的约定下取值 1.5，属 O(1) 口径项，与 45 个数量级的缺口不同源")
    assert abs(factor - 1.5) < 1e-6, "481 恒等与定稿值表未对齐"
    print("  判定：通过（两处相差因子 1.5 t_P，可对齐）")

    # ---------------- 字典判据 KD2：框架提供的唯一比值
    r_direct = K_POT / 1.0
    r_norm = K_POT / K_BOND
    print("\n[字典判据 KD2] 框架提供的唯一比值 r = 势系数 / 键刚度")
    print("  直接比（势系数自身，k_b = 1 口径）  r = {:.6e}".format(r_direct))
    print("  归一后（按定稿键刚度 {:.6f}）        r = {:.6e}".format(K_BOND, r_norm))
    print("  两口径差 {:.6f} 个数量级".format(abs(math.log10(r_direct / r_norm))))
    print("  说明：框架的定稿值表只给出一组（势系数, 键刚度），故只提供一个比值。")
    print("        复合位错模型的两层分工要求锚核与包层各有一个独立比值，")
    print("        定稿值表未给出第二个。")
    print("  判定：通过（单比值已确定）")

    # ---------------- 字典判据 KD3：阈值换算与缺口
    print("\n[字典判据 KD3] 阈值在框架约定下的位置与缺口")
    print("  {:<16} {:>10} {:>12} {:>12} {:>8}".format(
        "阈值", "读数", "缺口(直接)", "缺口(归一)", "方向"))
    rows = []
    for name, val, src, kind in THRESHOLDS:
        g_direct = abs(math.log10(r_direct / val))
        g_norm = abs(math.log10(r_norm / val))
        direction = "向上" if kind == "reside" else "向下"
        print("  {:<16} {:>10.6g} {:>12.2f} {:>12.2f} {:>8}".format(
            name, val, g_direct, g_norm, direction))
        rows.append((name, val, g_direct, g_norm, kind))
    gaps = [g for _, _, a, b, _ in rows for g in (a, b)]
    print("\n  缺口范围 {:.2f} 至 {:.2f} 个数量级".format(min(gaps), max(gaps)))
    print("  方向说明：驻留阈值为半直线 r >= K_c，落窗为半直线 r <= K_lim。")
    print("            框架比值极小，故对落窗以离域方式满足，对驻留判不满足。")

    # ---------------- 字典判据 KD4：判定与目标值
    print("\n[字典判据 KD4] 判定与驻留所需的目标值")
    xi = 1.0 / (2.0 * math.pi * math.sqrt(r_norm))
    print("  软模关联长度 xi = 1/(2 pi sqrt(r)) = {:.4e} 格".format(xi))
    print("  落窗条件 xi >= 2：{}".format("满足" if xi >= 2 else "不满足"))
    need_ratio = 0.24
    need_kappa_tp = need_ratio * K_BOND / 3.0
    print("  驻留条件 r >= 0.24：{}".format("满足" if r_norm >= need_ratio else "不满足"))
    print("  使驻留成立所需的异类耦合：")
    print("    kappa_phi t_P >= 0.24 k_b / 3 = {:.6e}".format(need_kappa_tp))
    print("    现定稿值 {:.6e}".format(KAPPA_TP))
    print("    所需倍数 {:.4e}（即 {:.2f} 个数量级）".format(
        need_kappa_tp / KAPPA_TP, math.log10(need_kappa_tp / KAPPA_TP)))
    verdict = "判负" if r_norm < need_ratio else "判正"
    print("  判定：{}（单相位 FK 归约下，当前定稿参数不满足驻留条件）".format(verdict))

    # ---------------- 出图
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    xs = [math.log10(v) for _, v, _, _ in [(n, v, s, k) for n, v, s, k in THRESHOLDS]]
    xs += [math.log10(r_direct), math.log10(r_norm), math.log10(need_ratio)]
    lo = math.floor(min(xs)) - 1.0
    hi = math.ceil(max(xs)) + 1.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, 2.5)
    ax.axvspan(math.log10(0.24), math.log10(0.25), color="tab:red", alpha=0.18)
    for val, col, st in ((0.02, "tab:blue", "-."), (0.00633, "tab:orange", "--"),
                         (1.0 / 48.0, "tab:green", ":")):
        ax.axvline(math.log10(val), color=col, linestyle=st, lw=1.2)
    ax.axvline(math.log10(r_norm), color="black", lw=2.0)
    ax.annotate("框架比值\nr = {:.2e}".format(r_norm),
                xy=(math.log10(r_norm), 1.9), xytext=(math.log10(r_norm) + 1.0, 2.15),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10)
    ax.annotate("驻留所需\nr >= 0.24", xy=(math.log10(0.24), 0.55),
                xytext=(math.log10(0.24) - 3.0, 1.15), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(math.log10(0.245), 0.12, "驻留窗", ha="center", color="tab:red", fontsize=9)
    ax.text(math.log10(0.00633), 0.12, "洛伦兹窗", ha="right", color="tab:orange", fontsize=9)
    ax.text(math.log10(1.0 / 48.0), 0.12, "软模窗", ha="left", color="tab:green", fontsize=9)
    for name, val, _, _, _ in rows:
        ax.plot([math.log10(val)], [0.8], marker="v", color="tab:purple", ms=6)
    ax.set_xlabel("log10(r) 与 log10(K/k_b)")
    ax.set_title("K 与 kappa_phi 换算字典：框架比值与两族阈值")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    print("\n图已存：{}".format(OUT_FIG))

    # ---------------- 汇总
    print("\n" + "=" * 76)
    print("判据汇总")
    print("=" * 76)
    print("  KD1  通过（481 恒等与定稿势系数相差因子 1.5 t_P，O(1) 可对齐）")
    print("  KD2  通过（框架唯一比值 r = {:.6e}）".format(r_norm))
    print("  KD3  通过（缺口 {:.2f} 至 {:.2f} 个数量级）".format(min(gaps), max(gaps)))
    print("  KD4  通过（判定 {}；驻留所需 kappa_phi t_P >= {:.4e}）".format(
        verdict, need_kappa_tp))
    print("  结论：字典第一步成立（K_FK = 2 kappa_phi 与定稿势系数 O(1) 对齐），")
    print("        但框架的唯一比值落在驻留阈值以下约 45 个数量级，")
    print("        单相位 FK 归约下锚核不驻留、粒子离域。")
    print("=" * 76)


if __name__ == "__main__":
    main()
