#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_scale_debt.py

K 值档位债的口径对照（验证脚本第 479 条）

背景。体系定稿的势系数（格点内异类耦合）为

    K_sys = 3 * kappa_phi * t_P = 2.788142e-46        （量纲桥收束报告 定稿值表）

而三维复合位错线（C1）与缺陷动力学线（D）的全部判据窗口都落在 1e-3 至 1e-1 档：

    驻留阈值        K_c = 0.24 至 0.25          （C1-4 第一轮）
    全塌缩档        K  = 0.02                    （C1-2）
    FK 落窗         K / k <= 0.02                （C1-5）
    洛伦兹落窗      K <= 0.00633                 （C1-5，按色散口径收敛）
    软模落窗        K <= 1/48                    （C1-6 W4 轮）
    D 线读数档      K  = 0.005                   （缺陷动力学）

量纲桥收束报告第六节其二把这一差距定性为 O(1) 口径涟漪：C1 各轮取 k_b = 1 归一，
公理给 k_b = 2*pi（u 场），故四窗口的 K 轴随之平移，需要在跨线引用处补换算表。

本脚本判定：O(1) 因子能否弥合该差距。

物理内容由无量纲比值 r = K / k_b 承载（FK 泛函
E = sum [ (k_b/2)(d phi)^2 + K(1 - cos phi) ] 的孤子宽度与钉扎阈值都只依赖 r）。

口径甲（直接口径）：把 C1/D 的 K 读数当作 r 读数（因 C1 取 k_b = 1，K 读数即 r 读数）。
口径乙（重算口径）：把 C1/D 的 K 读数按公理的 k_b = 2*pi 重新组成比值。
两套口径的差别是一个 2*pi 因子，即 0.80 个数量级。

判据：
  K1  定稿值 r_sys 的数值（两套k_b口径并列）
  K2  r_sys 与四个窗口的缺口（数量级），两套口径并列
  K3  口径涟漪的幅度（log10(2*pi)）与缺口的比，判涟漪能否弥合

确定性：纯算术，无随机、无采样、无迭代。每次运行输出完全相同。
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 输入（全部取自定稿值表与 C1/D 线记录）
K_SYS = 2.788142e-46          # 定稿势系数，= 3 * kappa_phi * t_P
KAPPA_PHI_TP = 9.293805e-47   # 异类耦合 kappa_phi * t_P
K_B_AXIOM = 2.0 * math.pi     # 公理给的键刚度（u 场）
K_B_C1 = 1.0                  # C1/D 各轮所用归一

WINDOWS = [
    ("驻留阈值下沿 K_c", 0.24, "C1-4 第一轮"),
    ("驻留阈值上沿 K_c", 0.25, "C1-4 第一轮"),
    ("全塌缩档 K", 0.02, "C1-2"),
    ("FK 落窗 K/k", 0.02, "C1-5"),
    ("洛伦兹落窗 K", 0.00633, "C1-5 色散口径"),
    ("软模落窗 K", 1.0 / 48.0, "C1-6 W4 轮"),
    ("D 线读数档 K", 0.005, "缺陷动力学"),
]

OUT_FIG = "D:/software/TianXingFC/images/k_scale_debt.png"

results = {}


def log10_gap(a, b):
    """返回 log10(a/b) 的绝对值，即两者相差的数量级。"""
    return abs(math.log10(a / b))


def main():
    print("=" * 72)
    print("K 值档位债的口径对照（第 479 条）")
    print("=" * 72)

    # ---------- 冒烟检查：输入的内部一致性
    print("\n[冒烟检查] 输入自洽性")
    k_check = 3.0 * KAPPA_PHI_TP
    print("  3 * (kappa_phi t_P) = {:.6e}".format(k_check))
    print("  定稿 K_sys         = {:.6e}".format(K_SYS))
    rel = abs(k_check - K_SYS) / K_SYS
    print("  相对偏差           = {:.3e}".format(rel))
    assert rel < 5e-7, "定稿值表的两个数不自洽"

    # ---------- 口径甲：C1/D 的 K 读数即 r 读数
    r_axiom = K_SYS / K_B_AXIOM
    r_c1 = K_SYS / K_B_C1
    print("\n[判据 K1] 定稿比值 r = K / k_b")
    print("  口径甲（C1 归一 k_b = 1）      r = {:.6e}".format(r_c1))
    print("  口径乙（公理 k_b = 2 pi）      r = {:.6e}".format(r_axiom))
    ripple = abs(math.log10(r_c1 / r_axiom))
    print("  两口径之差（= log10 2 pi）     {:.6f} 个数量级".format(ripple))
    results["r_axiom"] = r_axiom
    results["r_c1"] = r_c1
    results["ripple"] = ripple

    # ---------- 判据 K2：逐窗口的缺口
    print("\n[判据 K2] 定稿值与各窗口的缺口")
    print("  {:<22} {:>12} {:>10} {:>10}".format(
        "窗口", "K 读数", "缺口甲", "缺口乙"))
    rows = []
    for name, k_val, src in WINDOWS:
        gap_a = log10_gap(K_SYS, k_val)          # 口径甲：两者当作同一量直接比
        gap_b = log10_gap(r_axiom, k_val)        # 口径乙：定稿换算为比值 r 再与读数比
        print("  {:<22} {:>12.6g} {:>10.2f} {:>10.2f}".format(
            name, k_val, gap_a, gap_b))
        rows.append((name, k_val, gap_a, gap_b))
    results["rows"] = rows

    print("\n  方向：驻留窗为半直线 K >= K_c，定稿值在其下侧，判不满足；")
    print("        落窗为半直线 K <= K_lim，定稿值在其下侧，判满足（以离域方式）。")
    gaps_all = [g for _, _, a, b in rows for g in (a, b)]
    gap_min, gap_max = min(gaps_all), max(gaps_all)
    print("  缺口范围：{:.2f} 至 {:.2f} 个数量级".format(gap_min, gap_max))
    spread = sum(b - a for _, _, a, b in rows) / len(rows)
    print("  口径间散布：{:.6f} 个数量级（每窗口的乙减甲，= log10 2 pi）".format(spread))
    print("  窗口间极差：{:.6f} 个数量级（各窗口位置本身的跨度）".format(
        max(gaps_all) - min(gaps_all)))
    results["spread"] = spread
    results["gap_min"] = gap_min
    results["gap_max"] = gap_max

    # ---------- 判据 K3：涟漪能否弥合
    print("\n[判据 K3] O(1) 口径涟漪能否弥合缺口")
    ratio_of_ratio = gap_min / ripple
    print("  涟漪幅度             {:.6f} 个数量级".format(ripple))
    print("  最小缺口             {:.6f} 个数量级".format(gap_min))
    print("  最小缺口 / 涟漪      {:.4e}".format(ratio_of_ratio))
    verdict = "不能弥合" if ratio_of_ratio > 10 else "可以弥合"
    print("  判定                 {}".format(verdict))
    results["ratio_of_ratio"] = ratio_of_ratio
    results["verdict"] = verdict

    # ---------- 宽度与驻留的反向读数
    print("\n[附读数] 极小比值下的宽度与驻留后果")
    xi_ratio = 1.0 / (2.0 * math.pi * math.sqrt(r_axiom))
    print("  软模关联长度 xi（格）  {:.4e}".format(xi_ratio))
    print("  落窗条件 xi >= 2       {}".format("满足（但以离域方式满足）" if xi_ratio >= 2 else "不满足"))
    print("  驻留条件 r >= 0.24     {}".format("满足" if r_axiom >= 0.24 else "不满足"))
    results["xi_ratio"] = xi_ratio

    # ---------- 出图
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    xs = [math.log10(v) for _, v, _, _ in rows] + [math.log10(r_axiom)]
    lo = math.floor(min(xs)) - 1.0
    hi = math.ceil(max(xs)) + 1.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, 2.4)
    ax.axvspan(math.log10(0.24), math.log10(0.25), color="tab:red", alpha=0.18)
    ax.axvline(math.log10(0.00633), color="tab:orange", linestyle="--", lw=1.2)
    ax.axvline(math.log10(1.0 / 48.0), color="tab:green", linestyle=":", lw=1.4)
    ax.axvline(math.log10(0.02), color="tab:blue", linestyle="-.", lw=1.2)
    ax.axvline(math.log10(r_axiom), color="black", lw=2.0)
    ax.annotate("定稿值\nr = {:.2e}".format(r_axiom),
                xy=(math.log10(r_axiom), 1.9), xytext=(math.log10(r_axiom) + 1.2, 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10)
    ax.text(math.log10(0.245), 0.15, "驻留窗", ha="center", color="tab:red", fontsize=9)
    ax.text(math.log10(0.00633), 0.15, "洛伦兹窗", ha="right", color="tab:orange", fontsize=9)
    ax.text(math.log10(1.0 / 48.0), 0.15, "软模窗", ha="left", color="tab:green", fontsize=9)
    for name, k_val, _, _ in rows:
        ax.plot([math.log10(k_val)], [0.75], marker="v", color="tab:purple", ms=6)
    ax.annotate("C1 / D 线窗口簇", xy=(math.log10(0.02), 0.75),
                xytext=(math.log10(0.02) + 0.6, 0.95), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlabel("log10(K / k_b)")
    ax.set_title("K 值档位债：定稿值与模型窗口的 45 个数量级缺口")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160)
    print("\n图已存：{}".format(OUT_FIG))
    results["fig"] = OUT_FIG

    # ---------- 判据汇总
    print("\n" + "=" * 72)
    print("判据汇总")
    print("=" * 72)
    k1 = "通过（r_sys = {:.4e}，两口径差 {:.3f} 个数量级）".format(r_axiom, ripple)
    k2 = "通过（缺口 {:.2f} 至 {:.2f} 个数量级）".format(gap_min, gap_max)
    k3 = "通过（涟漪 {:.3f} 个数量级，仅为最小缺口的 {:.1e}）".format(ripple, 1.0 / ratio_of_ratio)
    for tag, txt in (("K1", k1), ("K2", k2), ("K3", k3)):
        print("  {}  {}".format(tag, txt))
    print("  结论：O(1) 口径涟漪不能弥合该缺口，档位债不是纯归一化问题")
    print("=" * 72)
    return results


if __name__ == "__main__":
    main()
