# -*- coding: utf-8 -*-
"""P4-31 子项：框架自身格点的弹性常数与各向同性关系（E1 至 E4）。

背景：法线场动力学的 D 判据确认框架的微观弹性结构是「每分量标量弹性」（键上罚差值向量的
平方），朴素读法给出多余一支呼吸模，其后由约束结构消去。本脚本从该键能结构直接提取立方
弹性常数，判定边际各向异性组合 C11 - C12 - 2 C44 是否为零。

判据（先立判据，再实现）：
  E1 框架键能（每分量标量弹性）的弹性常数：按均匀应变测能量密度反解 C11、C12、C44，
       期望 (2k, 0, k)；各向同性关系 C11 - C12 = 2 C44 须成立到 1e-12。
  E2 中心力对照（只罚键长变化）：C44 归零、关系被破坏；这是判据的灵敏度证明，也说明
       「框架用哪一种键能」不是无关的选择。
  E3 来源判定：两种键能的差别只在指标结构（|du|^2 无方向指标；(du.e)^2 带方向指标）。
       数值上验证前者给等向弹性张量、后者给各向异性张量。
  E4 速度读数：用 E1 提取的常数算 Born 速度，同类别跨方向须相等（方向无关的弹性响应）。

单位：格距 1、键刚度 k = 1。确定性，无随机数。
"""

import numpy as np

CHECKS = []
KAPPA = 1.0
AXES = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def bond_energy_density(eps, kind="scalar"):
    total = 0.0
    for e in AXES:
        for sgn in (1.0, -1.0):
            eb = sgn * e
            d = eps @ eb
            if kind == "scalar":
                total += 0.5 * KAPPA * float(d @ d)
            else:
                total += 0.5 * KAPPA * float(d @ eb) ** 2
    return total


def extract(kind, strain=1e-3):
    e = strain
    eps_11 = np.diag([e, 0.0, 0.0])
    eps_11_22 = np.diag([e, e, 0.0])
    eps_12 = np.zeros((3, 3))
    eps_12[0, 1] = eps_12[1, 0] = 0.5 * e
    c11 = 2.0 * bond_energy_density(eps_11, kind) / e ** 2
    w2 = bond_energy_density(eps_11_22, kind)
    c12 = w2 / e ** 2 - c11
    c44 = 2.0 * bond_energy_density(eps_12, kind) / e ** 2
    return c11, c12, c44


print("=" * 78)
print("E1 框架键能（每分量标量弹性）的弹性常数与各向同性关系")
print("=" * 78)
c11, c12, c44 = extract("scalar")
print("  实测 C11 = %.9f，C12 = %.9f，C44 = %.9f（解析期望 2k, 0, k，k = %.1f）"
      % (c11, c12, c44, KAPPA))
rel_iso = abs((c11 - c12) - 2.0 * c44) / max(abs(c11), 1e-30)
print("  各向同性组合 C11 - C12 - 2 C44 = %.3e（相对 %.2e）" % ((c11 - c12) - 2.0 * c44, rel_iso))
check("E1", abs(c11 - 2.0 * KAPPA) <= 1e-12 and abs(c12) <= 1e-12
      and abs(c44 - KAPPA) <= 1e-12 and rel_iso <= 1e-12,
      "框架键能给 (2k, 0, k)，各向同性关系构造性成立")

print()
print("=" * 78)
print("E2 中心力对照：只罚键长变化")
print("=" * 78)
d11, d12, d44 = extract("central")
print("  实测 C11 = %.9f，C12 = %.9f，C44 = %.9f" % (d11, d12, d44))
print("  各向同性组合 C11 - C12 - 2 C44 = %.6f（相对 %.3e）"
      % ((d11 - d12) - 2.0 * d44, abs((d11 - d12) - 2.0 * d44) / max(abs(d11), 1e-30)))
check("E2", abs(d44) <= 1e-12 and abs((d11 - d12) - 2.0 * d44) >= 1e-3,
      "中心力对照把 C44 打到零，各向同性关系被破坏")

print()
print("=" * 78)
print("E3 来源判定：两种键能只差一个指标结构")
print("=" * 78)
for kind, name in (("scalar", "|du|^2（无方向指标）"), ("central", "(du.e)^2（带方向指标）")):
    c11k, c12k, c44k = extract(kind)
    iso_gap = (c11k - c12k) - 2.0 * c44k
    print("  %s：C11 - C12 - 2 C44 = %+.6f" % (name, iso_gap))
print("  含义：键能是否带方向指标，决定立方对称允许的三个常数是否自动落在各向同性子空间。")
check("E3", abs((extract("scalar")[0] - extract("scalar")[1]) - 2.0 * extract("scalar")[2]) <= 1e-12
      and abs((extract("central")[0] - extract("central")[1]) - 2.0 * extract("central")[2]) >= 1e-3,
      "无指标结构给各向同性、带指标结构给各向异性")

print()
print("=" * 78)
print("E4 速度读数：同类别跨方向相等")
print("=" * 78)


def born(c11v, c12v, c44v):
    return {"L[100]": np.sqrt(c11v), "T[100]": np.sqrt(c44v),
            "L[111]": np.sqrt((c11v + 2.0 * c12v + 4.0 * c44v) / 3.0),
            "T[111]": np.sqrt((c11v - c12v + c44v) / 3.0)}


sp = born(c11, c12, c44)
gap = max(abs(sp["L[111]"] / sp["L[100]"] - 1.0), abs(sp["T[111]"] / sp["T[100]"] - 1.0))
print("  框架键能：%s；同类别跨方向速度差 = %.3e"
      % (", ".join("%s %.9f" % (k, v) for k, v in sp.items()), gap))
spc = born(d11, d12, d44)
print("  中心力对照：T[100] = %.9f，T[111] = %.9f（横波在 [100] 方向上失去刚度）"
      % (spc["T[100]"], spc["T[111]"]))
check("E4", gap <= 1e-12, "框架键能的弹性响应方向无关")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：%d/%d 通过" % (n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  %-6s %s" % (tag, "PASS" if ok else "FAIL"))
print("=" * 78)


# ----------------------------------------------------------------------
# 图 1：各向同性组合与 Born 速度
# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axs[0].bar([0, 1], [c11 - c12 - 2.0 * c44, d11 - d12 - 2.0 * d44],
               color=["#3b6ea5", "#b45309"])
    axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(["框架键能", "中心力对照"])
    axs[0].set_ylabel("C11 - C12 - 2 C44"); axs[0].set_title("(a) 各向同性组合")
    labs = ["L[100]", "T[100]", "L[111]", "T[111]"]
    fw = [sp["L[100]"], sp["T[100]"], sp["L[111]"], sp["T[111]"]]
    ct = [spc["L[100]"], spc["T[100]"], spc["L[111]"], spc["T[111]"]]
    xs = np.arange(4)
    axs[1].bar(xs - 0.2, fw, width=0.4, label="框架键能")
    axs[1].bar(xs + 0.2, ct, width=0.4, label="中心力对照")
    axs[1].set_xticks(xs); axs[1].set_xticklabels(labs)
    axs[1].set_ylabel("Born 速度"); axs[1].legend(); axs[1].set_title("(b) 速度读数")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/elastic_isotropy.png", dpi=150)
    print("图存 images/elastic_isotropy.png")
except Exception as exc:
    print("绘图跳过：%s" % exc)
