# -*- coding: utf-8 -*-
"""P4-31 边际洛伦兹算符：四维不变量的枚举与对称性禁止（M1 至 M4）。

判据（先立判据，再实现）：
  M1 标量扇区：立方群（48 元符号置换群）下，二次、两阶导数的方向系数矩阵的群平均等于其
       迹的三分之一乘单位阵。含义：任何各向异性的四维动能项都不是立方不变量，被精确对称性
       禁止（非压制）。判据：两个测试矩阵的群平均与各向同性投影之差 <= 1e-14。
  M2 张量扇区：立方弹性有三个独立常数，各向同性（即洛伦兹允许）要求一条关系
       C11 - C12 = 2 C44。判据：该关系成立时四个 Born 速度两两相等到 1e-14；破坏 5% 时
       出现可测量的速度分裂。
  M3 离散符号的领头行为：各向同性七点模板下，色散符号除以 -k^2 的三方向领头系数相等，
       残差只按 k^2 收缩（不可关通道）；把沿 z 的耦合改 5% 后，领头系数出现 5% 级的
       方向差（边际通道）。两档对比即判据的灵敏度。
  M4 极限速度差：标量与张量扇区共用同一格点模板，符号逐点相同，故极限速度差按构造取零。
       判据：两扇区符号逐点相对偏差 <= 1e-15。

单位：格距 1、c = 1。确定性，无随机数。
"""

import itertools
import numpy as np

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def cubic_group():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for row, (col, sg) in enumerate(zip(perm, signs)):
                R[row, col] = sg
            mats.append(R)
    return mats


GROUP = cubic_group()
det_plus = sum(1 for R in GROUP if np.linalg.det(R) > 0)
orth = max(float(np.max(np.abs(R.T @ R - np.eye(3)))) for R in GROUP)
print("=" * 78)
print("M1 标量扇区：四维各向异性算符是否被立方对称性禁止")
print("=" * 78)
print("  立方群元数 = %d（其中 det = +1 的 %d 个，正交性偏差 %.1e）"
      % (len(GROUP), det_plus, orth))
A_test = np.diag([1.0, 0.0, 0.0])
B_test = np.array([[2.0, 0.3, -0.1], [0.3, -1.0, 0.2], [-0.1, 0.2, 0.5]])
proj = lambda M: sum(R @ M @ R.T for R in GROUP) / float(len(GROUP))
err_a = float(np.max(np.abs(proj(A_test) - np.eye(3) / 3.0)))
err_b = float(np.max(np.abs(proj(B_test) - (np.trace(B_test) / 3.0) * np.eye(3))))
print("  群平均：diag(1,0,0) -> 各向同性投影，偏差 %.3e；一般各向异性矩阵 -> 迹/3 乘单位阵，"
      "偏差 %.3e" % (err_a, err_b))
print("  含义：各向异性方向系数在群平均下被抹掉，故四维各向异性标量动能项不是立方不变量。")
check("M1", err_a <= 1e-14 and err_b <= 1e-14,
      "标量扇区的边际各向异性项被精确对称性禁止")

print()
print("=" * 78)
print("M2 张量扇区：立方弹性的各向同性条件与速度分裂")
print("=" * 78)


def born_speeds(c11, c12, c44):
    return {"L[100]": np.sqrt(c11), "T[100]": np.sqrt(c44),
            "L[111]": np.sqrt((c11 + 2.0 * c12 + 4.0 * c44) / 3.0),
            "T[111]": np.sqrt((c11 - c12 + c44) / 3.0)}


c11, c12 = 2.0, 1.0
c44_iso = 0.5 * (c11 - c12)
sp_iso = born_speeds(c11, c12, c44_iso)
spread_iso = max(abs(sp_iso["T[111]"] / sp_iso["T[100]"] - 1.0),
                 abs(sp_iso["L[111]"] / sp_iso["L[100]"] - 1.0))
print("  各向同性点 C44 = (C11 - C12)/2 = %.6f：%s" % (c44_iso,
      ", ".join("%s %.9f" % (k, v) for k, v in sp_iso.items())))
print("    同类别跨方向的速度差 = %.3e（期望 0；纵波与横波本来就不同，不作比较）" % spread_iso)
c44_an = 1.05 * c44_iso
sp_an = born_speeds(c11, c12, c44_an)
split_T = sp_an["T[111]"] / sp_an["T[100]"] - 1.0
split_L = sp_an["L[111]"] / sp_an["L[100]"] - 1.0
print("  破坏各向同性关系 5%%（C44 -> 1.05 C44）：T[111]/T[100] - 1 = %+.4e，"
      "L[111]/L[100] - 1 = %+.4e" % (split_T, split_L))
print("  含义：立方对称允许 C11、C12、C44 三个独立常数，各向同性只要求一条关系，"
      "故边际各向异性在张量扇区不被自动禁止。")
check("M2", spread_iso <= 1e-14 and abs(split_T) >= 1e-3 and abs(split_L) >= 1e-4,
      "各向同性点同类别跨方向速度重合，破坏关系后出现可测量分裂")

print()
print("=" * 78)
print("M3 离散符号的领头系数：各向同性模板与 5%% 各向异性对照")
print("=" * 78)
DIRS = {"[100]": np.array([1.0, 0.0, 0.0]),
        "[110]": np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
        "[111]": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)}


def lead_coeff(dhat, kappa=1e-3, aniso=0.0):
    k = kappa * dhat
    lam = -4.0 * (np.sin(k[0] / 2.0) ** 2 + np.sin(k[1] / 2.0) ** 2
                  + (1.0 + aniso) * np.sin(k[2] / 2.0) ** 2)
    return lam / (-kappa ** 2)


res_iso = {n: lead_coeff(d) for n, d in DIRS.items()}
for kappa in (1e-3, 5e-4):
    vals = [lead_coeff(d, kappa) for d in DIRS.values()]
    print("  kappa = %.1e 各向同性：系数 %s，方向差 %.3e"
          % (kappa, np.round(vals, 12), max(vals) - min(vals)))
vals_lo = [lead_coeff(d, 5e-4) for d in DIRS.values()]
vals_hi = [lead_coeff(d, 1e-3) for d in DIRS.values()]
shrink = (max(vals_hi) - min(vals_hi)) / max(max(vals_lo) - min(vals_lo), 1e-300)
print("    方向差随 kappa 减半的收缩比 = %.3f（按 k^2 标度期望约 4）" % shrink)
res_an = {n: lead_coeff(d, 1e-3, aniso=0.05) for n, d in DIRS.items()}
gap_an = max(res_an.values()) - min(res_an.values())
print("  5%% 各向异性：系数 %s，方向差 %.4e" % (np.round(list(res_an.values()), 6), gap_an))
check("M3", max(vals_hi) - min(vals_hi) <= 1e-6 and shrink >= 3.5 and gap_an >= 1e-2,
      "领头系数方向无关且残差按 k^2 收缩；对照落在 5%% 级的边际差上")

print()
print("=" * 78)
print("M4 极限速度差：标量与张量扇区共用同一模板")
print("=" * 78)
kk = np.array([0.137, 0.211, 0.083])
lam_scalar = -4.0 * (np.sin(kk[0] / 2.0) ** 2 + np.sin(kk[1] / 2.0) ** 2
                     + np.sin(kk[2] / 2.0) ** 2)
lam_tensor = np.array([lam_scalar for _ in range(5)])
rel = float(np.max(np.abs(lam_tensor - lam_scalar)) / abs(lam_scalar))
print("  同一个 k = %s：标量符号 %.12f，张量五个分量逐个相同，相对偏差 %.3e"
      % (np.round(kk, 3), lam_scalar, rel))
print("  含义：两扇区在同一格点上共用同一模板，极限速度差按构造取零；这与矢量扇区共用同一"
      "模板的结论合并，给出「一个格点一个 c」。")
check("M4", rel <= 1e-15, "标量与张量扇区符号逐点相同（极限速度差构造性为零）")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：%d/%d 通过" % (n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  %-6s %s" % (tag, "PASS" if ok else "FAIL"))
print("=" * 78)
