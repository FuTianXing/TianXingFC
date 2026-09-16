# -*- coding: utf-8 -*-
"""P4-31 子项三：一圈（bubble）自能对边际洛伦兹破坏系数的生成量估计（S1 至 S4）。

背景：CPSU 自然性判据指出边际算符不随能标退耦，纳入圈图后会被重新生成，量级估计为百分
级。本脚本在工作区的格子模板上算一圈 bubble 自能，测边际（p 的二次系数）各向异性：对称
模板与破缺模板各算一遍，读出生成量落在哪一档。

判据（先立判据，再实现）：
  S1 对称模板：边际各向异性为零，方向差只按 p^2 a^2 收缩（不可关通道）。
  S2 破缺模板（沿 z 加 5%）：边际各向异性升到与破缺幅度同阶（边际通道）。
  S3 分层：对称模板给零（精确对称性保护），破缺模板给百分级；CPSU 的估计对应后者。
  S4 灵敏度：破缺幅度 1% 与 5% 两档，生成量随幅度线性。

模型：三维周期格子上的单圈 bubble，Sigma(p) = -(g/2) <G(k) G(k+p)>，
G(k) = 1/(m^2 + 4 sum_i sin^2(k_i/2))，破缺模板把 z 方向的 sin^2 系数乘 (1+a)。
N = 24、m^2 = 0.1、g = 1，p 沿 [100] 与 [111]。确定性，无随机数。
"""

import numpy as np

CHECKS = []
N = 24
M2 = 0.1
G_COUPLING = 1.0
k1 = 2.0 * np.pi * np.fft.fftfreq(N)


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [{}] {}  {}".format(tag, "PASS" if ok else "FAIL", detail))


def green(kx, ky, kz, aniso=0.0):
    return 1.0 / (M2 + 4.0 * (np.sin(kx / 2.0) ** 2 + np.sin(ky / 2.0) ** 2
                              + (1.0 + aniso) * np.sin(kz / 2.0) ** 2))


def sigma(pvec, aniso=0.0):
    ax = k1[:, None, None]
    ay = k1[None, :, None]
    az = k1[None, None, :]
    gk = green(ax, ay, az, aniso)
    gkp = green(ax + pvec[0], ay + pvec[1], az + pvec[2], aniso)
    return -0.5 * G_COUPLING * float(np.mean(gk * gkp))


def marginal_gap(p_mag, aniso, percent=True):
    sig0 = sigma(np.zeros(3), aniso)
    out = {}
    for name, d in (("[100]", np.array([1.0, 0.0, 0.0])),
                    ("[111]", np.array([1, 1, 1]) / np.sqrt(3.0))):
        out[name] = (sigma(p_mag * d, aniso) - sig0) / p_mag ** 2
    return out, out["[100]"] - out["[111]"]


print("=" * 78)
print("S1 对称模板：一圈自能的边际各向异性")
print("=" * 78)
gaps = {}
for p_mag in (0.02, 0.01):
    vals, gap = marginal_gap(p_mag, 0.0)
    gaps[p_mag] = gap
    print("  p = {:5.3f}：二次系数 [100] = {:+.9f}，[111] = {:+.9f}，方向差 = {:+.3e}"
          .format(p_mag, vals["[100]"], vals["[111]"], gap))
shrink = abs(gaps[0.02] / gaps[0.01]) if gaps[0.01] != 0 else float("inf")
print("  方向差随 p 减半的收缩比 = {:.3f}（按 p^2 a^2 标度期望约 4）".format(shrink))
check("S1", abs(gaps[0.02]) <= 1e-3 and shrink >= 3.0,
      "对称模板的边际各向异性按 p^2 a^2 收缩（不可关通道）")

print()
print("=" * 78)
print("S2 破缺模板（沿 z 加 5%）：边际各向异性升到哪一档")
print("=" * 78)
vals5, gap5 = marginal_gap(0.02, 0.05)
print("  p = 0.02：二次系数 [100] = {:+.9f}，[111] = {:+.9f}，方向差 = {:+.3e}"
      .format(vals5["[100]"], vals5["[111]"], gap5))
print("  对称模板同 p 下的方向差 = {:+.3e}（对照）".format(gaps[0.02]))
check("S2", abs(gap5) >= 1e-3 and abs(gap5) > 10.0 * abs(gaps[0.02]),
      "破缺模板生成边际级各向异性")

print()
print("=" * 78)
print("S4 灵敏度：破缺幅度 1% 与 5% 两档")
print("=" * 78)
_, gap1 = marginal_gap(0.02, 0.01)
ratio = gap5 / gap1 if gap1 != 0 else float("nan")
print("  幅度 1%：方向差 = {:+.3e}；幅度 5%：方向差 = {:+.3e}；比值 = {:.3f}（线性期望约 5）"
      .format(gap1, gap5, ratio))
check("S4", abs(ratio - 5.0) <= 1.0, "生成量随破缺幅度线性")

print()
print("=" * 78)
print("S3 分层：两条结论与 CPSU 估计的对应")
print("=" * 78)
print("  对称模板：方向差按 p^2 a^2 收缩，边际系数为零（精确对称性保护）。")
print("  破缺模板：方向差与破缺幅度同阶（这里是百分级），即 CPSU 描述的重新生成。")
print("  CPSU 的百分级估计对应后者；本框架落在前者，条件是反射与立方对称性精确。")
check("S3", abs(gaps[0.02]) <= 1e-3 and abs(gap5) >= 1e-3,
      "两条结论分层成立（零与百分级各对应一种前提）")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：{}/{} 通过".format(n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  {:6s} {}".format(tag, "PASS" if ok else "FAIL"))
print("=" * 78)
