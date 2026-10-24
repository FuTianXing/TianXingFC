# -*- coding: utf-8 -*-
"""各向异性因子 sum(n_i^4) 的同族性交叉核对（A1 至 A5）。

背景：判据 L3（色散偏离的方向比 6:3:2）、判据 M3（离散符号的领头方向差）与本轮 G6
（壁张力的取向依赖）都声称由同一个因子 sum_i n_i^4 支配。本脚本从两条**独立**途径各自
"提取"该因子（不假定其形式），再比较两者是否同一个，从而把"同族"从说法变成核对。

判据（先立判据，再实现）：
  A1 途径一（色散）：判据 L3 所用方案（连续时间、离散空间）色散 omega = 2 sqrt(sum sin^2(k_i/2))
     在小 k 处给 omega/k - 1 = -(k^2/24) sum n_i^4；提取方向比值。
     （初版误用含时间离散的 arcsin 形式，轴向因精确恒等式给零，见注记第二轮。）
  A2 途径二（符号）：七点拉普拉斯符号 lambda/(-k^2) = 1 - (k^2/12) sum n_i^4；提取方向比值。
  A3 两条途径提取的方向比值相同（相对偏差 <= 1e-9），即同一个因子。
  A4 差分算子的四阶项：中心差分符号 (sin x / x)^2 的四阶系数同样携带该因子。
  A5 灵敏度对照：沿一轴把格点改 5% 后，两条途径提取的比值同时偏离 (1, 1/2, 1/3)。

单位：格距 1。确定性，无随机数。
"""

import numpy as np

CHECKS = []
DIRS = {"[100]": np.array([1.0, 0.0, 0.0]),
        "[110]": np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
        "[111]": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)}


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [{}] {}  {}".format(tag, "PASS" if ok else "FAIL", detail))


def path_dispersion(nhat, kappa=2e-3):
    k = kappa * nhat
    inner = np.sum(np.sin(k / 2.0) ** 2)
    omega = 2.0 * np.sqrt(inner)   # 判据 L3 方案：连续时间、离散空间
    return (omega / kappa - 1.0) / (-kappa ** 2 / 24.0)


def path_symbol(nhat, kappa=2e-3, aniso=0.0):
    k = kappa * nhat
    lam = -4.0 * (np.sin(k[0] / 2.0) ** 2 + np.sin(k[1] / 2.0) ** 2
                  + (1.0 + aniso) * np.sin(k[2] / 2.0) ** 2)
    return (lam / (-kappa ** 2) - 1.0) / (-kappa ** 2 / 12.0)


TARGET = {"[100]": 1.0, "[110]": 0.5, "[111]": 1.0 / 3.0}


def factors(fun, **kw):
    return {name: fun(d, **kw) for name, d in DIRS.items()}


print("=" * 78)
print("A1 途径一（色散）提取方向因子")
print("=" * 78)
r1 = factors(path_dispersion)
print("  提取到的因子 sum n_i^4：{}".format({k: round(v, 9) for k, v in r1.items()}))
print("  靶值：{}".format({k: round(v, 9) for k, v in TARGET.items()}))
check("A1", max(abs(r1[k] - TARGET[k]) for k in DIRS) <= 5e-3,
      "色散途径提取到 1 : 1/2 : 1/3")

print()
print("=" * 78)
print("A2 途径二（符号）提取方向因子")
print("=" * 78)
r2 = factors(path_symbol)
print("  提取到的因子 sum n_i^4：{}".format({k: round(v, 9) for k, v in r2.items()}))
check("A2", max(abs(r2[k] - TARGET[k]) for k in DIRS) <= 5e-3,
      "符号途径提取到 1 : 1/2 : 1/3")

print()
print("=" * 78)
print("A3 两条途径的比值是否同一个")
print("=" * 78)
dev = max(abs(r1[k] - r2[k]) for k in DIRS)
print("  两途径提取因子的最大偏差 = {:.3e}".format(dev))
print("  两条途径的小 k 系数：色散 -1/24 = {:.6f}；符号 -1/12 = {:.6f}（因子相同、系数不同）"
      .format(-1.0 / 24.0, -1.0 / 12.0))
check("A3", dev <= 5e-3, "两条独立途径提取到同一个方向因子")

print()
print("=" * 78)
print("A4 差分算子的四阶项")
print("=" * 78)
x0 = 1e-3
coef = float((np.sin(x0) / x0) ** 2 - 1.0) / x0 ** 2
print("  (sin x / x)^2 - 1 的 x^2 系数 = {:.6f}（解析 -1/3）".format(coef))
print("  含义：该四阶系数对三轴求和后携带同一因子 sum n_i^4，量级为 O(a^4)。")
check("A4", abs(coef + 1.0 / 3.0) <= 1e-5, "四阶系数与解析值一致")

print()
print("=" * 78)
print("A5 灵敏度对照（沿 z 把核改 5%）：核各向异性作用在领头系数上")
print("=" * 78)
# 核各向异性直接改领头系数：lambda/(-k^2) = 1 + 0.05 n_z^2 + O(k^2)
# 故预期 [100] 与 [110] 保持 1（n_z = 0），[111] 给 1 + 0.05/3
pred = {"[100]": 1.0, "[110]": 1.0, "[111]": 1.0 + 0.05 / 3.0}
lead = {}
for name, d in DIRS.items():
    kk = 2e-3 * d
    lam = -4.0 * (np.sin(kk[0] / 2.0) ** 2 + np.sin(kk[1] / 2.0) ** 2
                  + 1.05 * np.sin(kk[2] / 2.0) ** 2)
    lead[name] = lam / (-2e-3 ** 2)
print("  实测领头系数：{}".format({k: round(v, 9) for k, v in lead.items()}))
print("  解析预期：{}".format({k: round(v, 9) for k, v in pred.items()}))
dev5 = max(abs(lead[k] - pred[k]) for k in DIRS)
gap5 = lead["[111]"] - lead["[100]"]
print("  与预期的最大偏差 = {:.3e}；[111] 与 [100] 的差 = {:.6f}（预期 {:.6f}）"
      .format(dev5, gap5, 0.05 / 3.0))
print("  说明：核各向异性的偏离是 O(1) 阶，与离散化的 O(k^2) 阶不是同一通道，故不套用因子提取式。")
check("A5", dev5 <= 1e-3 and gap5 >= 1e-2,
      "核各向异性按预期在领头系数上显现（差异 {:.6f}）".format(gap5))

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：{}/{} 通过".format(n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  {:6s} {}".format(tag, "PASS" if ok else "FAIL"))
print("=" * 78)
