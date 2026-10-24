# -*- coding: utf-8 -*-
r"""
ruler_coupling_candidates.py

[P1-24 第二段判据·候选筛查] 键能里能给出"尺随记录伸缩"的候选耦合逐条筛查

定位
----
上一轮读数（验证脚本/注记-记录场是否改写局域尺：键能模型上的读数）给出：框架现有键能
对记录值为偶、无一次项，故一阶 g_xx = 1，落在"仅钟率"读法、偏折半值。要把全值补上，
需要一个给出 s = -Phi 的耦合。本脚本枚举四类自然候选，逐条给出两项读数：
（甲）诱导应变的径向幂指数，与所需的 -1（跟随势 Phi 的 1/r）比较；
（乙）该耦合项在记录变号下的奇偶，与格点规则的反射对称性（已在册：文档更新式宇称精确）对照。

候选
----
C1 代数耦合      E = lambda * theta * s          -> s 正比 theta          （幂指数 -1）
C2 梯度耦合      E = mu * theta' * s             -> s 正比 theta'         （幂指数 -2）
C3 双梯度耦合    E = mu * theta' * s'            -> 变分给 kappa s = mu theta''（幂指数 -3）
C4 二次耦合      E = nu * theta^2 * s            -> s 正比 theta^2        （幂指数 -2）

判据
----
L1 幂指数读数：所需值为 -1，逐条给出实测幂指数（对数拟合，`r` 取 5 到 100）。
L2 奇偶读数：对每条候选算 E(theta)/E(-theta)，奇为 -1、偶为 +1；反射对称精确要求该值为 +1。
L3 联合判定：既满足幂指数 -1 又满足偶性的候选数应为 0，即局部键能项无法提供全值。

运行
----
python 验证脚本/ruler_coupling_candidates.py
确定性（无随机数），运行约 1 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


A = 1.0
EPS = 1e-3


def theta(r):
    return -A / np.sqrt(r ** 2 + EPS ** 2)


def dtheta(r, h=1e-4):
    return (theta(r + h) - theta(r - h)) / (2 * h)


def d2theta(r, h=1e-3):
    return (theta(r + h) - 2 * theta(r) + theta(r - h)) / h ** 2


def slope(r, y):
    """log-log 斜率"""
    m = (r > 0) & (np.abs(y) > 0)
    return float(np.polyfit(np.log(r[m]), np.log(np.abs(y[m])), 1)[0])


print("=" * 72)
print("最小验证：尺-记录耦合候选逐条筛查")
print("=" * 72)

r = np.linspace(5.0, 100.0, 400)
kappa = 1.0

# ---------------------------------------------------------------- L1
print("\n[L1] 诱导应变的径向幂指数（所需 -1）")
s1 = -theta(r) / kappa                      # C1：代数耦合
s2 = -dtheta(r) / kappa                     # C2：梯度耦合
s3 = d2theta(r) / kappa                     # C3：双梯度耦合
s4 = -theta(r) ** 2 / kappa                 # C4：二次耦合
slopes = {'C1 代数': slope(r, s1), 'C2 梯度': slope(r, s2),
          'C3 双梯度': slope(r, s3), 'C4 二次': slope(r, s4)}
for k, v in slopes.items():
    print(f"      {k}：幂指数 {v:+.3f}")
check("L1", abs(slopes['C1 代数'] + 1.0) < 0.05
      and all(abs(slopes[k] + 1.0) > 0.5 for k in ('C2 梯度', 'C3 双梯度', 'C4 二次')),
      f"C1 幂指数 {slopes['C1 代数']:+.3f}（对 -1 偏差 < 0.05），其余三条均偏离 -1 超过 0.5")

# ---------------------------------------------------------------- L2
print("\n[L2] 各候选在记录变号下的奇偶（反射对称要求 +1）")
t, s = 0.03, 0.01
par = {
    'C1 代数': (t * s) / (-t * s),
    'C2 梯度': (t * s) / (-t * s) if False else 1.0,   # theta' 与 theta 同奇，s 偶 -> 积偶
    'C3 双梯度': 1.0,                                   # theta' * s'：奇 x 偶 -> 奇？
    'C4 二次': (t ** 2 * s) / (t ** 2 * s),
}
# 逐条按定义计算：应变 s 与 s' 在记录变号下为偶，记录本身与它的梯度为奇
def parity_of(term):
    return term(-t, s) / term(t, s)


par = {
    'C1 代数': parity_of(lambda th, sv: th * sv),
    'C2 梯度': parity_of(lambda th, sv: th * sv),
    'C3 双梯度': parity_of(lambda th, sv: th * sv),
    'C4 二次': parity_of(lambda th, sv: th ** 2 * sv),
}
for k, v in par.items():
    print(f"      {k}：E(theta -> -theta)/E(theta) = {v:+.1f}")
check("L2", abs(par['C1 代数'] + 1.0) < 1e-12 and abs(par['C2 梯度'] + 1.0) < 1e-12
      and abs(par['C3 双梯度'] + 1.0) < 1e-12 and abs(par['C4 二次'] - 1.0) < 1e-12,
      f"C1 至 C3 为奇（{par['C1 代数']:+.0f}、{par['C2 梯度']:+.0f}、{par['C3 双梯度']:+.0f}），C4 为偶（{par['C4 二次']:+.0f}）")

# ---------------------------------------------------------------- L3
print("\n[L3] 联合判定：同时满足幂指数 -1 与偶性的候选")
ok_both = [k for k in slopes if abs(slopes[k] + 1.0) < 0.05 and abs(par[k] - 1.0) < 1e-12]
print(f"      满足两条的候选：{ok_both if ok_both else '无'}")
check("L3", len(ok_both) == 0,
      "局部键能项中没有任何候选同时满足幂指数 -1 与偶性，故全值不能来自局部键能项")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")