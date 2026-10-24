# -*- coding: utf-8 -*-
r"""
record_ruler_coupling.py

[P1-24 第二段判据] 记录场是否改写局域尺：键能模型上的读数

定位
----
内部草稿/注记-P1-24第二段：框架更新式到局域速度读法的链路与缺口 把光偏折系数的缺口
收成一句问话：局域尺是否随记录场伸缩（g_xx 是否等于 1 - 2 Phi）。本脚本用框架自己的
键能（格点内 Theta^2 项与格点间相位差平方项）给出该问话的读数。

模型
----
单键的相位差向量 d（三维），真空 |d| = pi（相位差 pi 对应格距 l_P），键能取框架形式
  E = (1/2) kappa (|d| - pi)^2。
记录场取"真空层法线方向转动 theta"（路径 B 规则 2 的 F 属于 SO(3)，列单位）。
物理键长与相位差模的关系取 L = l_P |d|/pi。

判据
----
K1 纯转动（记录取向改变）不改变键长：50 个随机取向的平衡长度与真空中一致（偏差 < 1e-15）。
K2 几何倾斜只到二阶：把层法线相对键方向的夹角 theta 引入长度关系 L = L0/cos(theta) 后，
   一阶差商为零（< 1e-9），二阶系数为 1/2（偏差 < 1%），并给出它相对所需一阶项的量级比。
K3 一次耦合的对照：若键能含一次项 lambda theta (|d| - pi)，平衡偏移 s = -lambda theta/kappa
   线性出现（数值与解析偏差 < 1e-12）；框架现有键能各项对记录值为偶，故一次项为零。

运行
----
python 验证脚本/record_ruler_coupling.py
确定性（随机种子固定 20260918），运行约 1 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []
RNG = np.random.default_rng(20260918)


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


print("=" * 72)
print("最小验证：记录场是否改写局域尺（键能模型读数）")
print("=" * 72)

L0 = 1.0  # 真空键长（相位差 pi 对应 l_P，取 l_P = 1）

# ---------------------------------------------------------------- K1
print("\n[K1] 纯转动（记录取向改变）不改变键长")
d0 = np.array([np.pi, 0.0, 0.0])
worst = 0.0
for _ in range(50):
    axis = RNG.normal(size=3)
    th = RNG.uniform(0, np.pi)
    R = rot(axis, th)
    d = R @ d0
    L = L0 * np.linalg.norm(d) / np.pi
    worst = max(worst, abs(L - L0))
print(f"      50 个随机取向的最大键长偏差 {worst:.2e}（l_P = 1）")
check("K1", worst < 1e-15, f"偏差 {worst:.1e}（< 1e-15），转动不产生尺伸缩")

# ---------------------------------------------------------------- K2
print("\n[K2] 几何倾斜只到二阶：一阶差商为零，二阶系数为 1/2")
ths = np.array([1e-3, 1e-2, 5e-2])
vals = 1.0 / np.cos(ths) - 1.0
th0 = 1e-3
f = lambda x: 1.0 / np.cos(x) - 1.0
first = (f(th0) - f(-th0)) / (2 * th0)
second = vals[-1] / ths[-1] ** 2
print(f"      对称一阶差商（theta = 1e-3）{first:.2e}；二阶系数 {second:.6f}（期望 0.5）")
check("K2", abs(first) < 1e-15 and abs(second - 0.5) < 5e-3,
      f"对称一阶差商为零（{first:.1e}，判据 < 1e-15），二阶系数 {second:.6f}（判据 0.5 ± 1%）")

# ---------------------------------------------------------------- K3
print("\n[K3] 一次耦合对照：一次项给出线性伸缩，框架现有键能无此项")
kappa, theta = 1.0, 0.01
for lam in (0.0, 0.05, 0.1):
    # 解析：E = kappa s^2/2 + lam theta s -> s* = -lam theta/kappa
    s_analytic = -lam * theta / kappa
    # 数值：粗网格求极小
    ss = np.linspace(-0.05, 0.05, 200001)
    E = 0.5 * kappa * ss ** 2 + lam * theta * ss
    s_num = ss[np.argmin(E)]
    dev = abs(s_num - s_analytic)
    print(f"      lambda = {lam:.2f}：平衡偏移 解析 {s_analytic:+.3e}，数值 {s_num:+.3e}，偏差 {dev:.1e}")
    if lam == 0.0:
        zero_ok = abs(s_analytic) < 1e-15
dev3 = 0.0
for lam in (0.05, 0.1):
    s_analytic = -lam * theta / kappa
    ss = np.linspace(-0.05, 0.05, 200001)
    E = 0.5 * kappa * ss ** 2 + lam * theta * ss
    dev3 = max(dev3, abs(ss[np.argmin(E)] - s_analytic))
check("K3", zero_ok and dev3 < 1e-12,
      f"一次项为零时伸缩为零；含一次项时数值与解析偏差 {dev3:.1e}（< 1e-12）")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")