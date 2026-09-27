# -*- coding: utf-8 -*-
"""多观测者全局协议收尾（第 472 条）：闭环 holonomy、双曲面积与信号层修正。

背景：P11 遗留 (c)（非局域协议的洛伦兹结构，含同时性定义严格化）；[[狭义相对论的定理化：从晶格几何到洛伦兹结构]] 遗留 (b)。
定义基础：同时性 = 振荡同步（同相）；爱因斯坦同步协议（第 470 条）；洛伦兹群晶格表达（第 471 条）。

判据：
  OG1 同步链（共线）：共线 boost 的环积为单位（机器精度）；相对速度与精确合成公式一致。
  OG2 闭环 holonomy：三点环（非共线）的传输积为纯转动；转动角 = 速度三角形双曲面积
      （Gauss-Bonnet：面积 = (n-2)pi - 角度和）；含静止观测者时与第 471 条 Wigner 闭式一致。
  OG3 协议一致性：整体 boost（共轭）下角度不变；反序遍历给逆转动；四边形分解满足
      传输因式分解与面积可加。
  OG4 信号层修正：有效信号速度 c_eff = cos(k l_P/2)（晶格色散群速度，第 471 条）；
      由信号推断的 holonomy 修正按 k^2 标度并随 k 消失。

确定性；约数秒。图存 images/multi_observer_protocol.png
"""
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "multi_observer_protocol.png")
T0 = time.time()
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
ETA = np.diag([1.0, -1.0, -1.0, -1.0])


def check(tag, ok, detail):
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))
    return bool(ok)


def gamma(v):
    return 1.0 / math.sqrt(1.0 - float(v @ v))


def boost(v):
    v = np.asarray(v, float)
    b2 = float(v @ v)
    if b2 < 1e-30:
        return np.eye(4)
    g = gamma(v)
    B = np.zeros((4, 4))
    B[0, 0] = g
    B[0, 1:] = g * v
    B[1:, 0] = g * v
    B[1:, 1:] = np.eye(3) + (g - 1.0) * np.outer(v, v) / b2
    return B


def u4(v):
    g = gamma(v)
    return np.array([g, *(g * np.asarray(v, float))])


def rel_velocity(a, b):
    w = np.linalg.inv(boost(a)) @ u4(b)
    return w[1:] / w[0]


def pure_boost(a, b):
    return boost(a) @ boost(rel_velocity(a, b)) @ np.linalg.inv(boost(a))


def loop_product(vs):
    H = np.eye(4)
    for i in range(len(vs)):
        H = pure_boost(vs[i], vs[(i + 1) % len(vs)]) @ H
    return H


def in_frame(H, v0):
    return np.linalg.inv(boost(v0)) @ H @ boost(v0)


def rot_defect(R):
    return max(abs(R[0, 0] - 1.0), float(np.max(np.abs(R[0, 1:]))), float(np.max(np.abs(R[1:, 0]))))


def rot_angle(R):
    tr = float(np.trace(R[1:, 1:]))
    return math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))


def rel_rap(a, b):
    w = rel_velocity(a, b)
    return math.acosh(1.0 / math.sqrt(1.0 - float(w @ w)))


def polygon_area(vs):
    n = len(vs)
    us = [u4(v) for v in vs]
    total = 0.0
    for i in range(n):
        a, b, c = us[(i - 1) % n], us[i], us[(i + 1) % n]
        t1 = a - float(a @ (ETA @ b)) * b
        t2 = c - float(c @ (ETA @ b)) * b
        n1 = math.sqrt(abs(float(t1 @ (ETA @ t1))))
        n2 = math.sqrt(abs(float(t2 @ (ETA @ t2))))
        cosang = -float(t1 @ (ETA @ t2)) / (n1 * n2)   # 诱导度规 = -eta 的限制
        total += math.acos(max(-1.0, min(1.0, cosang)))
    return (n - 2) * math.pi - total


print("第 472 条 多观测者全局协议：OG1 至 OG4")

# ---------------------------------------------------------------- OG1
print("OG1 同步链（共线）")
vs_col = [np.array([0.0, 0.0, 0.0]), 0.3 * X, 0.6 * X]
H = loop_product(vs_col)
dev_col = rot_defect(H)
Hr = loop_product(list(reversed(vs_col)))
dev_col_r = rot_defect(Hr)
check("OG1a", max(dev_col, dev_col_r) < 1e-13,
      "共线环积 = 单位（正序与反序）：最大缺陷 %.2e" % max(dev_col, dev_col_r))

va, vb, vc = 0.4 * X, 0.7 * X, 0.9 * X
u_ab = rel_velocity(va, vb)[0]
u_bc = rel_velocity(vb, vc)[0]
u_comp = (u_ab + u_bc) / (1.0 + u_ab * u_bc)
dev_comp = abs(u_comp - rel_velocity(va, vc)[0])
check("OG1b", dev_comp < 1e-13,
      "相对速度合成与精确公式一致：偏差 %.2e" % dev_comp)

# ---------------------------------------------------------------- OG2
print("OG2 闭环 holonomy（非共线）")
triangles = [
    [0.3 * X, 0.4 * Y, np.array([-0.25, -0.25, 0.0])],
    [0.2 * X, 0.5 * Y, np.array([-0.3, -0.3, 0.0])],
    [0.45 * X, np.array([-0.2, 0.5, 0.0]), np.array([0.1, -0.35, 0.2])],
    [np.array([0.25, 0.15, -0.1]), 0.55 * Y, np.array([-0.3, 0.1, 0.25])],
    [np.array([0.0, 0.0, 0.0]), 0.5 * X, 0.6 * Y],
]
defects, gaps = [], []
for vs in triangles:
    H = loop_product(vs)
    R = in_frame(H, vs[0])
    defects.append(rot_defect(R))
    om = rot_angle(R)
    gaps.append(abs(om - polygon_area(vs)))
check("OG2a", max(defects) < 1e-13,
      "非共线环积为纯转动：最大缺陷 %.2e（5 组）" % max(defects))
check("OG2b", max(gaps) < 1e-12,
      "转动角 = 双曲面积（Gauss-Bonnet）：最大偏差 %.2e" % max(gaps))

vs_rest = [np.array([0.0, 0.0, 0.0]), 0.3 * X, 0.4 * Y]
H = loop_product(vs_rest)
om = rot_angle(in_frame(H, vs_rest[0]))
a_, b_ = math.atanh(0.3), math.atanh(0.4)
formula = math.atan2(math.sinh(a_) * math.sinh(b_), math.cosh(a_) + math.cosh(b_))
check("OG2c", abs(om - formula) < 1e-14,
      "含静止观测者时与 Wigner 闭式一致：偏差 %.2e" % abs(om - formula))

# ---------------------------------------------------------------- OG3
print("OG3 协议一致性")
w = np.array([0.25, -0.15, 0.1])
def conj(vs):
    out = []
    for v in vs:
        x = boost(w) @ u4(v)
        out.append(x[1:] / x[0])
    return out
base = triangles[1]
o1 = rot_angle(in_frame(loop_product(base), base[0]))
b2 = conj(base)
o2 = rot_angle(in_frame(loop_product(b2), b2[0]))
check("OG3a", abs(o1 - o2) < 1e-12 and abs(polygon_area(base) - polygon_area(b2)) < 1e-12,
      "整体 boost（共轭）下角度与面积不变：偏差 %.2e" % max(abs(o1 - o2), abs(polygon_area(base) - polygon_area(b2))))

Hb = loop_product(base)
Hrev = loop_product([base[0], base[2], base[1]])   # 反序但保留起点，方为逆路径
dev_rev = float(np.max(np.abs(Hrev @ Hb - np.eye(4))))
o_rev = rot_angle(in_frame(Hrev, base[0]))
check("OG3b", dev_rev < 1e-13 and abs(o_rev - o1) < 1e-13,
      "反序遍历给逆转动：复合残差 %.2e，角度偏差 %.2e" % (dev_rev, abs(o_rev - o1)))

q = [np.array([0.3, 0.0, 0.0]), np.array([0.0, 0.45, 0.0]),
     np.array([-0.3, 0.0, 0.0]), np.array([0.0, -0.35, 0.0])]   # 共面（z = 0），面积公式适用
T12, T23, T34, T41 = (pure_boost(q[0], q[1]), pure_boost(q[1], q[2]),
                      pure_boost(q[2], q[3]), pure_boost(q[3], q[0]))
T13, T31 = pure_boost(q[0], q[2]), pure_boost(q[2], q[0])
outer = T41 @ T34 @ T23 @ T12
sub1 = T31 @ T23 @ T12
sub2 = T41 @ T34 @ T13
dev_fac = float(np.max(np.abs(outer - sub2 @ sub1)))
dev_inv = float(np.max(np.abs(T13 @ T31 - np.eye(4))))
a_q = polygon_area(q)
a_s = polygon_area([q[0], q[1], q[2]]) + polygon_area([q[0], q[2], q[3]])
check("OG3c", dev_fac < 1e-13 and dev_inv < 1e-13 and abs(a_q - a_s) < 1e-12,
      "四边形分解：因式分解 %.2e、边逆 %.2e、面积可加 %.2e" % (dev_fac, dev_inv, abs(a_q - a_s)))

# ---------------------------------------------------------------- OG4
print("OG4 信号层修正（有效信号速度 c_eff）")
base = triangles[1]
om0 = rot_angle(in_frame(loop_product(base), base[0]))
ks = np.array([0.1, 0.2, 0.4])
deltas = []
for k in ks:
    c_eff = math.cos(k / 2.0)
    vs2 = [v / c_eff for v in base]
    om2 = rot_angle(in_frame(loop_product(vs2), vs2[0]))
    deltas.append(om2 - om0)
deltas = np.array(deltas)
ratios = [deltas[1] / deltas[0], deltas[2] / deltas[1]]
check("OG4a", all(3.5 < r < 4.5 for r in ratios),
      "推断 holonomy 修正按 k^2 标度：减半比值 %.3f 与 %.3f" % (ratios[0], ratios[1]))
k_small = 0.01
c_small = math.cos(k_small / 2.0)
vs_small = [v / c_small for v in base]
om_small = rot_angle(in_frame(loop_product(vs_small), vs_small[0]))
check("OG4b", (om_small - om0) < 1e-4 * (deltas[0] / (0.1 ** 2)) and (om_small - om0) > 0,
      "连续极限恢复：k = 0.01 时修正 %.2e（向零）" % (om_small - om0))

# ---------------------------------------------------------------- 图
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
ax = axs[0, 0]
for i, v in enumerate(base):
    ax.arrow(0, 0, v[0], v[1], head_width=0.02, color="#2b6cb0", length_includes_head=True)
    ax.annotate("O%d" % (i + 1), (v[0], v[1]), fontsize=9)
tri = np.array([v[:2] for v in base] + [base[0][:2]])
ax.plot(tri[:, 0], tri[:, 1], "--", color="#555")
ax.set_aspect("equal")
ax.set_xlabel("v_x")
ax.set_ylabel("v_y")
ax.set_title("OG2 速度空间三角形（双曲面积 = holonomy）", fontsize=10)

ax = axs[0, 1]
oms, areas = [], []
for vs in triangles:
    R = in_frame(loop_product(vs), vs[0])
    oms.append(rot_angle(R))
    areas.append(polygon_area(vs))
ax.plot(areas, oms, "o", color="#2b6cb0")
lim = max(max(areas), max(oms)) * 1.1
ax.plot([0, lim], [0, lim], "--", color="gray", lw=0.9, label="y = x")
ax.set_xlabel("双曲面积（Gauss-Bonnet）")
ax.set_ylabel("holonomy 角")
ax.set_title("OG2b 角 = 面积（5 组）", fontsize=10)
ax.legend(fontsize=8)

ax = axs[1, 0]
labels = ["OG1a", "OG2a", "OG3a", "OG3b", "OG3c"]
vals = [max(dev_col, dev_col_r, 1e-17), max(max(defects), 1e-17),
        max(abs(o1 - o2), abs(polygon_area(base) - polygon_area(b2)), 1e-17),
        max(dev_rev, abs(o_rev - o1), 1e-17), max(dev_fac, dev_inv, abs(a_q - a_s), 1e-17)]
ax.bar(range(len(vals)), vals, color="#2b6cb0")
ax.set_yscale("log")
ax.set_ylim(1e-17, 1e-10)
ax.axhline(1e-12, color="gray", ls="--", lw=0.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("OG1 至 OG3 一致性残差（机器精度）", fontsize=10)

ax = axs[1, 1]
ax.loglog(ks, deltas, "o-", color="#2b6cb0", label="信号层推断修正")
ax.loglog(ks, deltas[0] * (ks / ks[0]) ** 2, "--", color="gray", lw=0.9, label="斜率 2 参照")
ax.set_xlabel("信号波数 k（l_P = 1）")
ax.set_ylabel("ΔΩ")
ax.set_title("OG4 修正按 k^2（c_eff 群速度）", fontsize=10)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存：%s（用时 %.1f s）" % (IMG, time.time() - T0))
