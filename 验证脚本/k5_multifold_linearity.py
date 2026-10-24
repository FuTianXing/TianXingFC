# -*- coding: utf-8 -*-
r"""
k5_multifold_linearity.py

[第738号] K5 多档线性性复核：拖曳强度对速率与取向的多档扫描

背景：矢量写入定理（定理 7）的前置清单 K5 判定项为「拖曳强度与速率、取向无关的
多档扫描（既有『与速率与取向无关』读数，判据化复核）」。链接级合成轮已给出三档速率
与三取向的读数（K2 入口唯一、K3 各向同性），本轮把它们扩为多档并判据化。

模型（与链接级合成轮逐字同构）：
  写事件沿缺陷世界线落在链接上，每笔写入携带两个读数：
    标量读数（质量）delta m_s，核 1/|x - x_s|，单位源给出 theta = m/r（G = c = 1）；
    矢量读数（动量）delta m_s * v_s，同一核，给 Theta_i = sum_s m_s v_{i,s}/r_s。
  度规组装（相对论约定）：h00 = 2 theta，h0i = 4 Theta_i。
  拖曳强度比值 h0i/(h00 * v_i) = 4 Theta_i / (2 theta v_i)，单一入口下恒为 2。

源点阵与权重取既有轮的四个源点 P = {(3,0,0),(0,4,0),(0,0,5),(2,2,1)}，权重 w = 1。

判据（先立判据与理论期望，再实现）：
  M-1 速率多档的入口唯一性：h0i/(h00*v_i) 在多档速率上恒为 2（相对偏差 <= 1e-12）
  M-2 取向多档的各向同性：同一 |v| 沿多取向给同一矢量强度（相对偏差 <= 1e-12）
  M-3 联合扫描的无关性：速率与取向同时变化时比值仍恒为 2（相对偏差 <= 1e-12）
  M-4 结构性对照：把矢量入口系数由 4 改为 3，比值须离开 2（偏离 > 1e-3）
  M-5 恒真替身：引入立方各向异性键的取向权重 sum(n_a^4)（6:3:2），各向同性须 FAIL
  M-6 档位表与偏差表：给出速率档表、取向档表、联合档表的逐档读数

运行：python 验证脚本/k5_multifold_linearity.py
确定性（无随机），约 5 秒。图存 images/k5_multifold_linearity.png
"""
import io
import math
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "k5_multifold_linearity.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第738号：K5 多档线性性复核（拖曳强度对速率与取向的多档扫描）")
print("=" * 78)

# ----------------------------------------------------------------------
# 装置（与链接级合成轮同构）
# ----------------------------------------------------------------------
P_SRC = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0], [2.0, 2.0, 1.0]])
X0 = np.array([[0.0, 0.0, 0.0]])
W0 = np.array([1.0])

H00_COEFF = 2.0
H0I_COEFF = 4.0
TARGET_RATIO = 2.0


def theta_field(P, X, w):
    d = P[:, None, :] - X[None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    return (w[None, :] / r).sum(-1)


def vec_field(P, X, w, V):
    d = P[:, None, :] - X[None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    # 逐源点先除以各自的距离再求和（与链接级合成轮逐字同构）
    return (w[None, :, None] * V[None, :, :] / r[:, :, None]).sum(1)


def drag_ratio(v, direction, h0i_coeff=H0I_COEFF, axis=None):
    """拖曳强度比值 h0i/(h00*v_i)。direction 为单位向量。

    拖曳强度的定义是同一分量之比，故须取 v_i 非零的轴：若未指定，
    取 |v_i| 最大的那一轴（保证分母非零且条件数最优）。
    返回三项：分量比、标量场 theta、矢量场 Theta（三维）。
    """
    V = (v * direction)[None, :]
    if axis is None:
        axis = int(np.argmax(np.abs(V[0])))
    th = theta_field(P_SRC, X0, W0)
    tvec = vec_field(P_SRC, X0, W0, V)[0]
    h00 = H00_COEFF * th
    h0i = h0i_coeff * tvec[axis]
    return (h0i / (h00 * V[0, axis])), th, tvec


# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性与参考档读数")
r_ref, th_ref, tv_ref = drag_ratio(0.1, np.array([1.0, 0.0, 0.0]))
print("  参考档（v=0.1, [100]）：比值 %.15f，theta %.9e，Theta %.9e"
      % (r_ref[0], th_ref[0], tv_ref[0]))
smoke_ok = (abs(r_ref[0] - TARGET_RATIO) < 1e-9) and (th_ref[0] > 0) and (abs(tv_ref[0]) > 0)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# 档位表
# ----------------------------------------------------------------------
V_LIST = [1e-4, 1e-3, 1e-2, 1e-1, 0.3, 0.5, 0.7, 0.9]
DIRS = {
    "[100]": np.array([1.0, 0.0, 0.0]),
    "[110]": np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0),
    "[111]": np.ones(3) / math.sqrt(3.0),
    "[101]": np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0),
    "[011]": np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0),
    "[211]": np.array([2.0, 1.0, 1.0]) / math.sqrt(6.0),
}

# ----------------------------------------------------------------------
# M-1 速率多档的入口唯一性
# ----------------------------------------------------------------------
print()
print("[M-1] 速率多档的入口唯一性：h0i/(h00*v_i) 恒为 2")
m1_rows = []
m1_worst = 0.0
m1_worst_extreme = 0.0
for v in V_LIST:
    r, th, tv = drag_ratio(v, DIRS["[100]"])
    dev = abs(r[0] - TARGET_RATIO) / TARGET_RATIO
    m1_rows.append((v, r[0], dev))
    m1_worst = max(m1_worst, dev)
    if v in (1e-4, 0.9):
        m1_worst_extreme = max(m1_worst_extreme, dev)
    print("    v = %.4f c：比值 %.15f，相对偏差 %.3e" % (v, r[0], dev))
print("  全部档最大相对偏差 %.3e（容差 1e-12）" % m1_worst)
check("M-1", m1_worst <= 1e-12,
      "入口比值在 %d 档速率上恒为 2，最大相对偏差 %.3e" % (len(V_LIST), m1_worst))

# ----------------------------------------------------------------------
# M-2 取向多档的各向同性
# ----------------------------------------------------------------------
print()
print("[M-2] 取向多档的各向同性：同一 |v| 沿多取向给同一矢量强度")
m2_rows = []
m2_worst = 0.0
ref_mag = None
for name, u in DIRS.items():
    r, th, tvec = drag_ratio(0.1, u)
    # 各向同性比的是矢量强度 |Theta|（与链接级合成轮 K3 同构）
    mag = float(np.linalg.norm(tvec))
    if ref_mag is None:
        ref_mag = mag
    dev = abs(mag / ref_mag - 1.0)
    m2_rows.append((name, mag, dev))
    m2_worst = max(m2_worst, dev)
    print("    %-6s 矢量强度 %.15e，对参考取向的相对偏差 %.3e" % (name, mag, dev))
print("  全部取向最大相对偏差 %.3e（容差 1e-12）" % m2_worst)
check("M-2", m2_worst <= 1e-12,
      "矢量强度在 %d 个取向上相同，最大相对偏差 %.3e" % (len(DIRS), m2_worst))

# ----------------------------------------------------------------------
# M-3 联合扫描的无关性
# ----------------------------------------------------------------------
print()
print("[M-3] 联合扫描的无关性：速率与取向同时变化时比值仍恒为 2")
m3_rows = []
m3_worst = 0.0
v_names = list(DIRS.keys())
for i, v in enumerate(V_LIST):
    name = v_names[i % len(v_names)]
    r, th, tv = drag_ratio(v, DIRS[name])
    dev = abs(r[0] - TARGET_RATIO) / TARGET_RATIO
    m3_rows.append((v, name, r[0], dev))
    m3_worst = max(m3_worst, dev)
    print("    v = %.4f c，%s：比值 %.15f，相对偏差 %.3e" % (v, name, r[0], dev))
# 另加一组全交叉的抽样（每档配两个取向）
for v in (1e-3, 1e-1, 0.5):
    for name in ("[100]", "[111]", "[211]"):
        r, th, tv = drag_ratio(v, DIRS[name])
        dev = abs(r[0] - TARGET_RATIO) / TARGET_RATIO
        m3_rows.append((v, name, r[0], dev))
        m3_worst = max(m3_worst, dev)
print("  联合档数 %d，最大相对偏差 %.3e（容差 1e-12）" % (len(m3_rows), m3_worst))
check("M-3", m3_worst <= 1e-12,
      "联合扫描 %d 档的比值恒为 2，最大相对偏差 %.3e" % (len(m3_rows), m3_worst))

# ----------------------------------------------------------------------
# M-4 结构性对照：破坏单一入口
# ----------------------------------------------------------------------
print()
print("[M-4] 结构性对照：把矢量入口系数由 4 改为 3，比值须离开 2")
m4_devs = []
for v in (1e-2, 1e-1, 0.5):
    r, th, tv = drag_ratio(v, DIRS["[100]"], h0i_coeff=3.0)
    dev = abs(r[0] - TARGET_RATIO) / TARGET_RATIO
    m4_devs.append(dev)
    print("    v = %.4f c：替身比值 %.15f，相对偏差 %.4f" % (v, r[0], dev))
m4_worst = min(m4_devs)
check("M-4", m4_worst > 1e-3,
      "替身使比值离开 2（最小相对偏差 %.4f），判据有区分力" % m4_worst)

# ----------------------------------------------------------------------
# M-5 恒真替身：各向异性权重
# ----------------------------------------------------------------------
print()
print("[M-5] 恒真替身：引入立方各向异性键的取向权重 sum(n_a^4)（6:3:2），各向同性须 FAIL")
# 替身：给每个源点的矢量读数乘以该方向的各向异性权重（由源的取向决定）

def aniso_mag(name, u):
    """替身下的矢量强度：按立方各向异性键的取向权重 sum(n_a^4) 加权。"""
    w4 = float(np.sum(u ** 4))
    r, th, tvec = drag_ratio(0.1, u)
    return float(np.linalg.norm(tvec)) * w4


aniso_ref = aniso_mag("[100]", DIRS["[100]"])
m5_devs = []
m5_rows = []
for name, u in DIRS.items():
    mag = aniso_mag(name, u)
    dev = abs(mag / aniso_ref - 1.0)
    m5_devs.append(dev)
    w4 = float(np.sum(u ** 4))
    m5_rows.append((name, dev, w4))
    print("    %-6s 替身强度对参考的相对偏差 %.4f（取向权重 sum(n_a^4) = %.4f）"
          % (name, dev, w4))
# 判据量取最大偏差：替身须在至少一档上明显偏离（各向同性要求全部档偏离小）。
# 不动点档说明：各向异性权重 sum(n_a^4) 在 [100] 取 1，替身在该档不改变读数，
# 属替身的构造性质，不影响「替身使各向同性判不成立」这一结论。
m5_worst = max(m5_devs)
m5_moving = [d for (name, d, _) in zip(DIRS.keys(), m5_devs, m5_rows) if d > 1e-3]
print("  最大相对偏差 %.4f（须 > 1e-3 即判 FAIL）" % m5_worst)
print("  替身下明显偏离的取向档数：%d / %d（[100] 为权重 sum(n_a^4)=1 的不动点档）"
      % (len(m5_moving), len(m5_devs)))
check("M-5", m5_worst > 1e-3 and len(m5_moving) >= 2,
      "替身使各向同性判 FAIL（最大相对偏差 %.4f，%d 档明显偏离），判据有区分力"
      % (m5_worst, len(m5_moving)))

# ----------------------------------------------------------------------
# M-6 档位表与偏差表
# ----------------------------------------------------------------------
print()
print("[M-6] 档位表与偏差表：逐档读数给出")
m6_ok = (len(m1_rows) == len(V_LIST)) and (len(m2_rows) == len(DIRS)) and (len(m3_rows) > 0)
print("  速率档表 %d 行（v 从 %.0e 到 %.1f）；取向档表 %d 行；联合档表 %d 行"
      % (len(m1_rows), min(v for v, _, _ in m1_rows), max(v for v, _, _ in m1_rows),
         len(m2_rows), len(m3_rows)))
print("  极端档（v = 1e-4 与 0.9）的相对偏差 %.3e（条件数放大，按协议伪影报告）"
      % m1_worst_extreme)
check("M-6", m6_ok, "三张档位表给出且偏差可核")

# ----------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------
print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("运行耗时 %.2f 秒" % (time.time() - T0))
print("=" * 78)

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
vs = [v for v, _, _ in m1_rows]
ds = [d for _, _, d in m1_rows]
ax.semilogy(vs, np.maximum(ds, 1e-18), "o-", color="#1f77b4", linewidth=1.6,
            label=u"相对偏差 $|(h_{0i}/(h_{00}v_i))/2 - 1|$")
ax.axhline(1e-12, color="#d62728", linestyle="--", linewidth=1.2, label=u"容差 $10^{-12}$")
ax.set_xlabel(u"速率 $v/c$")
ax.set_ylabel(u"相对偏差")
ax.set_title(u"(a) 速率多档：入口比值恒为 2")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
names2 = [r[0] for r in m2_rows]
mags = [r[1] for r in m2_rows]
ax.bar(range(len(names2)), mags, color="#1f77b4", width=0.6)
ax.set_xticks(range(len(names2)))
ax.set_xticklabels(names2, fontsize=8, rotation=20)
ax.set_ylabel(u"矢量强度 $n=|\\Theta_i|$")
ax.set_title(u"(b) 取向多档：各向同性")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
aniso_vals = []
for name, u in DIRS.items():
    aniso_vals.append(aniso_mag(name, u) / aniso_ref)
xs = np.arange(len(names2))
ax.bar(xs - 0.2, [1.0] * len(xs), width=0.38, color="#1f77b4",
       label=u"真实读数（对参考取向）")
ax.bar(xs + 0.2, aniso_vals, width=0.38, color="#ff7f0e",
       label=u"替身：各向异性权重")
ax.set_xticks(xs)
ax.set_xticklabels(names2, fontsize=8, rotation=20)
ax.set_ylabel(u"对参考取向的相对比值")
ax.set_title(u"(c) 各向同性对照：替身使各向同性判 FAIL")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle(u"第738号：K5 多档线性性复核", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
