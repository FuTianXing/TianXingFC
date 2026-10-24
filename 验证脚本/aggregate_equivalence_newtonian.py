# -*- coding: utf-8 -*-
"""
aggregate_equivalence_newtonian.py

多体聚合层阶段三（判据第 643 条）：等效性与牛顿极限判据轮。

判据：
  U1（牛顿极限的收敛率）：同一总质量的不同分配下，远区场 theta(R)*R/GM 趋于 1，
      偏差随 R 按 R^-2 衰减（四极主导；偶极项因质心取原点而消失）。双点与三点
      构型各测一条收敛率，斜率须落在 -2 附近。
  U2（等效性：质量重分配不变性，远区场）：三种分配（单点、双点、三点）在同一远区
      半径处的场值互差不超过多极预算（量级 (d/R)^2）。
  U3（等效性：远区试验粒子）：同一试验粒子在三种分配的场中，加速度互差不超过预算。
      注意与单源等效原理判据的区别：单源检验"轨迹与质量无关"，本条检验"轨迹与
      质量分布方式无关"，两者不可互推。
  U4（多源二阶项与交叉项）：G1b 完成形式 h00 = theta - theta^2/2 逐点作用于总场时，
      二阶项等于各源二阶项之和加交叉项，双源交叉项 = -theta_1*theta_2、三源
      交叉项 = -sum_{i<j} theta_i theta_j，残差须为机器零。
  U5（恒真反例，须 FAIL）：把场替身为"只取最近源"或"逐点取最大"，同一远区判据下
      须 FAIL。

模型设定（建模决策显式声明）：
  - 场取连续泊松解 theta(x) = sum_j G m_j / |x - x_j|（阶段一已确认注入加邻居抄写的
    稳态即该解，阶段二沿用），无限空间，无周期镜像。
  - 构型总质量相同 M = 3：单点（原点，质量 3）、双点（+-d/2，各 1.5）、三点
    （等边三角形边长 d，各 1）；源间距 d = 2，质心均取原点。
  - 单位取 G = 1。

边界声明：验证聚合层的远区行为与单源一致（牛顿极限、分配不变性）以及多源二阶项的
交叉项形式；不验证强场区（归黑洞线）、不验证 SEP（天体自身引力能，待立项）、不裁决
弱场二阶项对表的层级授予（作者裁决点 B，本轮按判据级登记）。

可逆性等级：判据稳健。远区收敛率由多极展开的阶数给出（质心在原点则偶极为零、四极
主导），换构型或换半径档不翻；恒真反例的分辨力为构造性。

协议伪影与物理结论分层：远区偏差的量级由 (d/R)^2 预算给出，属截断层读数；"远区只
依赖总质量"这一结构结论不受具体系数影响。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

G = 1.0
M_TOT = 3.0
D = 2.0
results = []
readings = []


def make_configs():
    A = [(np.zeros(3), M_TOT)]
    B = [(np.array([-D / 2, 0.0, 0.0]), M_TOT / 2),
         (np.array([D / 2, 0.0, 0.0]), M_TOT / 2)]
    R3 = D / np.sqrt(3.0)
    C = [(R3 * np.array([1.0, 0.0, 0.0]), M_TOT / 3),
         (R3 * np.array([-0.5, np.sqrt(3.0) / 2, 0.0]), M_TOT / 3),
         (R3 * np.array([-0.5, -np.sqrt(3.0) / 2, 0.0]), M_TOT / 3)]
    return [(u'\u5355\u70b9', A), (u'\u53cc\u70b9', B), (u'\u4e09\u70b9', C)]


def theta(x, src):
    return sum(G * m / np.linalg.norm(x - p) for p, m in src)


def grad_theta(x, src):
    g = np.zeros(3)
    for p, m in src:
        d = x - p
        r = np.linalg.norm(d)
        g += -G * m * d / r ** 3
    return g


DIRS = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
        np.array([1, 1, 1]) / np.sqrt(3.0), np.array([1, -1, 0]) / np.sqrt(2.0)]
R_LIST = [10.0, 20.0, 40.0, 80.0, 160.0]
CONF = make_configs()


def far_dev(src, R):
    """远区场偏差 |theta(R)*R/(G*M) - 1|，取方向上的最大值。"""
    return max(abs(theta(R * e, src) * R / (G * M_TOT) - 1.0) for e in DIRS)


# ---------- U1：牛顿极限的收敛率 ----------
slopes = {}
for name, src in CONF[1:]:
    devs = np.array([far_dev(src, R) for R in R_LIST])
    logs = np.log(devs)
    slope = float(np.polyfit(np.log(np.array(R_LIST)), logs, 1)[0])
    slopes[name] = (devs, slope)
    results.append(('U1-%s' % name,
                    u'远区偏差收敛斜率（%s 构型）' % name, slope,
                    u'≈ -2（四极主导）', -2.6 < slope < -1.4))
    readings.append(('U1-%s' % name, u'R=160 处远区偏差', devs[-1],
                     u'预算 (d/R)^2 = %.3g' % ((D / 160.0) ** 2)))

# ---------- U2：等效性（远区场分配不变性） ----------
R_REF = 160.0
vals = {name: [theta(R_REF * e, src) * R_REF / (G * M_TOT) for e in DIRS]
        for name, src in CONF}
max_pair = 0.0
for i in range(len(CONF)):
    for j in range(i + 1, len(CONF)):
        d_ij = max(abs(a - b) for a, b in zip(vals[CONF[i][0]], vals[CONF[j][0]]))
        max_pair = max(max_pair, d_ij)
budget2 = 2.0 * (D / R_REF) ** 2
results.append(('U2', u'三构型远区场互差（R=160）', max_pair,
                u'<= 预算 %.3g' % budget2, max_pair <= budget2))

# ---------- U3：等效性（远区试验粒子加速度） ----------
XP = np.array([0.0, 0.0, 60.0])
accs = {name: grad_theta(XP, src) for name, src in CONF}
max_acc = 0.0
for i in range(len(CONF)):
    for j in range(i + 1, len(CONF)):
        max_acc = max(max_acc, float(np.linalg.norm(accs[CONF[i][0]] - accs[CONF[j][0]])))
a_ref = float(np.linalg.norm(accs[u'\u5355\u70b9']))
rel_acc = max_acc / a_ref
results.append(('U3', u'远区试验粒子加速度互差 / 单点值', rel_acc,
                u'<= 预算 %.3g' % (4.0 * (D / 60.0) ** 2),
                rel_acc <= 4.0 * (D / 60.0) ** 2))

# ---------- U4：多源二阶项与交叉项 ----------
def comp(t):
    return t - 0.5 * t * t


# 用"逐源场"在给定点集上做：交叉项 = comp(sum) - sum(comp)
PTS = []
for e in DIRS:
    for R in (2.0, 4.0, 8.0):
        PTS.append(R * e)
PTS.append(np.array([0.0, 0.0, 3.0]))
PTS.append(np.array([1.0, 1.0, 1.0]))


def per_source_fields(src, pts):
    return [[G * m / np.linalg.norm(x - p) for x in pts] for p, m in src]


for name, src in CONF[1:]:
    fs = per_source_fields(src, PTS)
    tot = np.array([sum(f) for f in zip(*fs)])
    lhs = comp(tot) - sum(comp(np.array(f)) for f in fs)
    rhs = np.zeros_like(lhs)
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            rhs += -np.array(fs[i]) * np.array(fs[j])
    res = float(np.max(np.abs(lhs - rhs)) / max(np.max(np.abs(rhs)), 1e-300))
    tag = 'U4a' if name == u'\u53cc\u70b9' else 'U4b'
    results.append((tag, u'%s构型交叉项对 -sum theta_i theta_j 相对偏差' % name,
                    res, u'<= 1e-12（机器零）', res <= 1e-12))

# ---------- U5：恒真反例 ----------
def fake_nearest(x, src):
    ds = [np.linalg.norm(x - p) for p, m in src]
    i = int(np.argmin(ds))
    return G * M_TOT / ds[i]


def fake_max(x, src):
    return max(G * m / np.linalg.norm(x - p) for p, m in src)


srcB = CONF[1][1]
dev_nearest = max(abs(fake_nearest(R_REF * e, srcB) * R_REF / (G * M_TOT) - 1.0)
                  for e in DIRS)
dev_max = max(abs(fake_max(R_REF * e, srcB) * R_REF / (G * M_TOT) - 1.0)
              for e in DIRS)
results.append(('U5a', u'替身（只取最近源）远区偏差（须 FAIL）', dev_nearest,
                u'> 预算 %.3g' % budget2, dev_nearest > budget2))
results.append(('U5b', u'替身（逐点取最大）远区偏差（须 FAIL）', dev_max,
                u'> 预算 %.3g' % budget2, dev_max > budget2))

# ---------- 打印 ----------
print('=' * 78)
print(u'多体聚合层阶段三（第 643 条）：等效性与牛顿极限判据轮')
print('=' * 78)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-40s 实测 %.12g  期望 %-22s %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 78)
print(u'判据 %d/%d 通过' % (n_pass, len(results)))
print('-' * 78)
for tag, name, val, expect in readings:
    print('%s %-40s 实测 %.12g  （%s）' % (tag, name, val, expect))
print('-' * 78)
for name, (devs, slope) in slopes.items():
    print(u'%s 构型远区偏差：' % name + u', '.join('R=%.0f:%.3g' % (R, d)
                                                  for R, d in zip(R_LIST, devs)))
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 1：远区收敛 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
for name, src in CONF:
    devs = np.array([far_dev(src, R) for R in R_LIST])
    ax.loglog(R_LIST, np.maximum(devs, 1e-18), 'o-', label=u'%s构型' % name)
ax.loglog(R_LIST, [(D / R) ** 2 for R in R_LIST], 'k--', lw=1.2,
          label=u'$(d/R)^2$ 参考')
ax.set_xlabel(u'远区半径 R')
ax.set_ylabel(u'$|\\theta R/(GM)-1|$')
ax.set_title(u'(a) 远区场趋于 $GM/R$：偏差按 $R^{-2}$ 衰减')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

ax = axes[1]
for name, src in CONF:
    rr = np.linspace(4, 160, 200)
    ax.plot(rr, [theta(np.array([0, 0, r]), src) * r / (G * M_TOT) for r in rr],
            lw=1.6, label=u'%s构型' % name)
ax.axhline(1.0, color='k', ls='--', lw=1.2, label=u'牛顿极限 $GM/R$')
ax.set_xlabel(u'半径 r')
ax.set_ylabel(u'$\\theta(r)\\, r/(GM)$')
ax.set_title(u'(b) 三种质量分配在远区合流')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\aggregate_equivalence_newtonian.png', dpi=150)
print(u'figure saved: images/aggregate_equivalence_newtonian.png')

# ---------- 图 2：交叉项 ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 4.6))
for k, (name, src) in enumerate(CONF[1:]):
    fs = per_source_fields(src, PTS)
    tot = np.array([sum(f) for f in zip(*fs)])
    lhs = comp(tot) - sum(comp(np.array(f)) for f in fs)
    rhs = np.zeros_like(lhs)
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            rhs += -np.array(fs[i]) * np.array(fs[j])
    ax = axes2[k]
    ax.plot(range(len(PTS)), lhs, 'o', ms=6, label=u'实测交叉项')
    ax.plot(range(len(PTS)), rhs, 'x', ms=8, label=u'预测 $-\\sum_{i<j}\\theta_i\\theta_j$')
    ax.set_xlabel(u'测点序号')
    ax.set_ylabel(u'交叉项')
    ax.set_title(u'(%s) %s构型的交叉项形式' % ('ab'[k], name))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\aggregate_equivalence_cross.png', dpi=150)
print(u'figure saved: images/aggregate_equivalence_cross.png')
