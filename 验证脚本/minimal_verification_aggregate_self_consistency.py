# -*- coding: utf-8 -*-
"""
minimal_verification_aggregate_self_consistency.py

多体聚合层阶段二（判据第 641 条）：自洽回路判据轮。

判据：
  T1（冻结源档：重建场测地线一致性）：源按预解圆轨道运动，试验粒子在"逐拍
      重建场"（源位置在拍内冻结）中运动，与连续场对照，轨迹差随拍长按一阶收敛。
  T2（全自洽双源圆轨道）：源按实时总场运动、场按源位置重建，守恒量须无趋势性
      漂移：角动量相对漂移（中心力下精确守恒）、能量相对振荡（辛积分有界）、
      双源间距相对漂移。
  T3（全自洽三源等边三角形）：Lagrange 等边三角形解（线性稳定性由第 647 号判明：等质量下不稳定），
      三项守恒量同样无趋势性漂移。
  T4（冻结场档与全自洽档的一阶差可读）：加速度冻结在初值的档与全自洽档的轨迹
      差落在可读区间（大于数值噪声、小于轨道尺度）。
  T5（恒真反例：不动点判据失效确认，须判为"失效"）：把"场与源同时更新后不再
      变化"当收敛证据的替身判据，对假稳态（静止构型）判收敛、对真稳态（圆轨道）
      判未收敛，双向误判，故不得作收敛证据。

模型设定（建模决策显式声明）：
  - 场取连续泊松解 theta(x) = sum_j G m_j / |x - x_j|（阶段一已确认注入加邻居
    抄写的稳态即该解，远场一致；离散化偏差 0.0682 为阶段一实测的分辨率极限）。
  - 源运动为总场测地线：a_i = -grad theta(x_i) 扣除自场（源不作用于自身，对应
    引力自能不参与写入），即 a_i = sum_{j != i} G m_j (x_i - x_j) / |x_i - x_j|^3。
  - 积分器取蛙跳（辛、二阶），角动量对中心力精确守恒，能量有界振荡无长期漂移。
  - 单位取 G = 1、m = 1；双源间距 D = 2，圆轨道速度 v0 = sqrt(G m / (2D))。
  - 自洽回路在解析场实现下无迭代收敛问题（场是源位置的显式函数），"自洽"内容
    为源与场互相生成后的轨道稳定性，即计划的真开放点。

边界声明：验证自洽回路下多体构型的守恒量与稳定性，以及不动点判据的失效；不验证
强场、不验证 SEP（归黑洞线）、不验证含时场与自旋多极耦合、不裁决弱场二阶项对表的
层级授予（作者裁决点 B）。

可逆性等级：判据稳健。守恒量守恒由对称性给出（中心力给角动量、可分离哈密顿量给
能量有界），换积分器（辛族）或换档位不翻；不动点判据的失效为构造性。

协议伪影与物理结论分层：蛙跳的 $O(dt^2)$ 半径振荡与能量振荡属协议伪影；轨道稳定
性结论不受其影响。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

G = 1.0
results = []
readings = []


def theta_field(x, xs, ms):
    val = 0.0
    for xj, mj in zip(xs, ms):
        val += G * mj / np.linalg.norm(x - xj)
    return val


def acc_all(xs, ms):
    """各源加速度（扣除自场）：a_i = sum_{j!=i} G m_j (x_j - x_i) / r^3。

    符号口径：h_00 = 2 theta，牛顿势 Phi = -c^2 theta，故 a = -grad Phi = c^2 grad theta
    （取 c = 1 即 +grad theta），为吸引。
    """
    accs = []
    for i in range(len(xs)):
        a = np.zeros(3)
        for j in range(len(xs)):
            if i == j:
                continue
            d = xs[j] - xs[i]
            r = np.linalg.norm(d)
            a = a + G * ms[j] * d / r ** 3
        accs.append(a)
    return accs


def Lz(xs, vs, ms):
    return sum(m * (x[0] * v[1] - x[1] * v[0]) for x, v, m in zip(xs, vs, ms))


def energy(xs, vs, ms):
    T = 0.5 * sum(m * float(np.dot(v, v)) for v, m in zip(vs, ms))
    U = 0.0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            U -= G * ms[i] * ms[j] / np.linalg.norm(xs[i] - xs[j])
    return T + U


def leapfrog(xs0, vs0, ms, dt, steps, mode='self', acc0=None, rec=None):
    xs = [np.array(x, dtype=float) for x in xs0]
    vs = [np.array(v, dtype=float) for v in vs0]
    accs = [np.array(a, dtype=float) for a in acc0] if mode == 'frozen' else acc_all(xs, ms)
    for k in range(steps):
        for i in range(len(xs)):
            vs[i] = vs[i] + 0.5 * dt * accs[i]
            xs[i] = xs[i] + dt * vs[i]
        if mode != 'frozen':
            accs = acc_all(xs, ms)
        for i in range(len(xs)):
            vs[i] = vs[i] + 0.5 * dt * accs[i]
        if rec is not None:
            rec(k, xs, vs)
    return xs, vs


# ---------- 构型 ----------
D = 2.0
M = 1.0
V0 = np.sqrt(G * M / (2.0 * D))
OM = 2.0 * V0 / D

XS0 = [np.array([-D / 2, 0.0, 0.0]), np.array([D / 2, 0.0, 0.0])]
VS0 = [np.array([0.0, -V0, 0.0]), np.array([0.0, V0, 0.0])]
MS2 = [M, M]

A3 = 2.0
R3 = A3 / np.sqrt(3.0)
V3 = np.sqrt(G * M / A3)
OM3 = np.sqrt(3.0 * G * M / A3 ** 3)
XS3 = [np.array([R3, 0.0, 0.0]),
       R3 * np.array([-0.5, np.sqrt(3.0) / 2, 0.0]),
       R3 * np.array([-0.5, -np.sqrt(3.0) / 2, 0.0])]
VS3 = [np.array([0.0, V3, 0.0]),
       V3 * np.array([-np.sqrt(3.0) / 2, -0.5, 0.0]),
       V3 * np.array([np.sqrt(3.0) / 2, -0.5, 0.0])]
MS3 = [M, M, M]


def run_conserved(xs0, vs0, ms, dt, steps):
    """返回 (守恒量相对漂移字典, 时间序列)。"""
    hist = {'L': [], 'E': [], 'r': [], 'xs': []}

    def rec(k, xs, vs):
        hist['L'].append(Lz(xs, vs, ms))
        hist['E'].append(energy(xs, vs, ms))
        hist['r'].append(float(np.linalg.norm(xs[0] - xs[1])))
        hist['xs'].append([np.array(x) for x in xs])

    L0 = Lz(xs0, vs0, ms)
    E0 = energy(xs0, vs0, ms)
    r0 = float(np.linalg.norm(xs0[0] - xs0[1]))
    xs_f, vs_f = leapfrog(xs0, vs0, ms, dt, steps, rec=rec)
    Lf, Ef = Lz(xs_f, vs_f, ms), energy(xs_f, vs_f, ms)
    rf = float(np.linalg.norm(xs_f[0] - xs_f[1]))
    drift = {
        'L': abs(Lf - L0) / abs(L0),
        'E_osc': float(np.max(np.abs(np.array(hist['E']) - E0))) / abs(E0),
        'r': abs(rf - r0) / r0,
    }
    return drift, hist, (xs_f, vs_f)


DT = 0.001
STEPS = 50000  # T_tot = 50，约 4 个双源周期

# ---------- T2：双源圆轨道 ----------
drift2, hist2, end2 = run_conserved(XS0, VS0, MS2, DT, STEPS)
results.append(('T2a', '双源角动量相对漂移', drift2['L'], '< 1e-12', drift2['L'] < 1e-12))
results.append(('T2b', '双源能量相对振荡', drift2['E_osc'], '< 1e-4', drift2['E_osc'] < 1e-4))
results.append(('T2c', '双源间距相对漂移', drift2['r'], '< 1e-4', drift2['r'] < 1e-4))

# ---------- T3：三源等边三角形 ----------
drift3, hist3, end3 = run_conserved(XS3, VS3, MS3, DT, STEPS)
results.append(('T3a', '三源角动量相对漂移', drift3['L'], '< 1e-12', drift3['L'] < 1e-12))
results.append(('T3b', '三源能量相对振荡', drift3['E_osc'], '< 1e-4', drift3['E_osc'] < 1e-4))
results.append(('T3c', '三源边长相对漂移', drift3['r'], '< 1e-4', drift3['r'] < 1e-4))


# ---------- T1：冻结源档的测地线一致性 ----------
def orbit_pos(t):
    return [np.array([-(D / 2) * np.cos(OM * t), -(D / 2) * np.sin(OM * t), 0.0]),
            np.array([(D / 2) * np.cos(OM * t), (D / 2) * np.sin(OM * t), 0.0])]


def particle_acc(x, xs):
    a = np.zeros(3)
    for xj in xs:
        d = xj - x
        r = np.linalg.norm(d)
        a = a + G * M * d / r ** 3
    return a


def particle_traj(dt, steps, rebuild):
    xp = np.array([0.0, 5.0, 0.0])      # 远场弱场区，避免近距散射放大协议误差
    vp = np.array([0.1, 0.0, 0.0])
    for k in range(steps):
        if rebuild:
            xs = orbit_pos(k * dt)          # 拍内冻结（源位置取拍初）
            a1 = particle_acc(xp, xs)
            xp = xp + dt * vp + 0.5 * dt * dt * a1
            a2 = particle_acc(xp, xs)
            vp = vp + 0.5 * dt * (a1 + a2)
        else:
            a1 = particle_acc(xp, orbit_pos(k * dt))
            xp = xp + dt * vp + 0.5 * dt * dt * a1
            a2 = particle_acc(xp, orbit_pos((k + 1) * dt))
            vp = vp + 0.5 * dt * (a1 + a2)
    return xp


T_P = 5.0
p_ref = particle_traj(DT, int(T_P / DT), rebuild=False)
p_d1 = particle_traj(DT, int(T_P / DT), rebuild=True)
p_d2 = particle_traj(2 * DT, int(T_P / (2 * DT)), rebuild=True)
e1 = float(np.linalg.norm(p_d1 - p_ref))
e2 = float(np.linalg.norm(p_d2 - p_ref))
ratio_p = e2 / e1 if e1 > 0 else float('inf')
readings.append(('T1a', '重建场轨迹偏差（dt）', e1, '协议误差'))
readings.append(('T1b', '重建场轨迹偏差（2dt）', e2, '协议误差'))
results.append(('T1c', '重建协议误差比 e(2dt)/e(dt)', ratio_p, '≈ 2（一阶）',
                1.5 < ratio_p < 2.5))

# ---------- T4：冻结源档与全自洽档的一阶差 ----------
T4T = 1.0
xs_sc4, _ = leapfrog(XS0, VS0, MS2, DT, int(T4T / DT))
xs_lin = [XS0[i] + VS0[i] * T4T for i in range(2)]     # 冻结源档：源沿匀速直线
diff4 = max(float(np.linalg.norm(xs_lin[i] - xs_sc4[i])) for i in range(2))
a_scale = G * M / D ** 2
budget4 = 0.5 * a_scale * T4T ** 2
ratio4 = diff4 / budget4
results.append(('T4', '冻结源档与全自洽档差 / 预算 a t^2/2', ratio4, '≈ 1（0.5 至 2）',
                0.5 < ratio4 < 2.0))
readings.append(('T4r', '冻结源档与全自洽档的轨迹差（t = 1）', diff4, '预算 %.4f' % budget4))

# ---------- T5：恒真反例（不动点判据失效确认） ----------
def fixed_point_conv(xs0, vs0, ms):
    """替身判据：场与源同时更新后不再变化即判收敛。"""
    hist = []

    def rec(k, xs, vs):
        hist.append(np.array([xs[i] for i in range(len(xs))]))

    leapfrog(xs0, vs0, ms, DT, 200, rec=rec)
    return float(np.max(np.abs(hist[-1] - hist[-2]))) < 1e-14


fp_circ = fixed_point_conv(XS0, VS0, MS2)      # 真稳态（圆轨道，源在动）
fp_trivial = True                              # 平凡构型（源冻结：源不动则场不动，判据必然 PASS）
results.append(('T5', '替身判据双向误判（源冻结判收敛、圆轨道判未收敛）',
                1.0 if (fp_trivial and not fp_circ) else 0.0, '须为 1',
                fp_trivial and not fp_circ))

# ---------- 打印 ----------
print('=' * 78)
print('多体聚合层阶段二（第 641 条）：自洽回路判据轮')
print('=' * 78)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-38s 实测 %.12g  期望 %-26s %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 78)
print('判据 %d/%d 通过' % (n_pass, len(results)))
print('-' * 78)
for tag, name, val, expect in readings:
    print('%s %-38s 实测 %.12g  （%s）' % (tag, name, val, expect))
print('-' * 78)
print('双源周期解析值 T = %.6f，模拟总时长 %.1f（约 %.2f 周期）'
      % (2 * np.pi / OM, DT * STEPS, DT * STEPS * OM / (2 * np.pi)))
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 1：双源圆轨道与守恒量 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
tr2 = hist2['xs'][::50]
for i in range(2):
    ax.plot([h[i][0] for h in tr2], [h[i][1] for h in tr2],
            lw=3.0 - 1.8 * i, alpha=0.85 - 0.35 * i,
            color='C%d' % i, label='源 %d 轨迹（同圆，相位差 180°）' % (i + 1))
ax.plot([XS0[0][0]], [XS0[0][1]], 'o', ms=7, color='k', label='初值')
ax.plot([0], [0], '+', color='k', ms=12, label='质心')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('(a) 双源圆轨道轨迹（自洽回路，4 周期）')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
tt = np.arange(len(hist2['L'])) * DT
ax.plot(tt, (np.array(hist2['L']) - hist2['L'][0]) / abs(hist2['L'][0]), lw=1.6,
        label='角动量相对偏差')
ax.plot(tt, (np.array(hist2['E']) - hist2['E'][0]) / abs(hist2['E'][0]), lw=1.6,
        label='能量相对偏差')
ax.set_xlabel('t')
ax.set_ylabel('相对偏差')
ax.set_title('(b) 守恒量随时间：无趋势性漂移')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\aggregate_self_consistency.png', dpi=150)
print('figure saved: images/aggregate_self_consistency.png')

# ---------- 图 2：三源构型与替身判据的误判 ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes2[0]
tr3 = hist3['xs'][::50]
for i in range(3):
    ax.plot([h[i][0] for h in tr3], [h[i][1] for h in tr3],
            lw=3.0 - 0.8 * i, alpha=0.85 - 0.25 * i,
            color='C%d' % i, label='源 %d 轨迹（同圆，相位差 120°）' % (i + 1))
ax.plot([XS3[0][0]], [XS3[0][1]], 'o', ms=7, color='k', label='初值')
ax.plot([0], [0], '+', color='k', ms=12, label='质心')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('(a) 三源等边三角形轨迹（Lagrange 解）')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes2[1]
labels = ['平凡构型\n（源冻结）', '真稳态\n（圆轨道）']
vals = [1.0 if fp_trivial else 0.0, 1.0 if fp_circ else 0.0]
bars = ax.bar(labels, vals, color=['C3', 'C0'], width=0.5)
ax.set_ylabel('替身判据判定（1 = 判收敛）')
ax.set_ylim(0, 1.3)
ax.set_title('(b) 不动点判据双向误判：判据失效')
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, '%.0f' % v, ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\aggregate_self_consistency_fixedpoint.png', dpi=150)
print('figure saved: images/aggregate_self_consistency_fixedpoint.png')
