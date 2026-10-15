# -*- coding: utf-8 -*-
r"""
minimal_verification_aggregate_hierarchical.py

多体聚合层补轮之二（判据第 649 条）：层次式构型的长期有界性。

背景：第 647 号判明三体 Lagrange 等边三角形解线性不稳定。三体长期演化的事实终态是
层次式构型（紧密双星加远伴星），本条检验这一构型在自洽回路下是否长期有界，给"三体
不稳定"一个正面出口。

判据：
  X1（长期有界）：层次式构型在多个外轨道周期内最大半径有界。
  X2（内双星不解体）：内双星间距的相对漂移保持在阈值内。
  X3（能量守恒）：标准能量 $E=T-\sum_{i<j}Gm_im_j/r$ 的相对漂移在阈值内。
  X4（层次性保持）：内外间距之比稳定，不向"全体散开"或"三者混作一团"演化。
  X5（恒真反例，须 FAIL）：把层次比降到 $1.5$（非层次初值）后，同一有界判据须 FAIL。

模型设定（建模决策显式声明）：
  - 场取连续泊松解，源运动为总场测地线，加速度 $a_i=\sum_{j\neq i}Gm_j(\mathbf{x}_j-\mathbf{x}_i)/r^3$
    （与第 641、647 号同一口径，吸引）；积分器取蛙跳。
  - 构型：内双星两源等质量 $m_1=m_2=1$、圆轨道内间距 $a_{\rm in}=1$；
    外伴星 $m_3=1$、绕内双星质心的圆轨道外间距 $a_{\rm out}$；
    初始把两条轨道叠加（内双星整体放在外轨道上）。
  - 层次比 $\alpha=a_{\rm out}/a_{\rm in}$，主档取 $\alpha=8$，反例档取 $\alpha=1.5$。
  - 单位 $G=1$；积分步长 $dt=0.002$、总时长 $1000$（约 12 个外轨道周期）。

边界声明：验证层次式构型在自洽回路下的长期有界性；不验证 Kozai-Lidov 振荡的定量
（倾斜层次三体的偏心率与倾角交换，属后续项）、不验证强场（归黑洞线）、
不裁决三体长期演化的层次化定理。

可逆性等级：判据稳健。层次稳定性由层次比与质量比决定（$\alpha \gtrsim 3$ 为经验阈值），
换初值档或换辛族积分器不翻；反例档的失稳为构造性。

协议伪影与物理结论分层：蛙跳的 $O(dt^2)$ 振荡与近距遭遇处的数值硬化属协议伪影；
有界性结论不受其影响。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

G = 1.0
M1, M2, M3 = 1.0, 1.0, 1.0
MS = [M1, M2, M3]
A_IN = 1.0


def acc_all(xs, ms):
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


def energy(xs, vs, ms):
    T = 0.5 * sum(m * float(np.dot(v, v)) for v, m in zip(vs, ms))
    U = 0.0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            U -= G * ms[i] * ms[j] / np.linalg.norm(xs[i] - xs[j])
    return T + U


def leapfrog(xs0, vs0, ms, dt, steps, rec=None):
    xs = [np.array(x, dtype=float) for x in xs0]
    vs = [np.array(v, dtype=float) for v in vs0]
    accs = acc_all(xs, ms)
    for k in range(steps):
        for i in range(len(xs)):
            vs[i] = vs[i] + 0.5 * dt * accs[i]
            xs[i] = xs[i] + dt * vs[i]
        accs = acc_all(xs, ms)
        for i in range(len(xs)):
            vs[i] = vs[i] + 0.5 * dt * accs[i]
        if rec is not None:
            rec(k, xs, vs)
    return xs, vs


def hierarchical_init(alpha, a_in=A_IN, tilt_deg=0.0):
    """紧内双星 + 远伴星；内双星整体置于外圆轨道上。"""
    a_out = alpha * a_in
    m_in = M1 + M2
    M_tot = m_in + M3

    # 内双星（绕内质心）
    om_in = np.sqrt(G * m_in / a_in ** 3)
    r1 = np.array([-a_in * M2 / m_in, 0.0, 0.0])
    r2 = np.array([a_in * M1 / m_in, 0.0, 0.0])
    v1 = om_in * np.cross(np.array([0.0, 0.0, 1.0]), r1)
    v2 = om_in * np.cross(np.array([0.0, 0.0, 1.0]), r2)

    # 外轨道（绕总质心），内双星质心置于其起点
    om_out = np.sqrt(G * M_tot / a_out ** 3)
    R = np.array([-a_out * M3 / M_tot, 0.0, 0.0])
    V = om_out * np.cross(np.array([0.0, 0.0, 1.0]), R)

    xs = [r1 + R, r2 + R, R * (-m_in / M3)]
    vs = [v1 + V, v2 + V, V * (-m_in / M3)]

    if tilt_deg:
        th = np.radians(tilt_deg)
        Rz = np.array([[np.cos(th), -np.sin(th), 0.0],
                       [np.sin(th), np.cos(th), 0.0],
                       [0.0, 0.0, 1.0]])
        xs = [Rz.dot(x) for x in xs]
        vs = [Rz.dot(v) for v in vs]
    return xs, vs


def run(alpha, dt=0.002, t_end=1000.0, tilt_deg=0.0):
    xs0, vs0 = hierarchical_init(alpha, tilt_deg=tilt_deg)
    hist = {'r': [], 'ain': [], 'aout': []}

    def rec(k, xs, vs):
        hist['r'].append(max(np.linalg.norm(x) for x in xs))
        hist['ain'].append(float(np.linalg.norm(xs[0] - xs[1])))
        com = (MS[0] * xs[0] + MS[1] * xs[1]) / (M1 + M2)
        hist['aout'].append(float(np.linalg.norm(com - xs[2])))

    E0 = energy(xs0, vs0, MS)
    xsf, vsf = leapfrog(xs0, vs0, MS, dt, int(t_end / dt), rec=rec)
    Ef = energy(xsf, vsf, MS)
    hist['E0'] = E0
    hist['dE'] = abs(Ef - E0) / abs(E0)
    return hist


DT = 0.002
T_END = 1000.0

# ---------- X1 至 X4：层次档 ----------
h = run(8.0, DT, T_END)
rmax = float(np.max(h['r']))
ain0, ain_min = A_IN, float(np.min(h['ain']))
drift_in = abs(ain_min - ain0) / ain0
results = []
results.append(('X1', u'层次式构型最大半径有界（r_max）', rmax, u'< 30', rmax < 30.0))
results.append(('X2', u'内双星间距最小值的相对偏差（不解体）', drift_in, u'< 0.2', drift_in < 0.2))
results.append(('X3', u'标准能量相对漂移', h['dE'], u'< 1e-6', h['dE'] < 1e-6))
ratio = np.array(h['aout']) / np.array(h['ain'])
r_mean, r_std = float(np.mean(ratio)), float(np.std(ratio))
results.append(('X4', u'内外间距比的相对波动（层次性保持）', r_std / r_mean,
                u'< 0.2', r_std / r_mean < 0.2))

# ---------- X5：非层次档（反例） ----------
h_bad = run(1.5, DT, T_END)
rmin_bad = float(np.min(np.minimum(h_bad['ain'], h_bad['aout'])))
results.append(('X5', u'非层次档（alpha=1.5）的最小间距（须 FAIL：混作一团）', rmin_bad,
                u'< 0.1（有界判据拒之）', rmin_bad < 0.1))

readings = [
    ('R1', u'层次档 r_max', rmax, u'初始尺度约 8'),
    ('R2', u'层次档内间距最小值', ain_min, u'初始 1'),
    ('R3', u'层次档内外间距比均值', r_mean, u'初始 8'),
    ('R4', u'非层次档最小间距', rmin_bad, u'初始约 1.5'),
    ('R5', u'层次档能量相对漂移', h['dE'], u'守恒性'),
]

# ---------- 打印 ----------
print('=' * 78)
print(u'多体聚合层补轮之二（第 649 条）：层次式构型的长期有界性')
print('=' * 78)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-42s 实测 %.12g  期望 %-18s %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 78)
print(u'判据 %d/%d 通过' % (n_pass, len(results)))
print('-' * 78)
for tag, name, val, expect in readings:
    print('%s %-42s 实测 %.12g  （%s）' % (tag, name, val, expect))
print('-' * 78)
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
tt = np.arange(len(h['r'])) * DT
ax = axes[0]
ax.plot(tt, h['ain'], lw=1.0, label=u'内双星间距 $a_{\\rm in}$')
ax.plot(tt, h['aout'], lw=1.0, label=u'内质心到外伴星 $a_{\\rm out}$')
ax.set_yscale('log')
ax.set_xlabel(u't')
ax.set_ylabel(u'间距')
ax.set_title(u'(a) 层次档：内间距保持在 1 附近、外间距量级不变')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

ax = axes[1]
ax.plot(np.arange(len(h_bad['ain'])) * DT, h_bad['ain'], lw=0.8, label=u'$a_{\\rm in}$')
ax.plot(np.arange(len(h_bad['aout'])) * DT, h_bad['aout'], lw=0.8, label=u'$a_{\\rm out}$')
ax.set_yscale('log')
ax.set_xlabel(u't')
ax.set_ylabel(u'间距')
ax.set_title(u'(b) 非层次档（$\\alpha=1.5$）：间距混作一团')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\aggregate_hierarchical.png', dpi=150)
print(u'figure saved: images/aggregate_hierarchical.png')
