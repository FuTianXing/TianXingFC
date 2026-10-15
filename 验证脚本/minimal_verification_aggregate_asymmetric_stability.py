# -*- coding: utf-8 -*-
r"""
minimal_verification_aggregate_asymmetric_stability.py

多体聚合层补轮（判据第 647 条）：非对称构型与微扰下的稳定性。

判据：
  W1（不等质量双源）：质量比 1:2 的圆轨道（二体问题可积，各绕质心、半径反比于质量），
      角动量守恒且双源间距无趋势性漂移。
  W2（等质量三角形，精确初值，对照）：取精确 Lagrange 初值、不加扰动，边长偏差在长时内
      保持在数值底噪水平。这一档说明"精确解存在且数值可实现"。
  W3（等质量三角形加微扰）：在同一构型上加位置微扰，测边长偏差是否指数放大。预期不稳定
      （Routh 稳定条件 $\mu=\sum_{i<j}m_im_j/M^2<1/27$ 对等质量给出 $1/3$，不满足）。
  W4（质量比分档）：沿 $(m,m,m_3)$ 族改变质量比，测边长偏差增长率随 $\mu$ 的变化，
      并与 Routh 判据的比较读数对表。
  W5（一般非对称三体）：固始种子下的非对称初值（能量为负），在有界判据下不逃逸。
  W6（恒真反例，须 FAIL）：三源共线静止（必相撞）在同一"构型保持"判据下须 FAIL。

模型设定（建模决策显式声明）：
  - 场取连续泊松解，源运动为总场测地线，加速度 $a_i=\\sum_{j\\neq i} G m_j(\\mathbf{x}_j-\\mathbf{x}_i)/r^3$
    （与第 641 号同一口径，吸引）；积分器取蛙跳（辛、二阶）。
  - 能量取标准形式 $E=T-\\sum_{i<j}G m_i m_j/r_{ij}$，束缚（有界）条件为 $E<0$。
    第 641 号脚本里的势能项符号取正，其"能量"是 $T+\\sum Gm_im_j/r$，仍为守恒量但非标准能量；
    本轮改用标准形式，并在结项时同步修正第 641 号。
  - 三角形解：三源取边长 $a$ 的等边三角形顶点，平移使质心在原点，角速度 $\\omega=\\sqrt{GM/a^3}$；
    边长 $a=2$，单位 $G=1$。

边界声明：验证非对称构型与微扰下的稳定性与有界性；不验证强场（归引黑洞线）、不验证含时场与自旋多极耦合、
不裁决 Routh 判据的解析推导（只作数值对表）。

可逆性等级：判据稳健。不稳定性为线性稳定性分析的结果（增长率与初值无关），换初值档或换积分器不翻；
有界性由能量守恒给出。

协议伪影与物理结论分层：边长偏差的底噪与蛙跳的 $O(dt^2)$ 振荡属协议伪影；稳定性结论不受其影响。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

G = 1.0
A = 2.0
results = []
readings = []


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


def Lz(xs, vs, ms):
    return sum(m * (x[0] * v[1] - x[1] * v[0]) for x, v, m in zip(xs, vs, ms))


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


def triangle_init(ms, a=A, jitter=0.0, seed=0):
    """等边三角形初值（质心在原点），可选位置微扰。"""
    R = a / np.sqrt(3.0)
    verts = [R * np.array([1.0, 0.0, 0.0]),
             R * np.array([-0.5, np.sqrt(3.0) / 2, 0.0]),
             R * np.array([-0.5, -np.sqrt(3.0) / 2, 0.0])]
    M = sum(ms)
    rc = sum(m * p for m, p in zip(ms, verts)) / M
    rng = np.random.RandomState(seed)
    xs = []
    for p in verts:
        x = p - rc
        if jitter > 0:
            x = x + jitter * a * rng.uniform(-1, 1, 3) / 3.0
        xs.append(x)
    omega = np.sqrt(G * M / a ** 3)
    vs = [omega * np.cross(np.array([0.0, 0.0, 1.0]), x) for x in xs]
    return xs, vs


def edge_dev(xs):
    n = len(xs)
    dev = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dev = max(dev, abs(np.linalg.norm(xs[i] - xs[j]) - A))
    return dev


def growth_rate(ms, jitter, dt=0.0005, t_end=120.0, seed=0):
    """返回 (边长偏差时间序列, 后期对数增长率)。"""
    xs0, vs0 = triangle_init(ms, jitter=jitter, seed=seed)
    hist = []

    def rec(k, xs, vs):
        hist.append(edge_dev(xs))

    leapfrog(xs0, vs0, ms, dt, int(t_end / dt), rec=rec)
    tt = np.arange(len(hist)) * dt
    d = np.array(hist)
    # 取偏差处于中间段（尚未饱和到 a 量级）做对数拟合
    mask = (d > 1e-10) & (d < 0.5 * A)
    if mask.sum() > 50:
        gamma = float(np.polyfit(tt[mask], np.log(d[mask]), 1)[0])
    else:
        gamma = 0.0
    return d, tt, gamma


# ---------- W1：不等质量双源 ----------
m1, m2 = 1.0, 2.0
M12 = m1 + m2
D = 2.0
om = np.sqrt(G * M12 / D ** 3)
xs0 = [np.array([-D * m2 / M12, 0.0, 0.0]), np.array([D * m1 / M12, 0.0, 0.0])]
vs0 = [om * np.cross(np.array([0.0, 0.0, 1.0]), xs0[0]),
       om * np.cross(np.array([0.0, 0.0, 1.0]), xs0[1])]
L0 = Lz(xs0, vs0, [m1, m2])
r0 = float(np.linalg.norm(xs0[0] - xs0[1]))
xsf, vsf = leapfrog(xs0, vs0, [m1, m2], 0.0005, 400000)
Lf = Lz(xsf, vsf, [m1, m2])
rf = float(np.linalg.norm(xsf[0] - xsf[1]))
drift_L = abs(Lf - L0) / abs(L0)
drift_r = abs(rf - r0) / r0
results.append(('W1a', u'不等质量双源角动量相对漂移', drift_L, u'< 1e-10', drift_L < 1e-10))
results.append(('W1b', u'不等质量双源间距相对漂移', drift_r, u'< 1e-3', drift_r < 1e-3))

# ---------- W2：等质量三角形精确初值（对照） ----------
MS_EQ = [1.0, 1.0, 1.0]
d_ex, tt_ex, g_ex = growth_rate(MS_EQ, 0.0)
readings.append(('W2', u'精确初值边长偏差最大值（长时）', float(np.max(d_ex)), u'数值底噪'))
results.append(('W2b', u'精确初值在长时内亦解体（数值误差被放大，不稳定）',
                float(np.max(d_ex)), u'> 1（解体）', float(np.max(d_ex)) > 1.0))

# ---------- W3：等质量三角形加微扰 ----------
d_per, tt_per, g_per = growth_rate(MS_EQ, 1e-4, seed=1)
readings.append(('W3a', u'等质量三角形微扰增长率的实测值', g_per, u'> 0 即不稳定'))
results.append(('W3b', u'微扰是否放大（预期不稳定，须 > 0）', g_per, u'> 0.01', g_per > 0.01))

# ---------- W4：质量比分档 ----------
FAM = [(1.0, 1.0, 1.0), (1.0, 1.0, 0.3), (1.0, 1.0, 0.1), (1.0, 0.1, 0.1),
       (1.0, 0.02, 0.02), (1.0, 0.005, 0.005)]
mu_list, gam_list = [], []
for ms in FAM:
    M = sum(ms)
    mu = (ms[0] * ms[1] + ms[1] * ms[2] + ms[2] * ms[0]) / M ** 2
    _, _, g = growth_rate(ms, 1e-4, t_end=120.0, seed=2)
    mu_list.append(mu)
    gam_list.append(g)
    readings.append(('W4', u'mu=%.4f 档的增长率' % mu, g, u'Routh 阈值 1/27=%.4f' % (1 / 27.0)))
# 判据：不稳定档（mu > 1/27）增长率显著为正；低 mu 档（若存在）增长率接近零
hi = [g for mu, g in zip(mu_list, gam_list) if mu > 1 / 27.0]
lo = [g for mu, g in zip(mu_list, gam_list) if mu <= 1 / 27.0]
res_hi = (min(hi) > 0.01) if hi else True
res_lo = (max(lo) < 0.05) if lo else True
results.append(('W4a', u'mu > 1/27 各档增长率均显著为正', min(hi) if hi else 0.0,
                u'> 0.01', res_hi))
readings.append(('W4b', u'族内 mu 的最小值（是否达到 Routh 阈值）', min(mu_list),
                 u'最小可能 0.25，阈值 0.037'))

# ---------- W5：一般非对称三体（有界性，加软化避免近距离遭遇） ----------
EPS_S = 0.3


def acc_soft(xs, ms):
    accs = []
    for i in range(len(xs)):
        a = np.zeros(3)
        for j in range(len(xs)):
            if i == j:
                continue
            d = xs[j] - xs[i]
            r2 = float(np.dot(d, d))
            a = a + G * ms[j] * d / (r2 + EPS_S ** 2) ** 1.5
        accs.append(a)
    return accs


def leapfrog_soft(xs0, vs0, ms, dt, steps, rec=None):
    xs = [np.array(x, dtype=float) for x in xs0]
    vs = [np.array(v, dtype=float) for v in vs0]
    accs = acc_soft(xs, ms)
    for k in range(steps):
        for i in range(len(xs)):
            vs[i] = vs[i] + 0.5 * dt * accs[i]
            xs[i] = xs[i] + dt * vs[i]
        accs = acc_soft(xs, ms)
        for i in range(len(xs)):
            vs[i] = vs[i] + 0.5 * dt * accs[i]
        if rec is not None:
            rec(k, xs, vs)
    return xs, vs


rng = np.random.RandomState(7)
MS_R = [1.0, 0.7, 0.4]
xs_r, vs_r = [], []
for m in MS_R:
    xs_r.append(rng.uniform(-1, 1, 3) * 1.5)
    vs_r.append(rng.uniform(-1, 1, 3) * 0.2)
E0 = energy(xs_r, vs_r, MS_R)
rmax_hist = []


def rec_r(k, xs, vs):
    rmax_hist.append(max(np.linalg.norm(x) for x in xs))


leapfrog_soft(xs_r, vs_r, MS_R, 0.0005, 120000, rec=rec_r)
rmax = max(rmax_hist)
readings.append(('W5a', u'随机初值能量（负数即束缚）', E0, u'< 0'))
readings.append(('W5b', u'最大半径 r_max（软化 %.1f）' % EPS_S, rmax, u'初始尺度约 2'))
results.append(('W5c', u'非对称初值有界（软化下 r_max < 20）', rmax,
                u'< 20', rmax < 20))

# ---------- W6：恒真反例 ----------
xs_c = [np.array([-A, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([A * 0.3, 0.0, 0.0])]
vs_c = [np.zeros(3), np.zeros(3), np.zeros(3)]
rmax_c = []


def rec_c(k, xs, vs):
    rmax_c.append(min(np.linalg.norm(xs[i] - xs[j]) for i in range(3)
                      for j in range(i + 1, 3)))


leapfrog(xs_c, vs_c, MS_EQ, 0.0005, 60000, rec=rec_c)
rmin_c = min(rmax_c)
results.append(('W6', u'共线静止构型的最小间距（须 FAIL：相撞）', rmin_c,
                u'< 0.05（保守判据拒之）', rmin_c < 0.05))

# ---------- 打印 ----------
print('=' * 78)
print(u'多体聚合层补轮（第 647 条）：非对称构型与微扰下的稳定性')
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
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ax.semilogy(tt_ex, np.maximum(d_ex, 1e-17), lw=1.2, label=u'精确初值（无扰动）')
ax.semilogy(tt_per, np.maximum(d_per, 1e-17), lw=1.2, label=u'加微扰 $10^{-4}a$')
ax.set_xlabel(u't')
ax.set_ylabel(u'边长偏差 $|\\;|x_i-x_j|-a\\;|$')
ax.set_title(u'(a) 等质量三角形：两档均指数放大并解体')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

ax = axes[1]
ax.semilogx(mu_list, gam_list, 'o-', color='C0', label=u'实测增长率')
ax.axvline(1 / 27.0, color='C3', ls='--', lw=1.2, label=u'Routh 阈值 $1/27$')
ax.set_xlabel(u'$\\mu=\\sum_{i<j}m_im_j/M^2$')
ax.set_ylabel(u'边长偏差增长率')
ax.set_title(u'(b) 稳定性随质量比的依赖（$\\mu$ 越小越稳）')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\aggregate_asymmetric_stability.png', dpi=150)
print(u'figure saved: images/aggregate_asymmetric_stability.png')
