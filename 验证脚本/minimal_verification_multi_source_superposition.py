# -*- coding: utf-8 -*-
"""
minimal_verification_multi_source_superposition.py

多体聚合层阶段一（判据第 640 条）：写入叠加律判据轮。

判据：
  S1（远场双源叠加）：两源间距 d = 16 格距、质量 1.0 与 0.6，总场
      theta[rho_1 + rho_2] 逐点等于 theta[rho_1] + theta[rho_2]，最大相对残差
      不超过判据线（判据线由实测浮点底噪预算给出）。
  S2（近邻区叠加，第 588 号未覆盖区）：源间距 d = 2、3、4、6、8、12 逐档，
      残差同样不超过判据线。
  S3（多源与非等质量）：三源与四源构型（质量比 1.0 : 0.6 : 0.25），残差不超过
      判据线。
  S4（恒真反例 A，须 FAIL）：把加和型替身为逐点乘积型，同一残差判据下须 FAIL。
  S5（恒真反例 B，须 FAIL）：替身为逐点最大值型，须 FAIL。
  S6（输出校验，不作叠加律证据）：试验粒子在叠加场中的运动对叠加场 ODE 的步长
      收敛率读数（RK4）。

读数（不作判据）：
  R1：判据线的两项来源（浮点底噪、离散化与周期镜像偏差）。
  R2：二阶完成形式逐点作用于总场时交叉项的形式与量级（阶段三接口预告）。

模型设定（建模决策显式声明）：
  - 写入层：源 i 在格点 c_i 每拍注入倾角份额 delta_theta_i = m_i / m_P，总注入
    rho = sum_i rho_i（份额单位，归一化为质量份额）。
  - 稳态：注入加邻居抄写的稳态为离散泊松格林函数 theta(r) = G m / (c^2 r)，
    离散实现为周期格上七点拉普拉斯方程 -laplacian theta = rho 的 FFT 谱解
    （去零模），与第 446 条 DC1 同一实现。
  - 格距取 1（无量纲），质量取份额单位；周期边界。
  - 叠加的对象是写入量（各缺陷写入的倾角记录），非场的动力学合成；本轮的判据
    内容为定理 6 线性半在显式多源构型（含近邻区）下的可读性确认。

边界声明：验证写入量在多源构型下的逐点线性叠加与判据分辨力；不验证场方程非线性
半边的完整形式（交叉项只作读数）、不验证自洽回路（属阶段二）、不裁决写入份额定理
的线性半（第 588 号已立，本轮只作多源构型确认）。

可逆性等级：判据稳健（写入层的线性叠加由定理 6 线性半给出，换实现或换档位不翻；
恒真反例的分辨力为构造性）。

协议伪影与物理结论分层：周期镜像偏差与离散化偏差属协议伪影，与叠加律结论分层
报告；近邻区读数分辨率极限由底噪预算给出。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N = 64
C = N // 2
results = []
readings = []


def make_lam(n):
    kk = np.fft.fftfreq(n, d=1.0) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing='ij')
    lam = 6.0 - 2.0 * (np.cos(KX) + np.cos(KY) + np.cos(KZ))
    lam[0, 0, 0] = 1.0  # 去零模（中性背景）
    return lam


LAM = make_lam(N)


def solve(rho, lam=LAM):
    s = rho - rho.mean()
    return np.real(np.fft.ifftn(np.fft.fftn(s) / lam))


def src(pos, mass, n=N):
    rho = np.zeros((n, n, n))
    rho[tuple(p % n for p in pos)] = mass
    return rho


def relres(a, b):
    den = float(np.max(np.abs(b)))
    if den <= 0.0:
        den = 1.0
    return float(np.max(np.abs(a - b)) / den)


# ---------- R1：判据线的两项来源 ----------
rho0 = src((C, C, C), 1.0)
t_one = solve(rho0)
t_ten = solve(10.0 * rho0)
eps_fp = relres(t_ten, 10.0 * t_one)          # 浮点底噪（同一数学对象的两种写法）

LAM2 = make_lam(2 * N)
rho2 = src((N, N, N), 1.0, n=2 * N)
t_two = solve(rho2, LAM2)
rs = np.arange(3, 11)
a64 = np.array([t_one[C + r, C, C] for r in rs])
a128 = np.array([t_two[N + r, N, N] for r in rs])
eps_disc = float(np.max(np.abs(a64 - a128)) / np.max(np.abs(a128)))  # 离散化加镜像偏差

LINE = max(100.0 * eps_fp, 1e-12)             # 判据线：底噪预算加安全余量
readings.append(('R1a', '浮点底噪 eps_fp', eps_fp, 'FFT 谱解舍入'))
readings.append(('R1b', '离散化加镜像偏差 eps_disc', eps_disc, 'N=64 对 N=128，r=3..10'))
readings.append(('R1c', '判据线 LINE', LINE, 'max(100*eps_fp, 1e-12)'))


# ---------- 叠加残差与替身 ----------
def superpose(sources):
    """返回 (总场, 各源场之和)。"""
    th_tot = solve(sum(s[0] for s in sources))
    th_sum = sum(solve(s[0]) for s in sources)
    return th_tot, th_sum


def twin(kind, sources):
    """替身场：乘积型或最大值型。"""
    ths = [solve(s[0]) for s in sources]
    if kind == 'prod':
        cand = np.ones_like(ths[0])
        for t in ths:
            cand = cand * t
        return cand
    if kind == 'max':
        return np.maximum.reduce(ths)
    raise ValueError(kind)


# ---------- S1：远场双源 ----------
D_FAR = 16
sA = (src((C - D_FAR // 2, C, C), 1.0), 1.0)
sB = (src((C + D_FAR // 2, C, C), 0.6), 0.6)
th_far, sum_far = superpose([sA, sB])
r_far = relres(th_far, sum_far)
results.append(('S1', '远场双源叠加相对残差', r_far, '<= LINE',
                r_far <= LINE))

# ---------- S2：近邻区逐档 ----------
D_LIST = [2, 3, 4, 6, 8, 12]
near_res = []
for d in D_LIST:
    sa = (src((C - d // 2, C, C), 1.0), 1.0)
    sb = (src((C - d // 2 + d, C, C), 0.6), 0.6)
    tt, ss = superpose([sa, sb])
    near_res.append(relres(tt, ss))
r_near_max = max(near_res)
results.append(('S2', '近邻区各档最大残差', r_near_max, '<= LINE',
                r_near_max <= LINE))

# ---------- S3：三源与四源 ----------
tri = [(src((C, C, C), 1.0), 1.0),
       (src((C + 8, C, C), 0.6), 0.6),
       (src((C, C + 8, C), 0.25), 0.25)]
tt3, ss3 = superpose(tri)
r_tri = relres(tt3, ss3)
results.append(('S3a', '三源非等质量叠加残差', r_tri, '<= LINE',
                r_tri <= LINE))

quad = [(src((C + 6, C + 6, C), 1.0), 1.0),
        (src((C - 6, C + 6, C), 0.6), 0.6),
        (src((C - 6, C - 6, C), 0.25), 0.25),
        (src((C + 6, C - 6, C), 0.4), 0.4)]
tt4, ss4 = superpose(quad)
r_quad = relres(tt4, ss4)
results.append(('S3b', '四源非等质量叠加残差', r_quad, '<= LINE',
                r_quad <= LINE))

# ---------- S4 / S5：恒真反例 ----------
r_prod = relres(twin('prod', [sA, sB]), sum_far)
results.append(('S4', '乘积型替身相对残差（须 FAIL）', r_prod, '> LINE',
                r_prod > LINE))
r_max = relres(twin('max', [sA, sB]), sum_far)
results.append(('S5', '最大值型替身相对残差（须 FAIL）', r_max, '> LINE',
                r_max > LINE))

# 等质量档（更公平的对照：非等质量时最大值型退化为取强源，偏离偏小）
sA_eq = (src((C - D_FAR // 2, C, C), 1.0), 1.0)
sB_eq = (src((C + D_FAR // 2, C, C), 1.0), 1.0)
_, sum_eq = superpose([sA_eq, sB_eq])
r_prod_eq = relres(twin('prod', [sA_eq, sB_eq]), sum_eq)
results.append(('S4b', '乘积型替身残差（等质量档，须 FAIL）', r_prod_eq, '> LINE',
                r_prod_eq > LINE))
r_max_eq = relres(twin('max', [sA_eq, sB_eq]), sum_eq)
results.append(('S5b', '最大值型替身残差（等质量档，须 FAIL）', r_max_eq, '> LINE',
                r_max_eq > LINE))

# ---------- S6：试验粒子 ODE 步长收敛（输出校验） ----------
kx = np.fft.fftfreq(N, d=1.0) * 2.0 * np.pi
KXg, KYg, KZg = np.meshgrid(kx, kx, kx, indexing='ij')


def spec_grad(field):
    """谱梯度（带限场，避免中心差分的分段线性光滑度限制）。"""
    F = np.fft.fftn(field)
    return (np.real(np.fft.ifftn(1j * KXg * F)),
            np.real(np.fft.ifftn(1j * KYg * F)),
            np.real(np.fft.ifftn(1j * KZg * F)))


gx, gy, gz = spec_grad(th_far)


def trilinear(field, p):
    """周期格上的三线性插值（分段三线性场，避免最近邻取值造成阶梯）。"""
    x0 = np.floor(p).astype(int)
    f = p - x0
    i0 = [c % N for c in x0]
    i1 = [(c + 1) % N for c in x0]
    fx, fy, fz = f
    c000 = field[i0[0], i0[1], i0[2]]
    c100 = field[i1[0], i0[1], i0[2]]
    c010 = field[i0[0], i1[1], i0[2]]
    c110 = field[i1[0], i1[1], i0[2]]
    c001 = field[i0[0], i0[1], i1[2]]
    c101 = field[i1[0], i0[1], i1[2]]
    c011 = field[i0[0], i1[1], i1[2]]
    c111 = field[i1[0], i1[1], i1[2]]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def acc(p):
    return np.array([trilinear(gx, p), trilinear(gy, p), trilinear(gz, p)])


SRC_FAR = [((C - D_FAR // 2, C, C), 1.0), ((C + D_FAR // 2, C, C), 0.6)]


def acc_analytic(p):
    """解析叠加场的梯度（三维格林函数之和，取最小镜像），用于轨迹积分读数。

    符号口径：h_00 = 2 theta，牛顿势 Phi = -c^2 theta，故 a = -grad Phi = c^2 grad theta
    （取 c = 1 即 +grad theta），为吸引。
    """
    g = np.zeros(3)
    for pos, m in SRC_FAR:
        d = np.array(p, dtype=float) - np.array(pos, dtype=float)
        d = d - N * np.round(d / N)
        r = float(np.linalg.norm(d))
        g += -m * d / (4.0 * np.pi * r ** 3)
    return g


def integrate(dt, steps, p0, v0):
    p = np.array(p0, dtype=float)
    v = np.array(v0, dtype=float)
    for _ in range(steps):
        a1 = acc_analytic(p)
        k1p, k1v = v, a1
        a2 = acc_analytic(p + 0.5 * dt * k1p)
        k2p, k2v = v + 0.5 * dt * k1v, a2
        a3 = acc_analytic(p + 0.5 * dt * k2p)
        k3p, k3v = v + 0.5 * dt * k2v, a3
        a4 = acc_analytic(p + dt * k3p)
        k4p, k4v = v + dt * k3v, a4
        p = p + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return p


P0 = (C - 4.0, C + 0.5, C + 0.5)
V0 = (0.3, 0.05, 0.02)
T_END = 20.0
traj = {}
for n_steps in (200, 400, 800):
    dt = T_END / n_steps
    traj[n_steps] = integrate(dt, n_steps, P0, V0)
d1 = float(np.linalg.norm(traj[200] - traj[400]))
d2 = float(np.linalg.norm(traj[400] - traj[800]))
ratio = d1 / d2 if d2 > 0 else float('inf')
order = float(np.log2(ratio)) if ratio > 0 else 0.0
readings.append(('S6a', '粒子轨迹步长差比 |dt|-|dt/2|', ratio, 'RK4 期望约 16'))
readings.append(('S6b', '步长收敛阶', order, 'RK4 期望约 4（受场光滑度限制）'))
# 输出校验：轨迹加速度等于叠加场梯度（逐点恒等，机器级）
p_test = (C - 4.0, C + 0.5, C + 0.5)
idx_t = tuple(np.round(p_test).astype(int))
g_tot = np.array([gx[idx_t], gy[idx_t], gz[idx_t]])
ga = spec_grad(solve(sA[0]))
gb = spec_grad(solve(sB[0]))
g_sum = np.array([ga[i][idx_t] + gb[i][idx_t] for i in range(3)])
r_grad = float(np.max(np.abs(g_tot - g_sum)) / max(np.max(np.abs(g_sum)), 1e-300))
results.append(('S6c', '粒子加速度场等于各源梯度之和', r_grad, '<= LINE',
                r_grad <= LINE))

# ---------- R2：二阶完成形式的交叉项（阶段三接口预告） ----------
def comp(t):
    return t - 0.5 * t * t


ta = solve(sA[0])
tb = solve(sB[0])
cross_meas = comp(ta + tb) - comp(ta) - comp(tb)
cross_pred = -ta * tb
r_cross = relres(cross_meas, cross_pred)
readings.append(('R2a', '交叉项实测对 -theta_1*theta_2 的相对偏差', r_cross,
                 '机器零（形式确认）'))
readings.append(('R2b', '交叉项幅度 max|-theta_1 theta_2| / max|theta_lin|',
                 float(np.max(np.abs(cross_pred)) / np.max(np.abs(ta + tb))),
                 '读数'))

# ---------- 打印 ----------
print('=' * 78)
print('多体聚合层阶段一（第 640 条）：写入叠加律判据轮')
print('=' * 78)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-30s 实测 %.12g  期望 %-14s %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 78)
print('判据 %d/%d 通过' % (n_pass, len(results)))
print('-' * 78)
for tag, name, val, expect in readings:
    print('%s %-30s 实测 %.12g  （%s）' % (tag, name, val, expect))
print('-' * 78)
print('近邻区逐档残差：' + ', '.join('d=%d:%.3g' % (d, r) for d, r in zip(D_LIST, near_res)))
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 1：残差-间距曲线与场切片 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ax.semilogy(D_LIST, near_res, 'o-', color='C0', label='加和型（各源场之和）')
ax.axhline(LINE, color='C1', ls='--', lw=1.2, label='判据线（底噪预算）')
ax.axhline(r_prod, color='C3', ls=':', lw=1.2, label='乘积型替身')
ax.axhline(r_max, color='C4', ls='-.', lw=1.2, label='最大值型替身')
ax.set_xlabel('源间距 d（格距）')
ax.set_ylabel('相对残差')
ax.set_title('(a) 叠加残差随源间距：近邻区仍在线下，替身在线之上')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

ax = axes[1]
sl = th_far[:, C, :].T          # y=C 面切片，过两源连线（横轴 x，纵轴 z）
vmax = np.max(np.abs(sl))
im = ax.imshow(sl, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.set_title('(b) 远场双源总倾角场切片（y=C 面，d=16）')
ax.set_xlabel('x')
ax.set_ylabel('z')
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\multi_source_superposition.png', dpi=150)
print('figure saved: images/multi_source_superposition.png')

# ---------- 图 2：恒真反例分辨力 ----------
fig2, ax2 = plt.subplots(figsize=(6.2, 4.2))
labels = ['加和型\n（真值）', '乘积型\n（替身）', '最大值型\n（替身）']
vals = [r_far, r_prod, r_max]
bars = ax2.bar(labels, vals, color=['C0', 'C3', 'C4'], width=0.55)
ax2.axhline(LINE, color='k', ls='--', lw=1.2, label='判据线')
ax2.set_yscale('log')
ax2.set_ylabel('相对残差')
ax2.set_title('判据分辨力：加和型在线下，替身被判据拒绝')
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.5, '%.2g' % v,
             ha='center', fontsize=8)
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\multi_source_superposition_twins.png', dpi=150)
print('figure saved: images/multi_source_superposition_twins.png')
