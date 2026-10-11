# -*- coding: utf-8 -*-
# 第 608 号判据轮：D3p 载体合法性（V1a 联合规范不变性 / V1b 小角展开标度 / V1c 错写齿）
# 载体：U(1) 链接格点 + 复物质场 z（介质振子），非线性能量面
# H = sum E^2/2 + sum_plaq (curl theta)^2/2 + w0^2 sum |z|^2/2 - kappa sum Re(z^* e^{i theta} z')
# 规范变换：theta_pa -> theta_pa + chi_p - chi_{p+a}；z_p -> e^{i chi_p} z_p
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'D:\software\TianXingFC\images\608_gauge_invariance.png'
rng = np.random.default_rng(20261012)
N, W0, KAPPA = 8, 1.0, 0.6

def curl(theta):
    # theta: (N,N,2)；plaquette 环量（离散旋度）
    tx = theta[:, :, 0]
    ty = theta[:, :, 1]
    return (tx + np.roll(ty, -1, axis=0)
            - np.roll(tx, -1, axis=1) - ty)

def H_exact(theta, z, jump_mode='correct'):
    mag = 0.5 * np.sum(curl(theta) ** 2)
    matter = 0.5 * W0 ** 2 * np.sum(np.abs(z) ** 2)
    jump = 0.0
    for a, ax in enumerate((0, 1)):
        zp = np.roll(z, -1, axis=ax)
        link = np.exp(1j * theta[:, :, a])
        if jump_mode == 'wrong':  # 错写：右端共轭
            zp = np.conj(zp)
        jump = jump - KAPPA * np.sum(np.real(np.conj(z) * link * zp))
    return mag, matter, jump

def H_quad(theta, z):
    # 小角二次展开（保留到 θ² 与场二阶）：Re(z* e^{iθ} z') ≈ Re(w) - θ Im(w) - (θ²/2) Re(w)，w = z* z'
    mag = 0.5 * np.sum(curl(theta) ** 2)
    matter = 0.5 * W0 ** 2 * np.sum(np.abs(z) ** 2)
    jump = 0.0
    for a, ax in enumerate((0, 1)):
        zp = np.roll(z, -1, axis=ax)
        w = np.conj(z) * zp
        jump = jump - KAPPA * np.sum(np.real(w)
                                     - theta[:, :, a] * np.imag(w)
                                     - 0.5 * theta[:, :, a] ** 2 * np.real(w))
    return mag, matter, jump

def gauge_transform(theta, z, chi):
    th = theta.copy()
    th[:, :, 0] = th[:, :, 0] + chi - np.roll(chi, -1, axis=0)
    th[:, :, 1] = th[:, :, 1] + chi - np.roll(chi, -1, axis=1)
    return th, z * np.exp(1j * chi)

# ---------- 冒烟检查 ----------
th0 = rng.uniform(-0.3, 0.3, (N, N, 2))
z0 = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
chi0 = rng.normal(scale=0.5, size=(N, N))
m0, mt0, j0 = H_exact(th0, z0)
th1, z1 = gauge_transform(th0, z0, chi0)
m1, mt1, j1 = H_exact(th1, z1)
print('[smoke] H 总量 %.4f（非零），各分量量级 mag=%.3f matter=%.3f jump=%.3f' % (m0 + mt0 + j0, m0, mt0, j0))
print('[smoke] 变化轴 = 规范变换方向：Δmag=%.2e Δmatter=%.2e Δjump=%.2e' % (abs(m1 - m0), abs(mt1 - mt0), abs(j1 - j0)))

# ---------- V1a 联合规范不变性（20 组随机抽样） ----------
TOL = 1e-10
v1a_pass = True
v1a_rows = []
for trial in range(20):
    th = rng.uniform(-0.3, 0.3, (N, N, 2))
    z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    chi = rng.normal(scale=0.5, size=(N, N))
    H0 = sum(H_exact(th, z))
    H1 = sum(H_exact(*gauge_transform(th, z, chi)))
    d = abs(H1 - H0)
    ok = d <= TOL
    v1a_pass &= ok
    v1a_rows.append(d)
print('\n===== V1a 联合规范不变性 =====')
print('20 组 |ΔH| 最大 %.2e（容差 %.0e）→ %s' % (max(v1a_rows), TOL, 'PASS' if v1a_pass else 'FAIL'))

# ---------- V1b 小角二次展开的残差标度（多实现 RMS，随机行走量纲） ----------
# 残差领头项为四阶（Re(w)theta^4/24 与 Im(w)theta^3/6 乘一阶场；theta^2 与 theta·Im 交叉项
# 已被二次型完整吸收，三阶项不存在），同号系统项主导：每档 RMS ∝ eps^4，比值为 16。
# 过程教训：首版期望误写三阶（比值 8），被实测两档一致比值 16.8/17.0 抓出，逐阶重推订正。
print('\n===== V1b 展开残差标度（每档 30 实现 RMS，期望比值 16，四阶领头） =====')
eps_list = [0.02, 0.01, 0.005]
ntrial = 30
res = []
for eps in eps_list:
    ds = []
    for _ in range(ntrial):
        th = eps * rng.uniform(-1, 1, (N, N, 2))
        z = 1 + eps * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
        ds.append(abs(sum(H_exact(th, z)) - sum(H_quad(th, z))))
    res.append(float(np.sqrt(np.mean(np.array(ds) ** 2))))
    print('eps=%.3f  RMS|H_exact - H_quad| = %.6e' % (eps, res[-1]))
r1 = res[0] / res[1]
r2 = res[1] / res[2]
print('RMS 比：r1=%.3f r2=%.3f（16±2 为 PASS，30 实现 RMS 估计波动约 13%%）' % (r1, r2))
v1b_pass = abs(r1 - 16) <= 2 and abs(r2 - 16) <= 2
print('V1b 判定：', 'PASS' if v1b_pass else 'FAIL')

# ---------- V1c 错写齿（右端共轭破坏不变性） ----------
print('\n===== V1c 错写齿 =====')
dwrong = []
for trial in range(20):
    th = rng.uniform(-0.3, 0.3, (N, N, 2))
    z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    chi = rng.normal(scale=0.5, size=(N, N))
    H0 = sum(H_exact(th, z, jump_mode='wrong'))
    H1 = sum(H_exact(*gauge_transform(th, z, chi), jump_mode='wrong'))
    dwrong.append(abs(H1 - H0))
print('错写模式 20 组 |ΔH| 最小 %.3e（正确模式最大 %.2e）' % (min(dwrong), max(v1a_rows)))
v1c_pass = min(dwrong) > 100 * max(max(v1a_rows), 1e-12)
print('V1c 判定：', 'PASS' if v1c_pass else 'FAIL')

print('\n总判定：', 'PASS（3/3）' if (v1a_pass and v1b_pass and v1c_pass) else 'FAIL')

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.loglog(eps_list, res, 'o-', color='crimson', label='展开残差 RMS（30 实现）')
ref = [res[0] * (e / eps_list[0]) ** 4 for e in eps_list]
ax.loglog(eps_list, ref, '--', color='gray', label='四阶参考（比值 16）')
ax.set_xlabel('组态幅度 eps')
ax.set_ylabel('残差 RMS')
ax.set_title('V1b：小角二次展开残差 RMS\n（四阶领头标度，比值 16）')
ax.legend(fontsize=8)
ax = axes[1]
labels = ['正确模式\n（20 组最大）', '错写模式\n（20 组最小）']
vals = [max(v1a_rows), min(dwrong)]
ax.bar(labels, vals, color=['steelblue', 'crimson'], width=0.5)
ax.set_yscale('log')
ax.set_ylabel('|ΔH|（规范变换前后）')
ax.set_title('V1c：错写齿（右端共轭）\n不变性破坏显著，齿有牙')
for i, v in enumerate(vals):
    ax.text(i, v * 1.3, '%.2e' % v, ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print('图已存', OUT)
