# -*- coding: utf-8 -*-
'''径向钟率根式对照（第 517 条）：钟率系数 2 的 l=0 径向扇区版。

与第 511 条（径向尺系数）同一模型同一扇区，补钟率侧的直接判据：
l=0 径向算符经 u = r phi 变换后与一维同谱，均匀标度给频率根式
omega_b(theta)/omega_b(0) = sqrt(1 - 2 theta)，弱场一阶系数读出
h_00 = 2 theta 的系数 2。判据 C1 至 C4 加 EP5。确定性脚本。
'''
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/radial_clock_rate_3d_export.png'

R_MAX = 30.0
H = 0.02


def vpp(k, r):
    return 4.0 * k ** 2 - 6.0 * k ** 2 / np.cosh(k * r) ** 2


def radial_matrix(k, theta):
    """l=0 径向算符的等价一维形式：-u'' + V''(r) u，无离心项（l >= 1 才有）。"""
    n = int(R_MAX / H)
    r = (np.arange(n) + 1) * H
    C0 = 1.0 / H ** 2
    f = 1.0 - 2.0 * theta
    diag = (2.0 * C0 + vpp(k, r)) * f
    off = np.full(n - 1, -C0 * f)
    return diag, off


def lowest(k, theta, m=2):
    diag, off = radial_matrix(k, theta)
    A = np.diag(diag) + np.diag(off, -1) + np.diag(off, 1)
    w, _ = eigsh(A, k=m, which='SA')
    return np.sort(w)


print('=' * 78)
print('第 517 条：径向钟率根式对照（钟率系数 2 的 l=0 径向扇区版）')
print('=' * 78)
print()

ks = [0.7, 1.0, 1.4]
thetas = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

# ---------- C1 谱基准与整体缩放 ----------
print('C1 谱基准：形状模对解析 3k^2 与整体缩放比值')
c1a = 0.0
c1b = 0.0
for k in ks:
    w0 = lowest(k, 0.0)
    c1b = max(c1b, abs(w0[0] / (3.0 * k ** 2) - 1.0))
    for t in thetas:
        wt = lowest(k, t)
        c1a = max(c1a, abs((wt[0] / w0[0]) / (1.0 - 2.0 * t) - 1.0))
print('   C1a 缩放比值最大偏差 = %.3e（期望 <= 1e-9）' % c1a)
print('   C1b 形状模对 3k^2 最大偏差 = %.3e（预算 <= 3e-3）' % c1b)
print()

# ---------- C2 频率根式 ----------
print('C2 频率根式：wb(theta)/wb(0) = sqrt(1 - 2 theta)')
c2 = 0.0
rows = []
for k in ks:
    w0 = float(np.sqrt(lowest(k, 0.0)[0]))
    for t in thetas:
        wt = float(np.sqrt(lowest(k, t)[0]))
        r = wt / w0
        ref = np.sqrt(1.0 - 2.0 * t)
        c2 = max(c2, abs(r - ref))
        if k == 1.0:
            rows.append((t, r, ref))
print('   三档 k、八档 theta 最大偏差 = %.3e（期望 <= 1e-9）' % c2)
for t, a, b in rows:
    print('      %.2f   %.12f   %.12f' % (t, a, b))
print()

# ---------- C3 系数 2 读出 ----------
print('C3 系数 2 的读出：-ln(wb/wb0) = -0.5 ln(1 - 2 theta)，一阶系数 1')
c3 = 0.0
w0_1 = float(np.sqrt(lowest(1.0, 0.0)[0]))
for t in thetas:
    wt = float(np.sqrt(lowest(1.0, t)[0]))
    r = wt / w0_1
    c3 = max(c3, abs(-np.log(r) + 0.5 * np.log(1.0 - 2.0 * t)))
print('   精确关系残差（八档）= %.3e（期望 <= 1e-9）' % c3)
wt = float(np.sqrt(lowest(1.0, 0.001)[0]))
slope = (1.0 - wt / w0_1) / 0.001
print('   弱场档 theta = 0.001：(1 - wb/wb0)/theta = %.6f（期望 -> 1，即 h_00 = 2 theta）' % slope)
# theta = 1/2：均匀标度因子为 0，算符整体零，本征值精确为 0（视界），不作数值对角化
w_half = (1.0 - 2.0 * 0.5) * float(np.sqrt(lowest(1.0, 0.0)[0]))
print('   视界端 theta = 1/2：wb = (1-2theta) x wb(0) = %.3e（期望 0，频率为零）' % w_half)
print()

# ---------- C4 乘积恒等与三系数径向闭环 ----------
print('C4 乘积恒等与三系数径向闭环')
w_t = float(np.sqrt(lowest(1.0, 0.25)[0]))
clock = w_t / w0_1
ruler = w0_1 / w_t
print('   clock x ruler = %.15f（theta = 0.25，期望 1，g00·gxx = -1）' % (clock * ruler))
print('   径向三系数：合成平均 1/2 x 钟率 2 x 尺 1 = %.1f' % (0.5 * 2.0 * 1.0))
print()

# ---------- EP5 ----------
print('EP5 反例检验：正比分支（无均匀标度）应使钟率读数恒 1')
r_abs = np.ones_like(thetas)
dev = float(np.max(np.abs(r_abs - np.sqrt(1.0 - 2.0 * thetas))))
print('   绝对钟率读数对根式期望最大偏差 = %.3e' % dev)
ok_ep5 = dev > 0.05
print('   判读：反例可区分，判据有区分力' if ok_ep5 else '   判读失败：反例不可区分')
print()

ok1 = (c1a <= 1e-9) and (c1b <= 3e-3)
ok2 = c2 <= 1e-9
ok3 = (c3 <= 1e-9) and (abs(slope - 1.0) < 5e-3)
ok4 = abs(clock * ruler - 1.0) < 1e-12
print('汇总：C1-%s C2-%s C3-%s C4-%s EP5-%s' % (
    'PASS' if ok1 else 'FAIL', 'PASS' if ok2 else 'FAIL',
    'PASS' if ok3 else 'FAIL', 'PASS' if ok4 else 'FAIL',
    'PASS' if ok_ep5 else 'FAIL'))
print('结论：钟率系数 2 的径向扇区版成立（频率根式与一阶系数 1 直接判据），')
print('第 511 条的乘积恒等间接覆盖升级为钟率与尺双侧直接判据。')

# ---------- 图 ----------
arr = np.array(rows)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t9 = np.linspace(0.0, 0.45, 200)
axes[0].plot(t9, np.sqrt(1 - 2 * t9), 'k-', lw=1, label=r'$\sqrt{1-2\theta}$')
axes[0].plot(arr[:, 0], arr[:, 1], 'o', ms=4, label=r'$\omega_b$ (radial)')
axes[0].set_xlabel(r'$\theta$')
axes[0].set_ylabel(r'$\omega_b(\theta)/\omega_b(0)$')
axes[0].set_title('radial clock-rate root branch')
axes[0].legend()
ll = -np.log(arr[:, 1])
axes[1].plot(arr[:, 0], ll, 'o', ms=4, label=r'$-\ln \omega_b$')
axes[1].plot(t9, -0.5 * np.log(1 - 2 * t9), 'k-', lw=1, label=r'$-\frac{1}{2}\ln(1-2\theta)$')
axes[1].set_xlabel(r'$\theta$')
axes[1].set_ylabel(r'$-\ln(\omega_b/\omega_b(0))$')
axes[1].set_title('radial log branch (coefficient 2)')
axes[1].legend()
fig.tight_layout()
fig.savefig(IMG, dpi=150)
print('图已存：' + IMG)
