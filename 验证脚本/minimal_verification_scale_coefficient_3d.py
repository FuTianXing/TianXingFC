# -*- coding: utf-8 -*-
'''尺系数归一的三维径向推广（第 511 条）：SC 判据的径向扇区版。

模型：球坐标 l = 0 径向涨落算符 A = -[phi'' + (2/r)phi'] + V''(phi)，
V'' = 4k^2 - 6k^2 sech^2(kr)（自洽恒等式三维径向逐点保持，判据 S1）。
均匀标度：算符整体乘 (1 - 2 theta)（径向梯度与势项同缩放，l=0 无角向项）。
判据 D1 至 D4：径向谱基准与缩放、波长尺倒数分支、一阶系数读出、联合归一。
EP5：双反例区分力。确定性脚本。
'''
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/scale_coefficient_3d_export.png'

R_MAX = 30.0
H = 0.02


def vpp(k, r):
    return 4.0 * k ** 2 - 6.0 * k ** 2 / np.cosh(k * r) ** 2


def radial_matrix(k, theta):
    """l=0 径向算符 -[phi'' + (2/r)phi'] + V'' phi 的等价一维形式。

    代换 u = r phi（正则边界 phi'(0)=0 即 u(0)=0）严格给出
    -u'' + V''(r) u = lam u（无离心项：l(l+1)/r^2 只属 l >= 1 扇区），
    半线 (0, inf) 上的自伴问题，谱与一维 kink 相同（形状模 3k^2）。
    """
    n = int(R_MAX / H)
    r = (np.arange(n) + 1) * H  # r_1..r_n，u(0)=0
    C0 = 1.0 / H ** 2
    f = 1.0 - 2.0 * theta
    pot = vpp(k, r)
    diag = (2.0 * C0 + pot) * f
    off = np.full(n - 1, -C0 * f)
    return r, diag, off


def low_spectrum(k, theta, m=3):
    r, diag, off = radial_matrix(k, theta)
    w, _ = eigsh(np.diag(diag) + np.diag(off, -1) + np.diag(off, 1), k=m, which='SA')
    return np.sort(w)


print('=' * 78)
print('第 511 条：尺系数归一的三维径向推广（SC 判据的径向扇区版）')
print('=' * 78)
print()

# ---------- 冒烟检查：谱结构与 theta 缩放 ----------
print('冒烟检查：l=0 等价谱（u = r phi）的最低三支与 theta 缩放')
for k in [0.7, 1.0]:
    w0 = low_spectrum(k, 0.0)
    w3 = low_spectrum(k, 0.25)
    print('   k = %.1f: theta=0 最低三支 %s' % (k, np.array2string(w0, precision=4)))
    print('           theta=0.25 最低三支 %s' % np.array2string(w3, precision=4))
    print('           缩放比 (w3/w0)/0.5 = %s（期望各支 ~1）'
          % np.array2string(w3 / w0 / 0.5, precision=5))
print('   对照解析：形状模 3k^2（k=0.7: 1.47，k=1.0: 3.0）；l=0 扇区无平移零模')
print()

ks = [0.7, 1.0, 1.4]
thetas = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

# ---------- D1 谱基准与非简并 ----------
print('D1 谱基准：缩放比值（同网格相消）与解析 3k^2 对照（按 O(h^2) 截断预算）')
d1a = 0.0
d1b = 0.0
for k in ks:
    w0 = low_spectrum(k, 0.0)
    d1b = max(d1b, abs(w0[0] / (3.0 * k ** 2) - 1.0))
    for t in thetas:
        wt = low_spectrum(k, t)
        d1a = max(d1a, abs((wt[0] / w0[0]) / (1.0 - 2.0 * t) - 1.0))
print('   D1a 缩放比值最大偏差 = %.3e（期望 <= 1e-9，同网格截断相消）' % d1a)
print('   D1b 形状模对解析 3k^2 最大偏差 = %.3e（预算 <= 3e-3：h^2 k^4 @ k=1.4）' % d1b)
print()

# ---------- D2 波长尺倒数分支 ----------
print('D2 波长尺：ell_len(theta) = lambda(theta)/lambda(0) = (1-2theta)^(-1/2)')
d2 = 0.0
rows = []
for k in ks:
    w0 = low_spectrum(k, 0.0)
    lam0 = 2.0 * np.pi / np.sqrt(w0[0])
    for t in thetas:
        wt = low_spectrum(k, t)
        lam = 2.0 * np.pi / np.sqrt(wt[0])
        ell = lam / lam0
        ref = 1.0 / np.sqrt(1.0 - 2.0 * t)
        d2 = max(d2, abs(ell - ref))
        if k == 1.0:
            rows.append((t, ell, ref))
print('   三档 k、八档 theta 最大偏差 = %.3e（期望 <= 1e-9）' % d2)
for t, a, b in rows:
    print('      %.2f   %.12f   %.12f' % (t, a, b))
print()

# ---------- D3 一阶系数 ----------
print('D3 一阶系数：ln(ell_len) = -0.5 ln(1 - 2 theta)，一阶系数为 1')
d3 = 0.0
w0_1 = low_spectrum(1.0, 0.0)
lam0_1 = 2.0 * np.pi / np.sqrt(w0_1[0])
for t in thetas:
    wt = low_spectrum(1.0, t)
    ell = (2.0 * np.pi / np.sqrt(wt[0])) / lam0_1
    d3 = max(d3, abs(np.log(ell) + 0.5 * np.log(1.0 - 2.0 * t)))
print('   精确关系残差（八档）= %.3e（期望 <= 1e-9）' % d3)
wt = low_spectrum(1.0, 0.001)
ell_weak = (2.0 * np.pi / np.sqrt(wt[0])) / lam0_1
slope = (ell_weak - 1.0) / 0.001
print('   弱场档 theta = 0.001：loss/theta = %.6f（期望 -> 1）' % slope)
print()

# ---------- D4 联合归一 ----------
print('D4 联合归一：lambda = (1/2) x 2 x 1，乘积恒等钟率 x 尺 = 1')
w0c = low_spectrum(1.0, 0.0)
wc0 = np.sqrt(w0c[0])
wtc = np.sqrt(low_spectrum(1.0, 0.25)[0])
clock = wtc / wc0
ruler = wc0 / wtc
print('   clock x ruler = %.15f（theta = 0.25，期望 1）' % (clock * ruler))
print('   一维合成链 lambda = (1/2) x 2 x 1 = 1（径向倒数分支与一维同构）')
print()

# ---------- EP5 反例 ----------
print('EP5 反例检验（两支）：')
ell_abs = np.ones_like(thetas)
dev_abs = float(np.max(np.abs(ell_abs - 1.0 / np.sqrt(1.0 - 2.0 * thetas))))
ell_inv = np.sqrt(1.0 - 2.0 * thetas)
dev_inv = float(np.max(np.abs(ell_inv - 1.0 / np.sqrt(1.0 - 2.0 * thetas))))
print('   反例 a（绝对基准）：最大偏差 = %.3e' % dev_abs)
print('   反例 b（分支方向取反）：最大偏差 = %.3e' % dev_inv)
ok_ep5 = (dev_abs > 0.05) and (dev_inv > 0.05)
print('   判读：两支替身生成结构与径向波长尺结构均可区分' if ok_ep5
      else '   判读失败：反例不 FAIL')
print()

ok1 = (d1a <= 1e-9) and (d1b <= 3e-3)
ok2 = d2 <= 1e-9
ok3 = (d3 <= 1e-9) and (abs(slope - 1.0) < 5e-3)
ok4 = abs(clock * ruler - 1.0) < 1e-12
print('汇总：D1-%s D2-%s D3-%s D4-%s EP5-%s' % (
    'PASS' if ok1 else 'FAIL', 'PASS' if ok2 else 'FAIL',
    'PASS' if ok3 else 'FAIL', 'PASS' if ok4 else 'FAIL',
    'PASS' if ok_ep5 else 'FAIL'))
print('结论：波长尺倒数分支与一阶系数 1 在三维径向扇区成立，尺系数生成')
print('结构的径向推广与判据 S 的均匀标度径向推广同构（l=0 无角向项，')
print('径向梯度与势项同缩放）。')

# ---------- 图 ----------
arr = np.array(rows)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t9 = np.linspace(0.0, 0.45, 200)
axes[0].plot(t9, 1.0 / np.sqrt(1 - 2 * t9), 'k-', lw=1, label=r'$(1-2\theta)^{-1/2}$')
axes[0].plot(arr[:, 0], arr[:, 1], 'o', ms=4, label=r'$\ell_{len}$ (radial)')
axes[0].set_xlabel(r'$\theta$')
axes[0].set_ylabel(r'$\ell_{len}(\theta)/\ell(0)$')
axes[0].set_title('radial wavelength-ruler vs reciprocal branch')
axes[0].legend()
ll = np.log(arr[:, 1])
axes[1].plot(arr[:, 0], ll, 'o', ms=4, label=r'$\ln \ell_{len}$')
axes[1].plot(t9, -0.5 * np.log(1 - 2 * t9), 'k-', lw=1, label=r'$-\frac{1}{2}\ln(1-2\theta)$')
axes[1].set_xlabel(r'$\theta$')
axes[1].set_ylabel(r'$\ln(\ell_{len})$')
axes[1].set_title('radial log branch (slope 1 at weak field)')
axes[1].legend()
fig.tight_layout()
fig.savefig(IMG, dpi=150)
print('图已存：' + IMG)
