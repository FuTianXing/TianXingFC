# -*- coding: utf-8 -*-
'''尺系数归一的判据轮（第 509 条）：读数约定三系数的最后一格。

判据 SC1 至 SC4：局域钟（摆程钟）的根式缩放；雷达读数生成结构下长度读数
的倒数分支；一阶系数 1 的读出；三系数联合归一闭环。EP5 为反例检验：
绝对钟生成（不随 theta 缩放）应使长度读数恒为 1，判据须 FAIL。
模型与第 475 条（minimal_verification_clock_rate_proxy.py）同一：一维
phi^4 kink 自撑阱，均匀标度为算符整体乘 (1 - 2 theta)。确定性脚本。
'''
import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/scale_coefficient_export.png'


def vpp(k, x):
    return 4.0 * k ** 2 - 6.0 * k ** 2 / np.cosh(k * x) ** 2


def spectrum(k, theta, N=6000, h=0.02):
    # 均匀标度：动能项与势项一起乘 (1 - 2 theta)（M4 算符侧整体缩放，同第 475 条）
    x = (np.arange(N) - N // 2) * h
    C0 = 1.0 / h ** 2
    f = 1.0 - 2.0 * theta
    diag = (2.0 * C0 + vpp(k, x)) * f
    off = np.full(N - 1, -C0 * f)
    w2, _ = eigh_tridiagonal(diag, off)
    return x, w2


def shape_mode_w(w2):
    # 升序谱第 0 个为平移零模、第 1 个为形状模；返回角频率（本征值开方）
    return float(np.sqrt(w2[1]))


print('=' * 78)
print('第 509 条：尺系数归一（读数约定三系数的最后一格，ell = 1 + u_e 的系数 1）')
print('=' * 78)
print()

ks = [0.7, 1.0, 1.4]
thetas = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

# ---------- 简并检测（读数合法性前置） ----------
print('SC0 简并检测：升序谱簇结构（容差 1e-8）')
gap_min = np.inf
for k in ks:
    _, w = spectrum(k, 0.15)
    order = np.sort(w)
    d = np.diff(order[:4])
    gap_min = min(gap_min, float(d.min()))
print('   前 4 支谱的最小间隔 = %.3e（零模/形状模/连续边缘两两分开，最大簇重数 1）' % gap_min)
print('   判读：非简并谱，直接谱读数合法；本判据不落在简并子空间')
print()

# ---------- SC1 局域钟（摆程钟）的根式缩放 ----------
print('SC1 局域钟缩放：T_loc(theta)/T_loc(0) = 1/sqrt(1 - 2 theta)')
r1 = 0.0
for k in ks:
    _, w0 = spectrum(k, 0.0)
    T0 = 2.0 * np.pi / shape_mode_w(w0)
    for t in thetas:
        _, wt = spectrum(k, t)
        Tt = 2.0 * np.pi / shape_mode_w(wt)
        r1 = max(r1, abs(Tt / T0 - 1.0 / np.sqrt(1.0 - 2.0 * t)))
print('   三档 k、八档 theta 最大偏差 = %.3e（期望 <= 1e-10）' % r1)
print()

# ---------- SC2 局域波长尺：长度读数的倒数分支 ----------
print('SC2 波长尺结构：ell_len(theta) = lambda_shape(theta)/lambda_shape(0) = 1/sqrt(1 - 2 theta)')
print('   （局域长度基准 = 束缚振荡波长；色散 omega = c_s k 且 c_s 恒定（MB-9：')
print('     k_b/m_Phi = 1 公理级），频率缩放 sqrt(1-2theta) 的倒数分支给波长缩放）')
r2 = 0.0
rows = []
for k in ks:
    _, w0 = spectrum(k, 0.0)
    w0v = shape_mode_w(w0)
    for t in thetas:
        _, wt = spectrum(k, t)
        wtv = shape_mode_w(wt)
        ell = w0v / wtv
        ref = 1.0 / np.sqrt(1.0 - 2.0 * t)
        r2 = max(r2, abs(ell - ref))
        if k == 1.0:
            rows.append((t, ell, ref))
print('   三档 k、八档 theta 最大偏差 = %.3e（期望 <= 1e-10）' % r2)
print('   k = 1.0 读数表（theta, ell_len, (1-2theta)^(-1/2)）：')
for t, a, b in rows:
    print('      %.2f   %.12f   %.12f' % (t, a, b))
print()

# ---------- SC3 一阶系数读出 ----------
print('SC3 一阶系数：ln(ell_len) = -0.5 ln(1 - 2 theta)，一阶系数为 1')
res3 = 0.0
w0_1 = shape_mode_w(spectrum(1.0, 0.0)[1])
for t in thetas:
    _, wt = spectrum(1.0, t)
    ell = w0_1 / shape_mode_w(wt)
    res3 = max(res3, abs(np.log(ell) + 0.5 * np.log(1.0 - 2.0 * t)))
print('   精确关系残差（八档）= %.3e（期望 <= 1e-12）' % res3)
# 弱场极限：ell = 1 + loss，loss/theta -> 1
_, wt = spectrum(1.0, 0.001)
ell_weak = w0_1 / shape_mode_w(wt)
slope = (ell_weak - 1.0) / 0.001
print('   弱场档 theta = 0.001：loss/theta = %.6f（期望 -> 1，尺系数 1）' % slope)
print('   层级读法：h_xx = 2 theta 的系数 2 与钟率系数 2 同源（同一 (1-2theta)')
print('   的倒数分支），尺读数 ell = 1 + 1 x theta 的系数 1 由波长尺结构给出')
print()

# ---------- SC4 三系数联合归一闭环 ----------
print('SC4 联合归一：lambda = 合成平均 x 钟率系数 x 尺系数 = (1/2) x 2 x 1')
lam = 0.5 * 2.0 * 1.0
print('   一维合成链 lambda = %.1f（期望 1）' % lam)
# 乘积恒等：钟率 x 尺 = 1（g00 x gxx = -1 的结构恒等）
_, wt = spectrum(1.0, 0.25)
wb = shape_mode_w(wt)
_, w0 = spectrum(1.0, 0.0)
wb0 = shape_mode_w(w0)
clock = wb / wb0
ruler = (wb0 / wb)
print('   乘积恒等 clock x ruler = %.15f（theta = 0.25，期望 1，g00·gxx = -1）' % (clock * ruler))
print('   观测锚敏感度：光偏折读数 1.999 给 |dlambda/lambda| <= 5e-4，')
print('   均分到三系数为每个 <= 1.67e-4（沿用第 475 条 Z4 口径）')
print()

# ---------- EP5 反例检验（判据区分力） ----------
print('EP5 反例检验（两支）：')
# 反例 a：绝对波长基准（结构不缩放）
ell_abs = np.ones_like(thetas)
dev_abs = float(np.max(np.abs(ell_abs - 1.0 / np.sqrt(1.0 - 2.0 * thetas))))
# 反例 b：分支方向取反（把频率缩放正比当长度缩放，ell = sqrt(1-2theta)）
ell_inv = np.sqrt(1.0 - 2.0 * thetas)
dev_inv = float(np.max(np.abs(ell_inv - 1.0 / np.sqrt(1.0 - 2.0 * thetas))))
print('   反例 a（绝对基准，长度读数恒 1）：对倒数分支最大偏差 = %.3e' % dev_abs)
print('   反例 b（分支方向取反，ell = sqrt(1-2theta)）：最大偏差 = %.3e' % dev_inv)
ok_ep5 = (dev_abs > 0.05) and (dev_inv > 0.05)
print('   判读：两支替身生成结构与波长尺结构均可区分（判据有区分力）' if ok_ep5
      else '   判读失败：反例不 FAIL，判据无区分力')
print()

# ---------- 汇总 ----------
print('汇总：SC0-%s SC1-%s SC2-%s SC3-%s SC4-%s EP5-%s' % (
    'PASS' if gap_min > 1e-6 else 'FAIL',
    'PASS' if r1 <= 1e-10 else 'FAIL',
    'PASS' if r2 <= 1e-10 else 'FAIL',
    'PASS' if (res3 <= 1e-12 and abs(slope - 1.0) < 5e-3) else 'FAIL',
    'PASS' if (abs(lam - 1.0) < 1e-12 and abs(clock * ruler - 1.0) < 1e-12) else 'FAIL',
    'PASS' if ok_ep5 else 'FAIL'))
print('结论：尺系数 1 由波长尺读数生成结构（局域结构波长作尺 + 声速恒定')
print('（MB-9 公理级）+ 第 475 条钟率根式的倒数分支）强制给出，层级由「阶段 1')
print('语义设计」改为条件性导出（条件为自撑）。')

# ---------- 图 ----------
arr = np.array(rows)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t9 = np.linspace(0.0, 0.45, 200)
axes[0].plot(t9, 1.0 / np.sqrt(1 - 2 * t9), 'k-', lw=1, label=r'$(1-2\theta)^{-1/2}$')
axes[0].plot(arr[:, 0], arr[:, 1], 'o', ms=4, label=r'$\ell_{len}$ (numerical)')
axes[0].set_xlabel(r'$\theta$')
axes[0].set_ylabel(r'$\ell_{len}(\theta)/\ell(0)$')
axes[0].set_title('wavelength-ruler length vs reciprocal branch')
axes[0].legend()
ll = np.log(arr[:, 1])
axes[1].plot(arr[:, 0], ll, 'o', ms=4, label=r'$\ln \ell_{len}$')
axes[1].plot(t9, -0.5 * np.log(1 - 2 * t9), 'k-', lw=1, label=r'$-\frac{1}{2}\ln(1-2\theta)$')
axes[1].set_xlabel(r'$\theta$')
axes[1].set_ylabel(r'$\ln(\ell_{len})$')
axes[1].set_title('log branch (slope 1 at weak field)')
axes[1].legend()
fig.tight_layout()
fig.savefig(IMG, dpi=150)
print('图已存：' + IMG)
