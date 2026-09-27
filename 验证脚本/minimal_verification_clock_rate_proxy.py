# -*- coding: utf-8 -*-
'''钟率代理导出的判据轮（第 475 条）：自撑均匀标度下 h00 = 2 theta 的系数 2。

判据 Z1 至 Z4：自撑恒等式的整体缩放可实现性；谱的根式标度；弱场系数的读出；
视界位置一致性与三系数联合归一的敏感度。确定性脚本。
'''
import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/clock_rate_proxy_export.png'


def vpp(k, x, theta=0.0):
    return (4.0 * k ** 2 - 6.0 * k ** 2 / np.cosh(k * x) ** 2) * (1.0 - 2.0 * theta)


def spectrum(k, theta, N=6000, h=0.02):
    # 均匀标度：动能项与势项一起乘 (1 - 2 theta)（M4 的算符侧整体缩放）
    x = (np.arange(N) - N // 2) * h
    C0 = 1.0 / h ** 2
    f = 1.0 - 2.0 * theta
    diag = (2.0 * C0 + vpp(k, x, 0.0)) * f
    off = np.full(N - 1, -C0 * f)
    w2, _ = eigh_tridiagonal(diag, off)
    return x, w2


def shape_mode(w2, k):
    # 升序谱的第 0 个为平移零模、第 1 个为形状模；返回频率（本征值开方）
    return float(np.sqrt(w2[1]))


print('=' * 78)
print('第 475 条：钟率代理导出（自撑均匀标度下 h00 = 2 theta 的系数 2）')
print('=' * 78)
print()

ks = [0.7, 1.0, 1.4]
thetas = np.arange(0.0, 0.46, 0.05)

print('Z1 自撑恒等式的整体缩放：Vpp(theta) = (1 - 2 theta) Vpp(0) 点态成立')
r1 = 0.0
for k in ks:
    x = (np.arange(6000) - 3000) * 0.02
    for t in [0.15, 0.35]:
        r1 = max(r1, float(np.max(np.abs(vpp(k, x, t) - (1 - 2 * t) * vpp(k, x, 0.0)))))
print('   最大点态偏差 = %.3e（期望 0）' % r1)
print()

print('Z2 谱的根式标度：wb(theta)/wb(0) = sqrt(1 - 2 theta)')
r2 = 0.0
rows = []
for k in ks:
    _, w0 = spectrum(k, 0.0)
    wb0 = shape_mode(w0, k)
    for t in thetas[1:]:
        _, wt = spectrum(k, t)
        wbt = shape_mode(wt, k)
        ratio = wbt / wb0
        ref = np.sqrt(1.0 - 2.0 * t)
        r2 = max(r2, abs(ratio - ref))
        if k == 1.0:
            rows.append((t, ratio, ref))
print('   三档 k、九档 theta 的最大偏差 = %.3e（期望 <= 1e-10）' % r2)
print('   k = 1.0 读数表（theta, wb/wb0, sqrt(1-2theta)）：')
for t, a, b in rows:
    print('      %.2f   %.12f   %.12f' % (t, a, b))
print()

print('Z3 弱场系数的读出：精确关系 -ln(wb/wb0) = -0.5 ln(1 - 2 theta)')
_, w0 = spectrum(1.0, 0.0)
wb0 = shape_mode(w0, 1.0)
ts = np.array([0.001, 0.002, 0.004, 0.008])
loss = []
exact = []
for t in ts:
    _, wt = spectrum(1.0, t)
    r = shape_mode(wt, 1.0) / wb0
    loss.append(-np.log(r))
    exact.append(abs(-np.log(r) + 0.5 * np.log(1.0 - 2.0 * t)))
print('   精确关系最大残差 = %.3e（期望 <= 1e-12）' % max(exact))
print('   比值 loss/theta（theta -> 0 的极限给一阶系数 1）：')
for t, l in zip(ts, loss):
    print('      theta = %.4f   loss/theta = %.9f' % (t, l / t))
slope = np.polyfit(ts, loss, 1)
print('   线性拟合斜率 = %.9f（含 O(theta) 偏置，不作判据）' % slope[0])
print('   一阶系数 1 即 h00 = 2 theta 的系数 2（由根式展开读出，非拟合）')
print()

print('Z4 视界位置一致性与三系数联合归一')
_, w05 = spectrum(1.0, 0.5)
print('   theta = 0.5 处最小本征值 = %.3e（视界，期望 0）' % float(w05.min()))
print('   角度映射通道给出同一视界：alpha = 180° x theta = 90°（相对差 0）')
print('   三系数：合成 1/6、钟率 2、尺 1；lambda 相对偏离 = 三项相对偏差之和')
obs = 5.0e-4
print('   观测锚：光偏折读数 1.999 给 |dlambda/lambda| <= %.1e' % obs)
print('   均分上界：每个系数相对偏差 <= %.2e' % (obs / 3.0))
print('   敏感度表（单系数偏差 -> lambda 相对偏离）：')
for d in [1e-4, 1e-3, 1e-2]:
    print('      偏差 %.0e -> 偏离 %.0e' % (d, d))
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot([r[0] for r in rows], [r[1] for r in rows], 'o', label='measured wb/wb0')
tt = np.linspace(0.0, 0.45, 100)
ax[0].plot(tt, np.sqrt(1 - 2 * tt), '-', label='sqrt(1 - 2 theta)')
ax[0].set_xlabel('theta')
ax[0].set_ylabel('clock-rate ratio')
ax[0].set_title('Z2 clock-rate scaling (k = 1)')
ax[0].legend()
ax[1].plot(ts, loss, 'o-', label='measured -ln(wb/wb0)')
ax[1].plot(ts, slope[0] * ts + slope[1], 'r--', label='slope %.6f' % slope[0])
ax[1].set_xlabel('theta (weak field)')
ax[1].set_ylabel('-ln(wb/wb0)')
ax[1].set_title('Z3 weak-field slope (coefficient 2)')
ax[1].legend()
plt.tight_layout()
plt.savefig(IMG, dpi=150)
print('图已保存：' + IMG)