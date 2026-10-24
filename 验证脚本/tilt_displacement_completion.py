# -*- coding: utf-8 -*-
"""
tilt_displacement_completion.py

倾角与位移完成的最小验证（第 446 条）：写入时刻约束、两机制分离与两层位置。

判据：
  DC1（写入落点与中心在轴）：判据 L 分配规则框架下，站点写入（源 1.0 在单站点，
      六链接各承 1/6）的六邻居读数各向同性（"中心在轴"的离散读数）；链接中点
      写入的离散对应（源按该链接两端各承一半：0.5 @ c 与 0.5 @ c+ex）的写入
      邻域各向异性显著（"中心偏轴"的离散对应）。
  DC2（两机制分离）：取向偏角 alpha = 180°*theta 扫描。
      停滞机制（投影模型）：法线轴位移分量 cos(alpha) 的零点在 90°；离散螺旋
      N 拍累计：alpha = 90° 时轴向累计 |Z_N| ~ 0（停滞）、横向累计线性增长（脱轴）；
      alpha = 180° 时轴向累计 Z_N = -N（反向）、横向为零（崩溃）。
      两失效点分离（90° ≠ 180°），单阈值模型不完备，两层结构必要。
  DC3（两层位置）：史瓦西对齐 theta(r) = r_s/(2r) 下，theta = 1/2（90° 几何角，
      线性映射 alpha = 180°*theta）的面在 r = r_s（视界），theta = 1（180°）的面
      在 r = r_s/2（视界深处），半径比值恰为 2（代数恒等的浮点核对）。

模型设定（建模决策显式声明）：
  - 投影模型：位移沿法线轴的分量按取向偏角投影（cos），是"法线轴上的位移"的
    自然几何读法；180° 崩溃（取向-法线点积 = -1）不依赖投影模型（定义性）。
  - DC1 是作者连续图像（中心脱轴）在判据 L 站点分配框架上的离散对应，对称性
    即"中心在轴"的离散读数；连续偏轴模型的第一原理形式不在本轮射程内。
  - 离散螺旋：本体每拍沿取向方向前进 l_P（取向偏 alpha 时轴向分量 l_P*cos(alpha)），
    横向分量累积为横向漂移；相位每拍满周 2π（P1）。
  - DC1 周期域 FFT 谱解（去零模）：与判据 W/N 相同的周期镜像残差来源；对称性
    判据不受镜像影响（源居中，镜像保持对称）。

边界声明：验证两机制分离的几何结构与离散对应；不验证真实强场动力学、不裁决
连续偏轴模型的第一原理形式、不给出 90°/180° 的最终裁决（判据支持两层结构，
承载退出阈值绑定 180° 的结论是结构自洽论证加判据，非决定性实验）。

可逆性等级：判据敏感（投影模型更换会移动 90° 停滞点；180° 崩溃不受影响）。
协议伪影与物理结论分层：DC1 各向异性量值 O(0.5) 属模型内读数（非物理预测），
其结构（站点对称、链接各向异性）为判据内容。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R_NS = 1.0  # r_s 取 1（无量纲），结构不依赖标度
results = []

# ---------- DC1：写入落点与中心在轴（FFT 谱解离散泊松） ----------
NG = 48
idx = np.arange(NG)
kx = np.fft.fftfreq(NG, d=1.0) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
lam = 6.0 - 2.0 * (np.cos(KX) + np.cos(KY) + np.cos(KZ))  # -laplacian 特征值
lam[0, 0, 0] = 1.0  # 去零模（中性背景）

def solve_poisson(src):
    s = src - src.mean()
    return np.real(np.fft.ifftn(np.fft.fftn(s) / lam))

src_site = np.zeros((NG, NG, NG))
c = NG // 2
src_site[c, c, c] = 1.0
u_site = solve_poisson(src_site)

six_site = np.array([u_site[c+1, c, c], u_site[c-1, c, c], u_site[c, c+1, c],
                     u_site[c, c-1, c], u_site[c, c, c+1], u_site[c, c, c-1]])
spread_site = (six_site.max() - six_site.min()) / abs(six_site.mean())
results.append(('DC1a', '站点写入六邻居相对散布', spread_site, '<= 1e-10',
                spread_site <= 1e-10))

src_link = np.zeros_like(src_site)
src_link[c, c, c] = 0.5
src_link[c+1, c, c] = 0.5
u_link = solve_poisson(src_link)
six_link = np.array([u_link[c+1, c, c], u_link[c-1, c, c], u_link[c, c+1, c],
                     u_link[c, c-1, c], u_link[c, c, c+1], u_link[c, c, c-1]])
spread_link = (six_link.max() - six_link.min()) / abs(six_link.mean())
results.append(('DC1b', '链接写入六邻居相对散布（各向异性）', spread_link, '>= 0.1',
                spread_link >= 0.1))

# ---------- DC2：两机制分离 ----------
grid = np.linspace(0.0, np.pi, 1801)
cosg = np.cos(grid)
half = len(grid) // 2 + 1
i90 = np.argmin(np.abs(cosg[:half]))
alpha_zero = grid[i90]
results.append(('DC2a', '位移分量零点（停滞阈值）', alpha_zero, 'pi/2 ± 1e-3',
                abs(alpha_zero - np.pi/2) < 1e-3))

N = 1000
lP = 1.0

def spiral(alpha, n):
    return n * lP * np.cos(alpha), n * lP * np.sin(alpha)

Z90, D90 = spiral(np.pi/2, N)
Z180, D180 = spiral(np.pi, N)
results.append(('DC2b', 'alpha=90° 累计轴向 |Z_N|/N', abs(Z90)/N, '<= 1e-12',
                abs(Z90)/N <= 1e-12))
results.append(('DC2c', 'alpha=90° 横向漂移 D_N/N', D90/N, '≈ 1（线性脱轴）',
                abs(D90/N - 1.0) < 1e-12))
results.append(('DC2d', 'alpha=180° 累计轴向 Z_N/N', Z180/N, '≈ -1（反向）',
                abs(Z180/N + 1.0) < 1e-12))
results.append(('DC2e', 'alpha=180° 横向漂移 D_N/N', D180/N, '<= 1e-12',
                abs(D180/N) <= 1e-12))
sep = abs(np.pi/2 - np.pi)
results.append(('DC2f', '两失效点分离（90° 停滞 ≠ 180° 崩溃）', sep, '≈ pi/2',
                abs(sep - np.pi/2) < 1e-12))

# ---------- DC3：两层位置 ----------
def theta_r(r):
    return R_NS / (2.0 * r)

def bisect(target, lo, hi, it=200):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if theta_r(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

r_90 = bisect(0.5, 0.2, 5.0)
r_180 = bisect(1.0, 0.05, 1.5)
results.append(('DC3a', 'theta=1/2 面（90° 几何角）半径 / r_s', r_90 / R_NS, '≈ 1（视界）',
                abs(r_90 / R_NS - 1.0) < 1e-10))
results.append(('DC3b', 'theta=1 面（180°）半径 / r_s', r_180 / R_NS, '≈ 0.5（视界深处）',
                abs(r_180 / R_NS - 0.5) < 1e-10))
results.append(('DC3c', '两层半径比值 r(90°)/r(180°)', r_90 / r_180, '≈ 2（代数恒等）',
                abs(r_90 / r_180 - 2.0) < 1e-10))

# ---------- 打印 ----------
print('=' * 72)
print('倾角与位移完成的最小验证（第 446 条）：DC1 至 DC3')
print('=' * 72)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-34s 实测 %.12g  期望 %s  %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 72)
print('判据 %d/%d 通过' % (n_pass, len(results)))
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(np.degrees(grid), cosg, lw=2)
ax.axvline(90, color='C1', ls='--', lw=1, label='90° stagnation (cos=0)')
ax.axvline(180, color='C3', ls='-.', lw=1, label='180° collapse (reversal)')
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('tilt alpha (deg)')
ax.set_ylabel('axial displacement weight cos(alpha)')
ax.set_title('(a) projection model: stagnation vs collapse')
ax.legend(fontsize=8)

ax = axes[1]
Ns = np.arange(1, N + 1)
for a, cc in [(np.radians(60), 'C0'), (np.pi/2, 'C1'), (np.pi, 'C3')]:
    ax.plot(Ns, Ns * lP * np.cos(a), color=cc, ls='-', lw=1.5,
            label='axial, %d°' % round(np.degrees(a)))
    ax.plot(Ns, Ns * lP * np.sin(a), color=cc, ls=':', lw=1.2,
            label='lateral, %d°' % round(np.degrees(a)))
ax.set_xlabel('beat n')
ax.set_ylabel('cumulative displacement (l_P)')
ax.set_title('(b) discrete spiral accumulation (N=1000)')
ax.legend(fontsize=7)

ax = axes[2]
rr = np.linspace(0.3, 3.0, 600)
ax.plot(rr, theta_r(rr), lw=2, label='theta(r) = r_s/(2r)')
ax.axhline(0.5, color='C1', ls='--', lw=1)
ax.axhline(1.0, color='C3', ls='-.', lw=1)
ax.axvline(1.0, color='C1', ls='--', lw=1)
ax.axvline(0.5, color='C3', ls='-.', lw=1)
ax.set_xlabel('r / r_s')
ax.set_ylabel('theta(r)')
ax.set_title('(c) two layers: horizon (90°) vs full-tilt (180°)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\tilt_displacement_completion.png', dpi=150)
print('figure saved: images/tilt_displacement_completion.png')

# ---------- 图 2：DC1 两种写入的六邻居读数对比 ----------
fig2, ax2 = plt.subplots(figsize=(6, 4.2))
labels = ['+x', '-x', '+y', '-y', '+z', '-z']
xpos = np.arange(6)
ax2.bar(xpos - 0.2, six_site / abs(six_site).max(), width=0.4,
        label='site write (isotropic)', color='C0')
ax2.bar(xpos + 0.2, six_link / abs(six_link).max(), width=0.4,
        label='link write (anisotropic)', color='C3')
ax2.set_xticks(xpos)
ax2.set_xticklabels(labels)
ax2.set_ylabel('neighbor reading (normalized)')
ax2.set_title('DC1: site write symmetric, link write anisotropic')
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\tilt_displacement_dc1.png', dpi=150)
print('figure saved: images/tilt_displacement_dc1.png')
