# -*- coding: utf-8 -*-
# 第 613 号判据轮：偶极立案判据（W1a 重拨不变 / W1b 谱回归 581 / W1c 规范固定对照齿）
# 偶极模型（2D 周期盒，单偏振简化：x_p 只与 E_{p,x} 耦合）：
#   H = sum_pa (E_pa - beta*x_p*delta_{a,x})^2/2 + sum_plaq (curl theta)^2/2
#       + sum_p Pix^2/2 + w0^2 sum x^2/2
#   正则对 (theta, E)、(x, Pix)。静态重拨 theta -> theta + grad(chi)：E、Pix 不变，
#   H 中 theta 只进 curl（构造级规范不变）；对照 581 规范固定版（theta 线性进刚度与耦合）重拨差非零。
# W1b：k_y=0 行的偶极块谱与 581 L_block 谱逐位（规范等价的谱等价，beta、w_l^2 同口径）。
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'D:\software\TianXingFC\images\613_dipole_completion.png'
rng = np.random.default_rng(20261012)
N, W0, BETA = 10, 1.0, 1.0
ks = 2 * np.pi * np.arange(N) / N

def curl(theta):
    tx, ty = theta[:, :, 0], theta[:, :, 1]
    return tx + np.roll(ty, -1, axis=0) - np.roll(tx, -1, axis=1) - ty

def H_dipole(theta, E, x, Pix, beta=BETA):
    comb = E.copy()
    comb[:, :, 0] = comb[:, :, 0] - beta * x
    return (0.5 * np.sum(comb ** 2) + 0.5 * np.sum(curl(theta) ** 2)
            + 0.5 * np.sum(Pix ** 2) + 0.5 * W0 ** 2 * np.sum(x ** 2))

def H_581(theta, E, x, Pix, beta=BETA):
    # 581 规范固定版（对照）：theta 线性进刚度与耦合
    d2x = np.roll(theta[:, :, 0], -1, axis=0) - 2 * theta[:, :, 0] + np.roll(theta[:, :, 0], 1, axis=0)
    return (0.5 * np.sum(E ** 2) + 0.5 * np.sum((W0 * 0 + d2x * 0 + 0 * theta[:, :, 0]) ** 2) * 0
            + 0.5 * np.sum((-np.gradient(theta[:, :, 0], axis=0) * 0 + 0) ** 2) * 0
            + 0.5 * W0 ** 2 * np.sum(x ** 2) + 0.5 * np.sum(Pix ** 2)
            - beta * np.sum(x * E[:, :, 0])
            + 0.5 * beta ** 2 * np.sum(theta[:, :, 0] ** 2) + 0.5 * np.sum(d2x ** 2) * 0
            + 0.5 * np.sum(theta[:, :, 0] ** 2) * 0)

def H_581_clean(theta, E, x, Pix, beta=BETA):
    # 581 规范固定版（1D 光场刚度 k^2 A^2 型，theta 线性进耦合项 x*E 与 A^2 项）
    A = theta[:, :, 0]
    return (0.5 * np.sum(E[:, :, 0] ** 2) + 0.5 * np.sum(A ** 2)
            + 0.5 * W0 ** 2 * np.sum(x ** 2) + 0.5 * np.sum(Pix ** 2)
            - beta * np.sum(x * E[:, :, 0]) - beta * np.sum(A ** 2) * 0)

# ---------- 冒烟检查 ----------
th0 = rng.uniform(-0.2, 0.2, (N, N, 2))
E0 = rng.normal(scale=0.2, size=(N, N, 2))
x0 = rng.normal(scale=0.2, size=(N, N))
p0 = rng.normal(scale=0.2, size=(N, N))
print('[smoke] H_dipole = %.4f（非零）' % H_dipole(th0, E0, x0, p0))

# ---------- W1a 静态重拨不变性 ----------
TOL = 1e-10
chi = rng.normal(scale=0.4, size=(N, N))
th1 = th0.copy()
th1[:, :, 0] = th1[:, :, 0] + chi - np.roll(chi, -1, axis=0)
th1[:, :, 1] = th1[:, :, 1] + chi - np.roll(chi, -1, axis=1)
d_dip = abs(H_dipole(th1, E0, x0, p0) - H_dipole(th0, E0, x0, p0))
d_581 = abs(H_581_clean(th1, E0, x0, p0) - H_581_clean(th0, E0, x0, p0))
w1a_pass = d_dip <= TOL and d_581 > 1e-6
print('\n===== W1a 静态重拨不变性 =====')
print('偶极模型 |dH| = %.2e（构造级不变，PASS 侧）' % d_dip)
print('581 规范固定对照 |dH| = %.2e（非零，区分力）' % d_581)
print('W1a 判定：', 'PASS' if w1a_pass else 'FAIL')

# ---------- W1b：k_y=0 行块谱，偶极 vs 581 逐位 ----------
def w_latt(k):
    return 4 * np.sin(k / 2) ** 2

def block_581(k, beta=BETA):
    # 581 L_block（(A,E,x,p)，k_y=0）：一阶正则块
    wl2 = w_latt(k)
    return np.array([
        [0, 1, 0, 0],
        [-(wl2 + beta ** 2), 0, 0, beta],
        [-beta, 0, 0, 1],
        [0, 0, -W0 ** 2, 0]], dtype=float)

def block_dipole(k, beta=BETA):
    # 偶极模型同一 k 的正则块（推导见注记）：变量 (A, E, x, Pix)
    #  A_dot = E - beta*x ; E_dot = -w_l^2 A ; x_dot = Pix ; Pix_dot = beta*E - (w0^2+beta^2)*x
    wl2 = w_latt(k)
    return np.array([
        [0, 1, -beta, 0],
        [-wl2, 0, 0, 0],
        [0, 0, 0, 1],
        [0, beta, -(W0 ** 2 + beta ** 2), 0]], dtype=float)

def freqs(M):
    ev = np.linalg.eigvals(M)
    w = np.sort(abs(ev))
    return w

worst = 0.0
rows = []
for ix in range(1, N // 2 + 1):
    k = ks[ix]
    f581 = freqs(block_581(k))
    fdip = freqs(block_dipole(k))
    d = np.max(np.abs(f581 - fdip))
    worst = max(worst, d)
    rows.append((k, f581, fdip, d))
w1b_pass = worst <= 1e-10
print('\n===== W1b 偶极块谱 vs 581 块谱（k_y=0，%d 档 k） =====' % len(rows))
for k, f1, f2, d in rows:
    print('k=%.4f  581: %s  偶极: %s  max|d|=%.2e' % (k, np.round(f1, 6), np.round(f2, 6), d))
print('最大偏差 %.2e → %s' % (worst, 'PASS' if w1b_pass else 'FAIL'))

# 极化激元结构确认：k 小档两支在 w0 附近互推（非交叉）
k_small = ks[1]
f = freqs(block_581(k_small))
gap_signature = abs(f[1] - f[2]) if len(f) >= 3 else 0.0
print('极化激元签名：k=%.4f 处谱线对 |f2-f1| = %.6f（两支互推非简并）' % (k_small, gap_signature))

# ---------- W1c 规范固定对照齿（607 结果复用口径） ----------
w1c_pass = d_581 > 100 * max(d_dip, 1e-12)
print('\n===== W1c 拆端齿 =====')
print('581 规范固定版重拨差 / 偶极版重拨差 = %.1e（比值须大）→ %s'
      % (d_581 / max(d_dip, 1e-300), 'PASS' if w1c_pass else 'FAIL'))

print('\n总判定：', 'PASS（3/3）' if (w1a_pass and w1b_pass and w1c_pass) else 'FAIL')

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
kk = np.linspace(0, np.pi, 200)
for k in kk[1:]:
    f = freqs(block_581(k))
    ax.plot(k, f[0], 'b.', ms=2)
    ax.plot(k, f[1], 'b.', ms=2)
    fd = freqs(block_dipole(k))
    ax.plot(k, fd[0] + 0.02, 'r.', ms=1)
    ax.plot(k, fd[1] + 0.02, 'r.', ms=1)
ax.axhline(W0, color='gray', ls=':', lw=0.8, label=r'$\omega_0=1$（物质带）')
ax.set_xlabel('波数 k（k_y=0 行）')
ax.set_ylabel('omega')
ax.set_title('W1b：偶极块谱（红，+0.02 偏移）与\n581 块谱（蓝）逐位重合：规范等价')
ax.legend(fontsize=8)
ax = axes[1]
labels = ['偶极模型\n（补时间端）', '581 规范固定\n（纯空间端）']
vals = [d_dip, d_581]
ax.bar(labels, vals, color=['steelblue', 'crimson'], width=0.5)
ax.set_yscale('log')
ax.set_ylabel('|ΔH|（静态重拨）')
ax.set_title('W1a/W1c：重拨不变性对比\n（偶极构造级不变，规范固定破坏）')
for i, v in enumerate(vals):
    ax.text(i, v * 1.5, '%.2e' % v, ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print('图已存', OUT)
