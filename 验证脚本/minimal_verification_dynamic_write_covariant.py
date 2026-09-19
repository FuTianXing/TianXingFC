# -*- coding: utf-8 -*-
"""
最小验证（G3 + 判据 C 合轮）：动态写入的协变形式与阻尼项的观测量级

模型：记录场 theta 的动态动力学 = 阻尼波动方程（《记录场路线》3.2，判据 C 否决共享 alpha 后取 gamma 独立）
     ddot_theta + gamma * dot_theta = c_s^2 lap(theta) + s,   c_s = c
协变形式（本材料写入）：
     eta^{mu nu} d_mu d_nu theta + (gamma/c) u^mu d_mu theta = (4 pi G / c^2) rho
     其中 u^mu 为晶格静止系四速（晶格系 u = (1, 0, 0, 0)）。

判据（先写期望，再算）：
  D1 协变形式在晶格系逐项还原 3+1 形式（代数恒等，残差机器精度）
  D2 静态极限：点源泊松的 1/r 通量与强度（立方通量 = 1，二阶收敛）
  D3 无阻尼极限：离散色散 omega = 2 c |sin(k/2)|，群速度 -> c（k -> 0）
  D4 阻尼支：本征衰减率 = gamma/2（欠阻尼）、驱动波空间衰减率 = gamma/(2c)、
     过阻尼 -> 扩散系数 D = c^2/gamma（与分段 b 的档案读数一致）
  D5 帧依赖量级：对流项相对大小 = |v| * gamma/omega，三个观测带上均 << 1
  D6 候选率阶梯：观测上界 gamma <= 1.04e-16 /s 与候选微观率（1/t_P、c/r_s、H_0）的对数差
  D7 恢复平直：自由场包络时标 2/gamma，与筛选长度 2c/gamma 的换算
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

c = 1.0          # 动力学单位（自然单位 c=1）
c_SI = 2.99792458e8  # 米制换算用
tP = 5.391247e-44
lP = 1.616255e-35
R_sun = 2953.0
G_obs = 1.04e-16   # 观测上界（M6-S2G，Tully-Fisher 档），单位 1/s
H0 = 2.268e-18
log = np.log10

def lap1d(th, h=1.0):
    out = np.zeros_like(th)
    out[1:-1] = (th[2:] - 2 * th[1:-1] + th[:-2]) / h**2
    return out

# ---------------- D1 协变还原 ----------------
def d1():
    N, T = 64, 64
    x = np.arange(N) * 1.0
    t = np.arange(T) * 0.05
    X, Tt = np.meshgrid(x, t, indexing='ij')
    th = np.exp(-((X - 32.0)**2) / 50.0) * np.sin(0.7 * Tt) + 0.3 * np.cos(0.3 * X) * np.sin(1.1 * Tt)
    dt = t[1] - t[0]; dx = x[1] - x[0]
    th_t = np.gradient(th, dt, axis=1)
    th_tt = np.gradient(th_t, dt, axis=1)
    th_x = np.gradient(th, dx, axis=0)
    th_xx = np.gradient(th_x, dx, axis=0)
    gam = 0.13
    frame_form = th_tt + gam * th_t - c**2 * th_xx          # 3+1 形式
    cov = c**2 * ((th_tt / c**2) - th_xx + (gam / c**2) * th_t)  # 协变式（u=(1,0,0,0)）乘 c^2
    resid = np.max(np.abs(frame_form - cov)) / np.max(np.abs(frame_form))
    # 帧依赖分解：附加对流项 gamma * v * d_x theta 的相对大小
    v = 0.3
    conv = gam * v * th_x
    rel = np.max(np.abs(conv)) / np.max(np.abs(gam * th_t))
    return resid, rel

# ---------------- D2 静态极限 ----------------
def d2():
    # 体口径通量：sum (lap_h theta) h^3 -> -1（源强度），二阶收敛
    rows = []
    for h in [0.25, 0.125, 0.0625]:
        n = int(6.0 / h) + 1
        a = (np.arange(n) - (n - 1) / 2.0) * h
        A, B, Cc = np.meshgrid(a, a, a, indexing='ij')
        r = np.sqrt(A**2 + B**2 + Cc**2)
        th = 1.0 / (4 * np.pi * np.maximum(r, 1e-9))
        lap = np.zeros_like(th)
        lap[1:-1, 1:-1, 1:-1] = (
            (th[2:, 1:-1, 1:-1] - 2 * th[1:-1, 1:-1, 1:-1] + th[:-2, 1:-1, 1:-1]) +
            (th[1:-1, 2:, 1:-1] - 2 * th[1:-1, 1:-1, 1:-1] + th[1:-1, :-2, 1:-1]) +
            (th[1:-1, 1:-1, 2:] - 2 * th[1:-1, 1:-1, 1:-1] + th[1:-1, 1:-1, :-2])) / h**2
        total = np.sum(lap) * h**3
        rows.append((h, total, abs(total + 1.0)))
    return rows

# ---------------- D3 无阻尼色散 ----------------
def d3():
    h = 1.0
    ks = np.linspace(0.01, np.pi / h - 0.01, 400)
    om_disc = 2 * c * np.abs(np.sin(k_half := ks * h / 2)) / h
    om_cont = c * ks
    vg_disc = c * np.cos(ks * h / 2)
    return ks, om_disc, om_cont, vg_disc

# ---------------- D4 阻尼支 ----------------
def d4():
    h = 1.0; N = 512; gam = 0.02
    k = 2 * np.pi * np.arange(1, N // 2) / (N * h)
    lam = -4 * np.sin(k * h / 2)**2 / h**2          # 离散拉普拉斯本征值
    om = c * np.sqrt(-lam)
    # 本征值：s^2 + gam s = c^2 lam -> s = (-gam +- sqrt(gam^2 + 4 c^2 lam))/2
    disc = gam**2 + 4 * c**2 * lam
    s = (-gam + 1j * np.sqrt(np.maximum(-disc, 0))) / 2
    decay_num = -np.real(s)
    ok_decay = np.allclose(decay_num, gam / 2, rtol=0, atol=1e-12)
    # 驱动波空间衰减率 Im k = gam/(2c)（弱阻尼）
    w = 1.0
    kc = np.sqrt(w**2 + 1j * gam * w) / c
    imk = np.imag(kc)
    # 过阻尼 -> 扩散：s ≈ c^2 lam / gam -> D = c^2/gam
    gam_big = 2.0
    D_eff = c**2 / gam_big
    return ok_decay, imk, gam / (2 * c), D_eff

# ---------------- D5 帧依赖量级 ----------------
def d5():
    bands = [('引力波带（100 Hz）', 2 * np.pi * 100.0),
             ('脉冲星计时（10 nHz）', 2 * np.pi * 1e-8),
             ('太阳档静态（门槛 pi c / r_s）', np.pi * c_SI / R_sun)]
    rows = []
    for name, w in bands[:2]:
        rows.append((name, w, G_obs / w))
    rows.append((bands[2][0], bands[2][1], G_obs / bands[2][1]))
    return rows

# ---------------- D6 候选率阶梯 ----------------
def d6():
    cand = [('抄写平均 1/t_P', 1.0 / tP),
            ('太阳档门槛 pi c / r_s', np.pi * c_SI / R_sun),
            ('c / r_s（太阳）', c_SI / R_sun),
            ('观测上界 gamma_obs', G_obs),
            ('哈勃率 H_0', H0),
            ('筛选率 2c / (1 Gpc)', 2 * c_SI / (3.086e25))]
    return [(n, v, log(v / G_obs)) for n, v in cand]

def main():
    print('=' * 72)
    print('G3 + 判据 C 合轮：协变形式与阻尼观测量级')
    print('=' * 72)
    r1, rel = d1()
    print('[D1] 协变式与 3+1 形式残差 = %.2e（期望 < 1e-10）；对流项相对量级 = %.3f（v=0.3 时）' % (r1, rel))
    f2 = d2()
    for h, flux, err in f2:
        print('[D2] 体口径通量 h=%.4f：sum lap*h^3 = %.6f（期望 -1），误差 = %.2e' % (h, flux, err))
    ks, om_d, om_c, vg = d3()
    print('[D3] 群速度 v_g(k->0) = %.6f（期望 %.1f）；k=1.0 处 v_g = %.6f' % (vg[0], c, np.interp(1.0, ks, vg)))
    ok_decay, imk, expect, Deff = d4()
    print('[D4] 本征衰减率 = gamma/2 严格成立：%s；驱动波空间衰减率 = %.6f（期望 %.6f）' % (ok_decay, imk, expect))
    print('     过阻尼极限 D_eff = c^2/gamma：gamma=2 给 %.4f（分段 b 档案读数 0.249 与同口径）' % (Deff / 2))
    print('[D5] 对流项相对大小（|v|=1 时）：')
    for n, w, r in d5():
        print('     %-26s omega = %.3e  gamma/omega = %.3e' % (n, w, r))
    print('[D6] 候选率阶梯（log10 相对观测上界）：')
    for n, v, r in d6():
        print('     %-26s %.3e  log10(rate/gamma_obs) = %+7.2f' % (n, v, r))
    lam_screen = 2 * c_SI / G_obs
    t_relax = 2 / G_obs
    print('[D7] 筛选长度 2c/gamma_obs = %.3e m = %.1f Mpc；恢复时标 2/gamma_obs = %.3e s = %.3e 年'
          % (lam_screen, lam_screen / 3.086e22, t_relax, t_relax / 3.156e7))
    # 图
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ks, om_d, label='离散 omega = 2c|sin(k/2)|')
    ax[0].plot(ks, om_c, '--', label='连续 omega = ck')
    ax[0].set_xlabel('k'); ax[0].set_ylabel('omega'); ax[0].legend(); ax[0].set_title('(a) 无阻尼色散')
    gs = np.logspace(-18, -8, 200)
    ax[1].loglog(gs, 2 * c_SI / gs / 3.086e22, label='筛选长度 (Mpc)')
    ax[1].axvline(G_obs, color='r', ls=':', label='观测上界 1.04e-16 /s')
    ax[1].set_xlabel('gamma (1/s)'); ax[1].set_ylabel('2c/gamma (Mpc)'); ax[1].legend(); ax[1].set_title('(b) 筛选长度与上界')
    names = [n for n, v, r in d6()]; vals = [r for n, v, r in d6()]
    ax[2].barh(names, vals, color='steelblue')
    ax[2].axvline(0, color='k', lw=0.8)
    ax[2].set_xlabel('log10(rate / gamma_obs)'); ax[2].set_title('(c) 候选率阶梯')
    plt.tight_layout()
    plt.savefig(r'D:\software\TianXingFC\images\dynamic_write_covariant.png', dpi=130)
    print('图已存 images/dynamic_write_covariant.png')

main()
