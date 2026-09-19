# -*- coding: utf-8 -*-
"""
最小验证：h0i 的链接级合成（动量读数项的度规组装与拖曳强度；判据 K1 至 K5）

模型：写事件（拍）沿缺陷世界线落在链接上，每笔写入携带两个读数：
  标量读数（质量）：delta m_s，核 1/|x - x_s|，单位源给出 theta = m/r（G = c = 1）；
  矢量读数（动量）：delta m_s * v_s，同一核，给 Theta_i = sum_s m_s v_{i,s}/r_s。
度规组装（相对论约定，见判据 K2 与注记第三节）：h00 = 2 theta，h0i = 4 Theta_i。
标量侧链接合成已由 G6 给出（判据 J1 至 J4，同口径：单位源在链接层给 1/r，源强 4pi 归一）。

判据（先写期望，再算）：
  K1 静止极限：v = 0 时矢量读数逐点严格为零（与判据 V 的静态零读数相容）
  K2 入口唯一：h0i/(h00 v) = 2，与速率无关（结构性恒等：单一入口只出现一次）
  K3 各向同性：速度沿 [100]/[110]/[111] 给同一强度；对照立方各向异性键（6:3:2，已排除）
  K4 转动源偶极：Theta_phi = J/(2 rho^2)（远场），径向斜率 -2，角依赖 sin(theta)
  K5 回路签名：绕源的闭合回路积分 = 4 pi J / rho，离散样本与连续求积一致
"""
import numpy as np

def theta_field(P, X, w):
    d = P[:, None, :] - X[None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    return (w[None, :] / r).sum(-1)

def vec_field(P, X, w, V):
    d = P[:, None, :] - X[None, :, :]
    r = np.sqrt((d ** 2).sum(-1))
    return (w[None, :, None] * V[None, :, :] / r[:, :, None]).sum(1)

def ring_source(R, M, v, N, plane='xy'):
    ph = 2 * np.pi * (np.arange(N) + 0.5) / N
    X = np.stack([R * np.cos(ph), R * np.sin(ph), np.zeros(N)], axis=1)
    V = np.stack([-v * np.sin(ph), v * np.cos(ph), np.zeros(N)], axis=1)
    w = np.full(N, M / N)
    return X, w, V

def quad_ring(R, M, v, rho, nq=200000):
    ph = 2 * np.pi * (np.arange(nq) + 0.5) / nq
    xp = R * np.cos(ph); yp = R * np.sin(ph)
    r = np.sqrt((rho - xp) ** 2 + yp ** 2)
    lam = M / (2 * np.pi * R)
    cph = np.cos(ph); sph = np.sin(ph)
    Theta_x = lam * R * (-v * sph) / r
    Theta_y = lam * R * (v * cph) / r
    return (2 * np.pi / nq) * Theta_x.sum(), (2 * np.pi / nq) * Theta_y.sum()

def main():
    print('=' * 78)
    print('h0i 的链接级合成：动量读数项的度规组装与拖曳强度（判据 K1 至 K5）')
    print('=' * 78)
    P = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0], [2.0, 2.0, 1.0]])
    X0 = np.array([[0.0, 0.0, 0.0]]); w0 = np.array([1.0])

    # K1 静止极限
    V0 = np.array([[0.0, 0.0, 0.0]])
    Th = vec_field(P, X0, w0, V0)
    print('[K1] 静止极限 v = 0：矢量读数最大绝对值 = %.2e（期望 0，机器零）' % np.abs(Th).max())

    # K2 入口唯一
    print('[K2] 入口唯一：h0i/(h00 v) 期望 2（结构性恒等，单一入口）')
    for v in (0.01, 0.1, 0.5):
        V = np.array([[v, 0.0, 0.0]])
        th = theta_field(P, X0, w0)
        tv = vec_field(P, X0, w0, V)[:, 0]
        ratio = (4 * tv) / (2 * th * v)
        print('     v = %.2f c：比值 = %.15f（偏差 %.2e）' % (v, ratio[0], abs(ratio[0] - 2.0)))

    # K3 各向同性（含对照：立方各向异性键的取向权重）
    print('[K3] 各向同性：同一 |v| 沿三取向的矢量强度')
    dirs = {'[100]': np.array([1.0, 0, 0]), '[110]': np.array([1, 1.0, 0]) / np.sqrt(2), '[111]': np.ones(3) / np.sqrt(3)}
    mags = {}
    for name, u in dirs.items():
        V = (0.1 * u)[None, :]
        tv = vec_field(P, X0, w0, V)[0]
        mags[name] = np.linalg.norm(tv)
    ref = mags['[100]']
    for name in dirs:
        print('     %-6s 强度 = %.15e，相对偏差 %.2e' % (name, mags[name], abs(mags[name] / ref - 1.0)))
    weights = {name: float(np.sum(u ** 4)) for name, u in dirs.items()}
    print('     对照（立方各向异性键，取向权重 sum n_a^4）：[100] %.4f / [110] %.4f / [111] %.4f = 6:3:2' % (weights['[100]'], weights['[110]'], weights['[111]']))
    print('     该键在 [100] 方向横波刚度归零（2026-09-19 构造性判定），故框架取等向键；本条即其后果')

    # K4 转动源偶极
    R, M, v = 1.0, 1.0, 0.1
    J = M * v * R
    print('[K4] 转动源（R = %.1f，J = MvR = %.3f）：Theta_phi = J/(2 rho^2)（远场）' % (R, J))
    Pf = np.array([[rho, 0.0, 0.0] for rho in (5.0, 10.0, 20.0)])
    Xr, wr, Vr = ring_source(R, M, v, 4096)
    tv = vec_field(Pf, Xr, wr, Vr)
    analytic = J / (2 * np.array([5.0, 10.0, 20.0]) ** 2)
    for rho, got, want in zip((5.0, 10.0, 20.0), tv[:, 1], analytic):
        print('     rho = %5.1f：离散 %.10e，解析 %.10e，相对偏差 %.2e' % (rho, got, want, abs(got / want - 1.0)))
    rhos = np.array([8.0, 16.0, 32.0, 64.0])
    Pp = np.array([[rho, 0.0, 0.0] for rho in rhos])
    vals = vec_field(Pp, Xr, wr, Vr)[:, 1]
    slope = np.polyfit(np.log(rhos), np.log(np.abs(vals)), 1)[0]
    print('     径向衰减斜率（rho = 8 至 64）= %.4f（期望 -2）' % slope)
    for th in (90.0, 60.0, 30.0):
        t = np.deg2rad(th)
        p = np.array([[30.0 * np.sin(t), 0.0, 30.0 * np.cos(t)]])
        got = vec_field(p, Xr, wr, Vr)[0, 1]
        want = J * np.sin(t) / (2 * 30.0 ** 2)
        print('     极端角 theta = %2.0f deg：实测 %.6e，解析 %.6e，相对偏差 %.2e（有限尺寸修正约 (R/r)^2 = %.1e）' % (th, got, want, abs(got / want - 1.0), (R / 30.0) ** 2))
    print('     离散-连续：环采样的中点求和在解析周期被积函数上超代数收敛')
    qx, qy = quad_ring(R, M, v, 5.0)
    for N in (16, 32, 64, 256):
        Xn, wn, Vn = ring_source(R, M, v, N)
        got = vec_field(np.array([[5.0, 0.0, 0.0]]), Xn, wn, Vn)[0, 1]
        print('       N = %4d：Theta_y = %.12e，与求积偏差 %.3e' % (N, got, abs(got / qy - 1.0)))
    print('     偶极律的有限尺寸修正：取相对偏差与 (R/rho)^2 拟合，斜率即修正系数')
    rho_fit = np.array([4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 32.0])
    got_fit = vec_field(np.array([[r, 0.0, 0.0] for r in rho_fit]), Xr, wr, Vr)[:, 1]
    dev = got_fit / (J / (2 * rho_fit ** 2)) - 1.0
    xfit = (R / rho_fit) ** 2
    slope2 = np.polyfit(xfit, dev, 1)[0]
    print('       斜率 = %.4f（非零且为 O(1) 常数即几何修正，不引入新系数）' % slope2)

    # K5 回路签名
    print('[K5] 回路签名：绕源闭合回路积分（解析 4 pi J / rho）与标量侧对照')
    nq = 256
    ph = 2 * np.pi * (np.arange(nq) + 0.5) / nq
    for rho in (5.0, 10.0, 20.0):
        loop = np.stack([rho * np.cos(ph), rho * np.sin(ph), np.zeros(nq)], axis=1)
        h = 4 * vec_field(loop, Xr, wr, Vr)
        dl = np.stack([-np.sin(ph), np.cos(ph), np.zeros(nq)], axis=1) * (2 * np.pi * rho / nq)
        integral = np.sum(h * dl)
        qy5 = quad_ring(R, M, v, rho)[1]
        exact = 4 * 2 * np.pi * rho * qy5
        print('     rho = %5.1f：回路积分 = %.10e，连续求积 %.10e，相对偏差 %.2e；解析 4 pi J/rho = %.6e' % (rho, integral, exact, abs(integral / exact - 1.0), 4 * np.pi * J / rho))
    th_loop = theta_field(loop, Xr, wr)
    tele = np.sum(np.diff(np.append(th_loop, th_loop[0])))
    print('     标量侧对照：标量链接读数沿回路的望远镜求和 = %.3e（恒为零，G6 J1）；矢量侧同一回路为上面的非零值' % tele)
    print('     拖曳读法：omega = h_phi/rho = 2 J/rho^3（G = c = 1）；GPB 的广义相对论预期 -39.2 mas/yr 由此继承（登记项，见注记）')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rr = np.linspace(3.0, 40.0, 200)
    Pq = np.array([[r, 0.0, 0.0] for r in rr])
    ax[0].loglog(rr, np.abs(vec_field(Pq, Xr, wr, Vr)[:, 1]), 'o', ms=3, label='离散环源（N = 4096）')
    ax[0].loglog(rr, J / (2 * rr ** 2), '--', label='解析偶极 J/(2 rho^2)')
    ax[0].set_xlabel('rho'); ax[0].set_ylabel('Theta_phi'); ax[0].legend()
    ax[0].set_title('(a) 转动源的矢量读数：偶极 1/rho^2')
    ax[1].plot(xfit, dev, 'o', label='实测相对偏差')
    ax[1].plot(xfit, slope2 * xfit, '--', label='线性参考（斜率 %.3f）' % slope2)
    ax[1].set_xlabel('(R / rho)^2'); ax[1].set_ylabel('与偶极律的相对偏差'); ax[1].legend()
    ax[1].set_title('(b) 有限尺寸修正：偏差随 (R/rho)^2 线性')
    plt.tight_layout()
    plt.savefig(r'D:\software\TianXingFC\images\h0i_link_synthesis.png', dpi=130)
    print('图已存 images/h0i_link_synthesis.png')

main()
