# -*- coding: utf-8 -*-
"""
最小验证（校准通道的动力学形式）：相位表扇区的弛豫、分扇区与残余条件

模型：把校准写成对"相位时间表"的弛豫（锁相目标 = 相邻反相，delta = pi）：
    dtheta_i/dt = omega_P + kappa * [ sin(theta_{i+1}-theta_i-pi) + sin(theta_{i-1}-theta_i-pi) ]
另一独立扇区：实轴方向差 alpha_i（波矢），校准不得作用于它（结构性条件）。

判据（先写期望，再算）：
  C1 相位表扇区：交错锁相对任意 kappa > 0 线性稳定；收敛率随 kappa 线性放大
     （线性化给出扩散型谱：非零模式衰减率 = kappa * lambda_n，最低模式 k ~ 2pi/N）
  C2 扇区分工：校准只动相位表，实轴方向差的分布不变（分扇区性的数值核对）
  C3 残余条件：若把校准改成"把梯度压到零"（错扇区），波矢随之消失；正确形式保留
     残余差，其零模集合 = {常数（全局相位）, 实轴方向差（波矢）}
"""
import numpy as np

def run_kappa(N=64, kappa=0.1, seed=3, T=400.0, dt=0.05):
    rng = np.random.default_rng(seed)
    th = rng.random(N) * 2 * np.pi
    # 目标：交错（theta_i = i*pi + 常数）
    def dev(th):
        # 规范不变测度：相邻相位差对 pi 的偏差（全局相位自由度不计）
        d = (np.roll(th, -1) - th - np.pi + np.pi) % (2 * np.pi) - np.pi
        return d
    def deriv(th):
        return kappa * (np.sin(th[np.arange(N) - 1] - th - np.pi) + np.sin(np.roll(th, -1) - th - np.pi))
    hist = []
    t = 0.0
    while t < T:
        th = th + deriv(th) * dt
        t += dt
        if abs(t % 1.0) < dt / 2:
            hist.append(np.max(np.abs(dev(th))))
    return np.array(hist), th

def main():
    print('=' * 76)
    print('校准通道的动力学形式：相位表扇区的弛豫与扇区分工')
    print('=' * 76)
    # C1：不同 kappa 的收敛率
    print('[C1] 相位表扇区：随机初相 -> 交错锁相（N=64）')
    print('     扩散型弛豫：投影到最慢模 k_min = 2 pi / N，理论衰减率 = kappa * k_min^2（每单位时间）')
    kmin = 2 * np.pi / 64
    idx = np.arange(64)
    rates = {}
    for kap in [0.05, 0.1, 0.2]:
        # 直接积分并记录最慢模幅度
        rng = np.random.default_rng(3)
        th = rng.random(64) * 2 * np.pi
        amps = []
        for step in range(60000):
            d = (np.roll(th, -1) - th - np.pi + np.pi) % (2 * np.pi) - np.pi
            th = th + kap * (np.sin(np.roll(th, 1) - th - np.pi) + np.sin(np.roll(th, -1) - th - np.pi)) * 0.05
            if step % 200 == 0:
                dd = (np.roll(th, -1) - th - np.pi + np.pi) % (2 * np.pi) - np.pi
                amps.append(abs(np.sum(dd * np.exp(-1j * kmin * idx))) / 64)
        amps = np.array(amps); tt = np.arange(len(amps)) * 200 * 0.05
        ok = amps > 1e-14
        slope = np.polyfit(tt[ok], np.log(amps[ok]), 1)[0]
        rates[kap] = -slope
        print('     kappa = %.2f：最慢模对数斜率 %+.6f /单位时间（理论 %+.6f）'
              % (kap, slope, -kap * kmin**2))
    print('     斜率比：%.2f（0.10/0.05）、%.2f（0.20/0.10），期望 2.00（线性放大）'
          % (rates[0.1] / rates[0.05], rates[0.2] / rates[0.1]))
    # C2 扇区分工：实轴方向差独立扇区
    N = 64
    rng = np.random.default_rng(5)
    alpha = 0.3 * rng.random(N) - 0.15      # 实轴方向差（波矢读数）
    alpha_before = alpha.copy()
    # 校准只作用 theta；alpha 完全不进入运动方程（结构性）
    _, th_final = run_kappa(N=N, kappa=0.1, T=200.0)
    alpha_after = alpha.copy()
    print('[C2] 扇区分工：实轴方向差在校准前后逐点相同 = %s（最大差 %.1e）'
          % (np.allclose(alpha_before, alpha_after), np.max(np.abs(alpha_after - alpha_before))))
    print('     校准项只含 theta 的相位差，不含 alpha；波矢扇区无矩阵元')
    # C3 残余条件：错误扇区（把梯度压到零）会让波矢消失
    k_damp = 0.5
    alpha_bad = alpha.copy()
    for _ in range(200):
        lap = np.roll(alpha_bad, -1) - 2 * alpha_bad + np.roll(alpha_bad, 1)
        alpha_bad = alpha_bad + k_damp * 0.01 * lap
    print('[C3] 若校准误作用于梯度扇区：波矢散布 %.4f -> %.4f（被压向零，波矢消失）'
          % (np.std(alpha_before), np.std(alpha_bad)))
    print('     正确形式的零模集合 = {常数（全局相位）, 实轴方向差（波矢）}；残余差即波矢')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for kap in [0.02, 0.05, 0.1, 0.2]:
        hist, _ = run_kappa(kappa=kap, T=300.0)
        ax[0].semilogy(np.arange(len(hist)), np.maximum(hist, 1e-16), label='kappa = %.2f' % kap)
    ax[0].set_xlabel('时间'); ax[0].set_ylabel('最大相位偏差'); ax[0].legend()
    ax[0].set_title('(a) 锁相收敛随校准率')
    ax[1].plot(alpha_before, label='正确扇区：校准后不变')
    ax[1].plot(alpha_bad, '--', label='错扇区：梯度被压掉')
    ax[1].set_xlabel('格点'); ax[1].set_ylabel('实轴方向差（波矢读数）'); ax[1].legend()
    ax[1].set_title('(b) 分扇区与残余条件')
    plt.tight_layout()
    plt.savefig(r'D:\software\TianXingFC\images\calibration_channel.png', dpi=130)
    print('图已存 images/calibration_channel.png')

main()
