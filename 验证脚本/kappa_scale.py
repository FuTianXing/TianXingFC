# -*- coding: utf-8 -*-
"""
最小验证（kappa 的微观值）：候选阶梯、结构上界与可观测性判定

背景：校准通道的动力学形式已定（相位表扇区的锁相弛豫，判据 C1 至 C3），
      缺的是校准率 kappa 的定标。本材料给候选阶梯、结构上界、锁定条件与可观测性判定。

判据（先写期望，再算）：
  K1 候选阶梯：碰撞率（每摆程一次校准）、omega_P、1/t_P、H_0、3H_0
  K2 结构上界：kappa < omega_P（校准不得快于摆程本身，否则摆程定义失效）
  K3 锁定条件：kappa 须超过失同步源率；给出允许窗口的跨度
  K4 可观测性：分扇区使 kappa 不进色散；唯一可能签名是节拍噪声，量级 ∝ 1/kappa
"""
import numpy as np

tP = 5.391247e-44
omegaP = 2 * np.pi / tP
H0 = 2.2685e-18

def main():
    print('=' * 76)
    print('kappa 的微观值：候选阶梯、结构上界与可观测性')
    print('=' * 76)
    print('[K1] 候选阶梯（每秒）：')
    for n, v in [('碰撞率 1/t_P（每摆程一次校准）', 1 / tP), ('基模角频率 omega_P', omegaP),
                 ('落位率 3H_0', 3 * H0), ('哈勃率 H_0', H0)]:
        print('     %-28s %.4e   （相对 omega_P：%.3e）' % (n, v, v / omegaP))
    print('[K2] 结构上界：kappa < omega_P。取 kappa = 1/t_P 时 kappa/omega_P = %.4f（留 2 pi 倍余量）'
          % ((1 / tP) / omegaP))
    print('[K3] 锁定条件：kappa 须超过失同步源率。')
    for n, v in [('最慢源：宇宙年龄尺度 H_0', H0), ('最快源：摆程本身 1/t_P', 1 / tP)]:
        print('     %-24s 源率 %.3e /s ⇒ kappa 下界 %s' % (n, v, '与上界冲突' if v >= omegaP else '远低于上界'))
    print('     允许窗口：从 H_0 到 omega_P 跨 %.1f 个数量级；观测不能在此窗口内定值。' % (np.log10(omegaP / H0)))
    # 模型内读数：kappa = 1/t_P 时，域尺度 L 的锁相恢复时间
    print('[K3附] 取 kappa = 1/t_P，扩散律 tau ~ (L/2pi)^2 摆程：')
    for L in [16, 64, 1024]:
        tau = (L / (2 * np.pi))**2
        print('     域 L = %4d：恢复时间 %.1f 摆程 = %.2e 秒' % (L, tau, tau * tP))
    print('[K4] 可观测性：分扇区把 kappa 隔离在波扇区之外（判据 C2），故色散中无 kappa。')
    print('     唯一可能签名是节拍噪声：噪声幅度 ~ 1/kappa。取 kappa = 1/t_P ⇒ 每摆程校准一次，')
    print('     无累积；若 kappa 远小于 1/t_P，则累积噪声 ~ t/t_P * (kappa t_P)^{-1}，')
    print('     与"全体元相子时钟层完全同步、不可观测"的结构口径对照。')
    print('     判据形态：若节拍噪声可测且随 1/kappa 增大，则给出 kappa 的观测定值；')
    print('     若时钟层不可观测（框架口径），则 kappa 属结构性定标，观测无从定值。')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ['1/t_P', 'omega_P', '3H_0', 'H_0']
    vals = [1 / tP, omegaP, 3 * H0, H0]
    ax[0].barh(names, np.log10(vals))
    ax[0].axvline(np.log10(omegaP), color='r', ls='--', label='结构上界 omega_P')
    ax[0].set_xlabel('log10(kappa / s^-1)'); ax[0].legend(); ax[0].set_title('(a) 候选阶梯与上界')
    Ls = np.logspace(1, 4, 200)
    ax[1].loglog(Ls, (Ls / (2 * np.pi))**2, label='恢复时间（摆程）')
    ax[1].set_xlabel('域尺度 L（格）'); ax[1].set_ylabel('恢复时间（摆程）')
    ax[1].legend(); ax[1].set_title('(b) 锁相恢复时间（kappa = 1/t_P）')
    plt.tight_layout()
    plt.savefig(r'D:\software\TianXingFC\images\kappa_scale.png', dpi=130)
    print('图已存 images/kappa_scale.png')

main()
