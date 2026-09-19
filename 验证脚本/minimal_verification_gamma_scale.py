# -*- coding: utf-8 -*-
"""
最小验证（gamma 的量级判定）：记录阻尼来源的候选判定与所需观测精度

背景：判据 C 的量级侧已收口（观测上界 gamma <= 1.04e-16 /s，GW170817 汽笛对照），
      来源侧收窄为多体与场耦合一条通道。本材料判定该通道最经济的候选：
      膨胀落位率 3 H_0（框架内已有本体位置：落位概率 p = 3 H_0 t_P）。

判据（先写期望，再算）：
  H1 换算：上界对 3 H_0 的比、对每拍的比
  H2 候选预测：gamma = c_gamma * 3 H_0，给出当前允许的 c_gamma 上限
  H3 灵敏度：上界 gamma <= 2 c sigma_ln / D 所需 sigma_ln（当前精度与事件数、以及距离档）
  H4 相邻候选对照：抄写平均率、太阳档门槛、哈勃率相对上界的对数差
  H5 未来带：LISA/ET 级汽笛（Gpc 距离）可达的 gamma 灵敏度与可排除的 c_gamma
"""
import numpy as np

c = 2.99792458e8
tP = 5.391247e-44
Mpc = 3.0856775814913673e22
G_obs_TF = 1.04e-16      # Tully-Fisher 档上界（M6-S2G）
G_obs_FP = 7.89e-17      # 基本面档 1 sigma 上界
H0 = 70.0 * 1e3 / Mpc    # 70 km/s/Mpc -> 1/s
D_GW = 40.0 * Mpc        # GW170817 汽笛距离档
sig_ln = 0.1005          # 距离一致度散布（M6-S2G 基本面档读数）

def main():
    print('=' * 76)
    print('gamma 的量级判定：膨胀落位率候选与所需观测精度')
    print('=' * 76)
    r3H0 = 3 * H0
    print('[H1] H_0 = %.4e /s；3H_0 = %.4e /s' % (H0, r3H0))
    print('     上界/3H_0 = %.2f；上界*t_P = %.2e；3H_0*t_P = %.2e（= 落位概率 p）'
          % (G_obs_TF / r3H0, G_obs_TF * tP, r3H0 * tP))
    print('[H2] 候选 gamma = c_g x 3H_0：当前允许 c_g <= %.1f（TF 档）、<= %.1f（基本面档 1 sigma）'
          % (G_obs_TF / r3H0, G_obs_FP / r3H0))
    # H3
    def sig_req(gamma, D):
        return gamma * D / (2 * c)
    print('[H3] 灵敏度关系 gamma <= 2 c sigma_ln / D：')
    print('     排除 c_g = 1（gamma = %.2e /s）所需 sigma_ln = %.4f（D = 40 Mpc），当前 %.4f，需改善 %.1f 倍'
          % (r3H0, sig_req(r3H0, D_GW), sig_ln, sig_ln / sig_req(r3H0, D_GW)))
    print('     等价事件数（统计精度按 1/sqrt(N)）：约 %.0f 个同精度汽笛'
          % (sig_ln / sig_req(r3H0, D_GW)) ** 2)
    print('     或同等精度下所需距离：%.0f Mpc' % (D_GW * sig_ln / sig_req(r3H0, D_GW) / Mpc))
    # H4
    cand = [('抄写平均 1/t_P', 1 / tP), ('太阳档门槛 pi c / r_s', np.pi * c / 2953.0),
            ('上界（TF 档）', G_obs_TF), ('落位率 3H_0', r3H0), ('哈勃率 H_0', H0)]
    print('[H4] 候选率对照（相对上界的 log10）：')
    for n, v in cand:
        print('     %-22s %.3e   log10(rate/upper) = %+7.2f' % (n, v, np.log10(v / G_obs_TF)))
    # H5
    print('[H5] 未来带（同精度 sigma_ln = %.3f）：' % sig_ln)
    for name, D in [('ET 级 300 Mpc', 300 * Mpc), ('LISA 级 1 Gpc', 1e3 * Mpc), ('LISA 级 3 Gpc', 3e3 * Mpc)]:
        gmax = 2 * c * sig_ln / D
        print('     %-16s 可达上界 gamma <= %.3e /s，可排除 c_g >= %.2f' % (name, gmax, gmax / r3H0))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog([1e-3, 1e-2, 1e-1, 1.0], [(2 * c * s / D_GW) / r3H0 for s in [1e-3, 1e-2, 1e-1, 1.0]], 'o-')
    ax[0].axhline(1.0, color='r', ls='--', label='c_g = 1 判定线')
    ax[0].axvline(sig_ln, color='g', ls=':', label='当前 sigma_ln = 0.1005')
    ax[0].set_xlabel('距离一致度 sigma_ln'); ax[0].set_ylabel('可排除的 c_g 下限')
    ax[0].legend(); ax[0].set_title('(a) 40 Mpc 档：排除 c_g = 1 所需精度')
    Ds = np.logspace(1, 4, 200) * Mpc
    ax[1].loglog(Ds / Mpc, 2 * c * sig_ln / Ds / r3H0, label='sigma_ln = 0.1005')
    ax[1].axhline(1.0, color='r', ls='--', label='c_g = 1 判定线')
    ax[1].set_xlabel('汽笛距离 (Mpc)'); ax[1].set_ylabel('可排除的 c_g 下限')
    ax[1].legend(); ax[1].set_title('(b) 距离档：多远能否定 O(1) 耦合')
    plt.tight_layout()
    plt.savefig(r'D:\software\TianXingFC\images\gamma_scale_test.png', dpi=130)
    print('图已存 images/gamma_scale_test.png')

main()
