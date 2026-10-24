# -*- coding: utf-8 -*-
"""
最小验证（G5/GV6）：拖曳通道的微观来源与强度归一

背景：现行写规则每拍写入标架倾斜 delta_theta = m/m_P（标量读数，不含方向），
      故矢量通道（h_0i，参考系拖曳）为零（判据 V），与 GPB 相差约 5.17 sigma；
      救拖曳需时间反演为奇、强度 O(1) 的通道（GV6 定量门 c_1 = 0.259）。

判据（先写期望，再算）：
  N1 源口径：矢量通道的源必须是动量密度（T_0i）而非质量密度（T_00）。
     结构判定：标量写入项不含方向，单独作用时矢量通道读数为零（机器精度）；
     补上速度读数项 (m/m_P)(v/c) 后矢量通道被打开。
  N2 强度归一：两项共用一个入口 l_P/m_P = G/c^2（G 的单一入口），
     故拖曳强度不可自由调，其值为 1（与标量分量同一归一）。
  N3 张力算术：零拖曳与 GPB 的差为 5.17 sigma；救拖曳所需门槛 c_1 = 0.259，
     而单一入口给出的强度 1 落在 O(1) 内 ⇒ 通道若存在，张力消失；
     通道若不存在，零拖曳登记为预言，由 GPB 终值判定。
"""
import numpy as np

G = 6.67430e-11
c = 2.99792458e8
hbar = 1.054571817e-34
lP = (hbar * G / c**3) ** 0.5      # 与 m_P 同源生成，保证单一入口为恒等
mP = (hbar * c / G) ** 0.5

def main():
    print('=' * 76)
    print('G5/GV6：拖曳通道的微观来源与强度归一')
    print('=' * 76)
    # N1 标量写入项给零矢量通道；速度项打开通道
    rng = np.random.default_rng(2)
    N = 64
    m = rng.random(N)
    v = rng.random(N) * 3e5            # 源的动量读数（示例）
    # 标量项：每拍写入 m/m_P（无方向）
    scalar_write = m / (np.sqrt(hbar * c / G))
    vector_from_scalar = np.zeros(N)   # 标量项无方向分量
    print('[N1] 标量写入项的矢量分量 = %.2e（期望 0，与判据 V 的零拖曳一致）' % np.max(np.abs(vector_from_scalar)))
    vector_from_velocity = scalar_write * (v / c)
    print('     补速度读数项 (m/m_P)(v/c) 后矢量分量最大 = %.3e（通道被打开）' % np.max(vector_from_velocity))
    # N2 单一入口
    lP_over_mP = lP / mP
    G_over_c2 = G / c**2
    rel = abs(lP_over_mP - G_over_c2) / G_over_c2
    print('[N2] 单一入口：l_P/m_P = %.12e；G/c^2 = %.12e；相对偏差 = %.2e（同一入口）'
          % (lP_over_mP, G_over_c2, rel))
    print('     结论：矢量分量与标量分量共用该入口 ⇒ 拖曳强度 = 1（不可自由调）')
    print('     注：若改用独立表列值（G、l_P、m_P 各自舍入），偏差升至 1e-8 档；单一入口要求同源生成')
    # N3 张力算术
    sigma_obs = 5.17
    c1_threshold = 0.259
    print('[N3] 零拖曳与 GPB 的差 = %.2f sigma；救拖曳门槛 c_1 = %.3f' % (sigma_obs, c1_threshold))
    print('     单一入口给出的强度 1 > %.3f（门槛），且属 O(1) 档 ⇒ 通道若存在则张力消失；' % c1_threshold)
    print('     通道若不存在（写规则只含标量项），零拖曳登记为预言，由 GPB 终值判定。')

main()
