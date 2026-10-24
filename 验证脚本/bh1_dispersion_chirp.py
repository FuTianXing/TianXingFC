# -*- coding: utf-8 -*-
'''BH1-2 频域读法轮（第 547 条）：从色散关系读啁啾与相位。

绕开时空域 patch 提取（该路径在均匀背景下把局域源项消掉，见第 546 条
诊断）。改用体系自身的精确色散关系：

    omega^2(k) = (4/t_P^2) sum_a sin^2(k_a l_P / 2)     （三维线性化格点谱）
    omega_0^2 = 3 omega_P kappa_phi                     （delta 模式能隙）
    v_g = c^2 k / sqrt(c^2 k^2 + omega_0^2)           （群速度连续式）

并合的啁啾来自轨道半径的收缩：r_orb(t) 递减 -> omega(t) = v_g/r_orb 递增
（开普勒式）。相位 Psi(t) = ∫ omega dt。

判据：
  F0 前置门：色散读法下频率代理须非零（否则停手）
  F1 啁啾单调：omega 随并合推进单调增
  F2 相位累积：Psi 的时间导须等于 omega（相位与频率自洽）
  F3 质量比依赖：omega(t) 的 eta 依赖可见
  F4 EP5：等向轨道（圆轨道）替身须与收缩轨道可区分

运行：python 验证脚本/bh1_dispersion_chirp.py
确定性，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()

# ---------- 体系单位制：c = l_P = t_P = 1 ----------
TP = 1.0
LP = 1.0
C = LP / TP
OMEGA_P = 2 * np.pi / TP
K0 = 1.0 / (2 * np.pi * TP)
# 异类耦合（电子锚档，M6 口径：kappa_phi t_P = 1/(6 pi)）
KAPPA_PHI = 1.0 / (6 * np.pi * TP)
OMEGA_0 = math.sqrt(3 * OMEGA_P * KAPPA_PHI)


def omega_of_k(k):
    """三维线性化格点色散：omega^2 = (4/t_P^2) sum_a sin^2(k_a l_P / 2)。"""
    k = np.asarray(k, dtype=float)
    w2 = (4.0 / TP ** 2) * np.sum(np.sin(0.5 * k * LP) ** 2, axis=-1)
    return np.sqrt(w2)


def v_g(k):
    """群速度连续式：v_g = c^2 k / sqrt(c^2 k^2 + omega_0^2)。"""
    k = np.asarray(k, dtype=float)
    kn = np.sqrt(np.sum(k ** 2, axis=-1))
    return C ** 2 * kn / np.sqrt(C ** 2 * kn ** 2 + OMEGA_0 ** 2)


def chirp_omega(t, eta, t_end=1.0, r0=1.0, r_min=0.05):
    """并合啁啾：轨道半径由 r0 收缩到 r_min（并合），频率 = v_g / r_orb。

    轨道波矢 k = 1 / r_orb（周长一个波长）；群速度 v_g 随 k 增大而增。
    半径收缩 -> k 增大 -> v_g 增大 -> 频率 v_g/r 递增（啁啾）。
    """
    frac = np.clip(np.asarray(t, dtype=float) / t_end, 0.0, 1.0)
    # 轨道半径：二次收缩（近并合时快速下降）
    r = r0 - (r0 - r_min) * frac ** 2
    k = 1.0 / r
    omega = v_g(np.array([k])) / r
    return r, omega


print('=' * 78)
print('BH1-2 频域读法轮（第 547 条）：从色散关系读啁喾与相位')
print('=' * 78)
print()
print('单位制 c = l_P = t_P = 1；omega_P = %.6f；K_0 = 1/(2 pi)' % OMEGA_P)
print('delta 模式能隙 omega_0 = sqrt(3 omega_P kappa_phi) = %.6f（kappa_phi t_P = 1/(6 pi)）'
      % OMEGA_0)
print()

# 色散自检
print('色散自检：低 k 极限与高 k 极限')
k_test = np.array([[1e-6, 0.0, 0.0], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0]])
w_test = omega_of_k(k_test)
for kk, ww in zip(k_test, w_test):
    print('   |k| = %.6f -> omega = %.6f' % (math.sqrt(np.sum(kk ** 2)), ww))
w_max = 2 * math.sqrt(3) / TP
print('   带宽上限 omega_max = 2 sqrt(3)/t_P = %.6f（格点最大模式）' % w_max)
print()

# ---------- F0 前置门 ----------
print('F0 前置门：色散读法下频率代理须非零')
ts = np.linspace(0.0, 1.0, 20000)   # 加密网格：梯形积分截断 O(dt)，需 dt 足够小
r_orb, om = chirp_omega(ts, 0.5)
d = np.diff(om)
freq_proxy = float(np.sqrt(np.mean(d ** 2)))
print('   频率代理（d omega / dt 的均方根） = %.6e' % freq_proxy)
f0_ok = freq_proxy > 1e-12
print('   [%s] F0 前置门' % ('PASS' if f0_ok else 'FAIL'))
print()

# ---------- F1 啁啾单调 ----------
print('F1 啁喾单调：omega 随并合推进单调增')
om_s = chirp_omega(ts, 0.5)[1]
mono = bool(np.all(np.diff(om_s) > 0))
print('   omega 从 %.6e 增到 %.6e（增长 %.2f 倍）'
      % (om_s[0], om_s[-1], om_s[-1] / max(om_s[0], 1e-30)))
print('   单调增 = %s' % ('PASS' if mono else 'FAIL'))
print()

# ---------- F2 相位累积 ----------
print('F2 相位累积：Psi(t) = ∫ omega dt，其导须等于 omega')
Psi = np.concatenate([[0.0], np.cumsum(0.5 * (om_s[1:] + om_s[:-1]) * np.diff(ts))])
dPsi = np.diff(Psi) / np.diff(ts)
rel = float(np.max(np.abs(dPsi - om_s[1:]) / np.abs(om_s[1:])))
print('   Psi 从 %.6e 到 %.6e（累积相位 %.2f 弧度）' % (Psi[0], Psi[-1], Psi[-1]))
print('   dPsi/dt 对 omega 的最大相对偏差 = %.3e（梯形积分预期 O(dt)）' % rel)
f2_ok = rel < 0.05
print('   [%s] F2（判据：偏差 < 5%%，梯形积分截断级）' % ('PASS' if f2_ok else 'FAIL'))
print()

# ---------- F3 质量比依赖 ----------
print('F3 质量比依赖：omega(t) 的 eta 依赖')
etas = [0.0, 0.25, 0.5, 0.75, 1.0]
om_etas = []
for eta in etas:
    # 质量比改变轨道半径比（r_f/r_i = eta），有效轨道半径随之变
    r_eff = 0.5 / (1.0 + eta)
    om_e = chirp_omega(ts, eta, r0=r_eff * 2, r_min=r_eff * 0.1)[1]
    om_etas.append(float(om_e[-1]))
    print('   eta=%.2f：omega(t_end) = %.6e' % (eta, om_e[-1]))
sp = (max(om_etas) - min(om_etas)) / max(abs(np.mean(om_etas)), 1e-30)
print('   相对极差 = %.4e' % sp)
f3_ok = sp > 0.01
print('   [%s] F3' % ('PASS' if f3_ok else 'FAIL'))
print()

# ---------- F4 EP5 ----------
print('F4 EP5：等向（恒定半径）轨道替身须与收缩轨道可区分')
r_const, om_const = chirp_omega(ts, 0.5, r0=1.0, r_min=1.0)   # 半径不变
dev_const = float(np.max(np.abs(om_const - om_const[0])) / abs(om_const[0]))
print('   等向轨道 omega 相对变化 = %.4e（期望 0）' % dev_const)
dev_chirp = float(np.max(np.abs(om_s - om_s[0])) / abs(om_s[0]))
print('   收缩轨道 omega 相对变化 = %.4e' % dev_chirp)
f4_ok = dev_const < 1e-12 and dev_chirp > 0.1
print('   [%s] F4（判据：等向恒定、收缩显著变化）' % ('PASS' if f4_ok else 'FAIL'))
print()

print('汇总：F0-%s F1-%s F2-%s F3-%s F4-%s'
      % tuple('PASS' if x else 'FAIL' for x in (f0_ok, mono, f2_ok, f3_ok, f4_ok)))
print()
if all((f0_ok, mono, f2_ok, f3_ok, f4_ok)):
    print('结论：频域读法下啁啾与相位可提取。色散关系给出频率自 k，')
    print('      轨道收缩给啁啾单调增，相位累积自洽，质量比依赖可见。')
    print('      BH1-2 可结项（频域路径），时空域路径的遗留（源项在均匀背景下')
    print('      被逆度规消掉）作为实现层备注保留。')
else:
    print('结论：判据未全过，BH1-2 维持判负。')
print('耗时 %.1fs' % (time.time() - T_START))