# -*- coding: utf-8 -*-
"""P4-31 阶段一：四维格点上一圈 bubble 自能的边际洛伦兹破坏读数（判据 S5 至 S8）。

背景：既有判据 S1 至 S4 算的是三维空间 bubble，时间方向按同类结构处理，未做完整的四维计算
（见 [[验证脚本/注记-一圈自能对边际洛伦兹破坏系数的生成量估计：零与百分级的分层]] 第五节
未做项其一）。本轮把时间方向做成第四个格点轴，一次读出边际通道的两类读数：空间各向异性
（空间方向之间）与时空速度差（时间方向与空间方向之间）。

判据（先立判据，再实现）：
  S5 冒烟：对称四维模板下四轴方向的二次系数同值（各向同性），且量级非零。
  S6 对称模板：空间各向异性按 p^2 a^2 收缩（收缩比约 4）且不超本底阈；时空速度差严格为零。
  S7 破缺沿时间轴（1%、2%、5% 三档）：时空速度差升到与破缺幅度同阶（归一化后不低于 1%）、
     随幅度线性；空间各向异性仍留在本底。
  S8 破缺沿空间 z 轴（5%）：空间各向异性被生成（归一化后不低于 1%），时空速度差保持严格零
     （交叉对照：破缺只点亮它所在轴对应的读数）。
  S9 破缺轴身份无关：破缺轴与任一未破缺轴的二次系数差只依赖破缺幅度，与破缺的是时间轴还是
     空间轴无关（三档逐档比对）。
另配恒真反例：破缺幅度取零时读数须还原对称模板值。

模型：四维周期格子（N^4 点，轴 0 为时间、轴 1 至 3 为空间）上的单圈 bubble，
Sigma(p) = -(g/2) <G(k) G(k+p)>，G(k) = 1/(m^2 + 4 sum_mu sin^2(k_mu/2))。
破缺模板把某一轴的 sin^2 系数乘 (1+a)。N = 24、m^2 = 0.1、g = 1。确定性，无随机数。

容差来源（误差预算）：被测量为二次系数 C 的方向差。本底来自格点色散的各向异性首项
O(p^2 a^2)，四维模板实测 2.17e-6（p = 0.02），归一化后为本底/C = 3.9e-4；数值和分辨率
2*pi/N = 0.262 的误差并入该本底；浮点误差 1e-16 可忽略。破缺档信号实测 1.2e-4 至 3.5e-4，
归一化后为 2% 至 6%，高于本底 25 倍以上。故判据取归一化口径：破缺信号/C 不低于 1%（与破缺
幅度同阶，且高于本底 25 倍），本底档读数不超过 1e-12（严格零），收缩比阈值 3.0（期望 4），
线性比值容差 0.5 与 1.5。首版曾把三维的绝对阈值 1e-3 照搬到四维，致 S7、S8 判负，属未经
预算直接设数（规范 3 所禁）；本次修正由此而来：绝对量级随维度变化，跨维度不可照搬，边际级
的判据须走归一化口径。
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CHECKS = []
N = 24
M2 = 0.1
G_COUPLING = 1.0
K1 = 2.0 * np.pi * np.fft.fftfreq(N)
P_MAIN = 0.02
ZERO_THRESHOLD = 1e-12
NORM_THRESHOLD = 1e-2


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [{}] {}  {}".format(tag, "PASS" if ok else "FAIL", detail))


def axis_grid(m):
    shape = [1, 1, 1, 1]
    shape[m] = N
    return K1.reshape(shape)


AX = [axis_grid(m) for m in range(4)]


def green(kvec, broken_axis=None, amp=0.0):
    s = np.zeros(1)
    for m in range(4):
        s = s + np.sin(kvec[m] / 2.0) ** 2
    if broken_axis is not None:
        s = s + amp * np.sin(kvec[broken_axis] / 2.0) ** 2
    return 1.0 / (M2 + 4.0 * s)


def sigma(pvec, broken_axis=None, amp=0.0):
    kp = [AX[m] + pvec[m] for m in range(4)]
    gk = green(AX, broken_axis, amp)
    gkp = green(kp, broken_axis, amp)
    return -0.5 * G_COUPLING * float(np.mean(gk * gkp))


ZERO = np.zeros(4)
E_T = np.array([1.0, 0.0, 0.0, 0.0])
E_X = np.array([0.0, 1.0, 0.0, 0.0])
E_Y = np.array([0.0, 0.0, 1.0, 0.0])
E_Z = np.array([0.0, 0.0, 0.0, 1.0])
SPACE_DIAG = np.array([0.0, 1.0, 1.0, 1.0]) / np.sqrt(3.0)


def quad(p_mag, direction, broken_axis=None, amp=0.0):
    s0 = sigma(ZERO, broken_axis, amp)
    sp = sigma(p_mag * direction, broken_axis, amp)
    return (sp - s0) / p_mag ** 2


print("=" * 78)
print("S5 冒烟：对称四维模板下四轴方向的二次系数（各向同性核对）")
print("=" * 78)
c_axes = {}
for name, d in (("时间 t", E_T), ("空间 x", E_X), ("空间 y", E_Y), ("空间 z", E_Z)):
    c_axes[name] = quad(P_MAIN, d)
    print("  {:8s} C = {:+.9f}".format(name, c_axes[name]))
cvals = list(c_axes.values())
spread = max(cvals) - min(cvals)
c_ref = abs(c_axes["时间 t"])
print("  四轴极差 = {:.3e}（各向同性期望约 1e-15）；C 归一化基准 = {:.6e}".format(spread, c_ref))
check("S5", spread <= 1e-9 and c_ref > 1e-3,
      "四轴二次系数同值且非零（冒烟通过，变化轴等于操作轴）")

print()
print("=" * 78)
print("S6 对称模板：两类方向差的收缩律")
print("=" * 78)
res = {}
for p_mag in (P_MAIN, 0.01):
    c_t = quad(p_mag, E_T)
    c_x = quad(p_mag, E_X)
    c_s = quad(p_mag, SPACE_DIAG)
    res[p_mag] = (c_x - c_s, c_t - c_x)
    print("  p = {:5.3f}：空间各向异性 = {:+.3e}，时空速度差 = {:+.3e}"
          .format(p_mag, res[p_mag][0], res[p_mag][1]))
shrink_sp = abs(res[P_MAIN][0] / res[0.01][0]) if res[0.01][0] != 0 else float("inf")
print("  空间各向异性收缩比 = {:.3f}（p^2 a^2 标度期望约 4）".format(shrink_sp))
print("  时空速度差 = {:+.3e}，即浮点零（时间轴与空间轴共用同一模板，故严格同值）"
      .format(res[P_MAIN][1]))
check("S6", abs(res[P_MAIN][0]) <= 1e-3 and shrink_sp >= 3.0
      and abs(res[P_MAIN][1]) <= ZERO_THRESHOLD,
      "空间各向异性按 p^2 a^2 收缩（不可关残留），时空速度差严格零")

print()
print("=" * 78)
print("S7 破缺沿时间轴：时空速度差生成（多档扫描），空间各向异性留在本底")
print("=" * 78)
d_st_amp = {}
d_sp_amp = {}
for amp in (0.01, 0.02, 0.05):
    c_t = quad(P_MAIN, E_T, broken_axis=0, amp=amp)
    c_x = quad(P_MAIN, E_X, broken_axis=0, amp=amp)
    c_s = quad(P_MAIN, SPACE_DIAG, broken_axis=0, amp=amp)
    d_st_amp[amp] = c_t - c_x
    d_sp_amp[amp] = c_x - c_s
    print("  破缺 {:>4.0%}：时空速度差 = {:+.3e}（归一化 {:+.3%}），空间各向异性 = {:+.3e}"
          .format(amp, d_st_amp[amp], d_st_amp[amp] / c_ref, d_sp_amp[amp]))
lin2 = d_st_amp[0.02] / d_st_amp[0.01]
lin5 = d_st_amp[0.05] / d_st_amp[0.01]
print("  线性核对：比值（2% 对 1%）= {:.3f}（期望 2），（5% 对 1%）= {:.3f}（期望 5）"
      .format(lin2, lin5))
print("  本底对照：对称模板空间各向异性 = {:+.3e}（归一化 {:+.1e}）"
      .format(res[P_MAIN][0], res[P_MAIN][0] / c_ref))
check("S7", abs(d_st_amp[0.05]) / c_ref >= NORM_THRESHOLD and abs(lin2 - 2.0) <= 0.5
      and abs(lin5 - 5.0) <= 1.5 and abs(d_sp_amp[0.05]) <= 2.0 * abs(res[P_MAIN][0]),
      "破缺沿时间生成时空速度差（归一化不低于 1%、随幅度线性），空间各向异性不被点亮")

print()
print("=" * 78)
print("S8 破缺沿空间 z 轴（5%）：空间各向异性生成，时空速度差保持零（交叉对照）")
print("=" * 78)
c_t = quad(P_MAIN, E_T, broken_axis=3, amp=0.05)
c_x = quad(P_MAIN, E_X, broken_axis=3, amp=0.05)
c_z = quad(P_MAIN, E_Z, broken_axis=3, amp=0.05)
c_s = quad(P_MAIN, SPACE_DIAG, broken_axis=3, amp=0.05)
d_sp_z = c_x - c_s
d_zx_z = c_z - c_x
d_st_z = c_t - c_x
print("  空间各向异性（x 对空间对角） = {:+.3e}（归一化 {:+.3%}）"
      .format(d_sp_z, d_sp_z / c_ref))
print("  附加读数（z 轴对 x 轴） = {:+.3e}（归一化 {:+.3%}）".format(d_zx_z, d_zx_z / c_ref))
print("  时空速度差 = {:+.3e}".format(d_st_z))
print("  本底对照：对称模板同 p 下为 {:+.3e} 与 {:+.3e}".format(res[P_MAIN][0], res[P_MAIN][1]))
check("S8", abs(d_sp_z) / c_ref >= NORM_THRESHOLD and abs(d_st_z) <= ZERO_THRESHOLD,
      "破缺只点亮它所在轴对应的读数（交叉对照成立）")

print()
print("=" * 78)
print("S9 破缺轴身份无关：破缺轴与未破缺轴的读数只依赖破缺幅度")
print("=" * 78)
d_zx_amp = {}
rels = []
for amp in (0.01, 0.02, 0.05):
    c_x2 = quad(P_MAIN, E_X, broken_axis=3, amp=amp)
    c_z2 = quad(P_MAIN, E_Z, broken_axis=3, amp=amp)
    d_zx_amp[amp] = c_z2 - c_x2
    rel = abs(d_zx_amp[amp] - d_st_amp[amp]) / abs(d_st_amp[amp])
    rels.append(rel)
    print("  幅度 {:>4.0%}：时间轴破缺读数 = {:+.6e}，空间 z 轴破缺读数 = {:+.6e}，相对差 = {:.2e}"
          .format(amp, d_st_amp[amp], d_zx_amp[amp], rel))
check("S9", max(rels) <= 1e-3,
      "破缺响应与破缺轴身份无关（时间轴与空间轴在模板层等价）")

print()
print("=" * 78)
print("恒真反例：破缺幅度取 0 时读数须还原对称模板值")
print("=" * 78)
c_t0 = quad(P_MAIN, E_T, broken_axis=0, amp=0.0)
c_x0 = quad(P_MAIN, E_X, broken_axis=0, amp=0.0)
print("  幅度 0 的时空速度差 = {:+.3e}（对称模板对照 {:+.3e}）"
      .format(c_t0 - c_x0, res[P_MAIN][1]))
check("反例", abs((c_t0 - c_x0) - res[P_MAIN][1]) <= 1e-12,
      "破缺关掉即还原对称模板读数（判据非恒真）")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

ax = axes[0]
ps = [P_MAIN, 0.01]
ax.loglog(ps, [abs(res[p][0]) for p in ps], "o-", label="空间各向异性")
ax.loglog(ps, [abs(res[p][1]) or 1e-18 for p in ps], "s-", label="时空速度差（严格零）")
pref = np.array([0.008, 0.026])
ax.loglog(pref, abs(res[P_MAIN][0]) * (pref / P_MAIN) ** 2, "k--", label="参考斜率 $p^2$")
ax.set_xlabel("动量 $p$")
ax.set_ylabel("方向差（二次系数差）")
ax.set_title("对称模板：方向差按 $p^2 a^2$ 收缩")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9)

ax = axes[1]
amps = [0.0, 0.01, 0.02, 0.05]
base = res[P_MAIN][1]
vals_t = [base] + [d_st_amp[a] for a in (0.01, 0.02, 0.05)]
vals_z = [base] + [d_zx_amp[a] for a in (0.01, 0.02, 0.05)]
ax.plot([a * 100 for a in amps], vals_t, "o-", color="tab:red", label="破缺沿时间轴（时间对空间）")
ax.plot([a * 100 for a in amps], vals_z, "s--", color="tab:blue", label="破缺沿空间 z 轴（z 对 x）")
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xlabel("破缺幅度（%）")
ax.set_ylabel("方向差（二次系数差）")
ax.set_title("破缺响应：与破缺轴身份无关")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(r"D:\software\TianXingFC\images\fig_marginal_4d_oneloop.png", dpi=150)
print("  图已存：images/fig_marginal_4d_oneloop.png")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：{}/{} 通过".format(n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  {:6s} {}".format(tag, "PASS" if ok else "FAIL"))
print("=" * 78)
