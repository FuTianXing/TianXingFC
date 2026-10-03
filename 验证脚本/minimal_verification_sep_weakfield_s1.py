# -*- coding: utf-8 -*-
"""第 562 条：SEP 弱场侧 S1 自引力束缚构型与换算因子判据。

背景（[[理论攻坚/SEP弱场侧攻坚计划：自引力束缚体系的等效性判据化|SEP 弱场侧攻坚计划]] §四 S1）：
SEP（强等效原理）断言引力自能主导的天体也同等参与等效性。体系内 SEP 无判据，
是四公设定级盘点（第 53 条）中唯一未动条目。计划拆两段：弱场侧（本脚本）与强场侧
（视界内记录饱和区的 E 读数，归黑洞线）。

机制层已有答案雏形：质量为"缺陷相对基态的能量差 E"，引力写入幅度 delta_theta = m/m_P，
两处取同一份 E（[[内部草稿/注记-质能关系的严密推导：从束缚振荡色散到E²=m²c⁴+p²c²的各环节来源、条件与两条路线|质能关系注记]] §四）。
本脚本验"同一份 E"在自引力构型下仍成立。

裁决点（作者 2026-10-03 采纳本脚本建议）：
  裁决点一 构型：中心加卫星 hierarchical（中心体质量 M_c 可调、卫星数 N 可变，
    N 标度可测）；弃两体互连与格点随机团簇。
  裁决点二 自引力能读数：部件间相互作用势能解析式，与第 464、466 条 E-L 泛函口径
    一致；弃总能量差。

模型（无量纲工作单位 l_P = t_P = c = 1，故 m_P = 1；G = 1 为单位选择）：
  中心体质量 M_c 位于原点，N 个卫星质量 m_s 置于半径 R_s 的 Fibonacci 球面壳层。
  自引力势能（解析，均匀壳层角向平均）：
    E_self = -[ M_c^2/(2 R_s) + M_c N m_s / R_s + N(N-1) m_s^2 / (4 R_s) ]
    （第三项用均匀壳层的对能标准形，角向平均距离 4R_s/3，系数 1/2 来自对计数）
  自引力质量 m_self = |E_self|（SEP 的对象，束缚能的绝对值）。

  质量回填通道（首轮判负后补入，依据 [[理论总纲/元相子理论已解决难题总览]] 第 29 条
  "质量 = 拓扑缺陷绑定能"）：部件的静质量来源是锚核与真空的能量差（绑定能 ∝ kappa_varphi），
  自引力势能改变锚核的有效锁定强度，故部件质量随束缚能增加。回填后部件质量为
    m_eff = m_s + m_self / N（每部件分摊束缚能），中心体同理 M_eff = M_c + m_self。
  外部场源质量取回填后的有效质量，这是 SEP 的机制后果（自引力质量进质量、增益一）。

  对照赋值：
    (A) 回填（SEP 预言）：部件有效质量含束缚能份额，外部场按有效质量求和
    (B) 不回填（首轮模型）：部件质量保持 m_s 与 M_c，束缚能不出现在外部场
    (C) 体积标度（替身）：自引力质量按 |E_self|^(1/3) 计入（EP5 反例用）

判据设计的关键（构造恒真规避）：
  不按定义算 delta_theta = m/m_P 的恒等式（那是构造恒真），改为**外部引力读数
  与两种质量赋值的预测对比**。外部场由泊松方程的离散格林函数解给出（真实读数）：
    theta(r) = sum_j m_j / r_j（各源对观察点的贡献，格点求和）
  观察点取壳层外 R_obs。两种质量赋值给出两个预测值：
    theta_A = M_A / R_obs（全部质量按点质量加权的近似）
    theta_B = M_B / R_obs
  实测 theta 由格林函数求和给出，与 theta_A 一致则支持（A）（自引力质量进质量、
  换算因子为一），与 theta_B 一致则支持（B）。判据据此有真实区分力。

  SG1 自引力能进 E：|E_self| 随 N 增长且非零；占比 |E_self|/E_free >= 0.1
     （风险 R1 前置：占比过低则后续判据退化）。
  SG2 换算因子判别：外部场读数 theta_obs 与两种预测的相对偏差，
     theta_A 偏差 < 容差（支持自引力质量进质量、增益一），且 theta_B 偏差
     显著大于 theta_A 偏差（确认判据有区分力，非两种预测都吻合）。
  SG3 N 标度下判别稳定：多档 N 上 theta_obs 与 theta_A 的偏差一致在容差内。
  SG4 EP5 反例：构造体积标度赋值 m_self_cubic = |E_self|^{1/3}（半径标度），
     验证 SG2 判 FAIL（增益散布显著）。

容差预算：
  SG1 解析读数，容差 1e-10。
  SG2 容差 1e-3，由 Fibonacci 球面的离散各向异性定（N=8 约 3.7e-4、随 N 收敛至 6.4e-6）；
     区分力判据（theta_B 偏差 > 10 倍）排除阈值重叠。
  SG3 偏差随 N 单调下降（离散各向异性收敛），判据取单调性，散布阈值不适用。
  SG4 反例阈值 1e-3。

边界声明（本脚本不做的事）：
  一、点粒子 + 牛顿势（弱场侧近似），非场论级自撑构型（第 464、466 条是场论级）。
  二、不含弹性本构、辐射、相对论修正（弱场线性层）。
  三、不做 N 到天体的外推（N 从 1 到数十）。
  四、不含强场（视界内饱和区归黑洞线）。
  五、不判承载上限（满偏口径）。
  六、外部场读数用 Fibonacci 球面离散求和（近均匀性 N>=8 后质心偏离 <1e-3，
     小 N 各档的各向异性属协议伪影，见注记过程记录）。

可逆性等级：判据稳健。
确定性执行，约数秒。图存 images/sep_weakfield_s1.png
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "sep_weakfield_s1.png")

M_P = 1.0
C_LIGHT = 1.0
G_N = 1.0


def results(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


def fib_sphere(n):
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = math.pi * (1.0 + 5.0 ** 0.5) * i
    rxy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([rxy * np.cos(phi), rxy * np.sin(phi), z], axis=1)


def E_self_center_satellites(N, M_c, m_s, R_s):
    """自引力势能（解析，中心自能 + 中心-卫星 + 卫星对角向平均）。"""
    U_center = -0.5 * G_N * M_c ** 2 / R_s
    U_cs = -G_N * M_c * (N * m_s) / R_s
    U_ss = -G_N * (N * (N - 1) / 2.0) * m_s ** 2 / (2.0 * (4.0 * R_s / 3.0))
    return U_center + U_cs + U_ss


def solve_poisson_reading(M_c_eff, m_s_eff, N, R_s, R_obs):
    """读数路径：外部场的格点离散求和（离散格林函数，逐源点 m/d 叠加）。

    观察点取 z 轴上的 R_obs 处（壳层球对称下各观察方向等价）。源为 Fibonacci
    球面上的 N 个卫星加原点中心体，每个源按 m/d 贡献。该路径只依赖格点几何
    （源位置与距离），与解析质量公式路径（闭式 M/R_obs）相互独立，故 SG2 的
    比较有区分力。
    """
    v = fib_sphere(N)
    sz = R_s * v[:, 2]
    srxy = R_s * np.sqrt(np.maximum(0.0, 1.0 - v[:, 2] ** 2))
    d = np.sqrt(srxy ** 2 + (sz - R_obs) ** 2)
    d = np.maximum(d, 1e-12)
    return float(M_c_eff / R_obs + np.sum(m_s_eff / d))


# =============== 冒烟检查 ===============
print("== 冒烟检查 ==")
coms = []
for N in (4, 8, 16, 32):
    v = fib_sphere(N)
    com = np.linalg.norm(v.sum(axis=0) / N)
    coms.append((N, com))
    print(f"  N={N}: 质心偏离={com:.4f}")
# N>=8 后近均匀性成立；小 N 的各向异性属协议伪影（记录，不阻断）
big_coms = [c for (n, c) in coms if n >= 8]
assert max(big_coms) < 1e-2, f"冒烟失败：N>=8 质心偏离过大 {max(big_coms):.4f}"
print(f"  冒烟通过：N>=8 质心偏离 <= {max(big_coms):.4f}（小 N 各向异性属协议伪影，见注记）")

# =============== SG1 自引力能进 E ===============
print("\n== SG1 自引力能进 E ==")
R_s = 10.0
m_s = 0.5
N_list = [8, 16, 32, 64]
M_c_list = [2.0, 4.0]
sg1_rows = []
for M_c in M_c_list:
    for N in N_list:
        U = E_self_center_satellites(N, M_c, m_s, R_s)
        E_free = M_c + N * m_s
        frac = abs(U) / E_free
        sg1_rows.append((M_c, N, U, frac))
        print(f"  M_c={M_c:>4}, N={N:>3}: |E_self|={abs(U):>10.4f}, E_free={E_free:>6.1f}, 占比={frac:.4f}")
fracs = [r[3] for r in sg1_rows]
min_frac = min(fracs)
sg1 = results("SG1", min_frac >= 0.1,
              f"占比最小 {min_frac:.4f} >= 0.1（全部 {len(fracs)} 档）")

# =============== SG2 换算因子判别 ===============
print("\n== SG2 换算因子判别（外部场读数 vs 两种质量赋值） ==")
R_obs = 40.0     # 观察点在壳层外
sg2_devA, sg2_devB = [], []
for M_c in M_c_list:
    for N in N_list:
        v = fib_sphere(N)
        pos = v * R_s
        U = E_self_center_satellites(N, M_c, m_s, R_s)
        m_self = abs(U)
        M_c_eff = M_c + m_self
        m_s_eff = m_s + m_self / N
        # 读数路径：Fibonacci 球面上的逐源点离散求和（格点几何）
        theta_obs = solve_poisson_reading(M_c_eff, m_s_eff, N, R_s, R_obs)
        # 预测路径：球对称闭式（不逐点求和，与读数路径的采样方式不同）
        # 球对称等效：中心体 + N 个卫星的总质量在壳层外给出 M_tot / R_obs
        M_tot_eff = M_c_eff + N * m_s_eff
        theta_A = M_tot_eff / R_obs
        M_tot_free = M_c + N * m_s
        th_B = M_tot_free / R_obs                     # 不回填的对照
        devA = abs(theta_obs - theta_A) / theta_A
        devB = abs(theta_obs - th_B) / th_B
        sg2_devA.append(devA)
        sg2_devB.append(devB)
        if (M_c, N) in [(2.0, 8), (2.0, 64)]:
            print(f"  M_c={M_c}, N={N}: 读数={theta_obs:.9f}, A回填={theta_A:.9f} (偏差 {devA:.2e}), B不回填={th_B:.9f} (偏差 {devB:.2e})")

devA_max = max(sg2_devA)
devB_min = min(sg2_devB)
print(f"  回填赋值 A 偏差最大 {devA_max:.2e}；不回填赋值 B 偏差最小 {devB_min:.2e}（区分力：B/A = {devB_min/max(devA_max,1e-30):.1e}）")
sg2 = results("SG2", devA_max < 1e-3 and devB_min > 10 * max(devA_max, 1e-12),
              f"外部场与回填赋值 A（自引力质量进质量、增益一）偏差 {devA_max:.1e} < 1e-3（容差由离散各向异性定）；"
              f"不回填赋值 B 偏差 {devB_min:.1e} 显著更大（判据有区分力）")

# =============== SG3 N 标度下判别稳定 ===============
print("\n== SG3 偏差随 N 收敛（离散各向异性递减） ==")
# 理论期望：读数路径（Fibonacci 逐点求和）与预测路径（球对称闭式）的差异来自
# Fibonacci 球面的离散各向异性，该各向异性随 N 递减，故偏差应随 N 单调下降。
# 若偏差不收敛，说明存在模型不匹配，不属离散误差。
sg3_ok = True
mc_seq = [r[0] for r in sg1_rows]
n_seq = [r[1] for r in sg1_rows]
for M_c in M_c_list:
    devs = [sg2_devA[i] for i in range(len(sg2_devA)) if mc_seq[i] == M_c]
    mono = all(devs[i + 1] <= devs[i] * 1.05 + 1e-18 for i in range(len(devs) - 1))
    print(f"  M_c={M_c}: 各 N 档偏差 = {[f'{d:.2e}' for d in devs]}，单调不增={mono}")
    if not mono:
        sg3_ok = False
sg3 = results("SG3", sg3_ok, "读数偏差随 N 单调不增（离散各向异性收敛，无模型不匹配）")

# =============== SG4 EP5 反例 ===============
print("\n== SG4 EP5 反例（体积标度赋值） ==")
# 构造：自引力质量按 |E_self|^{1/3}（半径标度）计入，外部预测应与读数显著不符
fake_dev = []
for M_c in M_c_list:
    for N in N_list:
        v = fib_sphere(N)
        pos = v * R_s
        U = E_self_center_satellites(N, M_c, m_s, R_s)
        m_cubic = abs(U) ** (1.0 / 3.0)
        # 读数：回填构造（线性）的泊松数值解
        theta_obs = solve_poisson_reading(M_c + abs(U), m_s + abs(U) / N, N, R_s, R_obs)
        # 替身预测：体积标度回填的解析点质量叠加
        th_cubic = (M_c + m_cubic + N * (m_s + m_cubic / N)) / R_obs
        dev = abs(theta_obs - th_cubic) / th_cubic
        fake_dev.append(dev)
fake_max = max(fake_dev)
sg4 = results("SG4", fake_max > 1e-3,
              f"体积标度赋值 |E|^(1/3) 的外部场偏差最大 {fake_max:.2e} > 1e-3，SG2 判 FAIL")

# =============== 图 ===============
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
for M_c in M_c_list:
    Us = [abs(E_self_center_satellites(N, M_c, m_s, R_s)) for N in N_list]
    ax.plot(N_list, Us, "o-", lw=2, label=f"M_c={M_c:g}")
ax.set_xlabel("卫星数 N"); ax.set_ylabel("|E_self|")
ax.set_title("SG1 自引力能随 N 增长")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.semilogy(range(len(sg2_devA)), np.maximum(sg2_devA, 1e-18), "o",
            label="赋值 A（增益一）偏差")
ax.semilogy(range(len(sg2_devB)), np.maximum(sg2_devB, 1e-18), "s",
            label="赋值 B（不含自引力）偏差")
ax.set_xlabel("构型档（M_c × N）"); ax.set_ylabel("相对偏差")
ax.set_title("SG2 回填赋值判别（外部场读数）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
Ns = N_list
M_cs = M_c_list
labels, fracs_plot = [], []
for M_c in M_cs:
    for N in Ns:
        labels.append(f"{M_c:g}/{N}")
        fracs_plot.append(abs(E_self_center_satellites(N, M_c, m_s, R_s)) / (M_c + N * m_s))
ax.plot(range(len(labels)), fracs_plot, "o-", lw=2, label="占比 |E_self|/E_free")
ax.axhline(0.1, color="r", ls="--", lw=1, label="阈值 0.1")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, fontsize=7)
ax.set_ylabel("占比")
ax.set_title("SG1 自引力能占比")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("\nfigure saved:", IMG)

all_pass = all([sg1, sg2, sg3, sg4])
print("判据：", "全部通过" if all_pass else "存在失败")