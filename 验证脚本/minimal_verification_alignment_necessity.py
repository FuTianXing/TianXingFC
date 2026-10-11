# -*- coding: utf-8 -*-
"""
最小验证：对齐必然性轮——形式唯一性穷举抽查与失配判据（第 606 号，判据 A1 至 A3）
R 线阶段二第一轮（[[理论攻坚/响应侧测地线必然性攻坚计划：反例判据轮、前提精确化与定理化正稿]]，路线甲）

模型设定（移植自 minimal_verification_geodesic_lock.py 口径；统一取向-速度图像）：
  背景：中心源倾角场 theta(r) = theta_0 r_0 / r（r_0 = 1 格，c = 1 格/拍，每拍 dt = 1）。
  取向即速度方向（复合位错代理的内部朝向与运动方向一体）：粒子取向角 n_p、搬运帧角
  n_ref 各为角度变量；每拍帧扭转 delta_f = 2 c (v_hat x grad theta)_z dt（完整相对扭转），
  粒子按强度 kappa 应用：n_p += kappa * delta_f，n_ref += delta_f；轨迹沿 n_p 方向以 c 前进。
  kappa = 1 即锁定规则（精确旋转口径，与第 603 号 Euler+归一化口径差单拍 O(dv^3)，见 A3）。

判据与理论期望（先立后测）：
  A1 结构穷举（解析表 + 数值抽查）：局域数据（grad theta、v_hat）加六约束（局域性、各向同性、
    实值、2pi 周期、光速不变、弱场线性即更新一阶作用与写入份额同阶）下，一阶存活形式唯一为
    kappa * (v_hat x g) 旋转形式。数值抽查三条：
    (i) 零场零更新（lock 平坦空间 dv = 0，逐位）；
    (ii) 纵向候选 dv = v (v.g)（即 c^2 v_hat (v_hat.g) dt）破速率不变（|v| 单拍漂移 > 1e-8）；
    (iii) snap 候选（每拍完整转向 gh 方向）破弱场线性：单拍最大旋转角不随 theta_0 减半而减半
         （两档比值 ~1.00，对照 lock 单拍旋转角的比值 2.00）。
  A2 失配发散（强度归一，不经测地线一致性）：kappa = 1 时失配 m = |n_p - n_ref| 逐位恒零；
    kappa < 1 时每拍失配累积 (1 - kappa) * delta_f，随飞掠段数 M 线性无界（倍段比 2.000 +/- 0.05），
    且 m 随 (1 - kappa) 线性（M=4 档 m(0.25):m(0.5):m(0.75) = 3:2:1 +/- 5%，一阶路径不敏感性）。
    失锁取向与满周闭合锁相（R1b）不相容，故强度唯一为 1。
  A3 跨轮逐位自检：逐字移植第 603 号 run_particle，kappa = 1 基线偏转角与
    -0.00016001444680876443 逐位相同（跨轮移植教训：复合读数必须逐位复现自检）。

可逆性等级：A2 的 kappa=1 恒零半边与 A3 为判据稳健（逐位）；A1 抽查阈值与 A2 的线性比为
  判据敏感（档位依赖，外推须重跑）。
前提依赖：满周闭合锁相（R1b）、全局同步（R2）、写入平移不变（R3）、弱场静态（R4）、
  定义层识别式 h_00 = 2 theta 供给因子 2 与 c^2 搭桥（本轮不推导识别式本身）。
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 76)
print("对齐必然性轮：形式唯一性穷举抽查与失配判据（第 606 号，A1 至 A3）")
print("=" * 76)

C_SPEED = 1.0
r0 = 1.0
IMG_PATH = r"D:\software\TianXingFC\images\alignment_necessity.png"

VERDICT = {}

def grad_theta(x, y, theta_0):
    """nabla theta = -theta_0 r0 r_hat / r^2（r -> 0 时设零，避开源心）"""
    r = np.hypot(x, y)
    if r < 1e-6:
        return 0.0, 0.0
    g = -theta_0 * r0 / r ** 3
    return g * x, g * y

# ----------------------------------------------------------------------
# 移植件 A3：第 603 号 run_particle 逐字移植（Euler + 归一化口径）
# ----------------------------------------------------------------------
def run_particle_603(theta_0, b, L, kappa_r=1.0):
    x, y = -L, b
    v = np.array([1.0, 0.0])
    v = v / np.linalg.norm(v) * C_SPEED
    T = int(2.0 * L / C_SPEED)
    for _ in range(T):
        gx, gy = grad_theta(x, y, theta_0)
        g = np.array([gx, gy])
        dv = kappa_r * 2.0 * C_SPEED ** 2 * (g - v * (v @ g) / C_SPEED ** 2)
        v = v + dv
        v = v / np.linalg.norm(v) * C_SPEED
        x += v[0]
        y += v[1]
    return np.arctan2(v[1], v[0])

# ----------------------------------------------------------------------
# 统一取向追踪：一段飞掠（精确旋转口径）；角度经返回值回传
# ----------------------------------------------------------------------
def flyby(theta_0, b, L, kappa_r, n_p, n_ref):
    """一段飞掠：返回 (n_p, n_ref, 段内帧扭转合计, 最近点距)"""
    x, y = -L, b
    T = int(2.0 * L / C_SPEED)
    twist = 0.0
    rmin = math.hypot(x, y)
    for _ in range(T):
        gx, gy = grad_theta(x, y, theta_0)
        g = np.array([gx, gy])
        vhat = np.array([math.cos(n_p), math.sin(n_p)])
        cross = vhat[0] * g[1] - vhat[1] * g[0]
        df = 2.0 * C_SPEED * cross          # 帧扭转（每拍，dt = 1）
        n_ref += df
        n_p += kappa_r * df
        twist += df
        x += C_SPEED * math.cos(n_p)
        y += C_SPEED * math.sin(n_p)
        rmin = min(rmin, math.hypot(x, y))
    return n_p, n_ref, twist, rmin

def misalignment(kappa_r, M):
    """M 段连续飞掠（角度跨段持续，位置每段重置）：返回段末失配角"""
    n_p, n_ref = 0.0, 0.0
    for _ in range(M):
        n_p, n_ref, _, _ = flyby(th_ref, b, L, kappa_r, n_p, n_ref)
    return abs(n_p - n_ref)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print("\n[冒烟] 场量与扭转的量级与符号")
b, L, th_ref = 50.0, 6000, 0.002
gx0, gy0 = grad_theta(0.0, b, th_ref)
print(f"  最近点 |grad theta| = {math.hypot(gx0, gy0):.3e}（期望 ~ {th_ref / b ** 2:.3e}）")
cross0 = 1.0 * gy0 - 0.0 * gx0
print(f"  最近点单拍帧扭转 delta_f = {2.0 * cross0:.3e}（负号 = 朝源顺时针，量级 ~ 2 theta_0/b^2）")

# ----------------------------------------------------------------------
# A1. 结构穷举：解析排除 + 数值抽查
# ----------------------------------------------------------------------
print("\n[A1] 形式唯一性：六约束下存活集恰一（解析表 + 数值抽查）")

# (i) 零场零更新
g = np.array(grad_theta(0.0, b, 0.0))
v = np.array([1.0, 0.0])
dv_lock0 = 2.0 * C_SPEED ** 2 * (g - v * (v @ g) / C_SPEED ** 2)
flat_ok = (dv_lock0 == np.zeros(2)).all()
print(f"  (i) 零场 lock 更新 = ({dv_lock0[0]:.1e}, {dv_lock0[1]:.1e})，逐位零 = {flat_ok}（期望是）")

# (ii) 纵向候选破速率不变：dv = v (v.g)（即 c^2 v_hat (v_hat.g) dt，纯纵向，不旋转）
def longitudinal_speed_drift(theta_0, b, L):
    x, y = -L, b
    v = np.array([1.0, 0.0])
    T = int(2.0 * L)
    max_drift = 0.0
    for _ in range(T):
        gx, gy = grad_theta(x, y, theta_0)
        g = np.array([gx, gy])
        dv = v * (v @ g)
        v = v + dv
        max_drift = max(max_drift, abs(np.linalg.norm(v) - C_SPEED))
        x += v[0]
        y += v[1]
    return max_drift

drift_long = longitudinal_speed_drift(th_ref, b, L)
print(f"  (ii) 纵向候选 |v| 单拍最大漂移 = {drift_long:.3e}（期望 > 1e-8：破光速不变，出局）")

# (iii) snap 候选破弱场线性：单拍最大旋转角不随场强缩小
def snap_max_step_rotation(theta_0, b, L):
    x = -L
    max_rot = 0.0
    for _ in range(int(2.0 * L)):
        gx, gy = grad_theta(x, b, theta_0)
        gn = math.hypot(gx, gy)
        if gn > 0:
            max_rot = max(max_rot, abs(math.acos(max(-1.0, min(1.0, gx / gn)))))
        x += C_SPEED
    return max_rot

rot_1 = snap_max_step_rotation(0.002, b, L)
rot_2 = snap_max_step_rotation(0.001, b, L)
snap_ratio = rot_2 / rot_1
lock_rot_1 = 2.0 * abs(grad_theta(0.0, b, 0.002)[1])
lock_rot_2 = 2.0 * abs(grad_theta(0.0, b, 0.001)[1])
lock_ratio = lock_rot_2 / lock_rot_1
print(f"  (iii) snap 单拍最大旋转角（全路径，远端几何长尾主导）：theta_0=0.002 档 {math.degrees(rot_1):.2f} 度，"
      f"0.001 档 {math.degrees(rot_2):.2f} 度，减半档比值 {snap_ratio:.3f}"
      f"（期望 ~1.00：旋转角不随场强退场，破弱场线性，出局）")
print(f"        对照 lock 单拍旋转角减半档比值 = {lock_ratio:.3f}（期望 0.500：与写入份额同阶线性退场）")

print("  解析排除行：r_hat 类（源方位，破局域性）；标量 x 固定轴类（破各向同性）；"
      "吃绝对势/绝对旗向类（平坦空间无意义）；纵向类（破速率不变，实测见 ii）；"
      "snap 类（破弱场线性，实测见 iii）。存活集 = {kappa (v_hat x g) 旋转形式}，恰一。")
a1_pass = (flat_ok and drift_long > 1e-8 and abs(snap_ratio - 1.0) < 0.2
           and abs(lock_ratio - 0.5) < 0.05)
VERDICT["A1"] = a1_pass
print(f"  [A1] 判定：{'通过' if a1_pass else '未通过'}")

# ----------------------------------------------------------------------
# A2. 失配发散：强度归一（不经测地线一致性）
# ----------------------------------------------------------------------
print("\n[A2] 失配判据：kappa = 1 恒零，kappa < 1 随路径无界（锁相论证）")
kappas = (1.0, 0.75, 0.5, 0.25)
Ms = (1, 2, 4, 8)
m_table = {}
for kap in kappas:
    m_table[kap] = [misalignment(kap, M) for M in Ms]
    print(f"  kappa = {kap}: |m| = {['%.4e' % vv for vv in m_table[kap]]}（M = {list(Ms)}）")

n_p1, n_ref1, twist1, rmin1 = flyby(th_ref, b, L, 1.0, 0.0, 0.0)
print(f"  读数：单段帧扭转合计 = {twist1:.6e}（对照 603 偏转角 -1.600144e-04：同一量，搬运帧口径）")

r25 = m_table[0.25][1] / m_table[0.25][0]
r50 = m_table[0.5][1] / m_table[0.5][0]
r75 = m_table[0.75][1] / m_table[0.75][0]
ratios_ok = all(abs(r - 2.0) < 0.05 for r in (r25, r50, r75))
print(f"  倍段比（M=1 到 2）：kappa=0.25/0.5/0.75 -> {r25:.4f} / {r50:.4f} / {r75:.4f}（期望 2.000 +/- 0.05）")

m4 = (m_table[0.25][2], m_table[0.5][2], m_table[0.75][2])
lin = (m4[0] / m4[2], m4[1] / m4[2])
print(f"  (1-kappa) 线性（M=4）：m(0.25)/m(0.75) = {lin[0]:.4f}（期望 3.000），"
      f"m(0.5)/m(0.75) = {lin[1]:.4f}（期望 2.000）")
zero_ok = all(vv == 0.0 for vv in m_table[1.0])
print(f"  kappa = 1 恒零（M = 1/2/4/8 逐位）= {zero_ok}")

a2_pass = (zero_ok and ratios_ok and abs(lin[0] - 3.0) < 0.15 and abs(lin[1] - 2.0) < 0.1)
VERDICT["A2"] = a2_pass
print(f"  [A2] 判定：{'通过' if a2_pass else '未通过'}（失锁无界与满周闭合锁相不相容 -> 强度唯一为 1）")

# ----------------------------------------------------------------------
# A3. 跨轮逐位自检
# ----------------------------------------------------------------------
print("\n[A3] 跨轮逐位自检（第 603 号基线）")
d606 = run_particle_603(th_ref, b, L)
target = -0.00016001444680876443
print(f"  本轮移植件偏转角 = {d606!r}，603 基线 = {target!r}，逐位一致 = {d606 == target}")
a3_pass = (d606 == target)
VERDICT["A3"] = a3_pass
print(f"  [A3] 判定：{'通过' if a3_pass else '未通过'}")

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax = axes[0]
for kap in (0.25, 0.5, 0.75):
    ax.plot(Ms, m_table[kap], "o-", lw=1.2, label=f"kappa = {kap}")
ax.plot(Ms, m_table[1.0], "s-", lw=1.2, label="kappa = 1（恒零）")
ax.set_xlabel("飞掠段数 M")
ax.set_ylabel("失配 |m|（rad）")
ax.set_title("失配随路径线性无界（A2）：仅 kappa=1 锁相")
ax.legend(fontsize=9)

ax = axes[1]
kgrid = (0.25, 0.5, 0.75)
vals = [m_table[k][2] for k in kgrid]
kline = np.linspace(0.2, 1.0, 50)
ax.plot(kgrid, vals, "o", label="实测 |m|(M=4)")
ax.plot(kline, [(1 - k) * vals[2] / 0.25 for k in kline], "--", color="gray",
        label="(1 - kappa) 线性导引")
ax.plot([1.0], [0.0], "s", color="C3", label="kappa = 1：恒零")
ax.set_xlabel("对齐强度 kappa")
ax.set_ylabel("失配 |m|（rad，M=4）")
ax.set_title("失配随 (1-kappa) 线性（A2）：强度唯一")
ax.legend(fontsize=9)
fig.suptitle("对齐必然性轮（第 606 号）：形式唯一性与失配判据", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG_PATH, dpi=150)
print(f"\n图已存：{IMG_PATH}")

# ----------------------------------------------------------------------
# 总判定
# ----------------------------------------------------------------------
print("\n" + "=" * 76)
print("总判定")
print("=" * 76)
npass = 0
for k in ("A1", "A2", "A3"):
    ok = VERDICT[k]
    npass += int(ok)
    print(f"  {k}：{'通过' if ok else '未通过'}")
print(f"  合计：{npass}/3")
