# -*- coding: utf-8 -*-
"""
最小验证：守恒链判据轮（第 609 号，判据 Y1 至 Y4）
Y 线阶段二载体实现（候选二独立案）加阶段三平衡判据与对表
（[[理论攻坚/离散物质流攻坚计划：路线乙守恒链的载体立案、平衡判据与对表]]）

模型设定（场与度规口径逐字移植第 603 号；动力学原理换为守恒链）：
  背景：中心源倾角场 theta(r) = theta_0 r_0 / r；度规识别（定义层）
    g_00 = -(1-2 theta)，g_ij = (1+2 theta) delta_ij；c = 1 格/拍，每拍 dt = 1。
  载体（候选二，最小碰撞收支表）：尘埃颗粒携带局域动量方向角 pi_ang（局域系分量），
  搬运帧角 n_ref 随场扭转演化；每拍帧扭转 delta_f = 2 c (v_hat x grad theta)_z dt 为
  写入场给定（前提 R-Y：场的标架联络由写入记录供给，其与度规联络的一致性由
  判据第 603 号 G1c 机器级确认）；坐标速度方向 = n_ref + pi_ang。
  尘埃形态（被试对象）：内部动量源为零，pi_ang 逐位常数（收支表平衡的机制内容）；
  坐标动量的每拍变化全部由帧重表达承担（收支恒等式，残差逐位零）。

判据与理论期望（先立后测）：
  Y1 收支恒等与测地线映射：dust 档 pi_ang 逐位常数、收支残差
    max_k |(dir_{k+1}-dir_k) - delta_f_k| = 0（构造恒等式，结构层确认）；
    轨迹与真 null 测地线 ODE 末方向偏差 < 1e-10（实测预期机器级 ~1e-12，
    预算 = RK4 底噪 x 100）。
  Y2 形态替身（D2，区分力）：内部动量源替身 pi_ang += eps*dt（非尘埃形态：
    动量自我注入）在 eps = 1e-8/2e-8 下轨迹偏离测地线线性放大（比值 2.000 +/- 0.05）
    且远超 10 x Y1 偏差（FAIL），尘埃形态承重确认。
  Y3 对表（D3）：乙轨迹（eps=0）与甲锁定规则（第 603 号 Euler 口径逐字移植件）
    末方向偏差 < 1e-9（两者同为测地线的离散实现，方案差单拍 O(dv^3)）。
  Y4 收支替身（D1 对照，改掉它就 FAIL）：收支不平衡替身（每拍无账目对应的
    额外旋转 delta_b = 1e-8）使轨迹偏离测地线 > 10 x Y1 偏差（FAIL），
    收支平衡判据有牙。

可逆性等级：Y1 恒等式半边与 Y3 为判据稳健（逐位/机器级）；Y2、Y4 阈值类为
  判据敏感（档位依赖，外推须重跑）。
前提依赖：R-Y（场标架联络由写入记录供给，一致性归判据第 603 号 G1c）；
  定义层识别式 h_00 = 2 theta；尘埃形态与归一化为被试对象非前提。
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 76)
print("守恒链判据轮（第 609 号，Y1 至 Y4）：乙载体独立案加对表")
print("=" * 76)

C_SPEED = 1.0
r0 = 1.0
IMG_PATH = r"D:\software\TianXingFC\images\conservation_chain.png"

VERDICT = {}

def grad_theta(x, y, theta_0):
    """nabla theta = -theta_0 r0 r_hat / r^2（r -> 0 时设零，避开源心）"""
    r = np.hypot(x, y)
    if r < 1e-6:
        return 0.0, 0.0
    g = -theta_0 * r0 / r ** 3
    return g * x, g * y

# ----------------------------------------------------------------------
# 移植件：第 603 号脚本整体加载（geo_ode 与 run_particle 口径逐字）
# ----------------------------------------------------------------------
import io
import contextlib
_ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(open(r"D:\software\TianXingFC\验证脚本\minimal_verification_geodesic_lock.py",
                      encoding="utf-8").read(), "lock603", "exec"), _ns)
geo_ode = _ns["geo_ode"]
run_particle_603 = _ns["run_particle"]

# ----------------------------------------------------------------------
# 乙载体：局域动量守恒加帧重表达（候选二最小碰撞收支表）
# ----------------------------------------------------------------------
def y_evo(theta_0, b, L, eps=0.0, delta_b=0.0, track=False):
    """尘埃颗粒：pi_ang 逐位常数（dust，eps=0）；n_ref 按场扭转演化；
    坐标方向 = n_ref + pi_ang；delta_b 为收支不平衡替身。返回 (末方向角, 收支残差, 轨迹)"""
    x, y = -L, b
    n_ref, pi_ang = 0.0, 0.0
    T = int(2.0 * L / C_SPEED)
    resid = 0.0
    traj = []
    rmin = math.hypot(x, y)
    for _ in range(T):
        gx, gy = grad_theta(x, y, theta_0)
        g = np.array([gx, gy])
        dirc = n_ref + pi_ang
        vhat = np.array([math.cos(dirc), math.sin(dirc)])
        cross = vhat[0] * g[1] - vhat[1] * g[0]
        df = 2.0 * C_SPEED * cross
        n_ref += df
        pi_ang += eps                       # 内部源替身（dust：eps = 0 逐位）
        dirc_new = n_ref + pi_ang + delta_b  # 收支不平衡替身（dust：delta_b = 0）
        resid = max(resid, abs((dirc_new - dirc) - df - eps))
        n_ref += delta_b                     # 无收支对应的额外旋转
        if track and len(traj) < 4000:
            traj.append((x, y))
        x += C_SPEED * math.cos(n_ref + pi_ang)
        y += C_SPEED * math.sin(n_ref + pi_ang)
        rmin = min(rmin, math.hypot(x, y))
    return math.atan2(math.sin(n_ref + pi_ang), math.cos(n_ref + pi_ang)), resid, traj

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print("\n[冒烟] 场量与帧扭转向")
b, L, th_ref = 50.0, 6000, 0.002
gx0, gy0 = grad_theta(0.0, b, th_ref)
print(f"  最近点 |grad theta| = {math.hypot(gx0, gy0):.3e}（期望 ~ {th_ref / b ** 2:.3e}）")
cross0 = gy0
print(f"  最近点单拍帧扭转 = {2.0 * cross0:.3e}（负号朝源，与第 603/606 号同值口径）")

# ----------------------------------------------------------------------
# Y1. 收支恒等与测地线映射
# ----------------------------------------------------------------------
print("\n[Y1] 尘埃收支恒等与测地线映射（候选二载体）")
d_y, resid, traj_y = y_evo(th_ref, b, L, track=True)
print(f"  收支残差 max|(dir+ - dir) - df - eps| = {resid:.3e}（期望 < 1e-15：数学恒等式，"
      "残差为浮点结合序机器噪声）")
ode_full, _, _ = geo_ode(th_ref, b, L, spatial=True)
dev_y1 = abs(d_y - ode_full)
print(f"  乙轨迹末方向 = {d_y:.9e}，ODE 参照 = {ode_full:.9e}，偏差 = {dev_y1:.3e}（期望 < 1e-10）")
y1_pass = (resid < 1e-15 and dev_y1 < 1e-10)
VERDICT["Y1"] = y1_pass
print(f"  [Y1] 判定：{'通过' if y1_pass else '未通过'}")

# ----------------------------------------------------------------------
# Y2. 形态替身（尘埃形态承重）
# ----------------------------------------------------------------------
print("\n[Y2] 形态替身：内部动量源瓦解测地线（尘埃形态承重）")
devs = {}
for eps in (0.0, 1e-8, 2e-8):
    d_e, _, _ = y_evo(th_ref, b, L, eps=eps)
    devs[eps] = abs(d_e - ode_full)
    print(f"  eps = {eps:.1e}: 偏离测地线 {devs[eps]:.3e}")
ratio_y2 = devs[2e-8] / devs[1e-8]
print(f"  线性比 dev(2e-8)/dev(1e-8) = {ratio_y2:.4f}（期望 2.000 +/- 0.05）")
y2_pass = (devs[1e-8] > 10 * dev_y1 and devs[2e-8] > devs[1e-8]
           and abs(ratio_y2 - 2.0) < 0.05)
VERDICT["Y2"] = y2_pass
print(f"  [Y2] 判定：{'通过' if y2_pass else '未通过'}（尘埃形态是承重墙：内部动量源即偏离测地线）")

# ----------------------------------------------------------------------
# Y3. 对表（乙对甲）
# ----------------------------------------------------------------------
print("\n[Y3] 对表：乙守恒链轨迹对甲锁定规则轨迹（裁决点 2 材料其一）")
d_jia = run_particle_603(th_ref, b, L)[0]
dev_y3 = abs(d_y - d_jia)
target = -0.00016001444680876443
print(f"  甲（第 603 号 Euler 口径）= {d_jia!r}，基线逐位自检 = {d_jia == target}")
print(f"  乙（守恒链）= {d_y:.9e}，|乙 - 甲| = {dev_y3:.3e}（期望 < 1e-9：同为测地线离散实现，方案差单拍 O(dv^3)）")
y3_pass = (d_jia == target and dev_y3 < 1e-9)
VERDICT["Y3"] = y3_pass
print(f"  [Y3] 判定：{'通过' if y3_pass else '未通过'}")

# ----------------------------------------------------------------------
# Y4. 收支替身（平衡判据有牙）
# ----------------------------------------------------------------------
print("\n[Y4] 收支替身：无收支对应的额外旋转瓦解轨迹（平衡判据有牙）")
d_b, _, _ = y_evo(th_ref, b, L, delta_b=1e-8)
dev_y4 = abs(d_b - ode_full)
print(f"  delta_b = 1e-8/拍: 偏离测地线 {dev_y4:.3e}（期望 > 10 x {dev_y1:.1e}：FAIL）")
y4_pass = dev_y4 > 10 * dev_y1
VERDICT["Y4"] = y4_pass
print(f"  [Y4] 判定：{'通过' if y4_pass else '未通过'}")

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax = axes[0]
ax.plot([p[0] for p in traj_y], [p[1] - b for p in traj_y], "-", lw=1.3,
        label="乙守恒链轨迹（Y3 数值确认与甲重合）")
ax.axhline(0.0, color="gray", lw=0.6)
ax.set_xlim(-1500, 1500); ax.set_ylim(-0.55, 0.25)
ax.set_xlabel("x（格）"); ax.set_ylabel("y - b（格）")
ax.set_title("乙载体轨迹（dust 档）：与甲/测地线重合")
ax.legend(fontsize=9)

ax = axes[1]
eps_list = (1e-8, 2e-8)
ax.plot(eps_list, (devs[1e-8], devs[2e-8]), "o-", label="Y2 内部源替身（FAIL）")
ax.plot([1e-8], [dev_y4], "s", color="C3", label="Y4 收支替身（FAIL）")
ax.axhline(dev_y1, color="gray", ls="--", lw=1, label=f"Y1 机器级底噪 {dev_y1:.1e}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("替身强度（每拍）"); ax.set_ylabel("对测地线偏离（rad）")
ax.set_title("替身全部 FAIL（Y2/Y4）：判据有牙")
ax.legend(fontsize=9)
fig.suptitle("守恒链判据轮（第 609 号）：乙载体、形态承重与对表", fontsize=13)
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
for k in ("Y1", "Y2", "Y3", "Y4"):
    ok = VERDICT[k]
    npass += int(ok)
    print(f"  {k}：{'通过' if ok else '未通过'}")
print(f"  合计：{npass}/4")
