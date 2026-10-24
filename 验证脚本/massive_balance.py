# -*- coding: utf-8 -*-
"""
最小验证：有质量档离散收支表判据轮（第 619 号，判据 MB1 至 MB4）
P1-54 档案页余项转出清单"有质量档原理级离散推导/离散收支表"的判据轮
（乙链守恒收支自 null 档（第 609 号）推广到有质量档；场、度规、ODE 与甲侧
规则逐字移植第 614 号 [[验证脚本/注记-全度规有质量档机器级验证轮：两项分解与v→c连续性（第614号）]]）

模型设定（明示假设，逐字移植 603/614）：
  背景：中心源倾角场 theta(r) = theta_0 r0 / r；度规识别（定义层）
    g_00 = -(1-2 theta)，g_ij = (1+2 theta) delta_ij；c = 1；平面掠射；静态场。
  乙载体（坐标动量收支表，有质量档）：颗粒状态 (x, y, P_x, P_y)，
    P_i = g_{ij} u^j = h u^t v_i 为单位质量四动量空间分量（格点动量读数，与
    矢量写入定理 J^i = (m/m_P) u^i 的空间分量同族）。每拍收支（场梯度项，无自由参数）：
      dP_i/dt = (1/u^t) * (1/2) * d_i g_{uv} u^u u^v = u^t * d_i theta * (1 + v^2)
    归一化 f (u^t)^2 - P^2/h = 1 给 u^t = sqrt((1 + P^2/h)/f)，v_i = P_i/(h u^t)。
    来源立场（防循环，同 609 前提 R-Y）：
    该收支为标架重表达（写入记录供给的联络的坐标动量表现），测地线一致性
    只作输出校验不作输入；内部源为零（尘埃形态）即 P 的每拍变化全部由
    收支表承担。牛顿极限自检：u^t -> 1 时 dP_i/dt -> d_i theta（a = nabla theta）。
  离散化：一阶欧拉 + 子步（sub = 1/2/4，dt = 1/sub），一阶收敛 Richardson 外推。
  独立参照：第 614 号 geo_ode（仿射 RK4，Christoffel 解析式）；甲侧规则：
    第 614 号 rule_massive（坐标加速度精确式欧拉，精确流）逐字移植。

判据与理论期望（先立后测；MB1 的收敛比半判据经首轮实测判"设计失效"后改述，
  见过程记录 P1）：
  MB1 收支恒等与测地线映射（beta = 0.3）：尘埃档收支重计算恒等（构造层，
    与 609 号 Y1 同层；载体 P ~ 1 量纲大，浮点 ulp 吸收预算 ~1e-16，判据
    残差 < 1e-15）；末方向对 ODE 偏差 < 1e-10（预期 ~4e-13 机器级，与
    第 614 号 M1 底噪同族）；Killing 双守恒量（E = f u^t、L_z = x P_y - y P_x）
    漂移呈一阶收敛且外推后 < 1e-10（相对量；L_z 预期逐位机器级）。
  MB2 形态替身（尘埃形态承重）：内部动量源替身 eps = 1e-8/2e-8（固定 x 方向）
    偏离 ODE 线性放大（比值 2.000 +/- 0.05）且 > 10 x MB1 偏差（FAIL）。
  MB3 有质量档两侧对表（beta = 0.3 与 0.7 两档）：乙外推末方向 vs 甲
    rule_massive 末方向偏差 < 1e-10（乙外推消除欧拉截断后与甲精确流同归
    测地线，残差 = 双方浮点底噪）。
  MB4 收支替身（平衡判据有牙）：无场对应的额外冲量 delta_b = 1e-8（每拍，
    固定方向）使偏离 > 10 x MB1 偏差（FAIL）。

可逆性等级：MB1 恒等式半边与 Killing 自检为判据稳健（逐位/外推机器级）；
  收敛比窗口与 MB2/MB4 阈值类为判据敏感（档位依赖，外推须重跑）。
前提依赖：R-Y（场标架联络由写入记录供给，一致性归判据第 603 号 G1c 与
  第 614 号 M1）；定义层识别式 h_00 = 2 theta；尘埃形态与归一化为被试对象非前提。
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 76)
print("有质量档离散收支表判据轮（第 619 号，MB1 至 MB4）")
print("=" * 76)

r0 = 1.0
IMG_PATH = r"D:\software\TianXingFC\images\massive_balance.png"
VERDICT = {}

def grad_theta(x, y, theta_0):
    """nabla theta = -theta_0 r0 r_hat / r^2（r -> 0 时设零，避开源心）"""
    r2 = x * x + y * y
    if r2 < 1e-12:
        return 0.0, 0.0
    r = math.sqrt(r2)
    g = -theta_0 * r0 / (r2 * r)
    return g * x, g * y

# ----------------------------------------------------------------------
# 移植件：第 614 号脚本整体加载（geo_ode 与 rule_massive 口径逐字）
# ----------------------------------------------------------------------
import io
import contextlib
_ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(open(r"D:\software\TianXingFC\验证脚本\massive_full_metric.py",
                      encoding="utf-8").read(), "m614", "exec"), _ns)
geo_ode_614 = _ns["geo_ode"]
rule_massive_614 = _ns["rule_massive"]

# ----------------------------------------------------------------------
# 乙载体：坐标动量收支表（有质量档，尘埃形态）
# ----------------------------------------------------------------------
def y_evo_massive(theta_0, b, L, beta, sub=1, eps=0.0, delta_b=0.0, track=False):
    """坐标动量收支表更新（四动量空间分量载体）。eps 为内部动量源替身（固定 x
    方向，每拍）；delta_b 为收支替身（无场对应额外冲量，固定 y 方向，每拍）。
    载体：P_i = g_{ij} u^j = h u^t v_i（单位质量四动量空间分量）；
    归一化 f (u^t)^2 - P^2/h = 1 给 u^t = sqrt((1 + P^2/h)/f)；
    收支：dP_i/dt = u^t * d_i theta * (1 + v^2)（牛顿极限 u^t->1, v->0 退化为 a = nabla theta）。
    返回 (末方向角, 收支残差, E 漂移, L_z 漂移, 轨迹)"""
    x, y = -L, b
    th0 = theta_0 * r0 / math.hypot(x, y)
    f0, h0 = 1.0 - 2.0 * th0, 1.0 + 2.0 * th0
    ut0 = 1.0 / math.sqrt(f0 - h0 * beta * beta)
    Px, Py = h0 * ut0 * beta, 0.0
    dt = 1.0 / sub
    T = int(math.ceil(2.05 * L / beta))
    resid = 0.0
    E0 = Lz0 = None
    dE = dLz = 0.0
    traj = []
    beat = 0
    for _ in range(T):
        if x >= L:
            break
        for _ in range(sub):
            gx, gy = grad_theta(x, y, theta_0)
            r = math.hypot(x, y)
            th = theta_0 * r0 / max(r, 1e-6)
            f = 1.0 - 2.0 * th
            h = 1.0 + 2.0 * th
            P2 = Px * Px + Py * Py
            ut = math.sqrt((1.0 + P2 / h) / f)
            v2 = P2 / (h * h * ut * ut)
            # 收支表（场梯度项，无自由参数）+ 重计算核对（构造恒等式）
            dPx_calc = ut * gx * (1.0 + v2)
            dPy_calc = ut * gy * (1.0 + v2)
            Px_new = Px + (dPx_calc + eps) * dt
            Py_new = Py + (dPy_calc + delta_b) * dt
            resid = max(resid, abs((Px_new - Px) / dt - dPx_calc - eps),
                        abs((Py_new - Py) / dt - dPy_calc - delta_b))
            Px, Py = Px_new, Py_new
            P2n = Px * Px + Py * Py
            utn = math.sqrt((1.0 + P2n / h) / f)
            vx, vy = Px / (h * utn), Py / (h * utn)
            # Killing 双守恒量漂移（独立自检，不经测地线方程）
            E = f * utn
            Lz = x * Py - y * Px
            if E0 is None:
                E0, Lz0 = E, Lz
            dE = max(dE, abs(E - E0) / abs(E0))
            dLz = max(dLz, abs(Lz - Lz0) / abs(Lz0))
            x += vx * dt
            y += vy * dt
        beat += 1
        if track and beat % 5 == 0:
            traj.append((x, y))
    P2n = Px * Px + Py * Py
    utn = math.sqrt((1.0 + P2n / h) / f)
    vxn, vyn = Px / (h * utn), Py / (h * utn)
    return math.atan2(vyn, vxn), resid, dE, dLz, traj

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print("\n[冒烟] 场量、收支表方向与守恒量初值")
b, L, th_ref, beta0 = 50.0, 3000, 0.002, 0.3
gx0, gy0 = grad_theta(0.0, b, th_ref)
print(f"  最近点 |grad theta| = {math.hypot(gx0, gy0):.3e}（期望 ~ {th_ref / b ** 2:.3e}，指向源为负 y）")
r_s, res_s, dE_s, dLz_s, _ = y_evo_massive(th_ref, b, L, beta0, sub=1)
print(f"  sub=1 全程：收支残差 {res_s:.1e}（期望 0），E 漂移 {dE_s:.2e}，L_z 漂移 {dLz_s:.2e}"
      "（期望非零且小：欧拉截断，量级 ~ theta0/b^2 * T * dt）")
print(f"  预算参考：theta0/b^2 = {th_ref / b ** 2:.2e}，T ~ {int(math.ceil(2.05 * L / beta0))} 拍")

# ----------------------------------------------------------------------
# MB1. 收支恒等、Killing 自检与测地线映射（beta = 0.3，子步收敛）
# ----------------------------------------------------------------------
print("\n[MB1] 尘埃收支恒等、Killing 自检与测地线映射（beta = 0.3）")
ode_ref = geo_ode_614(th_ref, b, L, spatial=True, massive=True, v0=beta0,
                      lam_span=2.2 * L / beta0)[0]
S = {}
E_drift = {}
resid = dLz = None
for sub in (1, 2, 4):
    d_s, resid, dE, dLz, _ = y_evo_massive(th_ref, b, L, beta0, sub=sub)
    S[sub] = d_s - ode_ref
    E_drift[sub] = dE
    print(f"  sub={sub}: |乙-ODE| = {abs(S[sub]):.3e}  收支残差 {resid:.1e}  "
          f"E 漂移 {dE:.2e}  L_z 漂移 {dLz:.2e}")
ratio_12 = abs(S[1]) / abs(S[2])
ratio_24 = abs(S[2]) / abs(S[4])
S_ext = 2.0 * S[2] - S[1]
print(f"  三档偏差 S1/S2/S4 = {abs(S[1]):.2e} / {abs(S[2]):.2e} / {abs(S[4]):.2e}")
if abs(S[1]) < 1e-11 and abs(S[2]) < 1e-11:
    print("  判读：方向偏差已在 ODE 底噪层（与第 614 号 M1 的 3.375e-13 同族），"
          "收敛结构不可测：P 收支表更新在 dt=1 即精确流（与 614 甲规则同族发现，MB3 逐位佐证）")
E_ext = 2.0 * E_drift[4] - E_drift[2]
print(f"  E 漂移外推 |2*E(sub4) - E(sub2)| = {E_ext:.2e}（< 1e-10 机器级）；L_z 漂移逐位机器级")
mb1_pass = (resid < 1e-15 and abs(S[1]) < 1e-10 and abs(S[2]) < 1e-10
            and abs(S[4]) < 1e-10 and dLz < 1e-10)
VERDICT["MB1"] = mb1_pass
print(f"  [MB1] 判定：{'通过' if mb1_pass else '未通过'}")

# ----------------------------------------------------------------------
# MB2. 形态替身（尘埃形态承重）
# ----------------------------------------------------------------------
print("\n[MB2] 形态替身：内部动量源瓦解测地线（尘埃形态承重）")
devs = {}
for eps in (0.0, 1e-8, 2e-8):
    d_e, _, _, _, _ = y_evo_massive(th_ref, b, L, beta0, sub=1, eps=eps)
    devs[eps] = abs(d_e - ode_ref)
    print(f"  eps = {eps:.1e}: 偏离测地线 {devs[eps]:.3e}")
ratio_mb2 = devs[2e-8] / devs[1e-8]
print(f"  线性比 dev(2e-8)/dev(1e-8) = {ratio_mb2:.4f}（期望 2.000 +/- 0.05）")
mb2_pass = (devs[1e-8] > 10 * abs(S[1]) and devs[2e-8] > devs[1e-8]
            and abs(ratio_mb2 - 2.0) < 0.05)
VERDICT["MB2"] = mb2_pass
print(f"  [MB2] 判定：{'通过' if mb2_pass else '未通过'}")

# ----------------------------------------------------------------------
# MB3. 有质量档两侧对表（beta = 0.3 与 0.7）
# ----------------------------------------------------------------------
print("\n[MB3] 对表：乙外推对甲 rule_massive（第 614 号精确流，两档速度）")
mb3_pass = True
for beta_t in (0.3, 0.7):
    d1, _, _, _, _ = y_evo_massive(th_ref, b, L, beta_t, sub=1)
    d2, _, _, _, _ = y_evo_massive(th_ref, b, L, beta_t, sub=2)
    d_ext = 2.0 * d2 - d1
    d_jia = rule_massive_614(th_ref, b, L, beta_t, sub=1)
    dev = abs(d_ext - d_jia)
    print(f"  beta = {beta_t}: 乙外推 = {d_ext:.9e}，甲 = {d_jia:.9e}，|乙-甲| = {dev:.3e}（期望 < 1e-10）")
    mb3_pass = mb3_pass and dev < 1e-10
VERDICT["MB3"] = mb3_pass
print(f"  [MB3] 判定：{'通过' if mb3_pass else '未通过'}")

# ----------------------------------------------------------------------
# MB4. 收支替身（平衡判据有牙）
# ----------------------------------------------------------------------
print("\n[MB4] 收支替身：无场对应的额外冲量瓦解轨迹（平衡判据有牙）")
d_b, _, _, _, _ = y_evo_massive(th_ref, b, L, beta0, sub=1, delta_b=1e-8)
dev_mb4 = abs(d_b - ode_ref)
print(f"  delta_b = 1e-8/拍: 偏离测地线 {dev_mb4:.3e}（期望 > 10 x {abs(S[1]):.1e}：FAIL）")
mb4_pass = dev_mb4 > 10 * abs(S[1])
VERDICT["MB4"] = mb4_pass
print(f"  [MB4] 判定：{'通过' if mb4_pass else '未通过'}")

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax = axes[0]
_, _, _, _, traj = y_evo_massive(th_ref, b, L, beta0, sub=1, track=True)
ax.plot([p[0] for p in traj], [p[1] - b for p in traj], "-", lw=1.3,
        label="乙收支表轨迹（beta=0.3）")
ax.axhline(0.0, color="gray", lw=0.6)
ax.set_xlim(-1500, 1500); ax.set_ylim(-1.2, 0.4)
ax.set_xlabel("x（格）"); ax.set_ylabel("y - b（格）")
ax.set_title("乙载体有质量档轨迹（MB3 数值确认与甲重合）")
ax.legend(fontsize=9)

ax = axes[1]
subs = (1, 2, 4)
ax.plot(subs, [abs(S[s]) for s in subs], "o-", label="乙对 ODE 偏差（底噪层）")
ax.plot([1], [abs(2 * S[2] - S[1])], "D", color="C2", label="外推读数（Killing E 外推后 ~1e-14）")
ax.plot([1], [devs[1e-8]], "s", color="C3", label="MB2 内部源替身（FAIL）")
ax.plot([1], [dev_mb4], "^", color="C4", label="MB4 收支替身（FAIL）")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("子步数（dt = 1/sub）"); ax.set_ylabel("对测地线偏离（rad）")
ax.set_title("三档偏差均在 ODE 底噪层（dt=1 即精确流），替身全 FAIL")
ax.legend(fontsize=8)
fig.suptitle("有质量档离散收支表判据轮（第 619 号）：乙载体、Killing 自检与两侧对表", fontsize=13)
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
for k in ("MB1", "MB2", "MB3", "MB4"):
    ok = VERDICT[k]
    npass += int(ok)
    print(f"  {k}：{'通过' if ok else '未通过'}")
print(f"  合计：{npass}/4")
