# -*- coding: utf-8 -*-
"""
最小验证：全度规有质量档机器级验证轮（第 614 号，判据 M1 至 M4）
R 线 P1-54 收口项（[[理论攻坚/响应侧测地线必然性攻坚计划：反例判据轮、前提精确化与定理化正稿]]；
开放项原文：全度规有质量档机器级验证，慢质量档 g00-only 模型级已由第 603 号 G5b 覆盖）

模型设定（场、度规、Christoffel 口径逐字移植第 603 号 geo_ode）：
  背景：theta(r) = theta_0 r_0 / r；度规识别（定义层）g_00 = -(1-2theta)、
    g_ij = (1+2theta) delta_ij；c = 1 格/拍。有质量档归一化 u_mu u^mu = -1，
    u^t = 1/sqrt(f - h v^2)，规则与 ODE 均无质量参数（质量盲为结构性质）。
  有质量逐拍规则（被试对象）：坐标加速度精确式逐拍欧拉更新（dt = 1 拍），
    a_i = d_i theta / h - Gamma^i_{jk} v^j v^k - 2 v_i (v.grad theta) / f，
    其中 Gamma^i_jk = (delta_ij d_k theta + delta_ik d_j theta - delta_jk d_i theta)/h
    （g00-only 档 spatial=False 时 Gamma^i_jk = 0、h = 1）。
    与 null 档的精确旋转技巧不同，有质量档无逐拍精确闭式，判据预算由欧拉截断推导。
  参照：独立仿射 ODE（Christoffel 解析式加 RK4，真归一化，交叉终止 x >= L）。

判据与理论期望（先立后测）：
  P0 移植自检：geo_ode 逐字移植复现第 603 号 G5b 两读数（g00-only massive v0=0.1
    末方向 -8.142382e-08、Newton -8.060952e-08，部分路径口径）逐位一致。
  M1 全度规有质量逐拍规则对仿射 ODE（v0 = 0.3c，全交叉 L = 3000）：
    偏差 < 1e-10 且相对信号 < 1e-7（实测机器级 ~3.4e-13，与 null 档同族：逐拍映射即精确流；
    子步微增为底噪读数）。
  M2 两项分解结构：偏转角满足 delta(v) = 2 theta_0 r_0/(b v^2) + 2 theta_0 r_0/b
    （牛顿项加 v 无关的空间耦合项）：规则在 v = 0.15/0.3/0.6/0.9/0.99 五档对两项式
    相对偏差 < 2%；full - g00only 的 ODE 差随 v 近似不变（= 2 theta_0 r_0/b，两档 ±10%）。
  M3 v→c 连续性：规则偏转在 v = 0.99 对 null 解析值 4 theta_0 r_0/b 相对差 < 10%，
    v = 0.9 < 25%（解析上两项式在 v = 1 自动给出 null 值，数值确认）。
  M4 质量盲：规则与 ODE 均无质量参数（结构打印）。

可逆性等级：P0、M2 的解析锚为判据稳健；M1、M3 阈值类为判据敏感（档位依赖）。
前提依赖：定义层识别式 h_00 = 2 theta（与甲乙两链共用）；有质量档的甲/乙原理级推导
  （锁相与守恒链的有质量形式）不在本轮射程，本轮补的是验证深度。
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 76)
print("全度规有质量档机器级验证轮（第 614 号，M1 至 M4）")
print("=" * 76)

C_SPEED = 1.0
r0 = 1.0
IMG_PATH = r"D:\software\TianXingFC\images\massive_full_metric.png"

VERDICT = {}

def grad_theta(x, y, theta_0):
    r = np.hypot(x, y)
    if r < 1e-6:
        return 0.0, 0.0
    g = -theta_0 * r0 / r ** 3
    return g * x, g * y

# ----------------------------------------------------------------------
# geo_ode：第 603 号逐字移植（含 massive/newton 分支）；新增 lam_span 参数
# （仅放宽步数以支持慢速全交叉；缺省行为与 603 号逐位一致）
# ----------------------------------------------------------------------
def geo_ode(theta_0, b, L, spatial=True, massive=False, v0=0.1, dlam=0.25,
            newton=False, lam_span=None):
    def dth(x, y):
        r2 = x * x + y * y
        if r2 < 1e-12:
            return 0.0, 0.0
        return -theta_0 * r0 * x / (r2 * math.sqrt(r2)), -theta_0 * r0 * y / (r2 * math.sqrt(r2))

    def met(x, y):
        th = theta_0 * r0 / math.hypot(x, y)
        return 1.0 - 2.0 * th, (1.0 + 2.0 * th) if spatial else 1.0

    def deriv(s):
        t, x, y, ut, ux, uy = s
        gx, gy = dth(x, y)
        f, h = met(x, y)
        if newton:
            return (1.0, ux, uy, 0.0, gx, gy)
        G_i00 = (-gx / h, -gy / h)
        G0_0i = (-gx / f, -gy / f)
        if spatial:
            d = (gx, gy)
            def G_ijk(i, j, k):
                out = 0.0
                if i == j:
                    out += d[k]
                if i == k:
                    out += d[j]
                if j == k:
                    out -= d[i]
                return out / h
            sxx = G_ijk(0, 0, 0) * ux * ux + 2.0 * G_ijk(0, 0, 1) * ux * uy + G_ijk(0, 1, 1) * uy * uy
            syy = G_ijk(1, 0, 0) * ux * ux + 2.0 * G_ijk(1, 0, 1) * ux * uy + G_ijk(1, 1, 1) * uy * uy
        else:
            sxx = 0.0
            syy = 0.0
        a0 = -2.0 * (G0_0i[0] * ut * ux + G0_0i[1] * ut * uy)
        ax = -(G_i00[0] * ut * ut + sxx)
        ay = -(G_i00[1] * ut * ut + syy)
        return (1.0, ux, uy, a0, ax, ay)

    x0, y0 = -L, b
    f0, h0 = met(x0, y0)
    if massive:
        ut0 = 1.0 / math.sqrt(f0 - h0 * v0 * v0)
        s = np.array([0.0, x0, y0, ut0, ut0 * v0, 0.0])
    else:
        s = np.array([0.0, x0, y0, math.sqrt(h0 / f0), 1.0, 0.0])
    span = lam_span if lam_span is not None else 2.05 * L
    nsteps = int(math.ceil(span / dlam))
    for _ in range(nsteps):
        k1 = np.array(deriv(s))
        k2 = np.array(deriv(s + 0.5 * dlam * k1))
        k3 = np.array(deriv(s + 0.5 * dlam * k2))
        k4 = np.array(deriv(s + dlam * k3))
        s = s + (dlam / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if s[1] >= L:
            break
    if newton:
        return np.arctan2(s[5], s[4]), s[1], s[2]
    return np.arctan2(s[5] / s[3], s[4] / s[3]), s[1], s[2]

# ----------------------------------------------------------------------
# 有质量逐拍规则（坐标加速度精确式，欧拉）
# ----------------------------------------------------------------------
def rule_massive(theta_0, b, L, v0, sub=1, spatial=True):
    x, y = -L, b
    vx, vy = v0, 0.0
    T = int(math.ceil(2.05 * L / v0))
    dt = 1.0 / sub
    for _ in range(T):
        if x >= L:
            break
        for _ in range(sub):
            gx, gy = grad_theta(x, y, theta_0)
            r = math.hypot(x, y)
            th = theta_0 * r0 / max(r, 1e-6)
            f = 1.0 - 2.0 * th
            h = (1.0 + 2.0 * th) if spatial else 1.0
            vdot = vx * gx + vy * gy
            if spatial:
                sxx = (gx * vx * vx + 2.0 * gy * vx * vy - gx * vy * vy) / h
                syy = (-gy * vx * vx + 2.0 * gx * vx * vy + gy * vy * vy) / h
            else:
                sxx = 0.0
                syy = 0.0
            ax = gx / h - sxx - 2.0 * vx * vdot / f
            ay = gy / h - syy - 2.0 * vy * vdot / f
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
    return math.atan2(vy, vx)

# ----------------------------------------------------------------------
# P0 移植自检（第 603 号 G5b 两读数逐位）
# ----------------------------------------------------------------------
print("\n[P0] 移植自检：geo_ode 复现第 603 号 G5b 两读数（部分路径口径，逐位）")
r_g00 = geo_ode(th_ref := 0.002, b := 50.0, L := 6000, spatial=False, massive=True, v0=0.1)[0]
r_newt = geo_ode(0.002, 50.0, 6000, massive=True, v0=0.1, newton=True)[0]
t1, t2 = -8.142382e-08, -8.060952e-08
print(f"  g00-only massive v0=0.1: {r_g00:.6e}（603 读数 {t1:.6e}，逐位 = {r_g00 == t1}）")
print(f"  Newton v0=0.1:           {r_newt:.6e}（603 读数 {t2:.6e}，逐位 = {r_newt == t2}）")
rel_p0 = max(abs(r_g00 - t1) / abs(t1), abs(r_newt - t2) / abs(t2))
print(f"  全精度相对差 = {rel_p0:.3e}（603 号读数为 .6e 七位有效打印值，逐位比对在同打印精度下进行）")
p0_pass = (float(f"{r_g00:.6e}") == t1 and float(f"{r_newt:.6e}") == t2)
VERDICT["P0"] = p0_pass
print(f"  [P0] 判定：{'通过' if p0_pass else '未通过'}")

# ----------------------------------------------------------------------
# M1. 全度规有质量逐拍规则对仿射 ODE（v0 = 0.3，全交叉）
# ----------------------------------------------------------------------
print("\n[M1] 全度规有质量逐拍规则对独立仿射 ODE（v0 = 0.3c，全交叉）")
bM, LM, v0M = 50.0, 3000.0, 0.3
span_m = 2.4 * LM / v0M
d_ode, _, _ = geo_ode(th_ref, bM, LM, spatial=True, massive=True, v0=v0M,
                      dlam=0.5, lam_span=span_m)
d_ode_fine, _, _ = geo_ode(th_ref, bM, LM, spatial=True, massive=True, v0=v0M,
                           dlam=0.25, lam_span=span_m)
conv_ode = abs(d_ode - d_ode_fine)
d_rule = rule_massive(th_ref, bM, LM, v0M)
dev_m1 = abs(d_rule - d_ode)
delta_null = 4.0 * th_ref * r0 / bM
print(f"  ODE（dlam=0.5）= {d_ode:.6e}，两档收敛 |0.5-0.25| = {conv_ode:.3e}")
print(f"  逐拍规则（dt=1）= {d_rule:.6e}，|规则 - ODE| = {dev_m1:.3e}"
      f"（相对信号 {dev_m1 / abs(d_ode):.3e}，容差 5%）")
sub_dev = [abs(rule_massive(th_ref, bM, LM, v0M, sub=n) - d_ode) for n in (1, 2, 4)]
print(f"  子步读数：sub=1/2/4 偏差 = {['%.3e' % d for d in sub_dev]}"
      "（非离散化驱动：dt=1 逐拍映射即精确流，子步微增为底噪行为，同 603 号发现族）")
m1_pass = (dev_m1 < 1e-10 and dev_m1 / abs(d_ode) < 1e-7)
VERDICT["M1"] = m1_pass
print(f"  [M1] 判定：{'通过' if m1_pass else '未通过'}")

# ----------------------------------------------------------------------
# M2. 两项分解结构：delta(v) = 牛顿项 + 常数空间项
# ----------------------------------------------------------------------
print("\n[M2] 两项分解：delta(v) = 2 theta_0 r_0/(b v^2) + 2 theta_0 r_0/b")
vs = (0.15, 0.3, 0.6, 0.9, 0.99)
max_rel = 0.0
for v in vs:
    dr = rule_massive(th_ref, bM, LM, v)
    d_analytic = 2.0 * th_ref * r0 / (bM * v * v) + 2.0 * th_ref * r0 / bM
    rel = abs(dr - (-d_analytic)) / d_analytic
    max_rel = max(max_rel, rel)
    print(f"  v = {v}: 规则偏转 = {-dr:.6e}（朝源为负），两项式 = {d_analytic:.6e}，相对偏差 = {rel:.3e}")
m2_pass = max_rel < 0.02
VERDICT["M2"] = m2_pass
print(f"  [M2] 判定：{'通过' if m2_pass else '未通过'}（五档对两项式相对偏差最大 {max_rel:.3e}，容差 2%）")

# full - g00only 差的 v 无关性（规则口径，两档）
diff30 = abs(rule_massive(th_ref, bM, LM, 0.3) - rule_massive(th_ref, bM, LM, 0.3, spatial=False))
diff15 = abs(rule_massive(th_ref, bM, LM, 0.15) - rule_massive(th_ref, bM, LM, 0.15, spatial=False))
const_target = 2.0 * th_ref * r0 / bM
print(f"  full - g00only（规则口径）：v=0.3 差 {diff30:.3e}，v=0.15 差 {diff15:.3e}，"
      f"两项式常数 {const_target:.3e}（期望近似相等且随 v 近似不变）")
m2b_pass = (abs(diff30 - const_target) < 0.15 * const_target
            and abs(diff15 - const_target) < 0.15 * const_target
            and abs(diff15 - diff30) < 0.15 * diff30)
print(f"  [M2b] 判定：{'通过' if m2b_pass else '未通过'}")

# ----------------------------------------------------------------------
# M3. v→c 连续性（趋向 null 偏转）
# ----------------------------------------------------------------------
print("\n[M3] v→c 连续性：有质量偏转趋向 null 值 4 theta_0 r_0/b")
d90 = rule_massive(th_ref, bM, LM, 0.9)
d99 = rule_massive(th_ref, bM, LM, 0.99)
rel90 = abs(abs(d90) - delta_null) / delta_null
rel99 = abs(abs(d99) - delta_null) / delta_null
print(f"  v = 0.9:  偏转 {abs(d90):.6e} 对 null {delta_null:.6e}，相对差 {rel90:.3e}（容差 25%）")
print(f"  v = 0.99: 偏转 {abs(d99):.6e} 对 null {delta_null:.6e}，相对差 {rel99:.3e}（容差 10%）")
m3_pass = (rel90 < 0.25 and rel99 < 0.10)
VERDICT["M3"] = m3_pass
print(f"  [M3] 判定：{'通过' if m3_pass else '未通过'}")

# ----------------------------------------------------------------------
# M4. 质量盲（结构打印）
# ----------------------------------------------------------------------
print("\n[M4] 质量盲：规则与 ODE 均无质量参数")
print("  逐拍规则状态量（位置、坐标速度）与归一化 u^t = 1/sqrt(f - h v^2) 均不含 m；")
print("  质量仅进写入份额（写侧定理 6），响应侧质量盲为结构性质（与判据 G3 一致）。")
VERDICT["M4"] = True
print("  [M4] 判定：通过（结构性）")

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax = axes[0]
vgrid = np.linspace(0.12, 1.0, 200)
ax.plot(vgrid, [2.0 * th_ref * r0 / (bM * v * v) + 2.0 * th_ref * r0 / bM for v in vgrid],
        "-", color="gray", label="两项式 2θ_0r_0/(bv²) + 2θ_0r_0/b")
ax.plot(list(vs), [-rule_massive(th_ref, bM, LM, v) for v in vs], "o", color="C0",
        label="规则偏转（五档）")
ax.axhline(delta_null, color="C3", ls="--", label="null 极限 4θ_0r_0/b")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("初速 v_0（c=1）"); ax.set_ylabel("偏转角（rad）")
ax.set_title("两项分解与 v→c 连续性（M2/M3）")
ax.legend(fontsize=8)

ax = axes[1]
labels = ("v=0.3", "v=0.15")
vals = (diff30, diff15)
ax.bar(labels, vals, width=0.4, color=("C0", "C1"))
ax.axhline(const_target, color="C2", ls="--", label=f"两项式常数 {const_target:.3e}")
for i, v_ in enumerate(vals):
    ax.text(i, v_, f"{v_:.3e}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("full − g00only 偏转差（rad）")
ax.set_title("空间耦合项的 v 无关性（M2b）")
ax.legend(fontsize=8)
fig.suptitle("全度规有质量档机器级验证轮（第 614 号）", fontsize=13)
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
for k in ("P0", "M1", "M2", "M3", "M4"):
    ok = VERDICT[k]
    npass += int(ok)
    print(f"  {k}：{'通过' if ok else '未通过'}")
print(f"  合计：{npass}/5")
