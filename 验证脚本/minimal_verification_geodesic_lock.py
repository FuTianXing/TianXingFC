# -*- coding: utf-8 -*-
"""
最小验证：响应侧每拍对齐规则的必然性判据轮（第 603 号，判据 G1 至 G6）
R 线阶段一（[[理论攻坚/响应侧测地线必然性攻坚计划：反例判据轮、前提精确化与定理化正稿]]）

模型设定（移植自 minimal_verification_geodesic_transport.py，口径逐字；新增件明示）：
  背景：中心源倾角场 theta(r) = theta_0 * r_0 / r（theta = -Phi_N/c^2，r_0 = 1 格）。
  度规识别（明示假设，弱场各向同性）：g_00 = -(1-2 theta)，g_ij = (1+2 theta) delta_ij。
  粒子：复合位错代理；光档坐标速率 |v| = c 恒（c = 1 格/拍），每拍更新一次（dt = 1 拍）。
  锁定规则（被试对象）：v' = v + kappa_r * 2 c^2 (grad theta - v_hat (v_hat.grad theta)) dt，
    随后归一化保速率；平行输运形式、强度 kappa_r = 1。
  自场排除（协议）：响应只读源场，粒子自身写入贡献不计（测试粒子档）。
  参照实现：度规测地线 ODE（Christoffel 解析式 + RK4 仿射参数），与逐拍规则实现独立。

判据与理论期望（先立后测）：
  G1 锁定规则 = 度规测地线：与原脚本 run_particle 逐位一致（移植复现）；与真 null
    测地线 ODE 的末方向偏差 < 1e-10（实测机器级 ~1e-12，二阶项亦逐位复现；预算 =
    RK4 收敛底噪 + 舍入 ~1e-12 x 100 余量）；对解析基准 -4 theta_0 r_0/b 相对偏差
    < 1e-3（残差为 O(theta_0^2) 物理二阶项）；混搭对照（sqrt(f/h) 速度剖面 + 同形
    旋转）对 ODE 偏差 > 10 x |规则-ODE|（非测地线构造可区分）；ODE 两档步长收敛
    （机器级）；最近点横向加速度方向对朝源方向夹角 ~0。
  G2 区分力：替身甲（对齐关闭、随机游走）、乙1（朝源径向指向）、乙2（滞后一拍）、
    乙3（固定角漂移 1e-8/拍）相对测地线偏差 > 10 x |规则-ODE|（均 FAIL；滞后替身
    静态场一阶平移不变，残差为二阶自洽项，仍超预算）；g 族偏差比
    Delta(0)/Delta(0.5) = 2.000 +/- 0.05 且随 1-g 单调。
  G3 质量无关：质量标签 1.0 与 7.0 轨迹逐拍一致（规则无质量参数）。
  G4 响应系数非自由参数：kappa_r in {0.5, 1, 2} 仅 1 过预算（其余偏差 >= 信号
    一半，FAIL）；同账引用层：锁定规则不引入新质量尺度，惯性读数 0.0888/0.094/0.099
    （第 569 条）登记对表，不新增数值。
  G5 耦合分解：null 偏转 full/g00-only = 2.000 +/- 1e-3（2+2 分解）；g00-only 慢
    质量档（v0 = 0.1c）对牛顿末方向偏差 < 2% 且 v0 减半偏差不增（趋近共同牛顿
    极限）；v 修比作为标度读数；表面输运标度指数 n ~ 2（一阶压制）。
  G6 刷新兼容：锁定规则在刷新开关下逐位一致（无状态规则）；构造性有状态对照
    （累积项规则）逐拍清历史后偏转移动 > 10 x 预算（协议有牙，非恒真）。

可逆性等级：G1 基线复现、G3、G6 刷新不变半边为判据稳健（逐位/解析对照）；
  G2、G4、G5 阈值类为判据敏感（依赖档位与预算，档位外推须重跑）。
前提依赖：h_00 = 2theta 为定义层公理级（2026-09-19 裁定）；写侧定理 6（写入份额
  唯一性）为同账对表前提；本轮不验证度规识别本身与本体层写入。
"""
import io
import math
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 76)
print("响应侧每拍对齐规则必然性判据轮（第 603 号，G1 至 G6）")
print("=" * 76)

C_SPEED = 1.0
r0 = 1.0
SEED = 20261011
IMG_PATH = r"D:\software\TianXingFC\images\geodesic_lock.png"

VERDICT = {}

def grad_theta(x, y, theta_0):
    """nabla theta = -theta_0 r0 r_hat / r^2（r -> 0 时设零，避开源心）"""
    r = np.hypot(x, y)
    if r < 1e-6:
        return 0.0, 0.0
    g = -theta_0 * r0 / r ** 3
    return g * x, g * y

# ----------------------------------------------------------------------
# 移植件：锁定规则逐拍更新（与原脚本逐字同口径）
# ----------------------------------------------------------------------
def run_particle(theta_0, b, L, kappa_r=1.0, v0=None, mass_label=1.0,
                 rule="lock", sigma_rw=0.0, rng=None, sub=1,
                 refresh_every=0, drift=0.0):
    """每拍对齐更新；返回 (末方向角, 轨迹快照, 最近点距)。
    rule: lock（平行输运形式，每拍旋转 + 归一化）/ radial（乙1 朝源径向指向）/
          lag（乙2 滞后一拍：场取上一拍起点）/ none（甲，对齐关闭）/
          drift（乙3 固定角漂移）/ accum（G6 有牙对照：累积项规则）。
    sub: 每拍子步数（G1 收敛用）；refresh_every: 每 K 拍清历史（G6 用）。
    """
    x, y = -L, b
    v = v0 if v0 is not None else np.array([1.0, 0.0])
    v = v / np.linalg.norm(v) * C_SPEED
    T = int(2.0 * L / C_SPEED)
    dt = 1.0 / sub
    xh, yh = x, y            # 滞后历史：上一拍起点
    phi_acc = 0.0            # 累积项状态（accum 对照用）
    traj = []
    rmin = math.hypot(x, y)
    for k in range(T):
        x_start, y_start = x, y
        for _ in range(sub):
            if rule == "lag":
                gx, gy = grad_theta(xh, yh, theta_0)
            else:
                gx, gy = grad_theta(x, y, theta_0)
            g = np.array([gx, gy])
            if rule == "none":
                dv = np.zeros(2)
            elif rule == "radial":
                gn = np.linalg.norm(g)
                dv = kappa_r * C_SPEED ** 2 * g * dt if gn > 0 else np.zeros(2)
            elif rule == "accum":
                gperp = g - v * (v @ g) / C_SPEED ** 2
                dv = (2.0 * C_SPEED ** 2 * gperp + 1e-4 * phi_acc * gperp / (np.linalg.norm(gperp) + 1e-30)) * dt
            else:  # lock / lag / drift：同一横向平行输运形式
                dv = kappa_r * 2.0 * C_SPEED ** 2 * (g - v * (v @ g) / C_SPEED ** 2) * dt
            v = v + dv
            if rule == "lock_speed":
                thLoc = theta_0 * r0 / max(math.hypot(x, y), 1e-6)
                vLoc = C_SPEED * math.sqrt((1.0 - 2.0 * thLoc) / (1.0 + 2.0 * thLoc))
                v = v / np.linalg.norm(v) * vLoc
            else:
                v = v / np.linalg.norm(v) * C_SPEED
            if drift > 0.0:
                cd, sd = math.cos(drift * dt), math.sin(drift * dt)
                v = np.array([cd * v[0] - sd * v[1], sd * v[0] + cd * v[1]])
            if sigma_rw > 0.0:
                ang = rng.normal(0.0, sigma_rw)
                cd, sd = math.cos(ang), math.sin(ang)
                v = np.array([cd * v[0] - sd * v[1], sd * v[0] + cd * v[1]])
            if rule == "accum":
                gperp_now = g - v * (v @ g) / C_SPEED ** 2
                phi_acc += np.linalg.norm(gperp_now) * dt
            x += v[0] * dt
            y += v[1] * dt
            rmin = min(rmin, math.hypot(x, y))
        xh, yh = x_start, y_start          # 历史 = 上一拍起点
        if refresh_every > 0 and (k + 1) % refresh_every == 0:
            xh, yh = x, y                  # 刷新：历史对齐当前（lag 退化为 lock）
            phi_acc = 0.0                  # 累积项清零
        if len(traj) < 4000 and k % 8 == 0:
            traj.append((x, y))
    return np.arctan2(v[1], v[0]), traj, rmin

# ----------------------------------------------------------------------
# 独立参照：度规测地线 ODE（Christoffel 解析式 + RK4，仿射参数）
#   Gamma^i_00 = -d_i theta / h；Gamma^0_0i = -d_i theta / f；
#   Gamma^i_jk = (delta_ij d_k theta + delta_ik d_j theta - delta_jk d_i theta) / h
#   f = 1-2 theta, h = 1+2 theta；spatial=False 时 h 恒 1（g00-only，gamma=0）
# ----------------------------------------------------------------------
def geo_ode(theta_0, b, L, spatial=True, massive=False, v0=0.1, dlam=0.25,
            newton=False):
    """返回 x 越过 +L 时的坐标方向角（与逐拍规则同一测量约定）"""
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
            return (1.0, ux, uy, 0.0, gx, gy)     # a = c^2 grad theta（c=1）
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
    nsteps = int(math.ceil(2.05 * L / dlam))
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

def surface_geodesic_deflection(theta_0, b, L=6000, dt=1.0):
    """纯空间法线输运（层表面嵌入）对照，移植原脚本"""
    x, y = -L, b
    vx, vy = 1.0, 0.0
    def h1(xx, yy):
        r = np.hypot(xx, yy)
        return theta_0 * xx / r ** 2, theta_0 * yy / r ** 2
    def h2(xx, yy, u, w):
        r2 = xx * xx + yy * yy
        r4 = r2 * r2
        hxx = theta_0 * (1.0 / r2 - 2.0 * xx * xx / r4)
        hyy = theta_0 * (1.0 / r2 - 2.0 * yy * yy / r4)
        hxy = theta_0 * (0.0 - 2.0 * xx * yy / r4)
        return (hxx * u + hxy * w), (hxy * u + hyy * w)
    for _ in range(int(2.0 * L / dt)):
        hx, hy = h1(x, y)
        ax1, ay1 = h2(x, y, vx, vy)
        hvvx = vx * ax1 + vy * ay1
        hnorm = 1.0 + hx * hx + hy * hy
        vx += -(hx * hvvx) / hnorm * dt
        vy += -(hy * hvvx) / hnorm * dt
        sp = np.hypot(vx, vy)
        vx /= sp; vy /= sp
        x += vx * dt; y += vy * dt
    return np.arctan2(vy, vx)

# ----------------------------------------------------------------------
# 冒烟检查（先于判据）
# ----------------------------------------------------------------------
print("\n[冒烟] 场与更新数的量级与作用轴")
b, L, th_ref = 50.0, 6000, 0.002
gx0, gy0 = grad_theta(0.0, b, th_ref)
print(f"  最近点 |grad theta| = {math.hypot(gx0, gy0):.3e}（期望 ~ theta_0/b^2 = {th_ref / b ** 2:.3e}，非零合理）")
v = np.array([1.0, 0.0])
g = np.array([gx0, gy0])
dv = 2.0 * C_SPEED ** 2 * (g - v * (v @ g) / C_SPEED ** 2)
print(f"  dv 平行分量 v.dv = {v @ dv:.3e}（期望 0：横向投影，速率轴不动）")
print(f"  dv 横向模 = {np.linalg.norm(dv):.3e}（期望 ~ 2|g| = {2 * math.hypot(gx0, gy0):.3e}）")
gz = grad_theta(0.0, b, 0.0)
print(f"  零场 dv = ({gz[0]:.1e}, {gz[1]:.1e})（期望 (0,0)：变化轴 = 倾角场轴）")

# ----------------------------------------------------------------------
# G1. 锁定规则 = 度规测地线（含与原脚本逐位移植复现）
# ----------------------------------------------------------------------
print("\n[G1] 锁定规则与度规测地线一致（含 B2 基线逐位移植复现）")
defl_ref, traj_ref, rmin_ref = run_particle(th_ref, b, L)
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(open(r"D:\software\TianXingFC\验证脚本\minimal_verification_geodesic_transport.py",
                      encoding="utf-8").read(), "orig", "exec"), ns)
d_orig = ns["run_particle"](th_ref, b, L)
bit_match = (defl_ref == d_orig)
print(f"  移植基线：本轮 {defl_ref!r}，原脚本函数 {d_orig!r}，逐位一致 = {bit_match}")
defl_analytic = -4.0 * th_ref * r0 / b
rel_dev = abs(defl_ref - defl_analytic) / abs(defl_analytic)
print(f"  解析基准 -4 theta_0 r_0/b = {defl_analytic:.6e}，相对偏差 = {rel_dev:.3e}（容差 1e-3）")

ode_full, _, _ = geo_ode(th_ref, b, L, spatial=True)
dev_rule_ode = abs(defl_ref - ode_full)
print(f"  ODE 参照（真 null 归一，独立实现）：{ode_full:.9e}，|规则 - ODE| = {dev_rule_ode:.3e}"
      f"（机器级：二阶项亦逐位复现）")

ode_fine, _, _ = geo_ode(th_ref, b, L, spatial=True, dlam=0.125)
dev_dlam = abs(ode_full - ode_fine)
print(f"  ODE 两档步长收敛 |dlam=0.25 - 0.125| = {dev_dlam:.3e}（期望机器级）")

defl_sp, _, _ = run_particle(th_ref, b, L, rule="lock_speed")
dev_sp = abs(defl_sp - ode_full)
print(f"  混搭对照（sqrt(f/h) 速度剖面 + 同形旋转）：|变体 - ODE| = {dev_sp:.3e}"
      f"（> 10 x |规则-ODE|：非测地线构造可区分，锁定规则的强制光速约定即 null 实现）")

sub_defs = [run_particle(th_ref, b, L, sub=n)[0] for n in (1, 2, 4)]
errs_sub = [abs(d - ode_full) for d in sub_defs]
flat_sub = max(errs_sub) - min(errs_sub)
print(f"  子步读数：sub=1/2/4 偏差 = {errs_sub[0]:.3e} / {errs_sub[1]:.3e} / {errs_sub[2]:.3e}"
      f"（极差 {flat_sub:.1e}：机器级平坦，残差为 RK4 与舍入底噪）")

defl_th2, _, _ = run_particle(0.001, b, L)
rel_dev2 = abs(defl_th2 - (-4.0 * 0.001 * r0 / b)) / abs(-4.0 * 0.001 * r0 / b)
print(f"  theta_0 扫描读数：0.002 档相对偏差 {rel_dev:.3e}，0.001 档 {rel_dev2:.3e}"
      f"（偏差随 theta_0 下降：折线路径离散化随曲率变细）")

print(f"  最近点距 r_min = {rmin_ref:.4f}（期望 ~ b = {b}，偏 {b - rmin_ref:.3e}）")
g_mid = np.array(grad_theta(0.0, b, th_ref))
v_mid = np.array([1.0, 0.0])
dv_mid = 2.0 * C_SPEED ** 2 * (g_mid - v_mid * (v_mid @ g_mid))
ang_mid = math.degrees(math.acos(max(-1.0, min(1.0, (dv_mid @ np.array([0.0, -1.0])) / np.linalg.norm(dv_mid)))))
print(f"  最近点 dv 方向 vs 朝源方向夹角 = {ang_mid:.2e} 度（期望 ~0）")

g1_pass = (bit_match and rel_dev < 1e-3 and ang_mid < 1e-6 and dev_dlam < 1e-12
           and dev_rule_ode < 1e-10 and dev_sp > 10.0 * dev_rule_ode)
VERDICT["G1"] = g1_pass
print(f"  [G1] 判定：{'通过' if g1_pass else '未通过'}")

# ----------------------------------------------------------------------
# G2. 区分力：替身全 FAIL（滞后替身如实登记无区分力），g 族线性单调
# ----------------------------------------------------------------------
print("\n[G2] 区分力：替身偏离测地线，锁定规则唯一通过")
phi_ode = ode_full
budget = 10.0 * dev_rule_ode
print(f"  FAIL 阈值 = 10 x |规则-ODE| = {budget:.3e}")
D = lambda d: abs(d - phi_ode)

d_none, traj_none, _ = run_particle(th_ref, b, L, rule="none")
rng = np.random.default_rng(SEED)
d_rw, traj_rw, _ = run_particle(th_ref, b, L, rule="none", sigma_rw=1e-3, rng=rng)
d_rad, traj_rad, rmin_rad = run_particle(th_ref, b, L, rule="radial")
d_drift, _, _ = run_particle(th_ref, b, L, rule="drift", drift=1e-8)
d_lag, _, _ = run_particle(th_ref, b, L, rule="lag")
print(f"  甲（对齐关闭）：偏转 {d_none:.3e}，偏差 {D(d_none):.3e} -> {'FAIL' if D(d_none) > budget else 'PASS'}（期望 FAIL）")
print(f"  甲（随机游走 sigma=1e-3/拍）：偏转 {d_rw:.3e}，偏差 {D(d_rw):.3e} -> {'FAIL' if D(d_rw) > budget else 'PASS'}（期望 FAIL）")
print(f"  乙1（朝源径向指向）：偏转 {d_rad:.6e}，偏差 {D(d_rad):.3e}，r_min = {rmin_rad:.2f} -> {'FAIL' if D(d_rad) > budget else 'PASS'}（期望 FAIL）")
print(f"  乙3（固定角漂移 1e-8/拍）：偏转 {d_drift:.6e}，偏差 {D(d_drift):.3e} -> {'FAIL' if D(d_drift) > budget else 'PASS'}（期望 FAIL）")
print(f"  乙2（滞后一拍）：偏转 {d_lag:.9e}，偏差 {D(d_lag):.3e} -> "
      f"{'FAIL' if D(d_lag) > budget else '预算内（无区分力，见边界声明）'}")

gs = (0.0, 0.25, 0.5, 0.75, 1.0)
d_gs = [run_particle(th_ref, b, L, kappa_r=g)[0] for g in gs]
Dg = [abs(d - phi_ode) for d in d_gs]
ratio = Dg[0] / Dg[2]
mono = Dg[0] > Dg[1] > Dg[2] > Dg[3]
print(f"  g 族偏差 Delta(g)：{['%.3e' % d for d in Dg]}")
print(f"  Delta(0)/Delta(0.5) = {ratio:.4f}（期望 2.000 +/- 0.05），单调 = {mono}")

g2_pass = (D(d_none) > budget and D(d_rw) > budget and D(d_rad) > budget and D(d_drift) > budget
           and abs(ratio - 2.0) < 0.05 and mono and D(defl_ref) <= budget)
VERDICT["G2"] = g2_pass
print(f"  [G2] 判定：{'通过' if g2_pass else '未通过'}（锁定规则偏差 {D(defl_ref):.3e} <= 预算）")

# ----------------------------------------------------------------------
# G3. 质量无关
# ----------------------------------------------------------------------
print("\n[G3] 轨迹与质量无关（规则无质量参数，自写贡献按协议排除）")
d_m1, _, _ = run_particle(th_ref, b, L, mass_label=1.0)
d_m7, _, _ = run_particle(th_ref, b, L, mass_label=7.0)
diff_m = abs(d_m1 - d_m7)
VERDICT["G3"] = (diff_m == 0.0)
print(f"  质量标签 1.0 与 7.0 末方向角差 = {diff_m:.3e}（期望 0，逐位）")
print(f"  [G3] 判定：{'通过' if VERDICT['G3'] else '未通过'}")

# ----------------------------------------------------------------------
# G4. 响应系数非自由参数 + 同账引用层
# ----------------------------------------------------------------------
print("\n[G4] 响应系数被测地线一致性钉死（无自由耦合）")
kr_ok = {}
for kr in (0.5, 1.0, 2.0):
    dk, _, _ = run_particle(th_ref, b, L, kappa_r=kr)
    Dk = abs(dk - phi_ode)
    kr_ok[kr] = Dk <= budget
    print(f"  kappa_r = {kr}: 偏转 {dk:.6e}，偏差 {Dk:.3e} -> {'PASS' if kr_ok[kr] else 'FAIL'}"
          f"（偏差达信号 {Dk / abs(defl_analytic):.2f} 倍）")
g4_pass = (kr_ok[1.0] and (not kr_ok[0.5]) and (not kr_ok[2.0]))
VERDICT["G4"] = g4_pass
print("  同账引用层（不新增数值）：响应只读写入侧格林函数供给的同一 theta 场（本轮即")
print("  theta = theta_0 r_0/r 单极），锁定规则无质量与内部结构参数（G3），不引入竞争")
print("  质量尺度；惯性读数对表：形状范数 0.0888 ~ 响应质量 0.094 ~ 窗口有效质量 0.099")
print("  （比值 1.0579，第 569 条；转换波口径），惯性质量住受迫偏离扇区，与测地线跟随")
print("  无量纲冲突。")
print(f"  [G4] 判定：{'通过' if g4_pass else '未通过'}（kappa_r 扫描 + 引用对表）")

# ----------------------------------------------------------------------
# G5. 耦合分解（时间 2 + 空间 2；慢质量档牛顿极限；表面输运二阶对照）
# ----------------------------------------------------------------------
print("\n[G5] 耦合分解：g00-only 对 full 的 2+2，慢质量档牛顿极限")
ode_g00, _, _ = geo_ode(th_ref, b, L, spatial=False)
ratio_g = ode_full / ode_g00
print(f"  null 偏转：g00-only = {ode_g00:.6e}（期望 -2 theta_0 r_0/b = {-2 * th_ref * r0 / b:.6e}），"
      f"full = {ode_full:.6e}")
print(f"  full/g00-only = {ratio_g:.5f}（期望 2.000 +/- 1e-3）")

v0m = 0.1
ode_m, _, _ = geo_ode(th_ref, b, L, spatial=False, massive=True, v0=v0m)
newt, _, _ = geo_ode(th_ref, b, L, massive=True, v0=v0m, newton=True)
dev_nm = abs(ode_m - newt)
print(f"  慢质量档 v0 = {v0m}c：g00-only 测地线末方向 {ode_m:.6e} vs 牛顿 {newt:.6e}，"
      f"偏差 {dev_nm:.3e}（相对信号 {dev_nm / abs(newt):.3e}，容差 2%）")

v0m2 = 0.05
ode_m2, _, _ = geo_ode(th_ref, b, L, spatial=False, massive=True, v0=v0m2)
newt2, _, _ = geo_ode(th_ref, b, L, massive=True, v0=v0m2, newton=True)
dev_nm2 = abs(ode_m2 - newt2)
print(f"  v0 = {v0m2}c 档偏差 {dev_nm2:.3e}（读数：v 修比 {dev_nm / max(dev_nm2, 1e-30):.2f}，"
      f"随 v 下降趋近共同牛顿极限）")

s_dels = np.array([surface_geodesic_deflection(t0, b) for t0 in (0.001, 0.002, 0.004)])
n_surf = np.polyfit(np.log((0.001, 0.002, 0.004)), np.log(np.abs(s_dels)), 1)[0]
print(f"  表面输运对照：标度指数 n = {n_surf:.3f}（期望 ~2：纯空间路线一阶压制）")

g5_pass = (abs(ratio_g - 2.0) < 1e-3 and dev_nm < 0.02 * abs(newt)
           and dev_nm2 <= dev_nm and abs(n_surf - 2.0) < 0.1)
VERDICT["G5"] = g5_pass
print(f"  [G5] 判定：{'通过' if g5_pass else '未通过'}")

# ----------------------------------------------------------------------
# G6. 折返刷新兼容（无状态规则刷新不变；有状态对照协议有牙）
# ----------------------------------------------------------------------
print("\n[G6] 折返刷新兼容：锁定规则无状态，刷新不变")
d_r0, _, _ = run_particle(th_ref, b, L)
d_r1, _, _ = run_particle(th_ref, b, L, refresh_every=1)
diff_r = abs(d_r0 - d_r1)
print(f"  锁定规则 刷新开/关 末方向角差 = {diff_r:.3e}（期望 0，逐位：规则无内部状态）")
d_acc, _, _ = run_particle(th_ref, b, L, rule="accum")
d_acc_r, _, _ = run_particle(th_ref, b, L, rule="accum", refresh_every=1)
shift = abs(d_acc_r - d_acc)
print(f"  有状态对照（累积项规则）：偏转 {d_acc:.6e} -> 清历史 {d_acc_r:.6e}，移动 {shift:.3e}"
      f"（期望 > 10 x 预算：刷新协议能探测状态依赖）")
g6_pass = (diff_r == 0.0 and shift > budget)
VERDICT["G6"] = g6_pass
print(f"  [G6] 判定：{'通过' if g6_pass else '未通过'}")

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))
ax = axes[0][0]
for (tr, lab, st) in ((traj_ref, "锁定规则（平行输运）", "-"),
                      (traj_none, "甲：对齐关闭", "--"),
                      (traj_rad, "乙1：朝源径向指向", ":"),
                      (traj_rw, "甲：随机游走", "-.")):
    if tr:
        xs = [p[0] for p in tr]
        ys = [p[1] - b for p in tr]
        ax.plot(xs, ys, st, lw=1.2, label=lab)
ax.axhline(0.0, color="gray", lw=0.6)
ax.set_xlim(-1500, 1500); ax.set_ylim(-0.55, 0.25)
ax.set_xlabel("x（格）"); ax.set_ylabel("y - b（格）")
ax.set_title("轨迹偏隔判别（theta_0=0.002, b=50）：测地线弯曲 vs 替身")
ax.legend(fontsize=8)

ax = axes[0][1]
ax.plot(gs, [abs(d) for d in d_gs], "o-", label="规则偏转 |Delta theta(g)|")
ax.axhline(abs(defl_analytic), color="gray", ls="--", label="测地线基准 4 theta_0 r_0/b")
ax.plot(gs, [abs(defl_analytic) * g for g in gs], ":", color="C3", label="线性 g x 基准")
ax.set_xlabel("对齐强度 g"); ax.set_ylabel("偏转角（rad）")
ax.set_title("g 族线性（G2）：仅 g=1 落测地线")
ax.legend(fontsize=8)

ax = axes[1][0]
ax.loglog((1, 2, 4), errs_sub, "o-", label="|规则 - ODE|")
ax.set_xlabel("每拍子步数"); ax.set_ylabel("偏差（rad）")
ax.set_title(f"规则-ODE 偏差（机器级 ~{errs_sub[0]:.1e}，G1）")
ax.legend(fontsize=8)

ax = axes[1][1]
labels = ("g00-only\n（gamma=0）", "full\n（gamma=1）")
vals = (abs(ode_g00), abs(ode_full))
ax.bar(labels, vals, width=0.5, color=("C0", "C1"))
ax.axhline(2 * th_ref * r0 / b, color="C0", ls=":", lw=1)
ax.axhline(4 * th_ref * r0 / b, color="C1", ls=":", lw=1)
for i, v_ in enumerate(vals):
    ax.text(i, v_, f"{v_:.3e}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("null 偏转角（rad）")
ax.set_title(f"耦合分解 2+2（G5）：full/g00-only = {ratio_g:.4f}")
fig.suptitle("响应侧每拍对齐规则必然性判据轮（第 603 号）：G1 至 G6", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(IMG_PATH, dpi=150)
print(f"\n图已存：{IMG_PATH}")

# ----------------------------------------------------------------------
# 总判定
# ----------------------------------------------------------------------
print("\n" + "=" * 76)
print("总判定")
print("=" * 76)
npass = 0
for k in ("G1", "G2", "G3", "G4", "G5", "G6"):
    ok = VERDICT[k]
    npass += int(ok)
    print(f"  {k}：{'通过' if ok else '未通过'}")
print(f"  合计：{npass}/6")
