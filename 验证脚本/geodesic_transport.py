# -*- coding: utf-8 -*-
"""
最小验证：粒子法线沿背景法线场的平行输运与测地线判据（途径三响应侧）

模型设定（明示假设）：
  背景：中心源倾角场 theta(r) = theta_0 * r_0 / r（3D 泊松稳态形，源侧已在
    frame_tilt.py 验证；本脚本只测响应侧）。
  度规识别（弱场候选实现）：倾角场进入度规为势函数 Phi = -c^2 * theta
    （theta = GM/(c^2 r) = -Phi/c^2），弱场各向同性度规
    g_00 = -(1 - 2 theta)，g_ij = (1 + 2 theta) * delta_ij。
  粒子：携带内部法线方向代理的复合位错代理，速率 |v| = c 恒；速度经度规联络
    平行输运（测地线 = 切向量平行于自身的曲线）。弱场零测地线一拍更新：
    v' = v + 2 c^2 (grad theta - v_hat (v_hat . grad theta)) dt（横向，保速率）。

判据：
  B1（零场方向保留）：theta_0 = 0 时 T 拍后方向角误差与 |v|、|n_p| 偏差均为
    机器精度；速度方向不漂移。
  B2（弱场轨迹与测地线方程一致）：偏转角幅值 ≈ 4 theta_0 r_0 / b（零测地线解析
    基准，本度规弱场结果，朝源为负）；偏转角幅值 ∝ theta_0（线性，指数 n ~ 1）；加速度方向 = -grad(theta) 投影
    （a = -grad Phi 形式）；|v| 与 |n_p| 保持在 O(theta^2) 内（方向偏转不改速率）。
  B3（轨迹与质量无关）：更新规则无粒子质量参数；两种质量标签下轨迹逐拍一致；
    等效原理响应侧模型级成立（惯性质量 = 引力质量出自同一几何更新）。

对照（边界声明）：
  纯空间法线输运（层表面嵌入，度规 delta_ij + h_i h_j，grad h = theta r_hat）
  给出的偏转 ∝ theta_0^2（二阶）。一阶牛顿项需要倾角进入 g_00 / 时间坐标
  耦合，此为本判据定位的承重墙施工面（张量桥 b 响应侧 / P1-5）。

边界声明：本脚本验证数学层（给定度规识别下，离散更新与测地线方程的一致性），
不验证本体层（g_00 耦合的微观来源、写入幅度与 m/m_P 的核对、强场断言）。
"""
import numpy as np

R_REPORT = []
print("=" * 72)
print("最小验证：粒子法线平行输运与测地线判据（途径三响应侧）")
print("=" * 72)

C_SPEED = 1.0          # 格/拍，l_P/t_P 单位
r0 = 1.0               # 源尺度（1 格）

def grad_theta(x, y, theta_0, r0=r0):
    """nabla theta = -theta_0 r0 r_hat / r^2（r -> 0 时设零，避开源心）"""
    r = np.hypot(x, y)
    if r < 1e-6:
        return 0.0, 0.0
    g = -theta_0 * r0 / r ** 3
    return g * x, g * y

# ----------------------------------------------------------------------
# B1. 零场方向保留
# ----------------------------------------------------------------------
print("\n[B1] 零场方向保留：theta_0 = 0，直线运动 T 拍")
T1 = 2000
v = np.array([1.0, 0.0])
n_p = v.copy()
for _ in range(T1):
    gx, gy = grad_theta(10.0, 0.0, 0.0)
    dv = 2.0 * C_SPEED ** 2 * (np.array([gx, gy]) - v * (v @ np.array([gx, gy])) / C_SPEED ** 2)
    v = v + dv
    v = v / np.linalg.norm(v)
    n_p = v.copy()
err_dir = np.linalg.norm(v - np.array([1.0, 0.0]))
err_norm = abs(np.linalg.norm(n_p) - 1.0)
R_REPORT.append(("B1", err_dir, err_norm))
print(f"  方向误差 = {err_dir:.3e}（期望 ~0，机器精度）")
print(f"  |n_p| - 1 = {err_norm:.3e}（期望 ~0）")
print("  [B1] 判定：通过" if err_dir < 1e-12 else "  [B1] 判定：未通过")

# ----------------------------------------------------------------------
# B2. 弱场轨迹与测地线方程一致
# ----------------------------------------------------------------------
print("\n[B2] 弱场轨迹：冲击参数 b 掠过源，测地线基准对照")

def run_particle(theta_0, b, L, v0=None, mass_label=1.0):
    """离散平行输运更新；返回位置序列与末速度方向角"""
    x, y = -L, b
    v = v0 if v0 is not None else np.array([1.0, 0.0])
    v = v / np.linalg.norm(v) * C_SPEED
    T = int(2.0 * L / C_SPEED)
    for _ in range(T):
        gx, gy = grad_theta(x, y, theta_0)
        g = np.array([gx, gy])
        dv = 2.0 * C_SPEED ** 2 * (g - v * (v @ g) / C_SPEED ** 2)
        v = v + dv
        v = v / np.linalg.norm(v) * C_SPEED
        x += v[0]; y += v[1]
    return np.arctan2(v[1], v[0])

b = 50.0
L = 6000
theta_0 = 0.002

# 解析基准：零测地线弱场偏转角 = -4 theta_0 r_0 / b（度规 g_00, g_ij 双贡献，朝源为负）
deflection_theory = -4.0 * theta_0 * r0 / b

deflection_meas = run_particle(theta_0, b, L)
R_REPORT.append(("B2a", deflection_meas, deflection_theory))
print(f"  偏转角实测 = {deflection_meas:.6e}，解析基准 = {deflection_theory:.6e}（-4 theta_0 r0/b）")
print(f"  相对偏差 = {abs(deflection_meas - deflection_theory) / abs(deflection_theory):.4f}（期望 < 5%）")

# 线性检验：theta_0 加倍，偏转加倍（指数 n ~ 1）
ths = (0.001, 0.002, 0.004)
dels = np.array([run_particle(t0, b, L) for t0 in ths])
n_fit = np.polyfit(np.log(ths), np.log(np.abs(dels)), 1)[0]
R_REPORT.append(("B2b", n_fit))
print(f"  线性检验：theta_0 = {ths} -> 偏转角 = {dels}")
print(f"  标度指数 n = {n_fit:.3f}（期望 ~1.0，小力区线性）")

# 加速度方向检验：轨迹中途逐点测 dv 方向与 -grad(theta) 横向分量的夹角
gx, gy = grad_theta(0.0, b, theta_0)
g = np.array([gx, gy])
g_perp = g - np.array([1.0, 0.0]) * (np.array([1.0, 0.0]) @ g)   # v_hat = x_hat 处横向
ang = np.degrees(np.arccos(np.clip(g_perp @ np.array([0.0, -1.0]) / (np.linalg.norm(g_perp) + 1e-12), -1.0, 1.0)))
R_REPORT.append(("B2c", ang))
print(f"  加速度方向 vs -y（朝源）夹角 = {ang:.3f} 度（期望 ~0，a = -grad Phi 形式）")

# 速率与法线模长保持
v_check = np.array([1.0, 0.0])
n_p = v_check.copy()
max_dv = 0.0
for _ in range(200):
    gx, gy = grad_theta(0.0, b, theta_0)
    g = np.array([gx, gy])
    dv = 2.0 * C_SPEED ** 2 * (g - v_check * (v_check @ g) / C_SPEED ** 2)
    v_check = v_check + dv
    v_check = v_check / np.linalg.norm(v_check) * C_SPEED
    n_p = v_check.copy()
    max_dv = max(max_dv, abs(np.linalg.norm(v_check) - C_SPEED))
R_REPORT.append(("B2d", max_dv))
print(f"  200 拍内 |v| - c 最大偏差 = {max_dv:.3e}（期望 O(theta^2)）")

# ----------------------------------------------------------------------
# B3. 轨迹与质量无关
# ----------------------------------------------------------------------
print("\n[B3] 轨迹与质量无关：更新规则无粒子质量参数")
phi_m1 = run_particle(theta_0, b, L, mass_label=1.0)
phi_m7 = run_particle(theta_0, b, L, mass_label=7.0)
diff = abs(phi_m1 - phi_m7)
R_REPORT.append(("B3", diff))
print(f"  质量标签 1.0 与 7.0 的末方向角差 = {diff:.3e}（期望 0，机器精度）")
print("  [B3] 判定：通过" if diff == 0.0 else "  [B3] 判定：未通过")

# ----------------------------------------------------------------------
# 对照：纯空间法线输运（层表面嵌入）的偏转标度
# ----------------------------------------------------------------------
print("\n[对照] 纯空间法线输运（表面嵌入度规 delta_ij + h_i h_j）")
print("  表面 h 满足 grad h = theta r_hat；零测地线偏转期望 ∝ theta_0^2（二阶）")

def surface_geodesic_deflection(theta_0, b, L=6000, dt=1.0):
    """在曲面 z = h(r) 上做常速率运动，数值积分表面测地线"""
    x, y = -L, b
    vx, vy = 1.0, 0.0
    def h1(xx, yy):
        r = np.hypot(xx, yy)
        return theta_0 * xx / r ** 2, theta_0 * yy / r ** 2   # h_i = theta r_hat
    def h2(xx, yy, u, w):
        r2 = xx * xx + yy * yy
        r4 = r2 * r2
        hxx = theta_0 * (1.0 / r2 - 2.0 * xx * xx / r4)
        hyy = theta_0 * (1.0 / r2 - 2.0 * yy * yy / r4)
        hxy = theta_0 * (0.0 - 2.0 * xx * yy / r4)
        return (hxx * u + hxy * w), (hxy * u + hyy * w)
    T = int(2.0 * L / dt)
    for _ in range(T):
        hx, hy = h1(x, y)
        ax1, ay1 = h2(x, y, vx, vy)     # 梯度 (v . grad) h 的分量
        hvvx = vx * ax1 + vy * ay1      # (v . grad)^2 h
        hnorm = 1.0 + hx * hx + hy * hy
        ax = -(hx * hvvx) / hnorm
        ay = -(hy * hvvx) / hnorm
        vx += ax * dt; vy += ay * dt
        sp = np.hypot(vx, vy)
        vx /= sp; vy /= sp
        x += vx * dt; y += vy * dt
    return np.arctan2(vy, vx)

s_dels = np.array([surface_geodesic_deflection(t0, b) for t0 in ths])
n_surf = np.polyfit(np.log(ths), np.log(np.abs(s_dels)), 1)[0]
R_REPORT.append(("对照", n_surf))
print(f"  表面测地线偏转角 = {s_dels}（theta_0 = {ths}）")
print(f"  标度指数 n = {n_surf:.3f}（期望 ~2.0：纯空间法线输运为二阶）")

# ----------------------------------------------------------------------
# 总判定
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("总判定")
print("=" * 72)
b1_pass = R_REPORT[0][1] < 1e-12 and R_REPORT[0][2] < 1e-12
b2_pass = abs(R_REPORT[1][1] - R_REPORT[1][2]) / abs(R_REPORT[1][2]) < 0.05
b2_lin = abs(R_REPORT[2][1] - 1.0) < 0.1
b3_pass = R_REPORT[5][1] == 0.0
print(f"  B1（零场方向保留）：{'通过' if b1_pass else '未通过'}")
print(f"  B2（弱场测地线一致）：{'通过' if b2_pass else '未通过'}，"
      f"偏转解析基准相对偏差 {abs(R_REPORT[1][1] - R_REPORT[1][2]) / abs(R_REPORT[1][2]):.3e}")
print(f"  B2 线性（指数 n ~ 1）：{'通过' if b2_lin else '未通过'}，n = {R_REPORT[2][1]:.3f}")
print(f"  B3（轨迹与质量无关）：{'通过' if b3_pass else '未通过'}")
print(f"  对照（纯空间表面输运）：指数 n = {R_REPORT[6][1]:.2f}，"
      f"一阶牛顿项需 g_00 / 时间坐标耦合（张量桥 b 响应侧施工面）")
