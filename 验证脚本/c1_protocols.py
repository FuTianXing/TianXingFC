# -*- coding: utf-8 -*-
r"""
c1_protocols.py

[C1-1] 边界与协议确立：开放边界（吸收真空）与双协议等价（梯度流 + 惯性阻尼）。

定位
----
C1（三维复合位错）攻坚计划的第二项任务。C1-0 已交付初始化与横截面绕数测量装置
（vortex_init_apparatus.py，判据 M1 至 M12）；本脚本把
「边界协议」与「演化协议」立成公共资产，供 C1-2 起的形态枚举与驻留判定复用。

模型
----
kink3d A 部分（与 C1-0、环裁决实验同族）：
    H = (J0/2) sum_bonds (1 - cos dphi) + kappa sum_sites (1 - cos phi)
J0 = 1，kappa = 0.6，格距 1，相位差取最小像。开放边界 = 盒外一圈取真空 phi = 0
（垫片），能量只统计盒内键与格点；吸收性由演化健康性判据代理（B1 至 B4）。

协议
----
GF（梯度流 / 过阻尼）：phi <- phi - eta * gradH；eta = 0.02（主）。
ID（惯性阻尼）：velocity-Verlet 加阻尼 v <- (v + dt(a+a1)/2)/(1+gamma*dt)；
                dt = 0.1、gamma = 0.2（主）；dt 减半做步长收敛；gamma = 0 做守恒对照。

判据（先立判据，再实现；实测 vs 期望全部打印）
----
G0 梯度校核：三点有限差分 vs 解析梯度，最大绝对误差 < 1e-5。
G1 初态自检：标准环 R_c = 4、5，y = ±0.5 两格面、rad = 2 窗口，读数 +1。
B1 无锁定：R_c = 4 与 6 塌缩末态全盒 max|mi(phi)| < 1e-2（边界不留残迹）。
B2 丢失定位：绕数丢失的最大变化点距六边界 >= 5 格（丢失发生在体区）。
B3 能量健康（总能量 = 势 + 动）：gamma = 0.2 包络相对初值无 > 2% 回升；gamma = 0 的局域
   匹配脉冲 200 步漂移 < 1%（积分器守恒）；非局域初态在 gamma = 0 下的系统能量变化作
   边界垫片交换的定量读数报告。
B4 尺寸无关：N = 32 与 N = 48（同 R_c = 4、同协议）末态归属一致（|W| < 0.1 且 E < 1e-5）。
L1/L2 渐近真空单环构造（C1-0 登记的 C1-1 输入问题）：窗函数局域化后双格面读数 +1、盒边界
   max|phi| < 0.01；L3 其 gamma = 0 的边界交换 < 1%（对照标准环 8.24%）；GF 塌缩归属作读数。
P1 双协议归属：GF 与 ID 从同一标准环出发均塌缩（|W| < 0.1、E < 1e-5）；且丢失时刻的
   能量占比 E(t_lost)/E0 两协议相差 < 0.10。
P2 钉扎对照：管状冻结芯（w_pin = 1.0）在 GF 与 ID 下末态一致（|dW| <= 0.02、相对 dE <= 2%）。
   冻结为 C1-4 锚核的抽象代理；本判据只测协议等价，不测锚核物理。
P3 步长收敛：ID 在冻结构型上取 dt = 0.1 / 0.05 / 0.025 三档同 T = 240。P3a |dW| <= 0.02；
   P3b 收敛性：r1 = |E(0.1) - E(0.05)|、r2 = |E(0.05) - E(0.025)|，要求 r2 < r1 且 r1 相对 <= 1%。

运行
----
python 验证脚本/c1_protocols.py
确定性（无随机数），运行约 3 分钟。图存
D:/software/TianXingFC/images/c1_protocols.png。
"""

import time
import numpy as np

T_START = time.time()

J0, KAPPA = 1.0, 0.6
ETA, DT, GAMMA = 0.02, 0.1, 0.2
WPIN = 1.0

print("=" * 72)
print("[C1-1] 边界与协议确立：开放边界（吸收真空）与双协议等价")
print("=" * 72)

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def mi(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def grid(N):
    c = np.arange(N) - N / 2.0 + 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return c, X, Y, Z


def ring_phi(X, Z, RHO, Rc):
    return mi(np.arctan2(Z, RHO - Rc))


def energy(phi):
    E = KAPPA * float(np.sum(1.0 - np.cos(phi)))
    for ax in range(3):
        E += 0.5 * J0 * float(np.sum(1.0 - np.cos(mi(np.diff(phi, axis=ax)))))
    return E


def grad(phi):
    p = np.pad(phi, 1)
    ctr = (slice(1, -1),) * 3
    s = (np.sin(p[2:, 1:-1, 1:-1] - p[ctr]) + np.sin(p[:-2, 1:-1, 1:-1] - p[ctr])
         + np.sin(p[1:-1, 2:, 1:-1] - p[ctr]) + np.sin(p[1:-1, :-2, 1:-1] - p[ctr])
         + np.sin(p[1:-1, 1:-1, 2:] - p[ctr]) + np.sin(p[1:-1, 1:-1, :-2] - p[ctr]))
    return -0.5 * J0 * s + KAPPA * np.sin(phi)


def plaque_density(P):
    a00, a10, a11, a01 = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    return (mi(a10 - a00) + mi(a11 - a10) + mi(a01 - a11) + mi(a00 - a01)) / (2.0 * np.pi)


def box_vortex_scan(phi, thresh=0.5):
    """全盒涡量扫描：三个方向的全部格面，返回 (总绕数, 总数, 正数, 负数)。"""
    tot, n, npos, nneg = 0.0, 0, 0, 0
    for axis in range(3):
        for s in range(phi.shape[axis] - 1):
            dens = plaque_density(np.take(phi, s, axis=axis))
            vals = dens[np.abs(dens) > thresh]
            tot += float(vals.sum())
            n += int(vals.size)
            npos += int((vals > 0).sum())
            nneg += int((vals < 0).sum())
    return tot, n, npos, nneg


def winding_y(phi, coords, Rc, rad=2):
    """环顶点 theta=0 处的横截面绕数读数（C1-0 约定）：y = ±0.5 两格面、窗口 rad。"""
    s0 = int(np.argmin(np.abs(coords - 0.5)))
    s1 = int(np.argmin(np.abs(coords + 0.5)))
    ic = int(np.argmin(np.abs(coords - Rc)))
    iz = int(np.argmin(np.abs(coords - 0.0)))
    out = []
    for s in (s0, s1):
        dens = plaque_density(np.take(phi, s, axis=1))
        u0, u1 = max(ic - rad, 0), min(ic + rad, dens.shape[0])
        v0, v1 = max(iz - rad, 0), min(iz + rad, dens.shape[1])
        out.append(float(dens[u0:u1, v0:v1].sum()))
    return out


def relax(phi, coords, Rc, steps, protocol, eta=ETA, dt=DT, gamma=GAMMA,
          freeze_idx=None, freeze_val=None, snap=100, keep_frames=False):
    phi = phi.copy()
    out = {"tr": [], "frames": []}
    if protocol == "ID":
        v = np.zeros_like(phi)
        a = -grad(phi)
    for t in range(steps):
        if freeze_idx is not None:
            phi[freeze_idx] = freeze_val
        if protocol == "GF":
            phi = phi - eta * grad(phi)
            t_eff = (t + 1) * eta
        else:
            phi = phi + dt * v + 0.5 * dt ** 2 * a
            a1 = -grad(phi)
            v = (v + 0.5 * dt * (a + a1)) / (1.0 + gamma * dt)
            a = a1
            t_eff = (t + 1) * dt
        if freeze_idx is not None:
            phi[freeze_idx] = freeze_val
        if (t % snap == 0) or (t == steps - 1):
            w = float(np.mean(winding_y(phi, coords, Rc)))
            e_now = energy(phi)
            k_now = 0.5 * float(np.sum(v * v)) if protocol == "ID" else 0.0
            out["tr"].append((t_eff, w, e_now, k_now))
            if keep_frames:
                out["frames"].append(phi.copy())
    out["W"] = float(np.mean(winding_y(phi, coords, Rc)))
    out["E"] = energy(phi)
    out["maxphi"] = float(np.max(np.abs(mi(phi))))
    out["phi"] = phi
    return out


# ----------------------------------------------------------------------
# 网格与初态
# ----------------------------------------------------------------------
N = 48
coords, X, Y, Z = grid(N)
RHO = np.sqrt(X ** 2 + Y ** 2)
RC = 4.0
PHI0 = ring_phi(X, Z, RHO, RC)
E0 = energy(PHI0)

print(f"\n模型：J0 = {J0}，kappa = {KAPPA}，N = {N}，R_c = {RC}，E0 = {E0:.1f}")

# ----------------------------------------------------------------------
# G0 梯度校核 / G1 初态自检
# ----------------------------------------------------------------------
print("\n[G0] 梯度有限差分校核")
g = grad(PHI0)
err = 0.0
h = 1e-4
for idx in [(N // 2, N // 2, N // 2), (N // 2 + 3, N // 2, N // 2), (N // 2, N // 2 + 5, N // 2)]:
    pp = PHI0.copy(); pp[idx] += h
    pm = PHI0.copy(); pm[idx] -= h
    fd = (energy(pp) - energy(pm)) / (2 * h)
    err = max(err, abs(fd - g[idx]))
check("G0", err < 1e-5, f"max |analytic - finite diff| = {err:.2e}（期望 < 1e-5）")

print("\n[G1] 初态自检：标准环读数（y = ±0.5 两格面，rad = 2 窗口）")
g1_ok = True
for Rc_t in (4.0, 5.0):
    w = winding_y(ring_phi(X, Z, RHO, Rc_t), coords, Rc_t)
    g1_ok &= abs(w[0] - 1.0) < 1e-9 and abs(w[1] - 1.0) < 1e-9
    print(f"    R_c = {Rc_t}：W(+0.5) = {w[0]:+.4f}，W(-0.5) = {w[1]:+.4f}")
check("G1", g1_ok, "标准环双格面读数均为 +1.0000")

# ----------------------------------------------------------------------
# P1 裸环：GF 与 ID
# ----------------------------------------------------------------------
print("\n[P1] 裸环双协议：GF（eta = 0.02，2400 步）与 ID（dt = 0.1，gamma = 0.2，2400 步）")
t0 = time.time()
r_gf = relax(PHI0, coords, RC, 2400, "GF", keep_frames=True)
print(f"    GF 完成（{time.time() - t0:.0f} s）：W_f = {r_gf['W']:+.4f}，E_f = {r_gf['E']:.3e}")
t0 = time.time()
r_id = relax(PHI0, coords, RC, 2400, "ID", keep_frames=True)
print(f"    ID 完成（{time.time() - t0:.0f} s）：W_f = {r_id['W']:+.4f}，E_f = {r_id['E']:.3e}")


def loss_analysis(res):
    tr = res["tr"]
    for j in range(1, len(tr)):
        if abs(tr[j][1]) < 0.5:
            return tr[j][0], tr[j][2], j
    return float("nan"), float("nan"), -1


t_lost_gf, e_lost_gf, j_gf = loss_analysis(r_gf)
t_lost_id, e_lost_id, j_id = loss_analysis(r_id)
ratio_gf = e_lost_gf / E0
ratio_id = e_lost_id / E0
print(f"    GF：t_lost = {t_lost_gf:.1f}（t 单位 eta*t），E(t_lost)/E0 = {ratio_gf:.4f}")
print(f"    ID：t_lost = {t_lost_id:.1f}（t 单位 dt*t），E(t_lost)/E0 = {ratio_id:.4f}")
check("P1a", abs(r_gf["W"]) < 0.1 and r_gf["E"] < 1e-5 and abs(r_id["W"]) < 0.1 and r_id["E"] < 1e-5,
      f"双协议均塌缩：GF W = {r_gf['W']:+.4f}，ID W = {r_id['W']:+.4f}")
check("P1b", abs(ratio_gf - ratio_id) < 0.10,
      f"丢失能量占比差 = {abs(ratio_gf - ratio_id):.4f}（期望 < 0.10）")

# ----------------------------------------------------------------------
# B2 丢失定位（用 GF 轨迹的丢失帧对）
# ----------------------------------------------------------------------
print("\n[B2] 绕数丢失定位：丢失前后帧的最大变化点")
if j_gf >= 1:
    prev = r_gf["frames"][j_gf - 1]
    post = r_gf["frames"][j_gf]
    d = np.abs(mi(post - prev))
    idx = np.unravel_index(np.argmax(d), d.shape)
    dmin = min(min(i, N - 1 - i) for i in idx)
    print(f"    最大变化点格坐标 = {tuple(int(v) for v in idx)}（物理位置 x,y,z = "
          f"{coords[idx[0]]:+.1f}, {coords[idx[1]]:+.1f}, {coords[idx[2]]:+.1f}），"
          f"距最近边界 {dmin} 格")
    flat = np.argsort(d.ravel())[::-1][:5]
    for fl in flat:
        ii = np.unravel_index(fl, d.shape)
        iis = tuple(int(v) for v in ii)
        print(f"      前 5 变化点 {iis}：|dphi| = {d[ii]:.4f}，距最近边界 "
              f"{min(min(v, N - 1 - v) for v in iis)} 格")
    check("B2", dmin >= 5, f"丢失点距边界 {dmin} 格（期望 >= 5，体区丢失）")
else:
    check("B2", False, "GF 轨迹未检出丢失")

# ----------------------------------------------------------------------
# B1 无锁定：R_c = 6 补充实例
# ----------------------------------------------------------------------
print("\n[B1] 塌缩末态全盒审计：max|mi(phi)|（边界不留残迹）")
PHI6 = ring_phi(X, Z, RHO, 6.0)
r_gf6 = relax(PHI6, coords, 6.0, 1200, "GF")
print(f"    R_c = 4（复用 P1）：max|phi| = {r_gf['maxphi']:.3e}")
print(f"    R_c = 6：max|phi| = {r_gf6['maxphi']:.3e}，E_f = {r_gf6['E']:.3e}")
check("B1", r_gf["maxphi"] < 1e-2 and r_gf6["maxphi"] < 1e-2,
      "两实例末态 max|phi| < 1e-2（期望，真空回位无边界锁定）")

# ----------------------------------------------------------------------
# B3 能量健康：ID 回升界与 gamma=0 守恒
# ----------------------------------------------------------------------
print("\n[B3] 能量健康：ID 包络无回升（gamma = 0.2）与守恒对照（gamma = 0）")
r_b3a = relax(PHI0, coords, RC, 600, "ID", snap=10)
et = [row[2] + row[3] for row in r_b3a["tr"]]
rise = max(et) / et[0] - 1.0
print(f"    gamma = 0.2（非局域初态，600 步）：总能量包络最大回升 = {rise * 100:.4f}%（期望 < 2%）")

sig = 2.5
rr2 = X ** 2 + Y ** 2 + Z ** 2
PHI_P = mi(1.0 * np.exp(-rr2 / (2.0 * sig ** 2)))
r_b3b = relax(PHI_P, coords, RC, 200, "ID", gamma=0.0, snap=2)
etb = [row[2] + row[3] for row in r_b3b["tr"]]
drift_b = max(abs(e - etb[0]) for e in etb) / etb[0]
print(f"    gamma = 0（局域脉冲 sigma = {sig} 格，200 步）：总能量漂移 = {drift_b * 100:.5f}%（期望 < 1%）")

r_b3c = relax(PHI0, coords, RC, 600, "ID", gamma=0.0, snap=10)
etc = [row[2] + row[3] for row in r_b3c["tr"]]
delta = etc[-1] - etc[0]
print(f"    读数（非局域环，gamma = 0，600 步）：系统总能量变化 = {delta:+.1f}"
      f"（{delta / etc[0] * 100:+.2f}%；开放边界垫片与系统的能量交换读数，符号由初态配置决定）")
check("B3", rise < 0.02 and drift_b < 0.01,
      f"回升 {rise * 100:.4f}%，脉冲守恒漂移 {drift_b * 100:.5f}%")

# ----------------------------------------------------------------------
# [L] 渐近真空单环构造（C1-0 登记的 C1-1 输入问题）
# ----------------------------------------------------------------------
print("\n[L] 渐近真空单环构造：窗函数局域化与边界交换对照")
SIG_L = 3.0
core_dL = np.sqrt((RHO - RC) ** 2 + Z ** 2)
PHI_L = mi(np.arctan2(Z, RHO - RC) * np.exp(-core_dL ** 2 / (2.0 * SIG_L ** 2)))
w_l = winding_y(PHI_L, coords, RC)
bnd = np.zeros_like(PHI_L, dtype=bool)
bnd[0, :, :] = True; bnd[-1, :, :] = True
bnd[:, 0, :] = True; bnd[:, -1, :] = True
bnd[:, :, 0] = True; bnd[:, :, -1] = True
max_bnd = float(np.max(np.abs(mi(PHI_L)[bnd])))
E_L = energy(PHI_L)
print(f"    构造（sigma_L = {SIG_L} 格）：W = {w_l[0]:+.4f} / {w_l[1]:+.4f}，"
      f"边界 max|phi| = {max_bnd:.2e}，E0 = {E_L:.1f}（标准环 {E0:.1f}）")
check("L1", abs(w_l[0] - 1.0) < 1e-9 and abs(w_l[1] - 1.0) < 1e-9,
      "渐近真空环双格面读数 +1.0000")
check("L2", max_bnd < 0.01, f"盒边界 max|phi| = {max_bnd:.2e}（期望 < 0.01，真空渐近）")

r_L = relax(PHI_L, coords, RC, 600, "ID", gamma=0.0, snap=10)
etL = [row[2] + row[3] for row in r_L["tr"]]
dL = etL[-1] - etL[0]
print(f"    gamma = 0，600 步：系统总能量变化 = {dL:+.2f}（{dL / etL[0] * 100:+.3f}%；"
      f"对照标准环 {delta / etc[0] * 100:+.2f}%）")
check("L3", abs(dL / etL[0]) < 0.01,
      f"边界交换 {abs(dL / etL[0]) * 100:.3f}%（期望 < 1%，对照标准环 8.24%）")

r_Lg = relax(PHI_L, coords, RC, 2400, "GF")
print(f"    GF 2400 步读数：W_f = {r_Lg['W']:+.4f}，E_f = {r_Lg['E']:.3e}，"
      f"max|phi|_f = {r_Lg['maxphi']:.3e}")

# ----------------------------------------------------------------------
# B4 尺寸无关：N = 32
# ----------------------------------------------------------------------
print("\n[B4] 尺寸无关性：N = 32 同构型同协议")
N2 = 32
coords2, X2, Y2, Z2 = grid(N2)
RHO2 = np.sqrt(X2 ** 2 + Y2 ** 2)
r32 = relax(ring_phi(X2, Z2, RHO2, RC), coords2, RC, 2400, "GF")
print(f"    N = 32：W_f = {r32['W']:+.4f}，E_f = {r32['E']:.3e}")
check("B4", abs(r32["W"]) < 0.1 and r32["E"] < 1e-5, "N = 32 末态归属与 N = 48 一致（塌缩）")

# ----------------------------------------------------------------------
# P2 钉扎对照：管状冻结芯
# ----------------------------------------------------------------------
print("\n[P2] 钉扎对照：管状冻结芯（w_pin = 1.0）双协议末态")
core_d = np.sqrt((RHO - RC) ** 2 + Z ** 2)
fidx = tuple(np.argwhere(core_d < WPIN).T)
fval = ring_phi(X, Z, RHO, RC)[fidx]
print(f"    冻结格点数 = {fidx[0].size}（管状区，到芯圆距离 < {WPIN}）")
t0 = time.time()
r_p2g = relax(PHI0, coords, RC, 2400, "GF", freeze_idx=fidx, freeze_val=fval)
print(f"    GF 完成（{time.time() - t0:.0f} s）：W_f = {r_p2g['W']:+.4f}，E_f = {r_p2g['E']:.4f}")
t0 = time.time()
r_p2i = relax(PHI0, coords, RC, 2400, "ID", freeze_idx=fidx, freeze_val=fval)
print(f"    ID 完成（{time.time() - t0:.0f} s）：W_f = {r_p2i['W']:+.4f}，E_f = {r_p2i['E']:.4f}")
dW = abs(r_p2g["W"] - r_p2i["W"])
dE = abs(r_p2g["E"] - r_p2i["E"])
relE = dE / max(1.0, abs(r_p2g["E"]))
print(f"    |dW| = {dW:.4f}（期望 <= 0.02），相对 |dE| = {relE * 100:.3f}%（期望 <= 2%）")
check("P2", dW <= 0.02 and relE <= 0.02, f"dW = {dW:.4f}，relE = {relE * 100:.3f}%")
for tag, res in (("GF 末态", r_p2g), ("ID 末态", r_p2i)):
    tot_v, n_v, np_v, nn_v = box_vortex_scan(res["phi"])
    print(f"    全盒涡量扫描（{tag}）：净绕数 = {tot_v:+.0f}，超阈 plaquette 数 = {n_v}（正 {np_v} / 负 {nn_v}）")
tot_gf, n_gf, np_gf, nn_gf = box_vortex_scan(r_gf["phi"])
print(f"    （对照）P1 GF 塌缩末态：净绕数 = {tot_gf:+.0f}，plaquette 数 = {n_gf}")

# ----------------------------------------------------------------------
# P3 步长收敛：ID dt 减半（冻结构型）
# ----------------------------------------------------------------------
print("\n[P3] 步长收敛：ID dt = 0.05（4800 步）与 dt = 0.025（9600 步），同 T = 240")
t0 = time.time()
r_p3 = relax(PHI0, coords, RC, 4800, "ID", dt=0.05, freeze_idx=fidx, freeze_val=fval)
print(f"    dt = 0.05 完成（{time.time() - t0:.0f} s）：W_f = {r_p3['W']:+.4f}，E_f = {r_p3['E']:.4f}")
t0 = time.time()
r_p3q = relax(PHI0, coords, RC, 9600, "ID", dt=0.025, freeze_idx=fidx, freeze_val=fval)
print(f"    dt = 0.025 完成（{time.time() - t0:.0f} s）：W_f = {r_p3q['W']:+.4f}，E_f = {r_p3q['E']:.4f}")
dW3 = abs(r_p3["W"] - r_p2i["W"])
r1 = abs(r_p3["E"] - r_p2i["E"])
r2 = abs(r_p3q["E"] - r_p3["E"])
rel1 = r1 / max(1.0, abs(r_p2i["E"]))
print(f"    |dW| = {dW3:.4f}（期望 <= 0.02）")
print(f"    收敛读数：r1 = |E(0.1) - E(0.05)| = {r1:.4f}（相对 {rel1 * 100:.3f}%），"
      f"r2 = |E(0.05) - E(0.025)| = {r2:.4f}，比值 r2/r1 = {r2 / r1:.3f}（< 1 = 收敛）")
check("P3a", dW3 <= 0.02, f"dW = {dW3:.4f}")
check("P3b", r2 < r1 and rel1 <= 0.01,
      f"r2/r1 = {r2 / r1:.3f}，r1 相对 = {rel1 * 100:.3f}%（期望 r2 < r1 且 r1 相对 <= 1%）")

# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("判据汇总")
print("=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"  合计：{n_pass}/{len(CHECKS)} 通过")
print(f"  总运行时间：{time.time() - T_START:.0f} s")

# ----------------------------------------------------------------------
# 图
# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.5))

    a = ax[0, 0]
    tg = np.array([r[0] for r in r_gf["tr"]]); eg = np.array([r[2] for r in r_gf["tr"]])
    ti = np.array([r[0] for r in r_id["tr"]]); ei = np.array([r[2] for r in r_id["tr"]])
    a.plot(tg, eg, "-", color="#d62728", lw=1.4, label="GF: E(t)")
    a.plot(ti / ti[-1] * tg[-1], ei, "-", color="#1f77b4", lw=1.4, label="ID: E(t)（时间归一）")
    a.set_yscale("log")
    a.set_xlabel("有效时间（归一）"); a.set_ylabel("E")
    a.set_title("(a) 裸环塌缩：双协议能量轨迹")
    a.legend(fontsize=8)

    b = ax[0, 1]
    wg = np.array([r[1] for r in r_gf["tr"]]); wi = np.array([r[1] for r in r_id["tr"]])
    b.plot(tg, wg, "-", color="#d62728", lw=1.4, label="GF: W(t)")
    b.plot(ti / ti[-1] * tg[-1], wi, "-", color="#1f77b4", lw=1.4, label="ID: W(t)（时间归一）")
    b.axhline(0.5, color="gray", ls=":", lw=0.8)
    b.set_xlabel("有效时间（归一）"); b.set_ylabel("W")
    b.set_title("(b) 裸环塌缩：双协议绕数轨迹")
    b.legend(fontsize=8)

    c = ax[1, 0]
    c.plot([r[0] for r in r_p2g["tr"]], [r[2] for r in r_p2g["tr"]], "-", color="#d62728", lw=1.4, label="GF")
    c.plot([r[0] for r in r_p2i["tr"]], [r[2] for r in r_p2i["tr"]], "-", color="#1f77b4", lw=1.4, label="ID")
    c.set_xlabel("有效时间"); c.set_ylabel("E")
    c.set_title("(c) 冻结芯构型：双协议能量")
    c.legend(fontsize=8)

    d_ax = ax[1, 1]
    for res, col, lab in ((r_gf, "#d62728", "GF"), (r_id, "#1f77b4", "ID"), (r_p2g, "#2ca02c", "GF 冻结"), (r_p2i, "#9467bd", "ID 冻结")):
        wsp = np.array([r[1] for r in res["tr"]]); esp = np.array([r[2] for r in res["tr"]])
        d_ax.plot(esp, wsp, "-", color=col, lw=1.3, label=lab)
    d_ax.set_xscale("log")
    d_ax.set_xlabel("E"); d_ax.set_ylabel("W")
    d_ax.set_title("(d) W 对 E 相图（轨迹形状对照）")
    d_ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/c1_protocols.png", dpi=150)
    print("图已保存：images/c1_protocols.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print("\n[C1-1] 完成。")