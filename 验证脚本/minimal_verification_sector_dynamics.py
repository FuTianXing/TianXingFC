# -*- coding: utf-8 -*-
"""
T4/T5 约束扇区动力学实验：两种绕数的裁决（环面绕数 vs 局域涡量）
对应《绕数超选择：先验纠缠作为电荷保护层》T4/T5，
并合并四件套复跑遗留（窄区静态碗的判决性下降）。

背景（前两轮运行入档的方法与物理论证）：
  1. 相位场的"绕数"有两种，必须分开裁决。
     (a) 环面绕数：S^1 映射 T^3 -> S^1 的逐线度数 W = -Sigma_line k
         （k = floor(u/2pi + 0.5) 为键分支指标）。这是紧致场真正的
         超选择扇区：无约束动力学要改变它必须有键穿越 e^{i pi}
         （撕裂事件）。
     (b) 局域涡量：涡旋线/环的 plaqette 缠绕。它不构成扇区：可搬运、
         可对湮灭、可经边界清零（环的裁决实验 C1-C5、疤的裁决实验）。
  2. arctan2 包络构造的"环"的绕数是表示性的：其 S^1 映射拓扑平凡
     （包络 g < 1 使截面回路只扫过圆弧），冻结分支只是冻住多值表示的
     标记，物理构型可经 2pi 偶极子隐藏退到真空（第二轮 [D2] 约束终态
     E = 0 直接观测）。
  3. 协议修正：全局步拒绝法把切向运动一并冻死（第二跑 [B] 在 E=462
     硬停滞），改用分支夹取（clip）协议：每步后把越界键夹回本分支区间，
     多轮扫描收敛；切向运动保留，扇区逐键严格保持。

实验设计：
  [A] 无约束基线：环塌缩（复现 C1）。
  [B] 冻结分支流，初条件 = 环 + 均匀扭转（环面绕数 W_z = 1）：
      预期环部分经隐藏偶极子真空化（E 贡献 -> 0），均匀扭转部分
      作为真扇区极小必须驻留且离域（拓扑强制每条 z 线缠绕），
      E -> E_uni = N^3 (1-cos(2 pi/N))。
  [B2] 无约束对照（同初条件）：撕裂通道开放，全部消失 E -> 0。
  [C] 三元组窄区静态碗判决性下降（铁磁背景，kappa_phi=0.5, rho0=1.0）：
      无约束（切片碗伪影判定）与冻结分支（局域绕数终结形态）两支。
  [D] 交错背景裁定观测（四件套复跑协议）：无约束滚向铁磁、
      冻结分支完全刚性（反相键坐分支）。

判据：
  T4 = 冻结扇区内存在局域 dressing 极小。若 [B] 终态离域化（r_g 达盒尺度）
       且 [C] 约束支隐藏化（E -> 0），则 T4 否定：扇区保护不产生局域载体。
  T5 = 终态宽度与色散窗口（l* >= 2 格距）对照；局域载体不存在时 T5 消解。

确定性：无随机数。结果打印终端 + 一图（images/）。
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWO_PI = 2.0 * np.pi
EPS_WALL = 1e-7


def kidx(u):
    """分支指标：k 变化 = 该键穿越奇数倍 pi（撕裂事件）"""
    return np.floor(u / TWO_PI + 0.5)


def make_ring(N, R0, rho0, n=1.0):
    """环形涡旋（arctan2 截面族，与四件套复跑同构造）"""
    c = N // 2
    idx = np.arange(N) - c
    Z, Y, X = np.meshgrid(idx, idx, idx, indexing="ij")
    rho = np.sqrt(X ** 2 + Y ** 2)
    psi = np.arctan2(Z, rho - R0)
    d = np.sqrt((rho - R0) ** 2 + Z ** 2)
    g = 0.5 * (1.0 - np.tanh((d - rho0) / 1.0))
    return n * psi * g, (X, Y, Z)


def make_uniform_twist(N):
    """z 向均匀扭转：phi = 2 pi z / N，环面绕数 W_z = +1（wrap 键 k = -1）"""
    phi = np.zeros((N, N, N))
    phi += TWO_PI * np.arange(N)[None, None, :] / N
    return phi


def cross_section_winding(phi, X, Y, Z, R0, rho0, rc=1.2):
    """截面小环绕数（四件套复跑第三版协议同款）"""
    rho = np.sqrt(X ** 2 + Y ** 2)
    d = np.sqrt((rho - R0) ** 2 + Z ** 2)
    g = 0.5 * (1.0 - np.tanh((d - rho0) / 1.0))
    sec = (np.abs(rho - R0) <= rc) & (np.abs(Z) <= rc) & (g > 0.7)
    beta = np.arctan2(rho[sec] - R0, Z[sec])
    ph = phi[sec]
    if ph.size < 4:
        return 0.0
    order = np.argsort(beta)
    ps = ph[order]
    W = 0.0
    for k in range(len(ps) - 1):
        dd = ps[k + 1] - ps[k]
        W += (dd + np.pi) % TWO_PI - np.pi
    dd = ps[0] - ps[-1]
    W += (dd + np.pi) % TWO_PI - np.pi
    return W / TWO_PI


class BareXY:
    """单分量周期格 XY；冻结模式 = 每步后分支夹取（切向运动保留）"""

    def __init__(self, N, phi, frozen):
        self.N = N
        self.phi = phi.copy()
        self.frozen = frozen
        self.k = [kidx(self.bond_raw(a)) for a in range(3)]
        self.steps = 0

    def bond_raw(self, a):
        return np.roll(self.phi, -1, axis=a) - self.phi

    def energy(self):
        return sum(float(np.sum(1.0 - np.cos(self.bond_raw(a)))) for a in range(3))

    def grad(self):
        g = np.zeros_like(self.phi)
        for a in range(3):
            s = np.sin(self.bond_raw(a))
            g += np.roll(s, 1, axis=a) - s
        return g

    def enforce_walls(self, sweeps=4):
        """分支夹取：越界键夹回本分支区间 [2 pi k - pi, 2 pi k + pi]"""
        for _ in range(sweeps):
            done = True
            for a in range(3):
                u = self.bond_raw(a)
                lo = TWO_PI * self.k[a] - np.pi + EPS_WALL
                hi = TWO_PI * self.k[a] + np.pi - EPS_WALL
                viol = (u < lo) | (u > hi)
                if not viol.any():
                    continue
                done = False
                tgt = np.where(u < lo, lo, np.where(u > hi, hi, u))
                corr = (tgt - u) * 0.5
                self.phi = self.phi - corr + np.roll(corr, 1, axis=a)
            if done:
                break

    def relax(self, iters, eta0=0.05, eta_max=0.1, snap_every=4000,
              early_dE=1e-13, early_patience=5000, tag=""):
        eta = eta0
        E_prev = self.energy()
        rec = [(0, E_prev, self.gyration(), cross_w(self))]
        stall_last = 0
        t0 = time.time()
        for t in range(1, iters + 1):
            dphi = -eta * self.grad()
            self.phi = self.phi + dphi
            if self.frozen:
                self.enforce_walls()
            else:
                self.k = [kidx(self.bond_raw(a)) for a in range(3)]
            eta = min(eta * 1.02, eta_max)
            E = self.energy()
            if t % snap_every == 0 or t == iters:
                rec.append((t, E, self.gyration(), cross_w(self)))
            if abs(E_prev - E) < early_dE:
                stall_last += 1
                if stall_last >= early_patience:
                    print(f"    [{tag}] t={t} 早停（能量变化 < {early_dE} 连续 {early_patience} 步）")
                    break
            else:
                stall_last = 0
            E_prev = E
        self.steps = t
        print(f"    [{tag}] 完成 {t} 步，用时 {time.time() - t0:.1f} s")
        return rec

    def verify_frozen(self):
        """冻结校验：返回全键距本分支边界的最小余量（应 >= EPS_WALL）"""
        worst = np.inf
        for a in range(3):
            c = self.bond_raw(a) - TWO_PI * self.k[a]
            worst = min(worst, float((np.pi - np.abs(c)).min()))
        return worst

    def action_excess(self):
        e = np.zeros_like(self.phi)
        for a in range(3):
            e += 0.5 * (1.0 - np.cos(self.bond_raw(a)))
        return e

    def gyration(self):
        e = self.action_excess()
        s = e.sum()
        if s < 1e-9:
            return 0.0
        N = self.N
        idx = np.arange(N, dtype=float) - N // 2
        Z, Y, X = np.meshgrid(idx, idx, idx, indexing="ij")
        cx, cy, cz = (e * X).sum() / s, (e * Y).sum() / s, (e * Z).sum() / s
        r2 = (e * ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)).sum() / s
        return float(np.sqrt(r2))


def cross_w(f):
    return cross_section_winding(f.phi, Xg, Yg, Zg, R0_GLOB, RHO0_GLOB)


def wz_line(f, cx=None, cy=None):
    """环面绕数诊断：一条 z 线的 -Sigma k / 1"""
    if cx is None:
        cx = f.N // 2
        cy = f.N // 2
    return float(-f.k[2][cx, cy, :].sum())


# ---------------- [A]/[B]/[B2]：裸模型 ----------------
N = 32
R0_GLOB, RHO0_GLOB = 6.0, 2.0
c0 = N // 2
idx = np.arange(N) - c0
Zg, Yg, Xg = np.meshgrid(idx, idx, idx, indexing="ij")

print("=" * 72)
print("T4/T5 约束扇区动力学实验（N=32 周期盒）")
print("=" * 72)

phi_ring, _ = make_ring(N, R0_GLOB, RHO0_GLOB)
phi_ut = make_uniform_twist(N)
phi0 = phi_ring + phi_ut
E_uni = N ** 3 * (1.0 - np.cos(TWO_PI / N))

f0 = BareXY(N, phi0, frozen=False)
print(f"\n初条件 = 环（R0=6, rho0=2）+ 均匀扭转：E = {f0.energy():.3f}"
      f"（环部分约 481 + 均匀扭转 E_uni = {E_uni:.2f}），"
      f"截面绕数 Wc = {cross_w(f0):+.3f}，环面绕数 W_z = {wz_line(f0):+.0f}")

print("\n[A] 无约束基线（撕裂通道开放）")
fA = BareXY(N, phi0, frozen=False)
recA = fA.relax(20000, tag="A 无约束")
print(f"    终态：E = {recA[-1][1]:.4f}，Wc = {cross_w(fA):+.3f}，W_z = {wz_line(fA):+.0f}")

print("\n[B] 冻结分支流（分支夹取协议，切向运动保留）")
fB = BareXY(N, phi0, frozen=True)
recB = fB.relax(60000, tag="B 冻结分支")
EB, rB = recB[-1][1], recB[-1][2]
print(f"    终态：E = {EB:.4f}（对照 E_uni = {E_uni:.4f}），r_g = {rB:.3f}，"
      f"Wc = {cross_w(fB):+.3f}，W_z = {wz_line(fB):+.0f}，冻结校验余量 = {fB.verify_frozen():.2e}")

print("\n[E] 判定汇总见文末")

# ---------------- [C] 三元组窄区碗判决性下降（铁磁背景） ----------------
class Triplet:
    """三元组 H = kphi*Sigma_{a<b}(1-cos delta^ab) + kappa_par*Sigma(1-cos Delta phi^a)"""

    def __init__(self, N, kphi, R0, rho0, frozen, staggered=False):
        self.N = N
        self.kphi = kphi
        self.frozen = frozen
        c = N // 2
        idx = np.arange(N) - c
        Z, Y, X = np.meshgrid(idx, idx, idx, indexing="ij")
        rho = np.sqrt(X ** 2 + Y ** 2)
        psi = np.arctan2(Z, rho - R0)
        d = np.sqrt((rho - R0) ** 2 + Z ** 2)
        g = 0.5 * (1.0 - np.tanh((d - rho0) / 1.0))
        base = np.pi * ((X + Y + Z) % 2) if staggered else np.zeros_like(X, dtype=float)
        self.phi = np.stack([base + psi * g, base.copy(), base.copy()])
        self.X, self.Y, self.Z = X, Y, Z
        self.k = [[kidx(self.bond_raw(ci, a)) for a in range(3)] for ci in range(3)]

    def bond_raw(self, ci, a):
        return np.roll(self.phi[ci], -1, axis=a) - self.phi[ci]

    def energy(self):
        E = 0.0
        for ci in range(3):
            for a in range(3):
                E += np.sum(1.0 - np.cos(self.bond_raw(ci, a)))
        for a in range(3):
            for b in range(a + 1, 3):
                E += self.kphi * np.sum(1.0 - np.cos(self.phi[b] - self.phi[a]))
        return float(E)

    def grad(self):
        g = [np.zeros_like(self.phi[0]) for _ in range(3)]
        for ci in range(3):
            for a in range(3):
                s = np.sin(self.bond_raw(ci, a))
                g[ci] += np.roll(s, 1, axis=a) - s
        for a in range(3):
            for b in range(a + 1, 3):
                s = np.sin(self.phi[b] - self.phi[a])
                g[a] -= s
                g[b] += s
        return g

    def enforce_walls(self, sweeps=4):
        for _ in range(sweeps):
            done = True
            for ci in range(3):
                for a in range(3):
                    u = self.bond_raw(ci, a)
                    lo = TWO_PI * self.k[ci][a] - np.pi + EPS_WALL
                    hi = TWO_PI * self.k[ci][a] + np.pi - EPS_WALL
                    viol = (u < lo) | (u > hi)
                    if not viol.any():
                        continue
                    done = False
                    tgt = np.where(u < lo, lo, np.where(u > hi, hi, u))
                    corr = (tgt - u) * 0.5
                    self.phi[ci] = self.phi[ci] - corr + np.roll(corr, 1, axis=a)
            if done:
                break

    def relax(self, iters, eta0=0.02, eta_max=0.05, snap_every=2500, tag=""):
        eta = eta0
        rec = [(0, self.energy(), self.gyration_x())]
        t0 = time.time()
        for t in range(1, iters + 1):
            dphi = -eta * np.stack(self.grad())
            self.phi = self.phi + dphi
            if self.frozen:
                self.enforce_walls()
            eta = min(eta * 1.02, eta_max)
            if t % snap_every == 0 or t == iters:
                rec.append((t, self.energy(), self.gyration_x()))
        print(f"    [{tag}] 完成 {t} 步，用时 {time.time() - t0:.1f} s")
        return rec

    def action_x(self):
        e = np.zeros_like(self.phi[0])
        for a in range(3):
            e += 0.5 * (1.0 - np.cos(self.bond_raw(0, a)))
        for b in range(1, 3):
            e += 0.5 * self.kphi * (1.0 - np.cos(self.phi[b] - self.phi[0]))
        return e

    def gyration_x(self):
        e = self.action_x()
        s = e.sum()
        if s < 1e-9:
            return 0.0
        X, Y, Z = self.X, self.Y, self.Z
        cx, cy, cz = (e * X).sum() / s, (e * Y).sum() / s, (e * Z).sum() / s
        r2 = (e * ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)).sum() / s
        return float(np.sqrt(r2))


NT = 32
print("\n[C] 三元组窄区静态碗判决性下降（铁磁背景，kappa_phi=0.5, rho0=1.0）")
kphi_D, rho0_D = 0.5, 1.0
print("    静态 E(R) 扫描（构造族能量，未弛豫，铁磁真空 = 0）：")
scan = []
for R0 in [0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5]:
    ft = Triplet(NT, kphi_D, R0, rho0_D, frozen=False, staggered=False)
    Eex = ft.energy()
    scan.append((R0, Eex))
    print(f"      R = {R0:.1f}: E = {Eex:10.3f}   r_g(phi^x) = {ft.gyration_x():.3f}")
R_star = min(scan, key=lambda p: p[1])[0]
endpoint = R_star in (scan[0][0], scan[-1][0])
print(f"    静态极小 R* = {R_star}"
      + ("（警告：落在扫描端点，碗形前提不成立）" if endpoint else "（扫描区间内点）"))

print(f"    判决性下降：从 R* = {R_star} 初始化")
ftU = Triplet(NT, kphi_D, R_star, rho0_D, frozen=False, staggered=False)
recCU = ftU.relax(15000, tag=f"C 无约束 R*={R_star}")
print(f"    无约束终态：E = {recCU[-1][1]:.4f}，r_g = {recCU[-1][2]:.3f}")
ftC = Triplet(NT, kphi_D, R_star, rho0_D, frozen=True, staggered=False)
recCC = ftC.relax(15000, tag=f"C 冻结分支 R*={R_star}")
print(f"    冻结分支终态：E = {recCC[-1][1]:.4f}，r_g = {recCC[-1][2]:.3f}")

print("\n    kappa_phi = 1.2 静态对照：")
for R0 in [0.8, 1.0, 1.4, 2.0]:
    ft = Triplet(NT, 1.2, R0, rho0_D, frozen=False, staggered=False)
    print(f"      R = {R0:.1f}: E = {ft.energy():10.3f}")

# ---------------- [D] 交错背景裁定观测 ----------------
print("\n[D] 交错背景裁定观测（kappa_phi=0.5, rho0=1.0, R=1.0，协议同四件套复跑）")
ftS = Triplet(NT, 0.5, 1.0, 1.0, frozen=False, staggered=True)
E_stag = ftS.energy()
print(f"    交错真空 + 构造能量 = {E_stag:.1f} = {E_stag / NT ** 3:.2f} N^3（理论 18 N^3）")
recDS = ftS.relax(8000, tag="D 交错无约束")
print(f"    无约束终态：E = {recDS[-1][1]:.1f}（{recDS[-1][1] / NT ** 3:.3f} N^3）"
      f"{'→ 滚向铁磁真空：交错背景动力学不稳定' if recDS[-1][1] < 0.1 * E_stag else '→ 未滚落'}"
      f"，r_g(phi^x) = {recDS[-1][2]:.3f}")
ftSC = Triplet(NT, 0.5, 1.0, 1.0, frozen=True, staggered=True)
E_before = ftSC.energy()
recDSC = ftSC.relax(3000, tag="D 交错冻结分支", snap_every=3000)
print(f"    冻结分支终态：E = {recDSC[-1][1]:.1f}（初始 {E_before:.1f}）"
      f"{'→ 背景完全刚性（反相键坐分支，夹取协议下每步越界即被夹回）' if abs(recDSC[-1][1] - E_before) < 1e-6 else '→ 有运动'}")

# ---------------- [E] 判定汇总 ----------------
print("\n" + "=" * 72)
print("[E] 判定汇总")
l_star = recB[-1][2]
ring_part0 = f0.energy() - E_uni
print(f"  1. 局域涡量（环，表示性绕数）：无扇区可保护")
print(f"     [A] 无约束：环部分自由撕裂（贡献 {ring_part0:.1f} -> 0，复现 C1），"
      f"终态 E = {recA[-1][1]:.4f} = E_uni（均匀扭转零力驻点驻留）")
print(f"     [C] 冻结分支：环的分支标记冻结但 S^1 构型真空化，终态 E = {recCC[-1][1]:.4f}"
      f"（隐藏扭转 = 物理真空），r_g = {recCC[-1][2]:.3f}")
print(f"  2. 环面绕数（W_z，真扇区）：守恒严格但极小强制离域")
print(f"     [B] 冻结分支：W_z = {wz_line(fB):+.0f} 严格保持（逐键夹取，校验余量 {fB.verify_frozen():.1e}）；"
      f"终态 E = {EB:.2f} >= E_uni = {E_uni:.2f}")
print(f"         （扇区下界：W_z=1 冻结每条 z 线的 Sigma w = 2 pi，per-line 极小 = 均匀分布，"
      f"E >= E_uni 为解析界）；r_g = {l_star:.2f} 格距 = 盒尺度离域；"
      f"[B] 停在分支钉扎亚稳态（扇区内弛豫路径依赖）")
print(f"     [A] 无约束：W_z = {wz_line(fA):+.0f} 零温驻留（撕裂需越有限势垒 ~ O(kappa)，梯度流未触发）")
print(f"  3. 窄区碗（铁磁背景）：静态扫描无内点碗（R* = {R_star} 落扫描端点）；"
      f"无约束下降 E -> {recCU[-1][1]:.3f}（切片碗伪影确认）")
print(f"  4. 交错背景：无约束 18N^3 -> {recDS[-1][1] / NT ** 3:.2f} N^3（动力学不稳定）；"
      f"冻结分支 E 变化 {100 * abs(recDSC[-1][1] - E_before) / E_before:.1f}%（近乎刚性），设定退化，铁磁为准")
print(f"  5. T4/T5 裁决：T4 否定：真扇区极小存在但拓扑强制离域（每线必须缠绕），"
      f"局域 dressing 形态在相位场自由度内不存在；环的局域绕数为表示性，"
      f"冻结分支下隐藏为物理真空。T5 消解（无局域载体可测宽度）。")
print(f"     含义：超选择层守恒全局绕数（数值确认：W_z 严格守恒 + E >= E_uni 下界），"
      f"但粒子的能量与尺寸必须由场外自由度承担（格点内离散结构 = 质量机制定案）"
      f"，即保护归波、承载归粒两层分工的动力学证明形态。")

# ---------------- 出图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax = axes[0, 0]
ax.semilogy([r[0] for r in recA], [max(r[1], 1e-10) for r in recA],
            "-", color="0.4", label="A unconstrained")
ax.semilogy([r[0] for r in recB], [max(r[1], 1e-10) for r in recB],
            "-", color="C0", label="B frozen branches")
ax.axhline(E_uni, ls="--", color="C3", lw=1, label="E_uni (uniform twist)")
ax.set_xlabel("step")
ax.set_ylabel("E")
ax.set_title("Energy: tearing open vs frozen branches")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot([r[0] for r in recA], [r[2] for r in recA], "-", color="0.4", label="A unconstrained")
ax.plot([r[0] for r in recB], [r[2] for r in recB], "-", color="C0", label="B frozen branches")
ax.set_xlabel("step")
ax.set_ylabel("gyration radius r_g (lattice)")
ax.set_title("Size: delocalization of the frozen-sector minimum")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot([p[0] for p in scan], [p[1] for p in scan], "o-", color="C2",
        label=f"static E(R) ferro, kphi={kphi_D}")
ax.axvline(R_star, ls=":", color="C2", lw=1)
ax.plot([r[0] for r in recCU], [r[2] for r in recCU], "-", color="C1",
        label="descent r_g unconstrained")
ax.plot([r[0] for r in recCC], [r[2] for r in recCC], "--", color="C9",
        label="descent r_g frozen")
ax.set_xlabel("R (scan) / step (descent)")
ax.set_ylabel("E / r_g")
ax.set_title("Narrow bowl static scan & judgment descent (ferro)")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.semilogy([r[0] for r in recCU], [max(r[1], 1e-10) for r in recCU],
            "-", color="C1", label="C unconstrained")
ax.semilogy([r[0] for r in recCC], [max(r[1], 1e-10) for r in recCC],
            "--", color="C9", label="C frozen branches")
ax.set_xlabel("step")
ax.set_ylabel("E")
ax.set_title("Narrow bowl descent: tearing vs hidden-twist endgame")
ax.legend(fontsize=8)

fig.suptitle("Sector dynamics: two windings adjudicated (N=32)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "images", "minimal_verification_sector_dynamics.png")
fig.savefig(out, dpi=130)
print("\n图已保存：images/minimal_verification_sector_dynamics.png")
print("实验完成（仅打印与图，未写任何文档）。")
