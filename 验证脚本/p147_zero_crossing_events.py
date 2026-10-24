# -*- coding: utf-8 -*-
"""
p147_zero_crossing_events.py
判据第 666 号：P1-47 阶段二机制轮之二——振幅过零事件接口与温度淬火协议。

背景（判据第 665 号登记的两项缺口）：
  其一，提升边表示在振幅层把噪声荷冻结，成核事件在表示层不可记录；
  其二，固定强度噪声加线性穿越不构成温度淬火。
本判据补两项并判其是否打开成核读数。

模型层声明：复序参量 psi = rho * exp(i*beta)，自由能
  F = sum_bond (J/2)|psi_i - psi_j|^2 + sum_i [ -a(t)/2 |psi_i|^2 + U/4 |psi_i|^4 ]，
演化取含时 Ginzburg-Landau 梯度流加 Langevin 噪声。a > 0 为破缺相、a < 0 为
对称相；淬火协议令 a(t) 由 A_SYM 线性升到 A_BRK（穿越 a = 0），温度同步由
T0 降到 0，故「温度与控制参数同步」由同一时间表驱动。

振幅过零事件接口（本判据的核心构造）：当某格点 |psi| 落到 EPS_CORE 以下时，
该处相位失去定义，接口把该格点四条出边的提升值**重置为即时折叠差分**
（fold(Δbeta)）。重置不人为指定荷变化，只清掉该点的历史累计；由于
fold(Δbeta) ≡ Δbeta (mod 2π)，同余由构造自动保持（这是与「人为注入
2π*q/4」方案的关键区别，后者会破坏同余）。荷是否改变、改变多少，由
过零前后的真实相位决定，故该接口可记录事件而不预设事件的结果。

判据与理论期望（先立判据再实现）：
  E1 冒烟检查：深破缺相（a = A_BRK，无淬火）振幅收敛 rho0（相对差 < 20%）；
     相位增量沿两轴有变化性。
  E2 接口的记录力（区分力核心）：在深破缺相下人为制造一个过零事件（把某格点
     相位改成绕转一周的涡旋并置零振幅），接口触发且荷图改变、总荷仍为 0；
     不制造事件时接口不触发、荷图不变。
  E3 温度淬火下的成核：温度淬火协议下荷图变化步 > 0，残留对数偏离初始基线
     （与第 665 号零温冻结形成对照）。
  E4 淬火速率依赖：残留随 T_Q 变化，log-log 斜率实测登记（参考值 -1/2 与
     -1，不预设）。
  E5 读数合法性：总荷恒 0；提升边与即时折叠差分同余 < 1e-8。
  E6 无事件对照（恒真排除）：关掉接口后，同协议下荷图变化步应为 0。
多档：T_Q in {50, 200, 800}，种子 {1, 2, 3}。
"""
import math
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images"
OUT = IMG + r"\p147_zero_crossing_666_fig1.png"

N = 64
J, U, GAMMA, DT = 1.0, 1.0, 1.0, 0.02
REL = 2000
A_SYM, A_BRK, T0 = -0.5, 1.0, 0.20
EPS_CORE = 1.0e-3
TAUS = (50, 200, 800)
SEEDS = (1, 2, 3)
RHO0 = math.sqrt(A_BRK / U)


def fold(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def lap(psi):
    return (np.roll(psi, 1, 0) + np.roll(psi, -1, 0)
            + np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4.0 * psi)


class LiftedEdges:
    """第 658 号提升边表示（周期边界）：Eh[r,c] 为 (r,c)->(r,c+1) 的累计相位差。"""

    def __init__(self, beta):
        self.Eh = fold(np.roll(beta, -1, 1) - beta)
        self.Ev = fold(np.roll(beta, -1, 0) - beta)

    def update(self, delta):
        self.Eh += np.roll(delta, -1, 1) - delta
        self.Ev += np.roll(delta, -1, 0) - delta

    def charges(self):
        q = self.Eh + np.roll(self.Ev, -1, 1) - np.roll(self.Eh, -1, 0) - self.Ev
        return np.round(q / (2.0 * math.pi)).astype(int)

    def reset_at(self, r, c, beta):
        """振幅过零事件接口：把格点 (r,c) 四条出边的提升重置为即时折叠差分。

        只清历史，不指定荷变化；同余由 fold 的定义自动保持。
        """
        rc = int(r)
        cc = int(c)
        rn, cs = (rc + 1) % N, (cc + 1) % N
        rm, cp = (rc - 1) % N, (cc - 1) % N
        self.Eh[rc, cc] = fold(beta[rc, cs] - beta[rc, cc])
        self.Eh[rc, cp] = fold(beta[rc, cc] - beta[rc, cp])
        self.Ev[rc, cc] = fold(beta[rn, cc] - beta[rc, cc])
        self.Ev[rm, cc] = fold(beta[rc, cc] - beta[rm, cc])


def run(tau_q, seed, use_events=True, quench=True, inject=None, uniform=False):
    rng = np.random.default_rng(seed)
    if uniform:
        beta0 = np.zeros((N, N))
    else:
        beta0 = rng.uniform(0.0, 2.0 * math.pi, (N, N))
    psi = 0.05 * np.exp(1j * beta0)
    le = LiftedEdges(np.angle(psi))
    q_prev = le.charges()
    n0 = int(np.abs(q_prev).sum()) // 2
    q_changes = 0
    events = 0
    cong_max = 0.0
    min_abs = 1.0e9
    for t in range(tau_q + REL):
        if quench:
            frac = min(1.0, t / float(tau_q))
            a = A_SYM + (A_BRK - A_SYM) * frac
            temp = T0 * (1.0 - frac)
        else:
            a = A_BRK
            temp = 0.0
        psi_old = psi
        psi = psi + GAMMA * DT * (J * lap(psi) + a * psi - U * np.abs(psi) ** 2 * psi)
        if temp > 0.0:
            sc = math.sqrt(2.0 * GAMMA * temp * DT)
            psi = psi + sc * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / math.sqrt(2.0)
        if inject is not None and t == inject:
            r0, c0 = N // 2, N // 2
            rr, cc = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
            psi = psi * np.exp(1j * np.arctan2(rr - r0, cc - c0))
            psi[r0, c0] = 0.0
        le.update(np.angle(psi * np.conj(psi_old)))
        mabs = float(np.min(np.abs(psi)))
        if mabs < min_abs:
            min_abs = mabs
        if use_events:
            core = np.abs(psi) < EPS_CORE
            if core.any():
                beta_now = np.angle(psi)
                for (r, c) in np.argwhere(core):
                    le.reset_at(r, c, beta_now)
                    events += 1
                # 振幅归零但保留相位：若写成 psi=0，angle(0)=0 与提升边不一致，
                # 同余会立刻破坏（偏差 pi）。
                psi[core] = EPS_CORE * 1.0e-3 * np.exp(1j * beta_now[core])
        b = np.angle(psi)
        direct = fold(np.roll(b, -1, 1) - b)
        cong = float(np.max(np.abs(np.exp(1j * (le.Eh - direct)) - 1.0)))
        if cong > cong_max:
            cong_max = cong
        q = le.charges()
        if not np.array_equal(q, q_prev):
            q_changes += 1
            q_prev = q
    q = le.charges()
    return {
        "tau_q": tau_q, "seed": seed, "events": events,
        "n0": n0, "npair": int(np.abs(q).sum()) // 2, "qsum": int(q.sum()),
        "q_changes": q_changes, "cong_max": cong_max, "min_abs": min_abs,
        "rho_mean": float(np.mean(np.abs(psi))), "q": q,
    }


t0 = time.time()
print("=" * 72)
print("E1 冒烟检查（深破缺相 a = %.2f，期望振幅收敛 rho0 = %.3f）" % (A_BRK, RHO0))
r_smoke = run(TAUS[1], SEEDS[0], use_events=False, quench=False)
rel = abs(r_smoke["rho_mean"] - RHO0) / RHO0
print("  弛豫后平均振幅 = %.4f（期望 %.4f，相对差 %.2f%%）" % (r_smoke["rho_mean"], RHO0, 100 * rel))
rng = np.random.default_rng(7)
psi = 0.05 * np.exp(1j * rng.uniform(0.0, 2.0 * math.pi, (N, N)))
psi_old = psi
psi = psi + GAMMA * DT * (J * lap(psi) + A_BRK * psi - U * np.abs(psi) ** 2 * psi)
delta = np.angle(psi * np.conj(psi_old))
vx = float(np.std(np.diff(delta, axis=1)))
vy = float(np.std(np.diff(delta, axis=0)))
print("  相位增量逐轴变化性：x 轴 %.3e、y 轴 %.3e" % (vx, vy))
e1 = (rel < 0.20) and vx > 0.0 and vy > 0.0
print("  [E1 %s]" % ("PASS" if e1 else "FAIL"))

print("=" * 72)
print("E2 接口的记录力（均匀场对照；期望：无注入不触发；注入涡旋后触发且总荷为 0）")
r_noinj = run(TAUS[1], SEEDS[0], use_events=True, quench=False, uniform=True)
r_inj = run(TAUS[1], SEEDS[0], use_events=True, quench=False, inject=200, uniform=True)
print("  均匀场无注入：荷图变化步 %d，事件 %d，总荷 %d，n0 %d -> n %d"
      % (r_noinj["q_changes"], r_noinj["events"], r_noinj["qsum"], r_noinj["n0"], r_noinj["npair"]))
print("  均匀场注入涡旋：荷图变化步 %d，事件 %d，总荷 %d，n0 %d -> n %d"
      % (r_inj["q_changes"], r_inj["events"], r_inj["qsum"], r_inj["n0"], r_inj["npair"]))
e2 = ((r_noinj["q_changes"] == 0) and (r_noinj["events"] == 0)
      and (r_inj["q_changes"] > 0) and (r_inj["events"] > 0)
      and (r_inj["qsum"] == 0) and (r_inj["npair"] > r_inj["n0"]))
print("  [E2 %s]" % ("PASS" if e2 else "FAIL"))

print("=" * 72)
print("E3/E4 温度淬火下的成核与速率依赖（期望：荷图变化、残留偏离基线、随 T_Q 变化）")
recs = []
for tau in TAUS:
    for sd in SEEDS:
        r = run(tau, sd, use_events=True)
        recs.append(r)
        print("  T_Q=%4d 种子 %d：n0 = %d -> n = %d，荷图变化步 %d，事件 %d，同余 %.2e，min|psi| %.2e"
              % (tau, sd, r["n0"], r["npair"], r["q_changes"], r["events"], r["cong_max"], r["min_abs"]))
e3 = all(r["q_changes"] > 0 and r["npair"] != r["n0"] for r in recs)
means = [float(np.mean([r["npair"] for r in recs if r["tau_q"] == t])) for t in TAUS]
e4 = len(set(means)) > 1
slope = float(np.polyfit(np.log(np.array(TAUS, dtype=float)),
                         np.log(np.maximum(np.array(means), 0.5)), 1)[0])
print("  残留均值 n = %s，log-log 斜率 = %.3f（参考值 -0.5 与 -1，实测登记）" % (means, slope))
print("  [E3 %s] [E4 %s]" % ("PASS" if e3 else "FAIL", "PASS" if e4 else "FAIL"))

print("=" * 72)
print("E5 读数合法性（期望：总荷恒 0；同余 < 1e-8）")
e5a = all(r["qsum"] == 0 for r in recs)
cong_all = max(r["cong_max"] for r in recs)
e5b = cong_all < 1e-8
print("  总荷恒零 = %s；同余偏差最大 %.2e" % (e5a, cong_all))
print("  [E5a %s] [E5b %s]" % ("PASS" if e5a else "FAIL", "PASS" if e5b else "FAIL"))

print("=" * 72)
print("E6 无事件对照（恒真排除：关掉接口后荷图应不变）")
ctrl = [run(t, s, use_events=False) for t in TAUS for s in SEEDS]
e6 = all(r["q_changes"] == 0 for r in ctrl)
print("  关接口后荷图变化步 = %s（期望全 0）" % [r["q_changes"] for r in ctrl])
print("  [E6 %s]" % ("PASS" if e6 else "FAIL"))

print("=" * 72)
passed = sum([e1, e2, e3, e4, e5a, e5b, e6])
print("判据汇总：E1 %s / E2 %s / E3 %s / E4 %s / E5a %s / E5b %s / E6 %s（%d/7）"
      % ("PASS" if e1 else "FAIL", "PASS" if e2 else "FAIL", "PASS" if e3 else "FAIL",
         "PASS" if e4 else "FAIL", "PASS" if e5a else "FAIL", "PASS" if e5b else "FAIL",
         "PASS" if e6 else "FAIL", passed))
print("elapsed %.1f s" % (time.time() - t0))

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
ax = axes[0]
ax.semilogx(TAUS, means, "o-", color="#4472c4")
ax.set_xlabel(r"$T_Q$（淬火时长，步）")
ax.set_ylabel("残留缺陷对数 $n$")
ax.set_title("温度淬火协议下残留密度（斜率 %.2f）" % slope)
ax = axes[1]
if recs:
    ax.imshow(recs[0]["q"], cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
    ax.set_title("淬火后荷图快照（$T_Q=%d$）" % TAUS[0])
    ax.set_xlabel("$c$")
    ax.set_ylabel("$r$")
fig.tight_layout()
fig.savefig(OUT, dpi=160)
plt.close(fig)
print("figure saved:", OUT)
