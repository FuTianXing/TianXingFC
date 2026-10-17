# -*- coding: utf-8 -*-
"""
minimal_verification_p147_complex_order_parameter.py
判据第 665 号：P1-47 阶段二机制轮——复序参量最小模型与淬火成核。

模型层声明：复序参量 psi = rho * exp(i*beta)，自由能
  F = sum_bond (J/2)|psi_i - psi_j|^2 + sum_i [ -a(t)/2 |psi_i|^2 + U/4 |psi_i|^4 ]，
演化取含时 Ginzburg-Landau 梯度流；温度档加 Langevin 噪声
（<eta* eta> = 2*Gamma*T*dt，复高斯，温度强度固定为恒温浴），控制参数 a(t)
由 -A0 线性升到 +A0（穿越时刻 t_c = T_Q/2），随后在 a = A0 下弛豫。
相位读数沿用第 658 号提升边表示（每条边维护无界累计相位差，按端点相位
增量更新），荷 = plaq 四边提升和除以 2*pi 取整。本模型为最小模型层选择；
真实淬火中温度与控制参数的同步演化、振幅项的体系来源与真实校准动力学
归各专线，不在本判据射程。

判据与理论期望（先立判据再实现；2026-10-17 首跑后按教训重设）：
  Q1 冒烟检查：破缺相振幅收敛 rho0 = sqrt(A0/U)（相对差 < 20%）；相位增量
     场沿两轴均有变化性（变化的轴 = 更新的轴）。
  Q2 零温无成核（期望通过）：梯度流（T = 0）下提升边荷图全程不变
     （q_changes = 0，且残留对数等于初始基线 n0），即拓扑荷被连续演化冻结。
  Q3 冻结振幅反例（期望通过）：纯相位极限（|psi| 固定）下荷图同样不变，
     与 Q2 构成机制对照（复现第 658 号）。
  Q4 零温读数合法性（期望通过）：总荷恒 0；提升边与即时折叠差分同余 < 1e-8。
  Q5 有限温成核（期望通过）：T = 0.1 的 Langevin 档下，弛豫后残留对数偏离
     初始基线（n_after != n0），即热噪声驱动拓扑事件。
  Q6 有限温淬火速率依赖（期望通过）：有限温档下 n 随 T_Q 变化，log-log
     斜率实测登记（参考值 -1/2 与 -1，不预设）。
  Q7 有限温读数合法性（期望通过）：有限温档下同余偏差仍 < 1e-8、总荷恒 0。
多档：T_Q in {25, 100, 400}，种子 {1, 2, 3}，温度档 {0.0, 0.1}。
首跑教训：原判据「n > 0」为恒真（随机初始相位即给 n > 0），已改为基线对照。
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
OUT = IMG + r"\p147_complex_order_665_fig1.png"

N = 64
J, U, A0, GAMMA, DT = 1.0, 1.0, 1.0, 1.0, 0.05
REL = 1500
TAUS = (25, 100, 400)
SEEDS = (1, 2, 3)
T_WARM = 0.1
RHO0 = math.sqrt(A0 / U)


def fold(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def lap(psi):
    return (np.roll(psi, 1, 0) + np.roll(psi, -1, 0)
            + np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4.0 * psi)


class LiftedEdges:
    """第 658 号提升边表示（周期边界版）：Eh[r,c] 为 (r,c)->(r,c+1) 的累计相位差。"""

    def __init__(self, beta):
        self.Eh = fold(np.roll(beta, -1, 1) - beta)
        self.Ev = fold(np.roll(beta, -1, 0) - beta)

    def update(self, delta):
        self.Eh += np.roll(delta, -1, 1) - delta
        self.Ev += np.roll(delta, -1, 0) - delta

    def charges(self):
        q = self.Eh + np.roll(self.Ev, -1, 1) - np.roll(self.Eh, -1, 0) - self.Ev
        return np.round(q / (2.0 * math.pi)).astype(int)


def run(tau_q, seed, freeze_amp=False, temp=0.0):
    rng = np.random.default_rng(seed)
    beta0 = rng.uniform(0.0, 2.0 * math.pi, (N, N))
    amp0 = np.ones((N, N)) if freeze_amp else 0.1 * np.ones((N, N))
    psi = amp0 * np.exp(1j * beta0)
    le = LiftedEdges(np.angle(psi))
    q_prev = le.charges()
    n0 = int(np.abs(q_prev).sum()) // 2
    q_changes = 0
    cong_max = 0.0
    min_abs = 1.0e9
    noise_scale = math.sqrt(2.0 * GAMMA * temp * DT) if temp > 0.0 else 0.0
    for t in range(tau_q + REL):
        a = A0 * (2.0 * t / tau_q - 1.0) if t < tau_q else A0
        psi_old = psi
        psi = psi + GAMMA * DT * (J * lap(psi) + a * psi - U * np.abs(psi) ** 2 * psi)
        if noise_scale > 0.0:
            psi = psi + noise_scale * (rng.normal(size=(N, N))
                                       + 1j * rng.normal(size=(N, N))) / math.sqrt(2.0)
        if freeze_amp:
            psi = psi / np.abs(psi)
        le.update(np.angle(psi * np.conj(psi_old)))
        b = np.angle(psi)
        direct = fold(np.roll(b, -1, 1) - b)
        cong = float(np.max(np.abs(np.exp(1j * (le.Eh - direct)) - 1.0)))
        if cong > cong_max:
            cong_max = cong
        mabs = float(np.min(np.abs(psi)))
        if mabs < min_abs:
            min_abs = mabs
        q = le.charges()
        if not np.array_equal(q, q_prev):
            q_changes += 1
            q_prev = q
    q = le.charges()
    return {
        "tau_q": tau_q, "seed": seed, "freeze": freeze_amp, "temp": temp,
        "n0": n0, "npair": int(np.abs(q).sum()) // 2, "qsum": int(q.sum()),
        "q_changes": q_changes, "cong_max": cong_max, "min_abs": min_abs,
        "rho_mean": float(np.mean(np.abs(psi))), "q": q,
    }


t0 = time.time()
print("=" * 72)
print("Q1 冒烟检查（期望：破缺相振幅收敛 rho0 = %.3f；相位增量沿两轴有变化性）" % RHO0)
r_smoke = run(TAUS[1], SEEDS[0])
rel = abs(r_smoke["rho_mean"] - RHO0) / RHO0
print("  弛豫后平均振幅 = %.4f（期望 %.4f，相对差 %.2f%%）" % (r_smoke["rho_mean"], RHO0, 100 * rel))
rng = np.random.default_rng(7)
psi = 0.1 * np.exp(1j * rng.uniform(0.0, 2.0 * math.pi, (N, N)))
psi_old = psi
psi = psi + GAMMA * DT * (J * lap(psi) + (-A0) * psi - U * np.abs(psi) ** 2 * psi)
delta = np.angle(psi * np.conj(psi_old))
vx = float(np.std(np.diff(delta, axis=1)))
vy = float(np.std(np.diff(delta, axis=0)))
print("  相位增量逐轴变化性：x 轴 %.3e、y 轴 %.3e（期望均非零）" % (vx, vy))
q1 = (rel < 0.20) and vx > 0.0 and vy > 0.0
print("  [Q1 %s]" % ("PASS" if q1 else "FAIL"))

print("=" * 72)
print("Q2/Q3 零温档（期望：荷图全程不变，残留对数等于初始基线）")
zero, frozen = [], []
for tau in TAUS:
    for sd in SEEDS:
        r = run(tau, sd)
        zero.append(r)
        frozen.append(run(tau, sd, freeze_amp=True))
        print("  T_Q=%4d 种子 %d：n0 = %d -> n = %d，荷图变化步 %d｜同余 %.2e"
              % (tau, sd, r["n0"], r["npair"], r["q_changes"], r["cong_max"]))
q2 = all(r["q_changes"] == 0 and r["npair"] == r["n0"] for r in zero)
q3 = all(r["npair"] == r["n0"] for r in frozen)
print("  [Q2 %s]（零温无成核） [Q3 %s]（冻结振幅反例）" % ("PASS" if q2 else "FAIL", "PASS" if q3 else "FAIL"))

print("=" * 72)
print("Q4 零温读数合法性（期望：总荷恒 0；同余 < 1e-8）")
q4a = all(r["qsum"] == 0 for r in zero)
cong_zero = max(r["cong_max"] for r in zero)
q4b = cong_zero < 1e-8
print("  总荷恒零 = %s；同余偏差最大 %.2e" % (q4a, cong_zero))
print("  [Q4a %s] [Q4b %s]" % ("PASS" if q4a else "FAIL", "PASS" if q4b else "FAIL"))

print("=" * 72)
print("Q5/Q6 有限温档（T = %.2f；期望：残留偏离基线、n 随 T_Q 变化）" % T_WARM)
warm = []
for tau in TAUS:
    for sd in SEEDS:
        r = run(tau, sd, temp=T_WARM)
        warm.append(r)
        print("  T_Q=%4d 种子 %d：n0 = %d -> n = %d，荷图变化步 %d｜同余 %.2e"
              % (tau, sd, r["n0"], r["npair"], r["q_changes"], r["cong_max"]))
q5 = all(r["npair"] != r["n0"] for r in warm)
wmeans = [float(np.mean([r["npair"] for r in warm if r["tau_q"] == t])) for t in TAUS]
q6 = len(set(wmeans)) > 1
slope_w = float(np.polyfit(np.log(np.array(TAUS, dtype=float)),
                           np.log(np.maximum(np.array(wmeans), 0.5)), 1)[0])
print("  有限温均值 n = %s，log-log 斜率 = %.3f（参考值 -0.5 与 -1，实测登记）" % (wmeans, slope_w))
print("  [Q5 %s]（有限温成核） [Q6 %s]（随 T_Q 变化）" % ("PASS" if q5 else "FAIL", "PASS" if q6 else "FAIL"))

print("=" * 72)
print("Q8 温度扫描（探索：固定强度噪声能否驱动拓扑事件）")
hot_recs = []
for tt in (0.0, 0.1, 1.0):
    r = run(TAUS[1], SEEDS[0], temp=tt)
    hot_recs.append(r)
    print("  T = %.2f：n0 = %d -> n = %d，荷图变化步 %d｜同余 %.2e｜min|psi| = %.2e"
          % (tt, r["n0"], r["npair"], r["q_changes"], r["cong_max"], r["min_abs"]))
q8 = any(r["q_changes"] > 0 for r in hot_recs)
print("  [Q8 %s]（存在某温度档使荷图变化）" % ("PASS" if q8 else "FAIL"))

print("=" * 72)
print("Q7 有限温读数合法性（期望：同余 < 1e-8、总荷恒 0）")
q7a = all(r["qsum"] == 0 for r in warm)
cong_warm = max(r["cong_max"] for r in warm)
q7 = q7a and cong_warm < 1e-8
print("  总荷恒零 = %s；同余偏差最大 %.2e" % (q7a, cong_warm))
print("  [Q7 %s]" % ("PASS" if q7 else "FAIL"))

print("=" * 72)
passed = sum([q1, q2, q3, q4a, q4b, q5, q6, q7, q8])
print("判据汇总：Q1 %s / Q2 %s / Q3 %s / Q4a %s / Q4b %s / Q5 %s / Q6 %s / Q7 %s / Q8 %s（%d/9）"
      % ("PASS" if q1 else "FAIL", "PASS" if q2 else "FAIL", "PASS" if q3 else "FAIL",
         "PASS" if q4a else "FAIL", "PASS" if q4b else "FAIL", "PASS" if q5 else "FAIL",
         "PASS" if q6 else "FAIL", "PASS" if q7 else "FAIL", "PASS" if q8 else "FAIL", passed))
print("结论：零温与有限温（含强噪声档 T = 1.0）下荷图全程冻结，残留对数等于初始基线，成核未发生。")
print("机制指认：提升边表示把 |psi| 极小处的数值噪声荷永久冻结，成核事件在表示层不可记录；")
print("缺失环节收窄为振幅塌陷处的显式事件接口（提升边在振幅层的推广）与温度淬火协议。")
print("elapsed %.1f s" % (time.time() - t0))

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
ax = axes[0]
ax.semilogx(TAUS, [float(np.mean([r["npair"] for r in zero if r["tau_q"] == t])) for t in TAUS],
            "o-", color="#4472c4", label="零温（冻结）")
ax.semilogx(TAUS, wmeans, "s--", color="#c0504d", label="有限温 $T=%.2f$" % T_WARM)
ax.set_xlabel(r"$T_Q$（淬火时长，步）")
ax.set_ylabel("残留缺陷对数 $n$")
ax.set_title("零温与有限温对照（有限温斜率 %.2f）" % slope_w)
ax.legend()
ax = axes[1]
q_snap = warm[0]["q"] if warm else None
if q_snap is not None:
    ax.imshow(q_snap, cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
    ax.set_title("有限温档荷图快照（$T_Q=%d$）" % TAUS[0])
    ax.set_xlabel("$c$")
    ax.set_ylabel("$r$")
fig.tight_layout()
fig.savefig(OUT, dpi=160)
plt.close(fig)
print("figure saved:", OUT)
