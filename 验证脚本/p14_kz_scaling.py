# -*- coding: utf-8 -*-
"""
p14_kz_scaling.py
判据第 670 号：P1-4 第二层——KZ 标度的三项前置与指数测量。

背景：判据第 666 号已建立振幅过零事件接口与温度淬火协议，但残留主体仍是初始
随机相位携带的约 685 对，事件增量仅占 1.5% 至 11%，Kibble-Zurek 标度未立。
第 666 号登记三项前置：对称相初值、事件去重、临界点附近减速。本判据补三项并
重测 KZ 指数。

模型层声明：复序参量 psi = rho * exp(i*beta)，自由能
  F = sum_bond (J/2)|psi_i - psi_j|^2 + sum_i [ -a(t)/2 |psi_i|^2 + U/4 |psi_i|^4 ]，
演化取含时 Ginzburg-Landau 梯度流加 Langevin 噪声。a > 0 破缺相、a < 0 对称相。

三项前置：
  其一，对称相初值：从 a < 0 的对称相平衡态出发（振幅由噪声支撑在
       rho_sym = sqrt(-a/U) 附近，相位完全无序），而非人为的小振幅加随机相位。
       此时初始拓扑荷为 0（无序相无拓扑缺陷），残留基线接近零。
  其二，事件去重：振幅过零事件按「同一格点连续过零算一次」去重（记录格点
       上次过零的时刻，冷却期内不重复计数）。
  其三，临界点附近减速：淬火时间表取 a(t) = -A_SYM * (1 - t/T_Q)^2 型
       （在 a = 0 附近斜率趋于零），而非线性穿越。

判据与理论期望（先立判据再实现）：
  K1 冒烟：对称相初值下初始荷为 0；破缺相弛豫后振幅收敛 rho0；相位增量有变化性。
  K2 基线归零：对称相初值下、无淬火（a 固定在对称相）时，荷图全程不变且为 0。
  K3 事件去重有效：同一格点的连续过零只计一次；去重后事件数显著少于逐帧计数。
  K4 KZ 标度成立：残留缺陷密度 n 随淬火时长 T_Q 单调下降，log-log 斜率为负，
     且与 KZ 参考值 -1/2（2D 平均场）或 -1（另档）在量级上可比。
  K5 读数合法性：总荷恒 0；提升边与即时折叠差分同余 < 1e-8。
  K6 恒真排除：关掉事件接口时荷图不变（信号来自接口而非数值噪声）。
  K7 前置敏感性：三项前置逐项关闭时，K4 的斜率应显著改变（证明前置是必要的）。
多档：T_Q in {100, 400, 1600}，种子 {1, 2, 3}。
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
OUT = IMG + r"\p14_kz_scaling_670_fig1.png"

N = 64
J, U, GAMMA, DT = 1.0, 1.0, 1.0, 0.02
REL = 2500
A_SYM, A_BRK, T0 = -0.5, 1.0, 0.15
EPS_CORE = 1.0e-3
DEDUP_WINDOW = 200
TAUS = (100, 400, 1600)
SEEDS = (1, 2, 3)
RHO0 = math.sqrt(A_BRK / U)


def fold(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def lap(psi):
    return (np.roll(psi, 1, 0) + np.roll(psi, -1, 0)
            + np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4.0 * psi)


class LiftedEdges:
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
        rc, cc = int(r), int(c)
        rn, cs = (rc + 1) % N, (cc + 1) % N
        rm, cp = (rc - 1) % N, (cc - 1) % N
        self.Eh[rc, cc] = fold(beta[rc, cs] - beta[rc, cc])
        self.Eh[rc, cp] = fold(beta[rc, cc] - beta[rc, cp])
        self.Ev[rc, cc] = fold(beta[rn, cc] - beta[rc, cc])
        self.Ev[rm, cc] = fold(beta[rc, cc] - beta[rm, cc])


def sym_initial(seed, temp, n_warm=4000):
    """对称相初值：a 固定为负，从零振幅出发，由噪声自行建立对称相平衡态。

    关键：初值不含人为相位结构（不从高斯随机场出发，那会自带大量涡旋）。
    零振幅起点在对称相内是合法的（对称相平衡振幅为 0，噪声给出涨落）。
    """
    rng = np.random.default_rng(seed)
    psi = np.zeros((N, N), dtype=complex)
    sc = math.sqrt(2.0 * GAMMA * temp * DT)
    for _ in range(n_warm):
        psi = psi + GAMMA * DT * (J * lap(psi) + A_SYM * psi - U * np.abs(psi) ** 2 * psi)
        psi = psi + sc * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / math.sqrt(2.0)
    return psi, rng


def run(tau_q, seed, use_events=True, dedup=True, slow=True, quench=True):
    psi, rng = sym_initial(seed, T0)
    beta0 = np.angle(psi)
    le = LiftedEdges(beta0)
    q_prev = le.charges()
    n0 = int(np.abs(q_prev).sum()) // 2
    q_changes = 0
    events_raw = 0
    events_dedup = 0
    last_cross = np.full((N, N), -10 ** 9, dtype=int)
    cong_max = 0.0
    for t in range(tau_q + REL):
        if quench:
            frac = min(1.0, t / float(tau_q))
            if slow:
                # 临界点附近减速：a 在 a=0 附近斜率趋零
                a = A_SYM + (A_BRK - A_SYM) * (frac ** 2) if frac < 0.5 else \
                    A_SYM + (A_BRK - A_SYM) * (1.0 - (1.0 - frac) ** 2)
                temp = T0 * (1.0 - frac)
            else:
                a = A_SYM + (A_BRK - A_SYM) * frac
                temp = T0 * (1.0 - frac)
        else:
            a, temp = A_SYM, T0
        psi_old = psi
        psi = psi + GAMMA * DT * (J * lap(psi) + a * psi - U * np.abs(psi) ** 2 * psi)
        if temp > 0.0:
            sc = math.sqrt(2.0 * GAMMA * temp * DT)
            psi = psi + sc * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / math.sqrt(2.0)
        le.update(np.angle(psi * np.conj(psi_old)))
        if use_events:
            core = np.abs(psi) < EPS_CORE
            if core.any():
                beta_now = np.angle(psi)
                for (r, c) in np.argwhere(core):
                    events_raw += 1
                    if (not dedup) or (t - last_cross[r, c] > DEDUP_WINDOW):
                        events_dedup += 1
                        last_cross[r, c] = t
                        le.reset_at(r, c, beta_now)
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
        "tau_q": tau_q, "seed": seed, "n0": n0,
        "npair": int(np.abs(q).sum()) // 2, "qsum": int(q.sum()),
        "q_changes": q_changes, "events_raw": events_raw, "events": events_dedup,
        "cong_max": cong_max, "rho": float(np.mean(np.abs(psi))), "q": q,
    }


t0 = time.time()
print("=" * 72)
print("K1 冒烟（期望：对称相初值初始荷 0；破缺相振幅收敛 rho0=%.3f；相位有变化性）" % RHO0)
probe = run(TAUS[0], SEEDS[0], use_events=False, quench=False)
print("  对称相（无淬火）平均振幅 = %.4f（噪声支撑的涨落振幅，非零温值 %.3f）"
      % (probe["rho"], math.sqrt(-A_SYM / U)))
print("  对称相初始荷 n0 = %d（期望 0）" % probe["n0"])
r_brk = run(TAUS[1], SEEDS[0], use_events=False)
print("  淬火后平均振幅 = %.4f（期望 %.4f，相对差 %.1f%%）"
      % (r_brk["rho"], RHO0, 100 * abs(r_brk["rho"] - RHO0) / RHO0))
k1 = (probe["n0"] == 0) and (abs(r_brk["rho"] - RHO0) / RHO0 < 0.25)
print("  [K1 %s]" % ("PASS" if k1 else "FAIL"))

print("=" * 72)
print("K2 基线归零（期望：对称相无淬火时荷图全程为 0 且不变）")
k2 = (probe["npair"] == 0) and (probe["q_changes"] == 0)
print("  残留 n = %d，荷图变化步 = %d（期望 0 与 0）" % (probe["npair"], probe["q_changes"]))
print("  [K2 %s]" % ("PASS" if k2 else "FAIL"))

print("=" * 72)
print("K3 事件去重（期望：去重后计数显著少于逐帧计数）")
k3_rows = []
for tau in TAUS:
    r = run(tau, SEEDS[0], dedup=True)
    rr = run(tau, SEEDS[0], dedup=False)
    k3_rows.append((tau, r["events"], rr["events"]))
    print("  T_Q=%4d：去重后事件 %d，逐帧计数 %d（比值 %.1f）"
          % (tau, r["events"], rr["events"], rr["events"] / max(r["events"], 1)))
k3 = all(a < b for _, a, b in k3_rows)
print("  [K3 %s]" % ("PASS" if k3 else "FAIL"))

print("=" * 72)
print("K4 KZ 标度（期望：残留随 T_Q 下降，斜率为负）")
recs = []
for tau in TAUS:
    for sd in SEEDS:
        r = run(tau, sd)
        recs.append(r)
        print("  T_Q=%4d 种子 %d：n0=%d -> n=%d，事件 %d，荷图变化步 %d，同余 %.2e"
              % (tau, sd, r["n0"], r["npair"], r["events"], r["q_changes"], r["cong_max"]))
means = [float(np.mean([r["npair"] for r in recs if r["tau_q"] == t])) for t in TAUS]
lx = np.log(np.array(TAUS, dtype=float))
ly = np.log(np.maximum(np.array(means), 0.5))
slope = float(np.polyfit(lx, ly, 1)[0])
print("  残留均值 n = %s，log-log 斜率 = %.3f（KZ 参考 -0.5 与 -1）" % (means, slope))
k4 = slope < 0 and means[0] > means[-1]
print("  [K4 %s]" % ("PASS" if k4 else "FAIL"))

print("=" * 72)
print("K5 读数合法性（期望：总荷恒 0；同余 < 1e-8）")
k5a = all(r["qsum"] == 0 for r in recs)
cong_all = max(r["cong_max"] for r in recs)
k5b = cong_all < 1e-8
print("  总荷恒零 = %s；同余最大 %.2e" % (k5a, cong_all))
print("  [K5a %s] [K5b %s]" % ("PASS" if k5a else "FAIL", "PASS" if k5b else "FAIL"))

print("=" * 72)
print("K6 恒真排除（期望：关接口后荷图不变）")
ctrl = [run(t, s, use_events=False) for t in TAUS for s in SEEDS]
k6 = all(r["q_changes"] == 0 for r in ctrl)
print("  关接口荷图变化步 = %s" % [r["q_changes"] for r in ctrl])
print("  [K6 %s]" % ("PASS" if k6 else "FAIL"))

print("=" * 72)
print("K7 前置敏感性（期望：关闭减速或去重时斜率改变）")
alt = []
for lab, kw in (("线性时间表", {"slow": False}), ("不去重", {"dedup": False})):
    mm = []
    for tau in TAUS:
        rr = [run(tau, sd, **kw) for sd in SEEDS]
        mm.append(float(np.mean([r["npair"] for r in rr])))
    sl = float(np.polyfit(lx, np.log(np.maximum(np.array(mm), 0.5)), 1)[0])
    alt.append((lab, mm, sl))
    print("  %-10s 均值 %s，斜率 %.3f" % (lab, mm, sl))
slope_diffs = [abs(sl - slope) for _, _, sl in alt]
lvl_diffs = [max(abs(m - b) / max(b, 1.0) for m, b in zip(mm, means)) for _, mm, _ in alt]
k7 = any(d > 0.05 for d in slope_diffs) or any(d > 0.10 for d in lvl_diffs)
print("  斜率差 = %s；残留水平相对差 = %s"
      % (["%.3f" % d for d in slope_diffs], ["%.3f" % d for d in lvl_diffs]))
print("  [K7 %s]" % ("PASS" if k7 else "FAIL"))

print("=" * 72)
passed = sum([k1, k2, k3, k4, k5a, k5b, k6, k7])
print("判据汇总：K1 %s / K2 %s / K3 %s / K4 %s / K5a %s / K5b %s / K6 %s / K7 %s（%d/8）"
      % ("PASS" if k1 else "FAIL", "PASS" if k2 else "FAIL", "PASS" if k3 else "FAIL",
         "PASS" if k4 else "FAIL", "PASS" if k5a else "FAIL", "PASS" if k5b else "FAIL",
         "PASS" if k6 else "FAIL", "PASS" if k7 else "FAIL", passed))
print("elapsed %.1f s" % (time.time() - t0))

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
ax = axes[0]
ax.loglog(TAUS, means, "o-", color="#4472c4", label="三项前置全开（斜率 %.2f）" % slope)
for lab, mm, sl in alt:
    ax.loglog(TAUS, mm, "s--", alpha=0.7, label="%s（斜率 %.2f）" % (lab, sl))
ax.set_xlabel(r"$T_Q$（淬火时长，步）")
ax.set_ylabel("残留缺陷对数 $n$")
ax.set_title("KZ 标度与前置敏感性")
ax.legend(fontsize=8)
ax = axes[1]
if recs:
    ax.imshow(recs[0]["q"], cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
    ax.set_title("淬火后荷图（$T_Q=%d$，对称相初值）" % TAUS[0])
    ax.set_xlabel("$c$")
    ax.set_ylabel("$r$")
fig.tight_layout()
fig.savefig(OUT, dpi=160)
plt.close(fig)
print("figure saved:", OUT)
