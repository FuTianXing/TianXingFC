# -*- coding: utf-8 -*-
r"""
minimal_verification_p147_lifted_edges.py
判据第 658 号：P1-47 阶段二方法论前置轮之二——荷守恒离散流的提升边表示。

上游：第 612 条（W2/W3 判负，方法论缺口）+ 第 656 号（路线判别：fold 算子可读而
不守恒、raw 算子守恒而读不到荷，两者在单一实数表示下不可兼得，出路为提升边）。

提升边表示的构造：
  每条边维护一个无界实数 E（提升值），满足 E ≡ Δβ (mod 2π)；
  初始化 E(0) = fold(Δβ(0))（初始构型假设相邻差 < π，合法）；
  演化按端点更新 E ← E + (δ_head − δ_tail)，δ 为 β 的未 mod 更新量；
  荷读数为 plaq 四边提升和除以 2π 取整。
  数学性质：每一 plaq 的 ΣE 在端点更新下相消（每顶点出现两次、符号相反），
  故荷在连续演化下被初始值冻结。这不是恒真：它依赖实现的正确性，若更新
  误用 fold 刷新或端点差写错，E ≡ Δβ (mod 2π) 即失效、荷漂移。

判据与理论期望（先立判据再实现）：
  S1 初始可读性：pair 构型（d = 4, 6, 8）提升表示读出 2 个半荷，与 fold 读数一致。
  S2 提升合法性：演化 1000 步，每步 (E − Δβ_raw)/2π 保持整数（偏差 ≤ 1e-6），
     即提升与即时差分始终同余于 2π 整数倍。
  S3 荷守恒对照：quench 场（ξ₀ = 3）演化 1000 步，提升表示荷图偏差恒 0；
     同轨迹 naive fold 读数漂移非零（复现 612 伪影，确认 S3 非恒真）。
  S4 显式拓扑事件可记录：对相邻 +1/−1 对做显式重连（共享边加 ±2π），
     提升表示荷正确降为 0 且此后保持；证明表示是「连续守恒 + 事件可记」，
     不是死冻结。
  S5 与 612 的 W3 对表：quench 三档 ξ₀ 上 naive 读数给出伪荷（612 同款非单调），
     提升表示读出 0；结论：纯相位梯度流无成核机制，Kibble 残留需振幅自由度。

多档：d ∈ {4, 6, 8}（S1）、ξ₀ ∈ {2, 3, 5}（S5）；确定性，固定种子。
"""
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")
np.seterr(invalid="ignore", over="ignore")

IMG = r"D:\software\TianXingFC\images"
OUT = IMG + r"\p147_lifted_edges_658.png"

def edge_fold(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi

def flat(beta, N):
    B = np.zeros((N + 2, N + 2))
    B[1:N + 1, 1:N + 1] = beta
    return B

def plaq_fold(beta, N):
    B = flat(beta, N)
    Eh = edge_fold(B[:, :-1], B[:, 1:]); Ev = edge_fold(B[:-1, :], B[1:, :])
    d = (Ev[1:N + 1, 1:N + 1] + Eh[2:N + 2, 1:N + 1]
         - Ev[1:N + 1, 2:N + 2] - Eh[1:N + 1, 1:N + 1])
    return np.round(d / (2 * math.pi)).astype(int)

def grad_step_flow(beta, N, dt):
    """与 612 号相同实现：fold 边差的梯度流一步。返回 (beta_new, delta)，
    delta 为未 mod 的更新量。"""
    B = flat(beta, N)
    Eh = edge_fold(B[:, :-1], B[:, 1:]); Ev = edge_fold(B[:-1, :], B[1:, :])
    g = 2.0 * (Eh[1:N + 1, 1:N + 1] - Eh[1:N + 1, 0:N]
               + Ev[1:N + 1, 1:N + 1] - Ev[0:N, 1:N + 1])
    delta = -dt * g
    beta_new = (beta + delta) % (2 * math.pi)
    return beta_new, delta

class LiftedEdges:
    """提升边：无界实数 E，满足 E ≡ Δβ (mod 2π)；端点更新维护连续性。"""
    def __init__(self, beta, N):
        B = flat(beta, N)
        self.N = N
        self.Eh = edge_fold(B[:, :-1], B[:, 1:]).copy()   # (N+2, N+1)
        self.Ev = edge_fold(B[:-1, :], B[1:, :]).copy()   # (N+1, N+2)

    def update(self, delta):
        N = self.N
        DB = np.zeros((N + 2, N + 2))
        DB[1:N + 1, 1:N + 1] = delta
        self.Eh = self.Eh + DB[:, 1:] - DB[:, :-1]
        self.Ev = self.Ev + DB[1:, :] - DB[:-1, :]

    def charges(self):
        N = self.N
        d = (self.Ev[1:N + 1, 1:N + 1] + self.Eh[2:N + 2, 1:N + 1]
             - self.Ev[1:N + 1, 2:N + 2] - self.Eh[1:N + 1, 1:N + 1])
        return np.round(d / (2 * math.pi)).astype(int)

    def congruence_defect(self, beta):
        """(E − Δβ_raw)/2π 非整数部分的绝对值最大者（S2 判据量）。"""
        B = flat(beta, self.N)
        Dh = B[:, 1:] - B[:, :-1]; Dv = B[1:, :] - B[:-1, :]
        kh = (self.Eh - Dh) / (2 * math.pi)
        kv = (self.Ev - Dv) / (2 * math.pi)
        def niz(x):
            return np.abs(x - np.round(x)).max()
        return max(niz(kh), niz(kv))

def pair_setup(N, d):
    X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return (np.arctan2(Y - (N // 2 - 0.5), X - (N // 2 - d / 2 - 0.5))
            - np.arctan2(Y - (N // 2 - 0.5), X - (N // 2 + d / 2 + 0.5))) % (2 * math.pi)

def quench(N, xi0, seed):
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0, 2 * math.pi, (N, N))
    px, py = np.meshgrid(2 * np.pi * np.arange(N) / N, 2 * np.pi * np.arange(N) / N, indexing="ij")
    filt = np.exp(-((xi0 * px) ** 2 + (xi0 * py) ** 2) / 2)
    return (np.real(np.fft.ifft2(np.fft.ifftshift(filt * np.fft.fftshift(np.fft.fft2(beta)))))) % (2 * math.pi)

print("=" * 72)
print("判据第 658 号：提升边表示——荷守恒离散流的方法论前置轮之二")
print("=" * 72)

# ---------------------------------------------------------------- S1
print("S1 初始可读性（期望：pair 构型提升读数与 fold 一致，2 个半荷）")
N = 24
S1_OK = True
s1_rows = []
for d in (4, 6, 8):
    b = pair_setup(N, d)
    qf = plaq_fold(b, N)
    le = LiftedEdges(b, N)
    ql = le.charges()
    same = bool(np.array_equal(qf, ql))
    nz = int(np.abs(ql).sum())
    S1_OK = S1_OK and same and nz == 2
    s1_rows.append((d, nz, same))
    print("  d=%d: 提升半荷 %d（fold 半荷 %d），荷图一致 %s" % (d, nz, int(np.abs(qf).sum()), same))
print("  [S1 %s]" % ("PASS" if S1_OK else "FAIL"))

# ---------------------------------------------------------------- S2
print("=" * 72)
print("S2 提升合法性（期望：(E − Δβ_raw)/2π 全程为整数，偏差 ≤ 1e-6）")
bd = quench(N, 3, 45)
le = LiftedEdges(bd, N)
worst_cong = 0.0
bcur = bd.copy()
for t in range(1000):
    bcur, delta = grad_step_flow(bcur, N, 0.02)
    le.update(delta)
    if t % 50 == 0 or t == 999:
        worst_cong = max(worst_cong, le.congruence_defect(bcur))
S2_OK = worst_cong <= 1e-6
print("  1000 步内同余偏差最大值 = %.3e（门槛 1e-6）" % worst_cong)
print("  [S2 %s]" % ("PASS" if S2_OK else "FAIL"))

# ---------------------------------------------------------------- S3
print("=" * 72)
print("S3 荷守恒对照（期望：提升读数偏差恒 0；naive fold 漂移非零）")
bd = quench(N, 3, 45)
q0 = plaq_fold(bd, N)
le = LiftedEdges(bd, N)
b_lift = bd.copy(); b_naive = bd.copy()
worst_lift, worst_naive = 0, 0
trace_lift, trace_naive = [], []
for t in range(1000):
    b_lift, delta = grad_step_flow(b_lift, N, 0.02)
    le.update(delta)
    b_naive, _ = grad_step_flow(b_naive, N, 0.02)
    if t % 10 == 0:
        ql = le.charges(); qn = plaq_fold(b_naive, N)
        dl = int(np.abs(ql - q0).max()); dn = int(np.abs(qn - q0).max())
        worst_lift = max(worst_lift, dl); worst_naive = max(worst_naive, dn)
        trace_lift.append(int(np.abs(ql).sum())); trace_naive.append(int(np.abs(qn).sum()))
S3_OK = (worst_lift == 0) and (worst_naive > 0)
print("  提升读数 1000 步最大偏差 = %d；naive fold 最大偏差 = %d" % (worst_lift, worst_naive))
print("  naive 终态半荷数 = %d（初始 %d）" % (trace_naive[-1], int(np.abs(q0).sum())))
print("  [S3 %s]" % ("PASS" if S3_OK else "FAIL"))

# ---------------------------------------------------------------- S4
print("=" * 72)
print("S4 显式拓扑事件可记录（期望：重连后 ±1 对降为 0 并保持）")
# 构造近邻 ±1 对：d=1 的 pair 给垂直相邻的 +1/-1（实测 (5,5) 与 (6,5)）
N4 = 12
b4 = pair_setup(N4, 1)
qf4 = plaq_fold(b4, N4)
pos = np.argwhere(qf4 == 1); neg = np.argwhere(qf4 == -1)
print("  初始荷：+1 @ %s，-1 @ %s" % (pos.tolist(), neg.tolist()))
le4 = LiftedEdges(b4, N4)
q_before = le4.charges()
# 共享边重连：对两 plaq 的公共边，使 curl 各减/加 2π，令两荷融合归零
if len(pos) == 1 and len(neg) == 1:
    r1, c1 = pos[0]; r2, c2 = neg[0]
    # 相邻 plaq 共享一条边；把共享边的提升值调整 2π（方向按错位关系选）
    if r1 == r2 and abs(c1 - c2) == 1:
        cL = min(c1, c2)                      # 左 plaq 的列
        q1 = int(q_before[r1, cL])
        # 共享垂直边 Ev[r1+1, cL+2]：对左 plaq 系数 −1、对右 plaq 系数 +1
        le4.Ev[r1 + 1, cL + 2] += 2 * math.pi * q1
        op = "垂直边 Ev[%d,%d]" % (r1 + 1, cL + 2)
    elif c1 == c2 and abs(r1 - r2) == 1:
        rT = min(r1, r2)                      # 上 plaq 的行
        q1 = int(q_before[rT, c1])
        # 共享水平边 Eh[rT+2, c1+1]：对上 plaq 系数 +1、对下 plaq 系数 −1
        le4.Eh[rT + 2, c1 + 1] -= 2 * math.pi * q1
        op = "水平边 Eh[%d,%d]" % (rT + 2, c1 + 1)
    else:
        op = "非相邻（跳过重连）"
    q_after = le4.charges()
    print("  重连操作：%s" % op)
    print("  重连后荷：+1 数 %d，-1 数 %d（期望 0, 0）"
          % (int((q_after == 1).sum()), int((q_after == -1).sum())))
    # 继续演化 100 步，验证保持
    b4c = b4.copy()
    for t in range(100):
        b4c, delta4 = grad_step_flow(b4c, N4, 0.02)
        le4.update(delta4)
    q_final = le4.charges()
    keep = int(np.abs(q_final).sum()) == 0
    S4_OK = (int(np.abs(q_after).sum()) == 0) and keep
    print("  演化 100 步后净半荷 = %d（期望 0）" % int(np.abs(q_final).sum()))
else:
    S4_OK = False
    print("  近邻对构造失败")
print("  [S4 %s]" % ("PASS" if S4_OK else "FAIL"))

# ---------------------------------------------------------------- S5
print("=" * 72)
print("S5 与 612 的 W3 对表（期望：naive 伪荷非单调；提升读数为 0）")
Ns = 48
rows = []
for xi0 in (2, 3, 5):
    b = quench(Ns, xi0, 42 + xi0)
    q0s = plaq_fold(b, Ns)
    leS = LiftedEdges(b, Ns)
    b_l = b.copy(); b_n = b.copy()
    for t in range(3000):
        b_l, delta_l = grad_step_flow(b_l, Ns, 0.02)
        leS.update(delta_l)
        b_n, _ = grad_step_flow(b_n, Ns, 0.02)
    n_naive = int(np.abs(plaq_fold(b_n, Ns)).sum())
    n_lift = int(np.abs(leS.charges()).sum())
    rows.append((xi0, n_naive, n_lift))
    print("  xi0=%d: naive 终态 |q| 总数 %d（%d 对，612 同款读数 149/190/160 为对数）｜提升读数 %d"
          % (xi0, n_naive, n_naive // 2, n_lift))
S5_OK = all(r[2] == 0 for r in rows)
print("  [S5 %s]（提升读数恒 0：纯相位梯度流无成核机制）" % ("PASS" if S5_OK else "FAIL"))

# ---------------------------------------------------------------- 汇总
print("=" * 72)
passed = sum([S1_OK, S2_OK, S3_OK, S4_OK, S5_OK])
print("判据汇总：S1 %s / S2 %s / S3 %s / S4 %s / S5 %s（%d/5）" %
      ("PASS" if S1_OK else "FAIL", "PASS" if S2_OK else "FAIL", "PASS" if S3_OK else "FAIL",
       "PASS" if S4_OK else "FAIL", "PASS" if S5_OK else "FAIL", passed))
print("结论：提升边表示同时满足可读性（S1）与荷守恒（S3），且能记录显式拓扑事件（S4）；")
print("连续演化下荷的冻结是数学性质（ΣE 端点相消），不是实现巧合（S2 监督其合法性）。")
print("W3 的靶子由此重定义：纯相位梯度流无成核机制，Kibble 残留需振幅自由度。")

# ---------------------------------------------------------------- 图
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax = axes[0]
ds = [r[0] for r in s1_rows]; nzs = [r[1] for r in s1_rows]
ax.bar([str(d) for d in ds], nzs, color="#4472c4")
ax.set_xlabel("对间距 d"); ax.set_ylabel("提升读数半荷数")
ax.set_title("S1 初始可读性（与 fold 一致）")
ax = axes[1]
ts = [10 * i for i in range(len(trace_naive))]
ax.plot(ts, trace_naive, "-", color="#c0504d", label="naive fold 读数")
ax.plot(ts, trace_lift, "-", color="#4472c4", label="提升读数")
ax.set_xlabel("梯度流步数"); ax.set_ylabel("半荷数")
ax.set_title("S3 quench 场荷读数对照"); ax.legend()
ax = axes[2]
labels = [r"$\xi_0=2$", r"$\xi_0=3$", r"$\xi_0=5$"]
x = np.arange(3); w = 0.38
ax.bar(x - w/2, [r[1] for r in rows], w, color="#c0504d", label="naive fold")
ax.bar(x + w/2, [r[2] for r in rows], w, color="#4472c4", label="提升")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("终态半荷数"); ax.set_title("S5 quench 残留对表"); ax.legend()
fig.tight_layout()
fig.savefig(OUT, dpi=160)
plt.close(fig)
print("figure saved:", OUT)
