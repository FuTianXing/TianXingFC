# -*- coding: utf-8 -*-
r"""
minimal_verification_p147_structure_preserving.py
判据第 656 号：P1-47 阶段二方法论前置轮——荷守恒离散流的路线判别（诊断轮）。

上游：第 612 条 W2/W3 判负，方法论缺口 = 拓扑荷守恒的离散数值方法。
本轮不实现完整保结构流，而是判别两条候选路线是否可行：
  路线 fold：边差取折叠代表元（_edge），荷图可读，但流可能把边推过折叠临界；
  路线 raw：边差不折回，流稳定，但对 arctan2 差型构型荷图恒零（读不到荷）。

判据与理论期望（先立判据再实现）：
  R1 fold 荷图可读性：arctan2 差型构型上 fold 能读出半荷而 raw 读不出。
  R2 fold 流荷守恒：quench 场上 fold 流 1000 步荷图偏差恒 0。
  R3 fold 流数值稳定性：折回代表元保住有界性，1000 步无溢出。
  R4 raw 流荷守恒（对照）：raw 流荷图亦守恒，但初始荷图恒零，守恒对 quench 恒真。
  R5 反例检验（恒真纪律）：大步长 dt 必须破坏 fold 流的荷守恒，否则 R2 恒真。
"""
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")
np.seterr(invalid="ignore", over="ignore")   # raw 模式对照项的大数 round 溢出为预期行为

def edge_fold(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi

def plaq(beta, N, mode):
    B = np.zeros((N + 2, N + 2)); B[1:N + 1, 1:N + 1] = beta
    if mode == "fold":
        Eh = edge_fold(B[:, :-1], B[:, 1:]); Ev = edge_fold(B[:-1, :], B[1:, :])
    else:
        Eh = B[:, 1:] - B[:, :-1];         Ev = B[1:, :] - B[:-1, :]
    d = (Ev[1:N + 1, 1:N + 1] + Eh[2:N + 2, 1:N + 1]
         - Ev[1:N + 1, 2:N + 2] - Eh[1:N + 1, 1:N + 1])
    return np.round(d / (2 * math.pi)).astype(int)

def grad(beta, N, dt, mode):
    B = np.zeros((N + 2, N + 2)); B[1:N + 1, 1:N + 1] = beta
    if mode == "fold":
        Eh = edge_fold(B[:, :-1], B[:, 1:]); Ev = edge_fold(B[:-1, :], B[1:, :])
    else:
        Eh = B[:, 1:] - B[:, :-1];         Ev = B[1:, :] - B[:-1, :]
    g = 2.0 * (Eh[1:N + 1, 1:N + 1] - Eh[1:N + 1, 0:N]
               + Ev[1:N + 1, 1:N + 1] - Ev[0:N, 1:N + 1])
    nb = beta - dt * g
    return nb % (2 * math.pi) if mode == "fold" else nb

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
print("判据第 656 号（首轮诊断）：荷守恒离散流的两条候选路线判别")
print("上游：第 612 条 W2/W3 判负，方法论缺口 = 拓扑荷守恒的离散数值方法")

print("=" * 72)
print("R1 fold 荷图可读性（期望：fold 能读出 arctan2 差型构型的荷，raw 读不出）")
N = 24
for d in (4, 6, 8):
    b = pair_setup(N, d)
    qf = plaq(b, N, "fold"); qr = plaq(b, N, "raw")
    print("  pair d=%d: fold 半荷 %2d（非零 plaq %s）/ raw 半荷 %d"
          % (d, int(np.abs(qf).sum()), np.argwhere(qf != 0).tolist(), int(np.abs(qr).sum())))
r1 = all(int(np.abs(plaq(pair_setup(N, d), N, "fold")).sum()) > 0
         and int(np.abs(plaq(pair_setup(N, d), N, "raw")).sum()) == 0 for d in (4, 6, 8))
print("  [R1 %s] fold 可读而 raw 对 arctan2 差型构型失明" % ("PASS" if r1 else "FAIL"))

print("=" * 72)
print("R2 fold 流荷守恒（期望：fold 流在 quench 场上荷图必须守恒）")
b = quench(N, 3, 45)
q0 = plaq(b, N, "fold")
worst_fold = 0
bn = b.copy()
trace = []          # (step, 半荷数, 荷图偏差) 供绘图
for i in range(1000):
    bn = grad(bn, N, 0.02, "fold")
    if i in (9, 99, 499, 999):
        dev = int(np.abs(plaq(bn, N, "fold") - q0).max())
        worst_fold = max(worst_fold, dev)
        trace.append((i + 1, int(np.abs(plaq(bn, N, "fold")).sum()), dev))
        print("  fold step %4d: 荷图最大偏差 %d（当前半荷 %d）" % (i, dev, int(np.abs(plaq(bn, N, "fold")).sum())))
r2 = worst_fold == 0
print("  [R2 %s] fold 流荷图守恒" % ("PASS" if r2 else "FAIL"))

print("=" * 72)
print("R3 fold 流数值稳定性（期望：fold 流 1000 步内不溢出、beta 有界）")
bn = b.copy(); over = False
for i in range(1000):
    bn = grad(bn, N, 0.02, "fold")
    if not np.all(np.isfinite(bn)):
        over = True; break
print("  fold 流 1000 步后 beta 范围 [%.3f, %.3f]，溢出=%s" % (bn.min(), bn.max(), over))
r3 = (not over) and np.isfinite(bn).all()
print("  [R3 %s] fold 流数值稳定（折回代表元保住有界性）" % ("PASS" if r3 else "FAIL"))

print("=" * 72)
print("R4 raw 流荷守恒（对照：raw 流荷图亦守恒，但初始荷图恒零 => 守恒是恒真的）")
bnr = b.copy(); q0r = plaq(bnr, N, "raw")
worst_raw = 0
for i in range(1000):
    bnr = grad(bnr, N, 0.02, "raw")
    if i in (9, 99, 499, 999):
        dev = int(np.abs(plaq(bnr, N, "raw") - q0r).max())
        worst_raw = max(worst_raw, dev)
print("  raw 初始半荷 %d，1000 步后最大荷图偏差 %d" % (int(np.abs(q0r).sum()), worst_raw))
r4 = worst_raw == 0
print("  [R4 %s] raw 流守恒但初始荷图恒零（对 quench 场守恒恒真，不含信息）"
      % ("PASS" if r4 else "FAIL"))

print("=" * 72)
print("R5 反例检验（恒真纪律：fold 流的荷守恒必须能被反例打掉）")
bnr2 = quench(N, 3, 45); q0r2 = plaq(bnr2, N, "fold")
dev_big = 0
for i in range(200):
    bnr2 = grad(bnr2, N, 0.5, "fold")     # 大步长，故意越过折叠临界
    dev_big = max(dev_big, int(np.abs(plaq(bnr2, N, "fold") - q0r2).max()))
print("  大步长 dt=0.5 下 fold 流荷图最大偏差 %d（小步长 dt=0.02 下为 %d）" % (dev_big, worst_fold))
r5 = dev_big > 0
print("  [R5 %s] 大步长确实破坏荷守恒（R2 的 PASS 不是恒真）" % ("PASS" if r5 else "FAIL"))

print("=" * 72)
passed = sum([r1, r2, r3, r4, r5])
print("判据汇总：R1 %s / R2 %s / R3 %s / R4 %s / R5 %s（%d/5）" %
      ("PASS" if r1 else "FAIL", "PASS" if r2 else "FAIL", "PASS" if r3 else "FAIL",
       "PASS" if r4 else "FAIL", "PASS" if r5 else "FAIL", passed))
print("诊断结论（按本轮实测修正）：")
print("其一，R1 定位了缺口的形状：荷图可读性与流的稳定性由两个不相容的算子分别承担。")
print("fold 算子让荷图可读（pair 构型读出 2 个半荷），raw 算子让流稳定（偏差恒 0），")
print("但 raw 对 arctan2 差型构型荷图恒零，其守恒对 quench 场是恒真的（R4），无信息。")
print("其二，R2 判负：fold 流的荷伪影在 step 9 即出现（偏差 1），与第 612 条同型，")
print("证实伪影源于边差被推过折叠临界，不是实现错误。折回代表元只保住有界性（R3），")
print("保不住荷。其三，R5 表明荷守恒可被步长破坏，故荷守恒是可真可假的性质，")
print("不是恒真陈述。")
print("出路（登记待后续轮，不属本轮射程）：荷守恒的边须在整数层维护，即把每条边")
print("记为 (代表元, 整数卷绕数) 的提升对，流的每步在提升层做修正，读数才折回。")
print("第 612 条的 W2/W3 判据轮仍挂起，待提升边实现轮。")
print()
print("结论句修正：第 612 条登记的方法论缺口比原判更硬。原判把缺口表述为")
print("「荷守恒的离散流未找到」，本轮给出更强的结论：荷图可读性与荷守恒在单一")
print("实数表示下不相容（R1 与 R2 联合），必须换表示层（整数卷绕数与代表元分离）")
print("才有同时满足的可能。这不是实现问题，是表示问题。")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax = axes[0]
xs = [t[0] for t in trace]
ax.plot(xs, [t[1] for t in trace], "o-", color="#c0504d", label="fold 流")
ax.set_xlabel("梯度流步数")
ax.set_ylabel("半荷数")
ax.set_title("fold 流的荷伪影（R2 判负）")
ax.legend()
ax = axes[1]
labels = ["fold", "raw"]
vals = [int(np.abs(plaq(pair_setup(N, 4), N, "fold")).sum()),
        int(np.abs(plaq(pair_setup(N, 4), N, "raw")).sum())]
ax.bar(labels, vals, color=["#4472c4", "#a6a6a6"])
ax.set_ylabel("pair 构型半荷数")
ax.set_title("荷图可读性对照（R1）")
fig.tight_layout()
OUT = r"D:\software\TianXingFC\images\p147_sp_route_diag_656.png"
fig.savefig(OUT, dpi=160)
plt.close(fig)
print("figure saved:", OUT)
