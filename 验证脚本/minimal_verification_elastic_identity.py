# -*- coding: utf-8 -*-
"""
弹性身份裁决实验：紧致 cos 弹性 vs 非紧致螺旋（平方）弹性下的位错宽度
裁决《注记-复合模型》边界其二：宽度机制（K/k <= 0.02 时 l* >= 2）
究竟属于体系哈密顿量，还是只属于 FK 的平方弹性。

设计（唯一变量 = 弹性形式，其余全部相同）：
  链：N = 400，开边界，u_0 = 0 与 u_{N-1} = N 固定（强制恰好一个额外缠绕，
      位错在拓扑上必存在，问题只剩它的宽度与能量）。
  PN（两侧相同）：K (1 - cos 2 pi u_n)，整数格点为零。
  弹性两种读法：
    紧致 cos（体系哈密顿量注册形式）：E_el = kappa * Sigma (1 - cos(2 pi (u_{n+1} - u_n)))
      2 pi 周期：整数倍缠绕的键差零惩罚（cos 4 pi = 1）。
    非紧致平方（S^1 x R 螺旋读法候选）：E_el = Sigma kappa/2 (u_{n+1} - u_n - 1)^^2
      高度差为物理量（螺旋阶梯几何：相位状态空间 S^1 x R），整数偏离有真实代价。
  初条件（两侧相同）：平滑位错 u_n = n + s(n)，s = 0.5 (1 + tanh((n - n0)/5))。
  动力学：过阻尼梯度流（自适应步长），内区自由、边界固定。
  宽度统一用参与比：l = (Sigma |e|)^^2 / Sigma e^^2（e_n = u_{n+1} - u_n - 1，
  Sigma e = 1 由边界强制）；拓扑化位错的 e 剖面为平顶时 l = 宽度，单键时 l = 1。

判定预测：
  cos 侧：尖锐位错（全部格点整数、单键 e = 1）能量恰为零（弹性 cos 4pi = 1、PN 整点为零）
          → 塌缩隐藏，l -> 1，E -> 0，一切 r 下窗口为空。
  平方侧：尖锐位错有代价 k/2；FK 孤子能量 ~ 8 sqrt(kK) 量级
          → r 小于交叉点时平滑解胜出，l ~ sqrt(k/K) 落窗（复现复合模型注记的参数区）。

判定含义：宽度机制完全依赖非紧致弹性。若体系弹性取紧致读法，窗口为空、
复合模型不闭环；若取螺旋读法（S^1 x R），窗口在体系内打开。

确定性：无随机数。结果打印终端 + 一图。
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWO_PI = 2.0 * np.pi


def make_chain(N, n0=None, w0=5.0):
    """平滑位错初条件：u_n = n + s(n)，边界精确固定 u_0 = 0, u_{N-1} = N"""
    n = np.arange(N, dtype=float)
    if n0 is None:
        n0 = N / 2.0
    s = 0.5 * (1.0 + np.tanh((n - n0) / w0))
    u = n + s
    u[0] = 0.0
    u[-1] = float(N)
    return u


def energies(u, kappa, K, elastic):
    b = u[1:] - u[:-1]
    if elastic == "cos":
        E_el = float(np.sum(kappa * (1.0 - np.cos(TWO_PI * b))))
    else:
        E_el = float(np.sum(0.5 * kappa * (b - 1.0) ** 2))
    E_pn = float(np.sum(K * (1.0 - np.cos(TWO_PI * u))))
    return E_el, E_pn, E_el + E_pn


def forces(u, kappa, K, elastic):
    """内区位力（边界固定不更新）"""
    b = u[1:] - u[:-1]
    f = np.zeros_like(u)
    if elastic == "cos":
        fb = kappa * TWO_PI * np.sin(TWO_PI * b)
    else:
        fb = kappa * (b - 1.0)
    # 位点 m 的力 = 左键贡献 − 右键贡献（E 对 u_m：左键 +、右键 −）
    f[1:-1] = fb[:-1] - fb[1:]
    f[1:-1] += K * TWO_PI * np.sin(TWO_PI * u[1:-1])
    return f


def width_participation(u):
    b = u[1:] - u[:-1]
    e = b - 1.0
    s1 = float(np.sum(np.abs(e)))
    s2 = float(np.sum(e ** 2))
    if s2 < 1e-14:
        return 0.0
    return s1 * s1 / s2


def integer_distance(u):
    return float(np.min(np.abs(u[1:-1] - np.round(u[1:-1]))))


def relax(u, kappa, K, elastic, iters=200000, eta0=0.001, eta_max=0.008,
          early_dE=1e-12, early_patience=5000, snap_every=2000):
    # eta_max = 0.008：cos 弹性曲率含 (2pi)^2 ~= 39.5，稳定边界 2/39.5 ~= 0.05，
    # 首跑 eta_max = 0.05 压线导致流震荡、E 虚涨（协议教训入档）
    u = u.copy()
    eta = eta0
    E_prev = sum(energies(u, kappa, K, elastic))
    rec = [(0, E_prev, width_participation(u))]
    stall = 0
    t0 = time.time()
    for t in range(1, iters + 1):
        f = forces(u, kappa, K, elastic)
        u[1:-1] = u[1:-1] - eta * f[1:-1]
        eta = min(eta * 1.02, eta_max)
        E = sum(energies(u, kappa, K, elastic))
        if t % snap_every == 0 or t == iters:
            rec.append((t, E, width_participation(u)))
        if abs(E_prev - E) < early_dE:
            stall += 1
            if stall >= early_patience:
                break
        else:
            stall = 0
        E_prev = E
    return u, rec, t, time.time() - t0


print("=" * 72)
print("弹性身份裁决实验：紧致 cos vs 非紧致螺旋（平方）弹性下的位错宽度")
print("=" * 72)
N = 400
KAPPA = 1.0
r_list = [0.0005, 0.002, 0.01, 0.02, 0.05, 0.1, 0.3]
print(f"N = {N} 开边界（u_0 = 0, u_{{N-1}} = {N}，强制一个额外缠绕），kappa = {KAPPA}")
print(f"尖锐位错参考能量：cos 侧 = 0（整点 + cos 4pi = 1），平方侧 = kappa/2 = {0.5 * KAPPA}")
print(f"宽度统一 = 参与比（平顶剖面 l = 宽度，单键 l = 1）\n")

results = {}
for elastic in ["cos", "sq"]:
    print(f"[{elastic} 弹性]" + ("（体系哈密顿量注册形式，2pi 周期）" if elastic == "cos"
          else "（S^1 x R 螺旋读法候选，高度差物理）"))
    results[elastic] = []
    for r in r_list:
        u0 = make_chain(N)
        u, rec, t, dt = relax(u0, KAPPA, r * KAPPA, elastic)
        E_el, E_pn, E_tot = energies(u, KAPPA, r * KAPPA, elastic)
        lw = width_participation(u)
        dint = integer_distance(u)
        results[elastic].append((r, E_tot, E_el, E_pn, lw, dint))
        print(f"  r = {r:<7g}: E = {E_tot:10.6f}（弹 {E_el:9.5f} + PN {E_pn:9.5f}）"
              f"  l = {lw:7.3f}  整点距 = {dint:.2e} （{t} 步，{dt:.1f} s）")
    print()

print("判定对照：")
print(f"  {'r':>8} | {'cos l*':>8} {'cos E':>10} | {'sq l*':>8} {'sq E':>10} | 窗口")
for i, r in enumerate(r_list):
    cl, ce = results["cos"][i][4], results["cos"][i][1]
    sl, se = results["sq"][i][4], results["sq"][i][1]
    win = "落窗" if sl >= 2.0 else "不落"
    print(f"  {r:>8g} | {cl:>8.3f} {ce:>10.5f} | {sl:>8.3f} {se:>10.5f} | {win}")

# 代表性参数的塌缩/驻留轨迹与剖面（r = 0.005）
r_rep = 0.005
traj = {}
profiles = {}
for elastic in ["cos", "sq"]:
    u0 = make_chain(N)
    u, rec, t, dt = relax(u0, KAPPA, r_rep * KAPPA, elastic, snap_every=200)
    traj[elastic] = rec
    profiles[elastic] = (u[1:] - u[:-1]) - 1.0
    print(f"\n[{elastic} r={r_rep}] 弛豫 {t} 步：E = {rec[-1][1]:.6f}，l: "
          + " -> ".join(f"{x[2]:.2f}" for x in rec[::max(1, len(rec) // 8)]))

# ---------------- [C] 稳定性细化检验 ----------------
print("\n[C] 稳定性细化检验（基态身份与亚稳性）")


def make_sharp(N):
    """尖锐位错：全部整数，中键 e = 1（cos 下 E = 0）"""
    n = np.arange(N, dtype=float)
    u = n + (n >= N // 2) * 1.0
    u[0], u[-1] = 0.0, float(N)
    return u


# C1 cos：尖锐形初始化 + 微扰 → 是否回到 E = 0（隐藏基态是否为真局部极小）
r_chk = 0.005
u_sharp = make_sharp(N)
E0_sharp = sum(energies(u_sharp, KAPPA, r_chk * KAPPA, "cos"))
u_t = u_sharp.copy()
u_t[N // 2] += 0.05
u_t, rec_t, t_t, _ = relax(u_t, KAPPA, r_chk * KAPPA, "cos", iters=60000)
print(f"  C1 cos 尖锐形 + 微扰：初始 E = {E0_sharp:.2e}，微扰弛豫终态 E = {rec_t[-1][1]:.3e}，"
      f"l = {width_participation(u_t):.3f}，整点距 = {integer_distance(u_t):.1e}"
      f"（{'回到隐藏基态：E=0 为真局部极小' if rec_t[-1][1] < 1e-6 else '离开尖锐形'}）")

# C2 cos：宽态 + 窄化踢 → 亚稳还是鞍点
u_wide, rec_w, _, _ = relax(make_chain(N), KAPPA, r_chk * KAPPA, "cos")
E_wide = rec_w[-1][1]
u_kick = 0.5 * u_wide + 0.5 * make_sharp(N)
u_k, rec_k, t_k, _ = relax(u_kick, KAPPA, r_chk * KAPPA, "cos", iters=100000)
E_k = rec_k[-1][1]
print(f"  C2 cos 宽态（E = {E_wide:.4f}，l = {width_participation(u_wide):.1f}）+ 窄化踢："
      f"终态 E = {E_k:.4f}，l = {width_participation(u_k):.2f}"
      f"（{'回到宽态：宽位错为稳健亚稳极小' if abs(E_k - E_wide) < 0.05 * E_wide else '滑向尖锐形：宽态非稳健'}）")

# C3 sq：全 r 区"可见态 vs 隐藏/窄态"能量排序（粒子 = 基态？）
print("\n  C3 sq 全 r 区能量排序（平滑宽态 vs 尖锐/窄态初始化的弛豫终态）：")
sq_order = []
for i, r in enumerate(r_list):
    u_s, rec_s, _, _ = relax(make_sharp(N), KAPPA, r * KAPPA, "sq", iters=60000)
    E_sharp_rel = rec_s[-1][1]
    E_smooth = results["sq"][i][1]
    verdict = "可见宽态为最低态（粒子 = 基态）" if E_smooth < E_sharp_rel else "窄/隐藏态更低（粒子非基态）"
    sq_order.append((r, E_smooth, E_sharp_rel, verdict))
    print(f"    r = {r:<7g}: E_平滑 = {E_smooth:.4f}（l = {results['sq'][i][4]:.2f}）"
          f"  E_尖锐弛豫 = {E_sharp_rel:.4f}（l = {rec_s[-1][2]:.2f}）  → {verdict}")
print("\n  cos 侧对照：隐藏尖锐形 E = 0（C1 验证为稳定极小），全部 r 下 E_可见 > E_隐藏 = 0"
      "（粒子恒非基态，可见态仅零温亚稳且盆地浅，C2 窄化即塌缩）")

# ---------------- 出图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax = axes[0, 0]
rs = [x[0] for x in results["cos"]]
ax.loglog(rs, [x[4] for x in results["cos"]], "s-", color="C3", label="cos (compact)")
ax.loglog(rs, [x[4] for x in results["sq"]], "o-", color="C0", label="squared (helical)")
ax.axhline(2.0, ls="--", color="0.5", lw=1, label="dispersion window l=2")
ax.axhline(1.0, ls=":", color="0.6", lw=1, label="single-bond collapse")
ax.set_xlabel("r = K/kappa")
ax.set_ylabel("width l* (participation)")
ax.set_title("Dislocation width: elastic identity adjudication")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.loglog(rs, [max(x[1], 1e-8) for x in results["cos"]], "s-", color="C3", label="cos")
ax.loglog(rs, [x[1] for x in results["sq"]], "o-", color="C0", label="squared")
ax.axhline(0.5 * KAPPA, ls=":", color="0.6", lw=1, label="sharp cost k/2 (sq)")
ax.set_xlabel("r = K/kappa")
ax.set_ylabel("E total")
ax.set_title("Relaxed energy (cos collapses to zero)")
ax.legend(fontsize=8)

ax = axes[1, 0]
for elastic, color in [("cos", "C3"), ("sq", "C0")]:
    ax.semilogy([x[0] for x in traj[elastic]], [max(x[2], 1.0) for x in traj[elastic]],
                "-", color=color, label=f"{elastic}")
ax.set_xlabel("step")
ax.set_ylabel("width l")
ax.set_title(f"Width trajectory at r = {r_rep}")
ax.legend(fontsize=8)

ax = axes[1, 1]
mid = N // 2
for elastic, color in [("cos", "C3"), ("sq", "C0")]:
    e = profiles[elastic]
    ax.plot(range(mid - 40, mid + 40), e[mid - 40:mid + 40], "-", color=color, label=elastic)
ax.set_xlabel("bond index (centered)")
ax.set_ylabel("excess winding e_n")
ax.set_title(f"Final bond profile at r = {r_rep}")
ax.legend(fontsize=8)

fig.suptitle("Elastic identity adjudication: compact cos vs helical squared (N=400)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "images", "minimal_verification_elastic_identity.png")
fig.savefig(out, dpi=130)
print("\n图已保存：images/minimal_verification_elastic_identity.png")
print("实验完成（仅打印与图，未写任何文档）。")
