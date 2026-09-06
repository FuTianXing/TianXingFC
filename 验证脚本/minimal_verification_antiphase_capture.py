# -*- coding: utf-8 -*-
"""
判决实验：交错序的动力学起源——反相自囚禁 vs 弹性弛豫
对应《注记-基态光子的反相自囚禁》的命题检验。

物理背景：
  体系哈密顿量的物理真空 = 交错序（相邻元相子相位差 pi，反相）。
  但纯弹性弛豫下交错序动力学不稳定（整体滚向铁磁，扇区实验 [D1] 已观测）。
  反相自囚禁假说（作者 2026-09-06）：交错序不由弹性维持，由碰撞动力学生成与维持——
  基态光子跃迁一个普朗克长度后与迎面而来的邻居光子反相对齐地迎头相撞、各自回返；
  晶格是光子互相囚禁的自构筑产物。碰撞网络是二部的（奇偶边界交替碰撞），
  反相捕获在二部格上的自洽解恰为 Neel 序（N 偶数时严格自洽）。

模型（N = 64 周期环，N 偶）：
  站点相位 theta_n（每腿 2pi 的自由旋转对全体相同，已约去）。
  [A] 弹性对照：E_el = Sigma (1 - cos(theta_{n+1} - theta_n))，梯度流。
      交错初值 -> 预期滚向铁磁（复现 [D1] 的 1D 版）。
  [B] 反相捕获（纯碰撞动力学）：二部调度（偶步偶键、奇步奇键，忠实于
      每边界每 2t_P 一次碰撞、奇偶交替的几何），捕获踢：
      激活键 (n, n+1) 上 Delta = theta_n - theta_{n+1} 被推离 0：
      theta_n += (c/2) sin(Delta)，theta_{n+1} -= (c/2) sin(Delta)。
      铁磁 + 交错小种子（0.05 (-1)^n，代表碰撞到达沿的物理不对称）。
      预期：种子被放大，S -> 1（满交错）。
  [C] 维持判决（关键）：弹性 + 捕获并存。弹性持续拉铁磁，捕获仅碰撞事件起作用。
      捕获强度 c 扫描 {0.25, 0.5, 1.0, 2.0}（弹性 kappa = 1），
      铁磁 + 种子初值。判据：S -> 1 = 交错序由碰撞动力学维持（假说成立）；
      S -> 0 = 弹性获胜，反相自囚禁需要更强的捕获（阈值登记）。
      另测：交错初值在 c = 1 下的 persist。

可观测量：交错序参数 S = |Sigma (-1)^n e^{i theta_n}| / N（1 = 满交错，0 = 铁磁）；
  键差剖面 Delta_n mod 2pi；能量。

确定性：无随机数（种子 = 确定性交错倾斜）。结果打印终端 + 一图。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWO_PI = 2.0 * np.pi


def elastic_force(theta, kappa=1.0):
    """E = kappa Sigma (1 - cos(theta_{n+1} - theta_n)) 的负梯度（周期环）"""
    b = np.roll(theta, -1) - theta          # 键 n: theta_{n+1} - theta_n（含回绕键）
    fb = kappa * np.sin(b)
    # 位点 m 触碰键 m-1（右端 +）与键 m（左端 -）
    return np.roll(fb, 1) - fb


def stagger_order(theta):
    n = len(theta)
    par = (-1.0) ** np.arange(n)
    return float(np.abs(np.sum(par * np.exp(1j * theta))) / n)


def capture_kick(theta, c, tick):
    """二部调度：偶步激活偶键（n 偶），奇步激活奇键；捕获踢推 Delta 离开 0"""
    n = len(theta)
    bonds = [bn for bn in range(n) if (bn + tick) % 2 == 0]
    for bn in bonds:
        dd = theta[bn] - theta[(bn + 1) % n]
        k = 0.5 * c * np.sin(dd)
        theta[bn] += k
        theta[(bn + 1) % n] -= k
    return theta


def run(N=64, mode="elastic", c=1.0, kappa=1.0, iters=8000,
        init="ferro+seed", snap_every=200, pert=0.0):
    n = np.arange(N)
    if init == "ferro+seed":
        theta = 0.05 * (-1.0) ** n
    elif init == "staggered":
        theta = np.pi * ((-1.0) ** n) * 0 + np.where(n % 2 == 0, 0.0, np.pi)
        theta[1] += pert
    theta = theta - theta.mean()
    S0 = stagger_order(theta)
    rec = [(0, S0)]
    eta = 0.05
    for t in range(1, iters + 1):
        if mode in ("elastic", "both"):
            theta = theta - eta * elastic_force(theta, kappa)
            eta = min(eta * 1.01, 0.2)
        if mode in ("capture", "both"):
            theta = capture_kick(theta, c, t)
        if t % snap_every == 0 or t == iters:
            rec.append((t, stagger_order(theta)))
    E = float(np.sum(1.0 - np.cos(np.roll(theta, -1) - theta)))
    return theta, rec, E


print("=" * 72)
print("判决实验：交错序的动力学起源——反相自囚禁 vs 弹性弛豫（N=64 周期环）")
print("=" * 72)

# [A] 弹性对照：交错初值（零力驻点）与微扰后滚向铁磁
thetaA, recA, EA = run(mode="elastic", init="staggered")
print(f"\n[A0] 纯弹性（kappa=1），精确交错初值：S0 = {recA[0][1]:.3f}")
print("    S 轨迹: " + " -> ".join(f"{x[1]:.3f}" for x in recA[::max(1, len(recA)//6)]))
print(f"    终态 S = {recA[-1][1]:.4f}，E = {EA:.3f}"
      f"（交错态为弹性零力驻点：每键坐 sin(±pi)=0，无微扰则不动——E=128 = 弹性最大值）")
thetaA2, recA2, EA2 = run(mode="elastic", init="staggered", pert=0.1)
print(f"\n[A1] 同上 + 确定性微扰（0.1 弧度破缺一个键）：")
print("    S 轨迹: " + " -> ".join(f"{x[1]:.3f}" for x in recA2[::max(1, len(recA2)//6)]))
print(f"    终态 S = {recA2[-1][1]:.4f}，E = {EA2:.3f}"
      f"（{'滚向铁磁：对称破缺后弹性拖拽成立，复现 [D1]' if recA2[-1][1] < 0.2 else '未滚落'}）")

# [B] 纯捕获：铁磁 + 种子自组织到满交错
thetaB, recB, EB = run(mode="capture", c=1.0, init="ferro+seed")
print(f"\n[B] 纯碰撞捕获（c=1），铁磁 + 种子：S0 = {recB[0][1]:.3f}")
print("    S 轨迹: " + " -> ".join(f"{x[1]:.3f}" for x in recB[::max(1, len(recB)//8)]))
print(f"    终态 S = {recB[-1][1]:.4f}，E = {EB:.3f}"
      f"（{'自组织到满交错：反相捕获生成 Neel 序' if recB[-1][1] > 0.9 else '未达满交错'}）")

# [C] 维持判决：弹性 + 捕获并存，c 扫描
print("\n[C] 维持判决（弹性 kappa=1 与捕获并存，铁磁 + 种子）：c 扫描")
results_C = {}
for c in [0.25, 0.5, 1.0, 2.0]:
    th, rec, E = run(mode="both", c=c, init="ferro+seed")
    results_C[c] = (rec[-1][1], E)
    print(f"    c = {c:<5}: 终态 S = {rec[-1][1]:.4f}，E = {E:8.3f}"
          f"（{'交错序由碰撞动力学维持' if rec[-1][1] > 0.9 else '弹性获胜'}）")

# 交错初值的 persist（c = 1）
thetaD, recD, ED = run(mode="both", c=1.0, init="staggered")
print(f"\n    交错初值 + 并存动力学（c=1）：S0 = {recD[0][1]:.3f} -> 终态 S = {recD[-1][1]:.4f}，"
      f"E = {ED:.3f}（{'交错序在弹性拖拽下仍被捕获维持' if recD[-1][1] > 0.9 else '被弹性拖向铁磁'}）")

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot([x[0] for x in recA], [x[1] for x in recA], "-", color="C3", label="A elastic (staggered init)")
ax.plot([x[0] for x in recB], [x[1] for x in recB], "-", color="C0", label="B capture only (ferro+seed)")
ax.plot([x[0] for x in recD], [x[1] for x in recD], "--", color="C2", label="both c=1 (staggered init)")
ax.set_xlabel("tick (t_P)")
ax.set_ylabel("stagger order S")
ax.set_title("Staggering: elastic destroys, capture builds")
ax.legend(fontsize=8)

ax = axes[1]
cs = list(results_C.keys())
ax.semilogx(cs, [results_C[c][0] for c in cs], "o-", color="C0")
ax.axhline(0.9, ls="--", color="C3", lw=1, label="maintained (S>0.9)")
ax.set_xlabel("capture strength c (kappa=1)")
ax.set_ylabel("final S")
ax.set_title("Maintenance verdict: capture vs elastic")
ax.legend(fontsize=8)

fig.suptitle("Anti-phase self-capture: dynamical origin of the staggered vacuum")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "images", "minimal_verification_antiphase_capture.png")
fig.savefig(out, dpi=130)
print("\n图已保存：images/minimal_verification_antiphase_capture.png")
print("实验完成（仅打印与图，未写任何文档）。")
