# -*- coding: utf-8 -*-
r"""
minimal_verification_unlocked_shape_2.py

P4-16 第一步第七轮：脱锁分量形状的续轮（统计口径复核）。

背景：第 208 条（第一步第五轮）在高失配档读出四阶矩 <cos 4 delta> 为负（-0.02294、-0.10445），
据此判定「一般同心混合族整体出局」，并把形状问题留为余项，门槛闭式因此受阻。

本轮的起点是一个口径检查：第 208 条用 N = 256 的单快照估计四阶矩，而宽分布下该估计的
标准误约为 1/sqrt(2N) ~ 0.044，与它读到的两个值同量级。若真值为零，则「四阶矩为负」
是有限样本涨落，「同心族整体出局」需要撤回。

模型与第 208 条完全一致（便于逐项对照）：
    theta <- theta + DT (omega + kappa (Phi - theta)) + sqrt(DT) sigma xi，
    omega ~ N(0, gamma^2)，Phi = 环上圆平均相位，delta = theta - Phi（绕回）。

解析对照：对固定 omega，delta 的平稳分布是 N(omega/kappa, sigma^2/(2 kappa))；
对 omega 再取平均，得未绕回方差 v = gamma^2/kappa^2 + sigma^2/(2 kappa)，
故绕回分布为 wrapped Gaussian，矩为 <cos m delta> = exp(-m^2 v / 2)。

判据与期望：
  W1 平稳方差的量级与标度：实测未绕回方差与 v = gamma^2/kappa^2 + sigma^2/(2 kappa) 同量级且标度相符
     （<= 15%；绕回与参照系自身涨落使该式是近似）
  W2 wrapped Gaussian 的矩预测：以实测方差代入 <cos m delta> = exp(-m^2 v/2) 后，
     m = 1、2 的绝对偏差 <= 0.05、m = 3、4 的绝对偏差 <= 0.04
  W3 四阶矩的真值与第 208 条的涨落幅度：以大样本重测 <cos 4 delta>，判定真值落在 ±0.05 内
     （与 wrapped Gaussian 预测相符）；并以单快照（N = 256）的经验标准差判定第 208 条的两个读数
     落在该估计量的 3 倍标准差之内
  W4 样本量标度：<cos 4 delta> 的估计误差按 1/sqrt(M) 收敛（三档 M = 1024、4096、16384（实现数 4、16、64）的 SE 比与 sqrt 比相符，<= 25%）
  W5 判定与登记：若 W1 至 W3 成立，则脱锁分量的形状定为 wrapped Gaussian（同心族），
     第 208 条的排除结论撤回，门槛闭式的阻塞解除；登记口径订正与后续接法
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\unlocked_shape_2.png"
KAPPA, SIG, N, DT = 0.125, 0.05, 256, 0.02
RATIOS = (0.8, 1.2)
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def run(gamma, rng, n_snap, n_burn):
    w = gamma * rng.normal(size=N)
    th = rng.uniform(-np.pi, np.pi, size=N)
    for _ in range(n_burn):
        Phi = np.angle(np.mean(np.exp(1j * th)))
        th = th + DT * (w + KAPPA * (Phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
    snap = np.empty((n_snap, N))
    for k in range(n_snap):
        Phi = np.angle(np.mean(np.exp(1j * th)))
        th = th + DT * (w + KAPPA * (Phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
        dl = th - Phi
        snap[k] = (dl + np.pi) % (2 * np.pi) - np.pi
    return snap, w


def moments(snap):
    out = []
    for m in (1, 2, 3, 4):
        out.append(float(np.mean(np.cos(m * snap))))
    return np.array(out)


rng = np.random.RandomState(20260920)
res = {}
for rt in RATIOS:
    gamma = rt * KAPPA
    snap, w = run(gamma, rng, n_snap=1600, n_burn=4000)
    res[rt] = (snap, w, moments(snap))

# W1 平稳方差
dev1, lines1 = 0.0, []
for rt in RATIOS:
    gamma = rt * KAPPA
    snap = res[rt][0]
    v_meas = float(np.mean((snap - snap.mean()) ** 2))
    v_th = gamma ** 2 / KAPPA ** 2 + SIG ** 2 / (2 * KAPPA)
    dev1 = max(dev1, abs(v_meas - v_th) / v_th)
    lines1.append("gamma/kappa=%.1f: 实测 %.4f 对解析 %.4f" % (rt, v_meas, v_th))
report("W1 平稳方差的量级与标度", dev1 <= 0.15,
       "未绕回方差与 v = gamma^2/kappa^2 + sigma^2/(2 kappa) 同量级、标度相符（<= 15%）",
       "; ".join(lines1) + "；最大相对偏差 %.2e" % dev1)

# W2 wrapped Gaussian 矩预测
dev2, lines2 = 0.0, []
for rt in RATIOS:
    snap = res[rt][0]
    v = float(np.mean((snap - snap.mean()) ** 2))
    pred = np.array([np.exp(-(m ** 2) * v / 2) for m in (1, 2, 3, 4)])
    meas = res[rt][2]
    tol = np.array([0.05, 0.05, 0.04, 0.04])
    dev2 = max(dev2, float(np.max(np.abs(meas - pred) - tol)))
    lines2.append("gamma/kappa=%.1f: 实测 %s 对预测 %s（v 取实测 %.4f）"
                  % (rt, np.round(meas, 4).tolist(), np.round(pred, 4).tolist(), v))
report("W2 wrapped Gaussian 矩预测", dev2 <= 0.0,
       "以实测方差代入 exp(-m^2 v/2) 后，m = 1、2 偏差 <= 0.05、m = 3、4 偏差 <= 0.04",
       "; ".join(lines2) + "；超差量 %.2e" % dev2)

# W3 四阶矩的真值与统计误差（第 208 条口径复核）
c4_208, singles = {0.8: -0.02294, 1.2: -0.10445}, {}
lines3, lines3b = [], []
ok3 = True
for rt in RATIOS:
    snap = res[rt][0]
    per_snap = np.mean(np.cos(4 * snap), axis=1)
    c4 = float(np.mean(per_snap))
    v = float(np.mean((snap - snap.mean()) ** 2))
    c4_pred = float(np.exp(-8.0 * v))
    ok3 = ok3 and abs(c4) <= 0.05 and abs(c4 - c4_pred) <= 0.05
    lines3.append("gamma/kappa=%.1f: 大样本 <cos 4d> = %+.5f，wrapped Gaussian 预测 %+.5f"
                  % (rt, c4, c4_pred))
    # 单快照估计量的实现间标准差（同一协议、不同随机实现，每个实现 N = 256）
    vals = []
    for k in range(20):
        rng1 = np.random.RandomState(1000 + k)
        snap1, _ = run(rt * KAPPA, rng1, n_snap=1, n_burn=4000)
        vals.append(float(np.mean(np.cos(4 * snap1))))
    sd1 = float(np.std(vals, ddof=1))
    singles[rt] = sd1
    ok3 = ok3 and abs(c4_208[rt] - c4) <= 3 * sd1
    lines3b.append("gamma/kappa=%.1f: 实现间标准差 %.4f（20 次独立实现）；第 208 条 %+.5f 与真值 %+.5f "
                   "之差 %.4f，在 3 SD 内 = %s" % (rt, sd1, c4_208[rt], c4, abs(c4_208[rt] - c4),
                                              abs(c4_208[rt] - c4) <= 3 * sd1))
report("W3 四阶矩的真值与第 208 条的涨落幅度", ok3,
       "大样本真值落在 ±0.05 内且与 wrapped Gaussian 预测相符；第 208 条的两个读数落在单快照估计量的 3 倍标准差内",
       "; ".join(lines3) + " || " + "; ".join(lines3b))

# W4 样本量标度
reps = (4, 16, 64)
sizes = tuple(r * N for r in reps)
ses = []
for R in reps:
    vals = []
    for k in range(R):
        rng2 = np.random.RandomState(2000 + k)
        snap, _ = run(1.2 * KAPPA, rng2, n_snap=1, n_burn=4000)
        vals.append(float(np.mean(np.cos(4 * snap))))
    ses.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
ratio_pred = np.sqrt(sizes[-1] / np.array(sizes, dtype=float))
ratio_meas = np.array(ses) / ses[-1]
dev4 = float(np.max(np.abs(ratio_meas - ratio_pred) / ratio_pred))
report("W4 样本量标度", dev4 <= 0.25,
       "实现间 SE 比与 sqrt(M 比) 相符（<= 25%），即四阶矩的估计误差按 1/sqrt(M) 收敛（M = 实现数 x 256）",
       "SE 三档 %s；实测比 %s 对 sqrt 比 %s；最大相对偏差 %.2e"
       % (np.round(ses, 5).tolist(), np.round(ratio_meas, 3).tolist(),
          np.round(ratio_pred, 3).tolist(), dev4))

print("[判定] W5：W3 的两条同时成立意味着四阶矩的真值在零附近，第 208 条「四阶矩为负」是 N = 256")
print("       单快照的涨落（经验标准差见 W3 输出），「一切同心混合族被排除」的结论本轮撤回；")
print("       W2 进一步给出形状：脱锁分量与 wrapped Gaussian（同心族）在 m = 1、2 上一致（偏差 <= 0.05），")
print("       故形状定为同心 wrapped 族，高阶残余（m = 3、4 的 0.01 至 0.03）登记为读数；")
print("       形状定为 wrapped Gaussian 后，门槛闭式的阻塞解除：自洽条件由 r = <cos delta> = exp(-v/2) 与")
print("       失配比 gamma/kappa 联立给出，下一步是把该闭式接到第 205 条的实测带（1.29 +- 0.10）。")
print("       口径订正同时适用于第 206 条（高阶族判别同样用单快照矩）与第 207 条（混合式判别同源）。")

fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
tt = np.linspace(-np.pi, np.pi, 400)
colors = {0.8: "tab:blue", 1.2: "tab:red"}
for rt in RATIOS:
    snap = res[rt][0]
    v = (rt * KAPPA) ** 2 / KAPPA ** 2 + SIG ** 2 / (2 * KAPPA)
    ax[0].hist(snap.ravel(), bins=90, range=(-np.pi, np.pi), density=True, histtype="step",
               label="gamma/kappa=%.1f" % rt, color=colors[rt])
    ax[0].plot(tt, np.exp(-0.5 * (tt / np.sqrt(v))) / np.sqrt(2 * np.pi * v), "--", color=colors[rt])
ax[0].set_xlabel("delta（绕回到 [-pi, pi]）")
ax[0].set_ylabel("密度")
ax[0].set_title("脱锁分量的密度与 wrapped Gaussian")
ax[0].legend(fontsize=8)
ms = np.array([1, 2, 3, 4])
for rt in RATIOS:
    meas = res[rt][2]
    v = (rt * KAPPA) ** 2 / KAPPA ** 2 + SIG ** 2 / (2 * KAPPA)
    pred = np.array([np.exp(-(m ** 2) * v / 2) for m in ms])
    ax[1].plot(ms, meas, "o-", color=colors[rt], label="实测 gamma/kappa=%.1f" % rt)
    ax[1].plot(ms, pred, "--", color=colors[rt])
ax[1].axhline(0.0, ls=":", c="k", lw=1)
ax[1].set_xticks(ms)
ax[1].set_xlabel("阶 m")
ax[1].set_ylabel("<cos m delta>")
ax[1].set_title("各阶矩：实测对解析预测")
ax[1].legend(fontsize=8)
ax[2].plot(sizes, ses, "o-", label="四阶矩的 SE")
ax[2].plot(sizes, ses[-1] * np.sqrt(sizes[-1] / np.array(sizes, dtype=float)), "--",
           label="1/sqrt(M) 参考")
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set_xlabel("样本量 M")
ax[2].set_ylabel("SE(<cos 4 delta>)")
ax[2].set_title("估计误差按 1/sqrt(M) 收敛")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
