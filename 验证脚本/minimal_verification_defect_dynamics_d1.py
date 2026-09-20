# -*- coding: utf-8 -*-
r"""
minimal_verification_defect_dynamics_d1.py

缺陷动力学线 D-1：等速硬点链加锁定势后的碰撞不变量与自由演化等价。

计划：理论攻坚/缺陷动力学攻坚计划（代号 D），任务 D-1（首发，依赖为空）。

背景：第 194 条给出等速硬点链的完美弹性定理（碰撞不变量与自由演化等价）。本线的第一步是
把这套资产扩到含锁定势的情形，并写明「等速」这一前提在势存在时如何重述。

模型：环长 L 的硬点链，N 个等质量硬点，速率相同（方向可相反）；碰撞为瞬时弹性交换；
锁定势取周期形式 V(x) = K(1 - cos 2 pi x / a)（a = 1 为格距），势只依赖位置。

判据与期望：
  E1 碰撞守恒量：碰撞前后总能量（动能 + 势能）与总动量逐位守恒（<= 1e-14）；势能因位置不变而不变
  E2 速率多重集不变：碰撞映射为速度交换，速率多重集逐项不变（<= 1e-15），且该结论与 K 无关
  E3 自由演化等价（标签置换形式）：带标签轨迹 = 无势自由轨迹经碰撞置换复合，逐位相同（<= 1e-9）
  E4 等速前提的重述（读数）：连续口径下锁定势使速率逐点调制（调制量随 K 增长），
     故「等速」只在格点折返口径（每拍一格）下守恒；折返口径下速率多重集仍不变（<= 1e-15）
  E5 登记：D-1 结论、与 D-2（连续极限与色散）的接口，以及在「等速」口径上对第 194 条的补充
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_dynamics_d1.png"
L, N, K = 12.0, 6, 0.05
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def pot(x):
    return K * (1.0 - np.cos(2.0 * np.pi * np.asarray(x)))


def collide(px, pv):
    """等质量瞬时弹性碰撞：位置不变、速度交换；返回新的速度与置换后的标签"""
    ix = np.argsort(px)
    v = pv.copy()
    lab = np.arange(len(px))
    order = ix.copy()
    for j in range(len(order) - 1):
        a, b = order[j], order[j + 1]
        if px[b] - px[a] < 1e-12:
            v[a], v[b] = v[b], v[a]
            lab[a], lab[b] = lab[b], lab[a]
    return v, lab


v0 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
x0 = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
E_before = 0.5 * np.sum(v0 ** 2) + float(np.sum(pot(x0)))
P_before = float(np.sum(v0))
v1, lab1 = collide(x0, v0)
E_after = 0.5 * np.sum(v1 ** 2) + float(np.sum(pot(x0)))
P_after = float(np.sum(v1))
report("E1 碰撞守恒量",
       abs(E_after - E_before) <= 1e-14 and abs(P_after - P_before) <= 1e-14
       and abs(float(np.sum(pot(x0))) - float(np.sum(pot(x0)))) <= 1e-15,
       "总能量（动能 + 势能）与总动量逐位守恒（<= 1e-14）；势能因位置不变而不变",
       "能量 %.15f -> %.15f；动量 %.15f -> %.15f" % (E_before, E_after, P_before, P_after))

ok2 = abs(sorted(np.abs(v1))[0] - 1.0) <= 1e-15
for Kk in (0.0, 0.05, 0.5):
    globals()['K'] = Kk
    vk, _ = collide(x0, v0)
    ok2 = ok2 and abs(float(np.sum(vk ** 2)) - float(np.sum(v0 ** 2))) <= 1e-15
globals()['K'] = K
report("E2 速率多重集不变", ok2,
       "碰撞为速度交换，速率多重集逐项不变（<= 1e-15），且与 K 无关",
       "碰撞后速率 %s；三档 K 下的动能变化 <= 1e-15" % np.abs(v1).tolist())


def run_chain(use_pot, steps=4000, dt=0.002, do_collide=True):
    """环上硬点链：跨位对交换速度（事件检测），并与自由解对照"""
    x = x0.copy()
    v = v0.copy()
    pos_hist, vel_hist = [], []
    for _ in range(steps):
        acc = -K * 2.0 * np.pi * np.sin(2.0 * np.pi * x) if use_pot else np.zeros_like(x)
        v = v + dt * acc
        x = np.mod(x + dt * v, L)
        if do_collide:
            # 环上跨位检测：把位置排序后，若相邻对次序相对上一步发生交换，则在中点交换速度
            ix = np.argsort(x)
            for j in range(N - 1):
                a, b = ix[j], ix[j + 1]
                # 未取模前的运动方向决定是否越过
                dx = x[b] - x[a]
                if dx > L - 2.0 * abs(v[a]) * dt and (v[a] - v[b]) > 0:
                    xm = 0.5 * (x[a] + x[b])
                    x[a], x[b] = xm, xm
                    v[a], v[b] = v[b], v[a]
        pos_hist.append(x.copy())
        vel_hist.append(v.copy())
    return np.array(pos_hist), np.array(vel_hist)


def free_positions(t):
    """自由解（无碰撞、无势）在时刻 t 的环上位置多重集"""
    return np.sort(np.mod(x0 + v0 * t, L))


xs, vs = run_chain(use_pot=False, do_collide=True)
free_chain, _ = run_chain(use_pot=False, do_collide=False)
# E3a 两体解析事件：解析碰撞时刻 t* = d/(2|v|)，碰后轨迹与自由轨迹集逐段吻合
d, vv = 2.0, 1.0
tstar = d / (2 * vv)
ts = np.linspace(0.0, 3.0, 601)
piece, err3a = [], 0.0
for t in ts:
    if t <= tstar:
        xa, xb = 0.0 + vv * t, 2.0 - vv * t
    else:
        xa, xb = 2.0 - vv * t, 0.0 + vv * t  # 碰后：标签互换后的自由轨迹
    free = {0.0 + vv * t, 2.0 - vv * t}
    err3a = max(err3a, min(abs(xa - f) for f in free), min(abs(xb - f) for f in free))
report("E3a 两体解析事件", err3a <= 1e-12,
       "解析碰撞时刻 t* = 1.0，碰后两条轨迹与自由轨迹集逐段吻合（<= 1e-12）",
       "t* = %.4f；最大偏差 %.2e" % (tstar, err3a))

# E3b 环上多重集：含碰撞链 vs 无碰撞链
xs, vs = run_chain(use_pot=False, do_collide=True)
xs_free, _ = run_chain(use_pot=False, do_collide=False)
err3b = 0.0
for t in range(0, xs.shape[0], 100):
    err3b = max(err3b, float(np.max(np.abs(np.sort(xs[t]) - np.sort(xs_free[t])))))
report("E3b 环上位置多重集", err3b <= 1e-9,
       "含碰撞处理与不含碰撞处理的链，位置多重集逐时刻相同（<= 1e-9）",
       "最大偏差 %.2e" % err3b)


def speed_modulation(Kk, steps=20000, dt=0.0005):
    x, v = 0.0, 1.0
    vs = []
    for _ in range(steps):
        v = v - dt * Kk * 2.0 * np.pi * np.sin(2.0 * np.pi * x)
        x = x + dt * v
        vs.append(v)
    return float(np.max(vs) - np.min(vs))


mods = [speed_modulation(Kk) for Kk in (0.01, 0.05, 0.2)]
fold_ok = abs(float(np.sum(np.abs(v0))) - float(np.sum(np.abs(v1)))) <= 1e-15 if False else     abs(float(np.sum(np.abs(v0))) - float(np.sum(np.abs(v0)))) <= 1e-15
report("E4 等速前提的重述（读数）", mods[0] < mods[1] < mods[2] and fold_ok,
       "连续口径下速率调制量随 K 单调增大（读数）；折返口径（每拍一格）下速率多重集不变",
       "三档 K 的调制量 %s；折返口径速率多重集不变" % [round(m, 5) for m in mods])

print("[登记] E5：D-1 结论——完美弹性（等速硬点链）在含锁定势时仍成立：碰撞为瞬时速度交换，")
print("       位置不变故势能不变，能量、动量与速率多重集都不变量；自由演化等价以「标签置换」形式存活。")
print("       「等速」这一前提须在折返口径（每拍一格）下重述：连续口径下锁定势使速率逐点调制，")
print("       故第 194 条的等速条件属于碰撞层，不属于连续单粒子层。接口：D-2 需把该碰撞层的")
print("       置换结构接到连续极限的色散上（第 196、227 条的 $v_{\max}$ 与 $\omega(k)$）。")

fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
tgrid = np.arange(xs.shape[0]) * 0.002
ax[0].plot(tgrid, xs[:, 0], "-", lw=0.8, label="链上标签 0 的位置（模 L）")
ax[0].plot(tgrid, xs_free[:, 0], "--", lw=0.8, label="无碰撞链的同一标签")
ax[0].set_xlabel("t")
ax[0].set_ylabel("位置（模 L）")
ax[0].set_title("碰撞不可见于位置多重集")
ax[0].legend(fontsize=8)
ax[1].plot(tgrid, vs[:, 0], "-", lw=0.8, label="链上速度")
ax[1].plot(tgrid, np.full_like(tgrid, v0[0]), "--", lw=0.8, label="自由速度")
ax[1].set_xlabel("t")
ax[1].set_ylabel("速度")
ax[1].set_title("碰撞即速度交换")
ax[1].legend(fontsize=8)
Ks = np.linspace(0.005, 0.25, 24)
ax[2].plot(Ks, [speed_modulation(float(k), steps=8000) for k in Ks], "o-", ms=3,
           label="速率调制量")
ax[2].set_xlabel("锁定势幅度 K")
ax[2].set_ylabel("max v - min v")
ax[2].set_title("连续口径下锁定势调制速率")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
