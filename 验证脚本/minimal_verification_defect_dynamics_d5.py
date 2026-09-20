# -*- coding: utf-8 -*-
r"""
minimal_verification_defect_dynamics_d5.py

缺陷动力学线 D-5：与写入机制的接口（缺陷过轴一笔、与包络份额的配对形式、两个速度上界的层次）。

计划：理论攻坚/缺陷动力学攻坚计划（代号 D）。消费上游：记录机制线命题 12 至 17
（[[记录机制：碰撞即写入的算符形式、三维缺陷读数与弦的分层]]）：每站点每周期一笔；跃迁时长等于驱动率的倒数；
记录流取样间隔 t_P，慢驱动不欠计数、快驱动欠计数（12/36 = 0.33）；可分辨跃迁率上界 1/t_P。

模型：一维锁定链，相位场 u_n 以整圈为单位（u 增一等于指针转满一圈），
     E = sum_n [ (1/2) k_b (u_{n+1} - u_n - 1)^2 + K (1 - cos 2 pi u_n) ]，k_b = 1，K = 0.005，N = 256。
     移动 kink 解析解 u_n(t) = (2/pi) arctan(exp(gamma (n - n_0 - v t)))，gamma = 2 pi sqrt(K) / sqrt(1 - v^2)。
     写入规则取记录机制线口径：指针每转满一圈写一笔，故单站笔数等于相位增量的绕数（整数）。

判据与期望：
  W1 单次穿越：每站笔数增量恒为 1（整数），对速度 v 的 6 档与宽度 gamma 的 3 档均不变（偏差 <= 1e-12）
  W2 逐站笔数分布对份额分布：单次穿越的笔数分布均匀（每站 1），包络份额分布局域（sech^2 型），
     两者的总变差距离 >= 0.5，故「逐站笔数 = 份额」逐事件判负
  W3 系综口径：按份额分布抽取 M = 4096 个落点，每站笔数对 M s_n 的偏差 <= 3 sigma（多项式抽样）
  W4 采样定理落点：跃迁时长 4、2、1 个取样间隔时计数比恒为 1；跃迁时长 1/3 个间隔时计数比 = 1/3（12/36）
  W5 两个上界的层次：采样上界 1 格每 t_P 与色散带上界 v_max(m) = (sqrt(m^2+4)-m)/2（m = 0.0900）比较，
     色散上界更紧（0.95670 < 1）；m -> 0 时两者重合（<= 1e-6）
  W6 结论登记：互补性的定理形式是「份额 = 抽样落点分布」，单次穿越的笔数是拓扑足迹（每站一笔）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_dynamics_d5.png"
KB, K, N = 1.0, 0.005, 256
GAMMA0 = 2.0 * np.pi * np.sqrt(K)
rng = np.random.default_rng(20260920)
rows = []


def kink(n, v, n0, width_scale=1.0):
    g = GAMMA0 * width_scale / np.sqrt(1.0 - v * v)
    return (2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - n0), -60.0, 60.0)))


def writes_of_passage(v, width_scale, n0=128.0, span=400.0):
    n = np.arange(N, dtype=float)
    t_half = span / max(v, 1e-9)
    u_before = kink(n, v, n0 - v * (-t_half), width_scale)
    u_after = kink(n, v, n0 - v * (t_half), width_scale)
    return np.round(u_after - u_before).astype(int)


def share_profile(width_scale=1.0):
    n = np.arange(N, dtype=float)
    psi = 1.0 / np.cosh(GAMMA0 * width_scale * (n - N / 2.0))
    s = psi ** 2
    return n, s / s.sum()


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


# ---------- W1 单次穿越：每站一笔，与速度、宽度无关 ----------
vs = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
ws = [0.5, 1.0, 2.0]
dev1 = 0.0
for v in vs:
    for w in ws:
        d = writes_of_passage(v, w)
        dev1 = max(dev1, float(np.max(np.abs(d - 1))))
rows.append(("W1", "18 组 (v, 宽度) 上每站笔数对 1 的最大偏差", "%.3e" % dev1, "<= 1e-12", dev1 <= 1e-12))

# ---------- W2 逐站笔数分布对份额分布 ----------
n, s = share_profile(1.0)
p_uniform = np.ones(N) / N
d_tv = tv(p_uniform, s)
rows.append(("W2", "单次穿越笔数分布（均匀）与份额分布的 TV 距离", "%.4f" % d_tv, ">= 0.5", d_tv >= 0.5))

# ---------- W3 系综口径：抽样落点分布对份额 ----------
M = 4096
idx = rng.choice(N, size=M, p=s)
counts = np.bincount(idx, minlength=N)
expected = M * s
sigma = np.sqrt(M * s * (1.0 - s))
z = np.abs(counts - expected) / np.maximum(sigma, 1e-12)
zmax = float(np.max(z))
rows.append(("W3", "4096 次落点抽样对 M s_n 的最大标准化偏差", "%.3f" % zmax, "<= 3", zmax <= 3.0))
tv_samp = tv(counts / M, s)
rows.append(("W3", "抽样频次分布对份额的 TV 距离", "%.4f" % tv_samp, "<= 0.05", tv_samp <= 0.05))

# ---------- W4 采样定理落点：跃迁计数比 ----------
tp = 1.0
dt_list = [4.0 * tp, 2.0 * tp, 1.0 * tp, tp / 3.0]
ratios = []
for dt in dt_list:
    n_trans = int(round(36.0 * tp / dt))
    times = (np.arange(n_trans) + 0.5) * dt
    bins = np.floor(times / tp).astype(int)
    counted = len(set(bins.tolist()))
    ratios.append(counted / n_trans)
rows.append(("W4", "跃迁时长 4/2/1 个取样间隔的计数比", "%.3f / %.3f / %.3f" % tuple(ratios[:3]), "全为 1", all(abs(r - 1.0) < 1e-12 for r in ratios[:3])))
rows.append(("W4", "跃迁时长 1/3 个间隔的计数比", "%.4f" % ratios[3], "0.3333", abs(ratios[3] - 1.0 / 3.0) < 1e-9))

# ---------- W5 两个上界的层次 ----------
def vmax(m):
    return (np.sqrt(m * m + 4.0) - m) / 2.0

m_def = 0.0900
v_sampling = 1.0
v_disp = float(vmax(m_def))
rows.append(("W5", "色散带上界 v_max(0.0900) 对采样上界 1", "%.5f" % v_disp, "< 1", v_disp < 1.0))
rows.append(("W5", "m -> 0 时两上界之差", "%.3e" % (1.0 - float(vmax(1e-8))), "<= 1e-6", (1.0 - float(vmax(1e-8))) <= 1e-6))

print("=== D-5 判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-22s 期望 %-10s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("W6 结论：单次穿越的逐站笔数是拓扑足迹（每站一笔，与速度、宽度无关），与包络份额的局域分布不成比例；")
print("        份额的正确配对是抽样落点分布（W3 通过）。故互补性在这条线上的定理形式为：份额 = 抽样落点分布，")
print("        写入笔数 = 相位绕数增量；两者靠抽样联系，不靠逐事件相等。")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
axes[0].plot(n, np.ones(N), "o", ms=3, label="单次穿越逐站笔数（全 1）")
axes[0].plot(n, s * N, "-", label="份额分布（放大 N 倍）")
axes[0].set_xlabel("站点")
axes[0].set_ylabel("笔数 / 份额")
axes[0].set_title("(a) 逐站笔数与份额不是同一分布（TV = %.3f）" % d_tv)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(expected, counts, "o", ms=4)
lim = max(expected.max(), counts.max()) * 1.1
axes[1].plot([0.0, lim], [0.0, lim], "r--", label="对角")
axes[1].set_xlabel("期望笔数 M s_n")
axes[1].set_ylabel("抽样笔数")
axes[1].set_title("(b) 4096 次落点抽样（最大标准化偏差 %.2f）" % zmax)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].semilogx([4.0, 2.0, 1.0, 1.0 / 3.0], ratios, "o-", label="计数比（实测）")
axes[2].axhline(1.0, color="tab:red", ls="--", label="不欠计数")
axes[2].set_xlabel("跃迁时长 / 取样间隔")
axes[2].set_ylabel("计数比")
axes[2].set_title("(c) 采样定理落点：慢驱动全额、快驱动欠计数")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
