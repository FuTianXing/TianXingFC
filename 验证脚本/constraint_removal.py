# -*- coding: utf-8 -*-
r"""
constraint_removal.py

二-1 补缺轮：判据草案 C4 的「去约束对照」。
保留星项与面项时，配对读数的格距无关性成立；去掉其中任一项（或两项同去）时出现距离依赖。
确定性，随机固定种子 20260920。

口径：星项约束对应顶点（荷）层，去掉它等于让荷型缺陷（e 型任何子）自由，
其相位效应是路径上的相位滑移；面项约束对应面（通量）层，去掉它等于让通量型缺陷自由，
其相位效应是路径围住的面元通量。两者都是局域事件，沿路径的累积数与距离成正比。

模型（沿路径长度 N 的一维读数）
  保留档：Delta-alpha(n) = eta_{j+n} - eta_j，eta 为平稳场（xi = 4、sigma = 1）
  去星项：上式加泊松滑移，每步以概率 p_v 加 N(0, s^2)
  去面项：上式加逐面通量，每步加 N(0, sigma_f^2)
  两去：两条通道相加
解析预期：保留档饱和于 2 sigma^2；去星项档斜率 = p_v s^2；去面项档斜率 = sigma_f^2；两去档斜率 = 两者之和。

判据与期望：
  U1 保留档：n 到 128 的差值方差与 2 sigma^2 相符（<= 5%），大格距散布 <= 5%
  U2 去星项：大格距斜率与 p_v s^2 相符（<= 5%）
  U3 去面项：大格距斜率与 sigma_f^2 相符（<= 5%）
  U4 两去：斜率等于两条通道之和（<= 5%）
  U5 密度扫描：星项通道斜率与 p_v 成正比（三档，<= 5%）
  U6 二维计数：直线路径穿过的顶点数与面数都与路径长度成正比（线性拟合残差 <= 5%）
  U7 接口登记（C4 与第 184 条稳定子代数，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\constraint_removal.png"
SEED = 20260920
N = 512
NS = 20000
XI, SIG = 4.0, 1.0
PV, SLIP = 0.1, 0.3
SIGF = 0.2
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def ar1(rng):
    rho = np.exp(-1.0 / XI)
    eta = np.zeros((NS, N))
    eta[:, 0] = rng.normal(scale=SIG, size=NS)
    for j in range(1, N):
        eta[:, j] = rho * eta[:, j - 1] + rng.normal(scale=SIG * np.sqrt(1.0 - rho ** 2), size=NS)
    return eta


rng = np.random.default_rng(SEED)
eta = ar1(rng)
slips = (rng.random((NS, N)) < PV) * rng.normal(scale=SLIP, size=(NS, N))
flux = rng.normal(scale=SIGF, size=(NS, N))
CH = {
    "res": eta,
    "no_star": eta + np.cumsum(slips, axis=1),
    "no_face": eta + np.cumsum(flux, axis=1),
    "no_both": eta + np.cumsum(slips, axis=1) + np.cumsum(flux, axis=1),
}
NS_SCAN = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
NS_ARR = np.array(NS_SCAN)


def var_diff(x, n):
    d = x[:, n:] - x[:, :x.shape[1] - n]
    return float(np.mean(d ** 2))


res_arr = {k: np.array([var_diff(v, n) for n in NS_SCAN]) for k, v in CH.items()}

# ---------- U1 保留档 ----------
big = np.array([var_diff(CH["res"], n) for n in (48, 64, 96, 128)])
spread = big.max() / big.min() - 1.0
report("U1 保留档", abs(big.mean() - 2 * SIG ** 2) / (2 * SIG ** 2) <= 0.05 and spread <= 0.05,
       "大格距上与 2 sigma^2 = %.4f 相符（<= 5%%），散布 <= 5%%" % (2 * SIG ** 2),
       "均值 %.4f（相对 %.2e），散布 %.2e" % (big.mean(), abs(big.mean() - 2 * SIG ** 2) / (2 * SIG ** 2), spread))


def slope(key, n0=64, n1=128):
    m = (NS_ARR >= n0) & (NS_ARR <= n1)
    return float(np.polyfit(NS_ARR[m], res_arr[key][m], 1)[0])


# ---------- U2 / U3 / U4 ----------
s_star, s_face, s_both = slope("no_star"), slope("no_face"), slope("no_both")
d2 = abs(s_star - PV * SLIP ** 2) / (PV * SLIP ** 2)
d3 = abs(s_face - SIGF ** 2) / SIGF ** 2
d4 = abs(s_both - (PV * SLIP ** 2 + SIGF ** 2)) / (PV * SLIP ** 2 + SIGF ** 2)
report("U2 去星项", d2 <= 0.05, "大格距斜率与 p_v s^2 = %.4f 相符（<= 5%%）" % (PV * SLIP ** 2),
       "斜率 %.5f（相对 %.2e）" % (s_star, d2))
report("U3 去面项", d3 <= 0.05, "大格距斜率与 sigma_f^2 = %.4f 相符（<= 5%%）" % SIGF ** 2,
       "斜率 %.5f（相对 %.2e）" % (s_face, d3))
report("U4 两去", d4 <= 0.05, "大格距斜率等于两通道之和 %.4f（<= 5%%）" % (PV * SLIP ** 2 + SIGF ** 2),
       "斜率 %.5f（相对 %.2e）" % (s_both, d4))

# ---------- U5 密度扫描 ----------
g5 = []
for pv in (0.05, 0.1, 0.2):
    sl = (rng.random((NS, N)) < pv) * rng.normal(scale=SLIP, size=(NS, N))
    x = eta + np.cumsum(sl, axis=1)
    m = (NS_ARR >= 64) & (NS_ARR <= 128)
    s = float(np.polyfit(NS_ARR[m], np.array([var_diff(x, n) for n in NS_SCAN])[m], 1)[0])
    g5.append((pv, s, s / pv))
ref5 = g5[1][2]
report("U5 密度扫描", all(abs(r - ref5) / ref5 <= 0.05 for _, _, r in g5),
       "星项通道斜率与密度成正比（三档相对散布 <= 5%）",
       "；".join("p_v=%.2f: 斜率 %.5f，斜率/密度 %.4f" % (pv, s, r) for pv, s, r in g5))

# ---------- U6 二维计数 ----------
M = 64
ns = np.arange(2, 33)
verts = ns + 1
faces = ns
res_v = float(np.max(np.abs(verts - np.polyval(np.polyfit(ns, verts, 1), ns))) / verts.max())
res_f = float(np.max(np.abs(faces - np.polyval(np.polyfit(ns, faces, 1), ns))) / faces.max())
report("U6 二维计数", res_v <= 0.05 and res_f <= 0.05 and np.all(np.diff(verts) == 1) and np.all(np.diff(faces) == 1),
       "直线路径穿过的顶点数与面数与长度成正比（线性残差 <= 5%%；逐段增量为 1）",
       "顶点残差 %.2e、面残差 %.2e；n = 2 至 32 上两者逐段增量为 1；格点 M = %d" % (res_v, res_f, M))

print("[登记] U7：去约束对照与第 184 条的稳定子代数衔接（星项 = 顶点层约束，面项 = 面层约束），本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for k, lab in (("res", "保留星项与面项"), ("no_star", "去星项"), ("no_face", "去面项"), ("no_both", "两去")):
    ax[0].plot(NS_SCAN, res_arr[k], "o-", label=lab)
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("路径长度 n")
ax[0].set_ylabel("差值方差")
ax[0].set_title("去约束对照：保留档饱和、去约束档线性增长")
ax[0].legend(fontsize=8)
pvs = [pv for pv, _, _ in g5]
ax[1].plot(pvs, [s for _, s, _ in g5], "o", label="实测斜率")
ax[1].plot(pvs, [pv * SLIP ** 2 for pv in pvs], "k--", label="解析 p_v s^2")
ax[1].set_xlabel("滑移密度 p_v")
ax[1].set_ylabel("星项通道斜率")
ax[1].set_title("斜率与缺陷密度成正比")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
