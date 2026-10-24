# -*- coding: utf-8 -*-
r"""
harmonic_expansion.py

逐阶展开轮：用固定指数基做最小二乘，逐阶定系数，替代两参拟合。

来源：作者 2026-09-20 指令（同意按逐阶展开做）。上一轮（判据 T1 至 T5）否定了两种两参候选。

方法：交叉项积分算精确的 V(d)。把 Y = V exp(gamma d) 展成固定基的最小二乘：
      Y(d) = a0 + a1 d + a2 exp(-gamma d) + a3 d exp(-gamma d) + a4 exp(-2 gamma d) + a5 d exp(-2 gamma d) + ...
      基函数固定（指数为 gamma 的整数倍、前因子为 d 的多项式），只解系数，故不发生「两参模型互相补偿」的问题。
      Y 作因变量也避免了大 d 上基函数指数小时的条件数问题。

扫描：固定 k_b = 1 变 K（0.005、0.002）；固定 K = 0.005 变 k_b（4）。窗口 theta = gamma d 从 2 到 24。

判据（先行）：
  U1 单项模型（仅常数）残差 >= 0.10（单项不足）
  U2 六项基（到二阶、含 d 前因子）的残差 <= 0.01
  U3 四阶基（1、d、e^{-gamma d}、d e^{-gamma d}）对六项基的残差比 >= 2（说明二阶项必要），否则登记
  U4 主系数 a0/M 的跨档散布 <= 0.10
  U5 登记：逐阶系数表与所需阶数
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\harmonic_expansion.png"
N = 1024
n = np.arange(N, dtype=float)
rows = []


def kink(kb, K, c):
    g = 2.0 * np.pi * np.sqrt(K / kb)
    return -(2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - c), -800.0, 800.0)))


def cross(kb, K, d, c0):
    w1 = kink(kb, K, c0 - 0.5 * d)
    w2 = kink(kb, K, c0 + 0.5 * d)
    A = 2.0 * np.pi * w1
    B = 2.0 * np.pi * w2
    pot = K * ((1.0 - np.cos(B)) * (np.cos(A) - 1.0) + np.sin(A) * np.sin(B))
    d1 = w1[1:] - w1[:-1]
    d2 = w2[1:] - w2[:-1]
    return float(np.sum(kb * d1 * d2 + 0.5 * (pot[1:] + pot[:-1])))


cases = [(1.0, 0.005), (1.0, 0.002), (4.0, 0.005)]
tab = {}
for kb, K in cases:
    g = 2.0 * np.pi * np.sqrt(K / kb)
    xi = 1.0 / g
    M = 4.0 * np.sqrt(kb * K) / np.pi
    c0 = N / 2.0
    th = np.linspace(2.0, 24.0, 45)
    ds = th / g
    V = np.array([cross(kb, K, float(d), c0) for d in ds])
    Y = V * np.exp(g * ds)
    x = np.exp(-g * ds)
    bases = {
        "M1 常数": [np.ones_like(ds)],
        "M2 常数+d": [np.ones_like(ds), ds],
        "M3 常数+e^-gd": [np.ones_like(ds), x],
        "M4 常数+d+e^-gd+d e^-gd": [np.ones_like(ds), ds, x, ds * x],
        "M6 到二阶含 d": [np.ones_like(ds), ds, x, ds * x, x ** 2, ds * x ** 2],
    }
    res = {}
    for name, cols in bases.items():
        Mat = np.array(cols).T
        coef, *_ = np.linalg.lstsq(Mat, Y, rcond=None)
        r = float(np.max(np.abs(Y - Mat @ coef)) / np.max(np.abs(Y)))
        res[name] = (r, coef)
    tab[(kb, K)] = dict(ds=ds, th=th, Y=Y, res=res, M=M, xi=xi, g=g)
    rows.append(("U1", "k_b = %.2f, K = %.3f 单项残差" % (kb, K), "%.4f" % res["M1 常数"][0], ">= 0.10", res["M1 常数"][0] >= 0.10))
    rows.append(("U2", "同档六项残差" % (), "%.5f" % res["M6 到二阶含 d"][0], "<= 0.01", res["M6 到二阶含 d"][0] <= 0.01))
    rows.append(("U3", "同档四项对六项的残差比" % (), "%.2f" % (res["M4 常数+d+e^-gd+d e^-gd"][0] / max(res["M6 到二阶含 d"][0], 1e-12)), ">= 2", (res["M4 常数+d+e^-gd+d e^-gd"][0] / max(res["M6 到二阶含 d"][0], 1e-12)) >= 2.0))

a0s = [tab[c]["res"]["M6 到二阶含 d"][1][0] / tab[c]["M"] for c in cases]
sp = abs((max(a0s) - min(a0s)) / (sum(a0s) / len(a0s)))
rows.append(("U4", "主系数 a0/M 跨三档散布（均值）", "%.4f（%.3f）" % (sp, sum(a0s) / len(a0s)), "<= 0.10", sp <= 0.10))

print("=== 逐阶展开轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-44s 实测 %-20s 期望 %-14s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
for c in cases:
    t = tab[c]
    print("k_b = %.2f, K = %.3f（M = %.4f）：各模型残差 %s"
          % (c[0], c[1], t["M"], "；".join("%s %.4f" % (k.split()[0], v[0]) for k, v in t["res"].items())))
    coef = t["res"]["M6 到二阶含 d"][1]
    print("    六项系数（a0、a1、a2、a3、a4、a5）= %s" % ", ".join("%.4e" % v for v in coef))
    print("    归一化：a0/M = %.3f，a1/(M xi) = %.4f，a2/M = %.3f，a3/(M xi) = %.3f，a4/M = %.3f，a5/(M xi) = %.3f"
          % (coef[0] / t["M"], coef[1] / (t["M"] * t["xi"]), coef[2] / t["M"], coef[3] / (t["M"] * t["xi"]), coef[4] / t["M"], coef[5] / (t["M"] * t["xi"])))

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
for c in cases:
    t = tab[c]
    axes[0].plot(t["th"], t["Y"], "o", ms=3, label="k_b=%.2f K=%.3f" % c)
axes[0].set_xlabel("theta = gamma d")
axes[0].set_ylabel("Y = V exp(gamma d)")
axes[0].set_title("(a) 三档参数上的 Y(theta)")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

names = list(tab[cases[0]]["res"].keys())
for c in cases:
    axes[1].semilogy(names, [tab[c]["res"][k][0] for k in names], "o-", ms=3, label="k_b=%.2f K=%.3f" % c)
axes[1].set_ylabel("相对残差")
axes[1].set_title("(b) 嵌套基的残差")
axes[1].tick_params(axis="x", labelrotation=20)
axes[1].legend(fontsize=6)
axes[1].grid(True, which="both", alpha=0.3)

for i, lab in enumerate(["a0/M", "a2/M", "a4/M"]):
    vals = [tab[c]["res"]["M6 到二阶含 d"][1][[0, 2, 4][i]] / tab[c]["M"] for c in cases]
    axes[2].plot(range(len(cases)), vals, "o-", label=lab)
axes[2].set_xticks(range(len(cases)))
axes[2].set_xticklabels(["%.2f/%.3f" % c for c in cases], fontsize=7)
axes[2].set_ylabel("归一化系数")
axes[2].set_title("(c) 逐阶系数的跨档值")
axes[2].legend(fontsize=7)
axes[2].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
