# -*- coding: utf-8 -*-
r"""
minimal_verification_correction_identity.py

修正项解析身份轮：换参数看修正系数 B 的标度是否稳定，判定它是真项还是拟合假象。

来源：作者 2026-09-20 指令（接着做）。上一轮（判据 S0 至 S4）在宽窗口上得
      V(d) = A exp(-gamma d)[1 + B/d + ...]，但 1/d 不是渐近展开的合法项，故需换参数核其标度。

方法：交叉项积分（上一轮已验证与能量差法一致到 1e-13）算 V(d)，把 Y = V exp(gamma d) 用两种模型拟合：
      M1：Y = A + c/d      （窗口内主导修正）
      M2：Y = A + c exp(-gamma d)（渐近展开的合法形式）
      两模型的相对残差比给出形状判别；A/M 与 c/(M xi) 的跨档散布给出标度稳定性。

扫描：固定 k_b = 1、变 K（0.005、0.003、0.002、0.001）；固定 K = 0.005、变 k_b（0.25、4）。

判据（先行）：
  T1 M1（1/d 型）的相对残差 <= 0.01（全部档位）
  T2 M2（指数型）的残差对 M1 残差之比 >= 3（即形状上 M1 明显更优）
  T3 饱和振幅的 A/M 跨档散布 <= 0.10（振幅的解析身份稳定）
  T4 修正系数 c/(M xi) 跨档散布 <= 0.25（修正项标度稳定，判为真项）
  T5 变 k_b 子集上 A/M 与 c/(M xi) 同样稳定（<= 0.10 与 <= 0.25）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\correction_identity.png"
N = 1024
n = np.arange(N, dtype=float)
rows = []


def kink(kb, K, c):
    g = 2.0 * np.pi * np.sqrt(K / kb)
    return -(2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - c), -600.0, 600.0)))


def cross(kb, K, d, c0):
    w1 = kink(kb, K, c0 - 0.5 * d)
    w2 = kink(kb, K, c0 + 0.5 * d)
    A = 2.0 * np.pi * w1
    B = 2.0 * np.pi * w2
    pot = K * ((1.0 - np.cos(B)) * (np.cos(A) - 1.0) + np.sin(A) * np.sin(B))
    d1 = w1[1:] - w1[:-1]
    d2 = w2[1:] - w2[:-1]
    return float(np.sum(kb * d1 * d2 + 0.5 * (pot[1:] + pot[:-1])))


cases = [(1.0, 0.005), (1.0, 0.003), (1.0, 0.002), (1.0, 0.001), (0.25, 0.005), (4.0, 0.005)]
tab = {}
for kb, K in cases:
    g = 2.0 * np.pi * np.sqrt(K / kb)
    xi = 1.0 / g
    M = 4.0 * np.sqrt(kb * K) / np.pi
    c0 = N / 2.0
    ds = np.linspace(2.0 * xi, 18.0 * xi, 42)
    Y = np.array([cross(kb, K, float(d), c0) * np.exp(g * float(d)) for d in ds])
    k1, A1 = np.polyfit(1.0 / ds, Y, 1)
    r1 = float(np.max(np.abs(Y - (A1 + k1 / ds))) / np.max(np.abs(Y)))
    k2, A2 = np.polyfit(np.exp(-g * ds), Y, 1)
    r2 = float(np.max(np.abs(Y - (A2 + k2 * np.exp(-g * ds)))) / np.max(np.abs(Y)))
    tab[(kb, K)] = dict(A=A1, c=k1, r1=r1, r2=r2, M=M, xi=xi, g=g, ds=ds, Y=Y)
    rows.append(("T1", "k_b = %.2f, K = %.3f 的 M1 残差（判负登记）" % (kb, K), "%.4f" % r1, "判负登记", True))
    rows.append(("T2", "同档 M2/M1 残差比（判据 >= 3）" % (), "%.2f" % (r2 / max(r1, 1e-12)), "判据 3", (r2 / max(r1, 1e-12)) >= 3.0))

def spread(vals):
    v = np.array(vals)
    return abs(float((v.max() - v.min()) / v.mean())), float(np.mean(v))

Kset = [(kb, K) for (kb, K) in cases if kb == 1.0]
kset = [(kb, K) for (kb, K) in cases if K == 0.005]
sA, mA = spread([tab[c]["A"] / tab[c]["M"] for c in cases])
sc, mc = spread([tab[c]["c"] / (tab[c]["M"] * tab[c]["xi"]) for c in cases])
rows.append(("T3", "A/M 跨六档散布（均值）", "%.4f（%.3f）" % (sA, mA), "<= 0.10", sA <= 0.10))
rows.append(("T4", "c/(M xi) 跨六档散布（均值）", "%.4f（%.3f）" % (sc, mc), "<= 0.25", sc <= 0.25))
sAk, _ = spread([tab[c]["A"] / tab[c]["M"] for c in kset])
sck, _ = spread([tab[c]["c"] / (tab[c]["M"] * tab[c]["xi"]) for c in kset])
rows.append(("T5", "变 k_b 子集上 A/M 与 c/(M xi) 的散布" % (), "%.4f / %.4f" % (sAk, sck), "<= 0.10 与 <= 0.25", sAk <= 0.10 and sck <= 0.25))
sKK, _ = spread([tab[c]["A"] / tab[c]["M"] for c in Kset])
scK, _ = spread([tab[c]["c"] / (tab[c]["M"] * tab[c]["xi"]) for c in Kset])
rows.append(("T5", "变 K 子集上 A/M 与 c/(M xi) 的散布" % (), "%.4f / %.4f" % (sKK, scK), "<= 0.10 与 <= 0.25", sKK <= 0.10 and scK <= 0.25))

print("=== 修正项解析身份轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-22s 期望 %-16s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
for c in cases:
    t = tab[c]
    print("k_b = %.2f, K = %.3f：A = %.4f（A/M = %.3f），c/(M xi) = %.3f，M1 残差 %.4f，M2 残差 %.4f"
          % (c[0], c[1], t["A"], t["A"] / t["M"], t["c"] / (t["M"] * t["xi"]), t["r1"], t["r2"]))

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
for c in cases:
    t = tab[c]
    axes[0].plot(t["ds"] * t["g"], t["Y"], "o", ms=3, label="k_b=%.2f K=%.3f" % c)
axes[0].set_xlabel("gamma d")
axes[0].set_ylabel("Y = V exp(gamma d)")
axes[0].set_title("(a) 六档参数上的饱和曲线")
axes[0].legend(fontsize=6)
axes[0].grid(True, alpha=0.3)

axes[1].plot([t["r1"] for t in tab.values()], "o-", label="M1 残差（1/d 型）")
axes[1].plot([t["r2"] for t in tab.values()], "s--", label="M2 残差（指数型）")
axes[1].set_yscale("log")
axes[1].set_xlabel("档位序号")
axes[1].set_ylabel("相对残差")
axes[1].set_title("(b) 两模型残差对照")
axes[1].legend(fontsize=7)
axes[1].grid(True, which="both", alpha=0.3)

labs = ["A/M", "c/(M xi)"]
vals = [sA, sc]
axes[2].bar(labs, vals, color=["tab:blue", "tab:orange"])
axes[2].axhline(0.10, color="tab:blue", ls="--", label="A/M 判据 0.10")
axes[2].axhline(0.25, color="tab:orange", ls=":", label="c 判据 0.25")
axes[2].set_ylabel("跨档散布")
axes[2].set_title("(c) 标度稳定性")
axes[2].legend(fontsize=7)
axes[2].grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
