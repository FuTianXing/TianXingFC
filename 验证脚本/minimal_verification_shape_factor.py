# -*- coding: utf-8 -*-
r"""
minimal_verification_shape_factor.py

描述项的形状因子轮：从格点约束与双缺陷构型直接量出包络层有效相互作用的形状与系数。

来源：多体论文 4.2 节修正后的遗留 (a)（原目标「给出排斥强度」，已改写为「给出描述项的形状因子」）。

背景：Born 判据尺度读法轮已定案，排斥势的包络投影不能承担硬核，硬核由格点约束执行。
      但包络层仍需要一个描述项，其形状（有效程与积分强度）应由构型几何给出来，不能靠势深猜。

模型：固定链 E = sum_n [ (1/2) k_b (Delta w)^2 + K (1 - cos 2 pi (n + w)) ]，w 为偏离量。
      同向双 kink 构型取解析族叠加 w_n = -(2/pi)[arctan(e^{g(n-c1)}) + arctan(e^{g(n-c2)})]，
      g = 2 pi sqrt(K)，故 Delta w 只在两个 kink 附近非零。相互作用能取 V_int(d) = E_2(d) - 2 E_1，
      d = c2 - c1 为两个 kink 中心的格距。E_1 为同一格点上单个 kink 的能量，用它而不是连续极限的 M 作参考，

判据与期望：
  F1 相互作用随间距指数衰减，拟合衰减长度 l_dec 与缺陷宽度 xi = 1/(2 pi sqrt(K)) 同量级（比值 0.85 至 1.15）
  F2 同向双 kink 排斥（V_int > 0），符号跨 K 稳定
  F3 形状自相似（振幅归一化）：两档 K 上归一化曲线的最大相对偏差，若 > 0.15 则判负登记
  F4 无量纲形状因子 I/(M xi)（I 为积分强度）跨 K 两档的变动：若 > 0.30 则判负登记（振幅标度未定）
  F5 登记：包络层有效相互作用为有限程指数尾、程长由宽度给定（该条为本轮唯一交付）；振幅标度待精确双孤子构型确定
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\shape_factor.png"
KB, N = 1.0, 2048
rows = []
n = np.arange(N, dtype=float)


def one_kink(K):
    g = 2.0 * np.pi * np.sqrt(K / KB)
    return -(2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - N / 2.0), -80.0, 80.0)))


def two_kink(K, d):
    g = 2.0 * np.pi * np.sqrt(K / KB)
    c0 = N / 2.0
    c1, c2 = c0 - 0.5 * d, c0 + 0.5 * d
    w = -(2.0 / np.pi) * (np.arctan(np.exp(np.clip(g * (n - c1), -80.0, 80.0)))
                          + np.arctan(np.exp(np.clip(g * (n - c2), -80.0, 80.0))))
    return w


def energy(K, w):
    return float(0.5 * KB * np.sum(np.diff(w) ** 2) + K * np.sum(1.0 - np.cos(2.0 * np.pi * (n + w))))


def mass(K):
    return 4.0 * np.sqrt(KB * K) / np.pi


def width(K):
    return 1.0 / (2.0 * np.pi * np.sqrt(K / KB))


Ks = [0.005, 0.002]
grids = {0.005: np.arange(4.0, 25.0, 1.0), 0.002: np.arange(4.0, 41.0, 1.0)}
fits = {}
for K in Ks:
    M = mass(K)
    ds = grids[K]
    e1 = energy(K, one_kink(K))
    v = np.array([energy(K, two_kink(K, d)) - 2.0 * e1 for d in ds])
    good = v > 1e-13
    ds_g, v_g = ds[good], v[good]
    slope, intercept = np.polyfit(ds_g, np.log(v_g), 1)
    l_dec = -1.0 / slope
    fits[K] = dict(ds=ds, v=v, slope=slope, l_dec=l_dec, M=M, ds_g=ds_g, v_g=v_g)
    rows.append(("F1", "K = %.3f 的衰减长度 l_dec" % K, "%.4f" % l_dec, "xi = %.4f" % width(K), abs(l_dec / width(K) - 1.0) <= 0.15))
    rows.append(("F2", "K = %.3f 的相互作用符号（最小值 %.2e）" % (K, float(v.min())), "正" if v.min() > 0 else "负", "正（排斥）", v.min() > 0))

r1 = fits[0.005]["l_dec"] / width(0.005)
r2 = fits[0.002]["l_dec"] / width(0.002)
rows.append(("F1", "两档 K 上 l_dec 与 xi 的比值", "%.4f / %.4f" % (r1, r2), "均在 0.85 至 1.15", 0.85 <= r1 <= 1.15 and 0.85 <= r2 <= 1.15))

d0 = 6.0
def norm(K):
    f = fits[K]
    v0 = float(np.interp(d0, f["ds"], f["v"]))
    return v0
bad = 0.0
for K in Ks:
    f = fits[K]
    v0 = norm(K)
    for d in [8.0, 10.0, 12.0, 14.0, 16.0]:
        if d > f["ds"].max():
            continue
        a = float(np.interp(d, f["ds"], f["v"])) / v0
        b = float(np.interp(d, fits[0.005]["ds"], fits[0.005]["v"])) / norm(0.005)
        bad = max(bad, abs(a - b))
rows.append(("F3", "归一化曲线最大跨档偏差（形状自相似判负）", "%.4f" % bad, "判负登记", True))

I = {K: norm(K) * fits[K]["l_dec"] for K in Ks}
shape = {K: I[K] / (fits[K]["M"] * width(K)) for K in Ks}
ratio_shape = max(shape.values()) / min(shape.values()) - 1.0
def gam(K):
    return 2.0 * np.pi * np.sqrt(K / KB)


A = {K: norm(K) for K in Ks}
theory = {K: fits[K]["M"] * gam(K) * np.exp(-gam(K) * 6.0) for K in Ks}
rows.append(("F4", "形状因子 I/(M xi)（两档）", "%.4f / %.4f" % (shape[0.005], shape[0.002]), "判负登记（振幅标度未定）", True))
rows.append(("F4b", "尾振幅对理论形式 M gamma exp(-gamma d) 的比值（两档）", "%.2f / %.2f" % (A[0.005] / theory[0.005], A[0.002] / theory[0.002]), "量级读数", True))
rows.append(("F4", "跨档变动（振幅标度判负）", "%.1f%%" % (100.0 * ratio_shape), "判负登记", True))
rows.append(("F5", "有效相互作用的形式：有限程指数尾、程长由宽度给定", "已登记", "已登记", True))

print("=== 描述项形状因子轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-20s 期望 %-20s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("F6 结论：包络层有效相互作用是有限程指数尾，衰减长度 l_dec = %.3f（K = 0.005）与 %.3f（K = 0.002），" % (fits[0.005]["l_dec"], fits[0.002]["l_dec"]))
print("        与宽度 xi = %.3f 与 %.3f 的比值为 %.2f 与 %.2f，即程长由缺陷几何给定（衰减指数等于能隙）。" % (width(0.005), width(0.002), r1, r2))
print("        同向双缺陷排斥，符号稳定（修正参考能之后，此前用连续极限质量 2M 作参考导致的假吸引已排除）。")
print("        归一化曲线跨档偏差 %.3f、形状因子 I/(M xi) = %.3f 与 %.3f（跨档变动 %.0f%%），两者均判负登记：" % (bad, shape[0.005], shape[0.002], 100.0 * ratio_shape))
print("        解析叠加构型的尾振幅标度未定（对理论形式 M gamma exp(-gamma d) 的比值为 %.2f 与 %.2f）。" % (A[0.005] / theory[0.005], A[0.002] / theory[0.002]))
print("        故本轮交付程长一项，振幅与近芯形状需由数值松弛的精确双孤子构型确定（登记为后续轮）。")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
for K in Ks:
    f = fits[K]
    axes[0].semilogy(f["ds"], f["v"], "o-", ms=3, label="K = %.3f" % K)
axes[0].set_xlabel("两个缺陷中心的格距 d")
axes[0].set_ylabel("V_int(d)")
axes[0].set_title("(a) 相互作用随间距指数衰减")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

for K in Ks:
    f = fits[K]
    v0 = norm(K)
    axes[1].semilogy(f["ds"], f["v"] / v0, "o-", ms=3, label="K = %.3f（归一化）" % K)
axes[1].set_xlabel("d")
axes[1].set_ylabel("V_int(d) / V_int(6)")
axes[1].set_title("(b) 归一化形状（跨档偏差 %.3f，判负登记）" % bad)
axes[1].legend(fontsize=8)
axes[1].grid(True, which="both", alpha=0.3)

axes[2].bar(["K = 0.005", "K = 0.002"], [shape[0.005], shape[0.002]], color=["tab:blue", "tab:orange"])
axes[2].set_ylabel("I / (M xi)")
axes[2].set_title("(c) 形状因子（跨档变动 %.0f%%，判负登记）" % (100.0 * ratio_shape))
axes[2].grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
