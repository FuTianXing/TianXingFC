# -*- coding: utf-8 -*-
"""判据第 633 号：号差来源接入——复结构 i 落在粗格生成元上的读数。

背景：判据第 632 号在粗格上得到两枚生成元 gamma_1 = sigma_3、gamma_2 = sigma_1，
      平方同为 +1（欧氏型），洛伦兹号差不在此层。本轮把体系内的复结构 i 接上去。

判据与先立的期望：
  U1 复结构是中心元，故把 i 乘在任一枚生成元上不破坏反对易（期望反对易子为零）。
  U2 号差唯一性：穷举 i 加在 0、1、2 枚生成元上，只有"恰一枚"给出一负一正。
  U3 色散类型：由 M^2 的两项系数符号读椭圆型或双曲型（读数）。
  U4 来源唯一性：把 i 换成 e^{i theta} 扫描 theta，只有 theta = pi/2 与 3pi/2 使平方为 -1。
  U5 恒真反例：把 i 换成实数 1，号差仍同号，洛伦兹型不成立。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\signature_source.png"

S3 = np.array([[1, 0], [0, -1]], dtype=complex)
S1 = np.array([[0, 1], [1, 0]], dtype=complex)


def sig_count(gens):
    """生成元平方为本征值 +1 / -1 的个数（先按 sqrt(|sq|) 归一，只读符号）。"""
    out = []
    for g in gens:
        sq = g @ g
        out.append(int(np.round(np.real(np.trace(sq)) / 2.0)))
    return out


def dispersion_coeffs(gens):
    """M(k) = i s1 g1 + i s2 g2，返回 M^2 的两项系数 (c1, c2)。"""
    def M(x, y):
        return 1j * x * gens[0] + 1j * y * gens[1]
    m2 = M(1, 0) @ M(1, 0)
    c1 = np.real(np.trace(m2) / 2.0)
    m2 = M(0, 1) @ M(0, 1)
    c2 = np.real(np.trace(m2) / 2.0)
    return c1, c2


print("=== 判据第 633 号 ===")
cases = {
    "甲：不加 i（632 基线）": [S3, S1],
    "乙：i 加在第一枚": [1j * S3, S1],
    "丙：i 加在第二枚": [S3, 1j * S1],
    "丁：i 加在两枚": [1j * S3, 1j * S1],
}
for name, gens in cases.items():
    ac = np.abs(gens[0] @ gens[1] + gens[1] @ gens[0]).max()
    sg = sig_count(gens)
    c1, c2 = dispersion_coeffs(gens)
    kind = "双曲型（洛伦兹）" if c1 * c2 < 0 else "椭圆型（欧氏）"
    neg = sum(1 for x in sg if x < 0)
    print("  %s：反对易子=%.1e  平方号=%s（负号 %d 枚）  色散系数=(%+.0f, %+.0f)  %s"
          % (name, ac, sg, neg, c1, c2, kind))

neg_counts = sorted({sum(1 for x in sig_count(g) if x < 0) for g in cases.values()})
print("U2  三档负号枚数 = %s；洛伦兹型要求恰一枚 → 唯一性%s"
      % (neg_counts, "成立" if neg_counts == [0, 1, 2] and 1 in neg_counts else "不成立"))

# U4：相位扫描
ths = np.linspace(0, 2 * np.pi, 721)
mods, args, negs = [], [], []
for th in ths:
    ph = np.exp(1j * th)
    g = ph * S3
    sq = g @ g
    ev = np.linalg.eigvalsh((sq + sq.conj().T) / 2)
    mods.append(abs(ev[0]))
    args.append(np.angle(ev[0]))
    negs.append(1 if abs(ev[0] + 1) < 1e-9 else 0)
mods = np.array(mods); args = np.array(args); negs = np.array(negs)
hit = ths[negs == 1]
print("U4  使平方为 -1 的相位 = %s 弧度（共 %d 个采样点）"
      % (np.round(hit, 4), len(hit)))

# U5：恒真反例
g0 = 1.0 * S3
print("U5  反例（把 i 换成实数 1）：平方号 = %s，%s"
      % (sig_count([g0, S1]),
         "仍同号，洛伦兹不成立" if len(set(sig_count([g0, S1]))) == 1 else "异号"))

# 图
fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.2))

names = list(cases.keys())
negs_per = [sum(1 for x in sig_count(g) if x < 0) for g in cases.values()]
ax[0].bar(range(4), negs_per, color=["tab:gray", "tab:blue", "tab:blue", "tab:red"], width=0.55)
ax[0].axhline(1, color="tab:green", ls="--", lw=1.5, label="洛伦兹型要求恰一枚")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["不加 i", "加一枚", "另一枚", "加两枚"], fontsize=9)
ax[0].set_ylabel("负平方枚数"); ax[0].set_ylim(0, 2.5); ax[0].legend(fontsize=9)
ax[0].set_title("(a) 号差唯一性\n只有恰一枚给洛伦兹型")

k1, k2 = np.meshgrid(np.linspace(-np.pi, np.pi, 161), np.linspace(-np.pi, np.pi, 161))
lv = [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
for a, gens, ttl in [(ax[1], [S3, S1], "(b) 不加 i：椭圆型"),
                     (ax[2], [1j * S3, S1], "(c) i 加一枚：双曲型")]:
    c1, c2 = dispersion_coeffs(gens)
    f = c1 * np.sin(k1) ** 2 + c2 * np.sin(k2) ** 2
    cs = a.contour(k1, k2, f, levels=lv, cmap="coolwarm")
    a.clabel(cs, inline=True, fontsize=7)
    a.set_title(ttl); a.set_xlabel("$k_1$"); a.set_ylabel("$k_2$")

ax[3].plot(ths, np.abs(np.cos(2 * ths)), color="tab:purple", lw=2, label=r"平方的模 $|\cos 2\theta|$")
ax[3].plot(hit, [1] * len(hit), "o", color="tab:green", ms=8, label=r"平方 $=-1$（两处）")
ax[3].axvline(np.pi / 2, color="tab:green", ls="--", lw=1.2)
ax[3].set_xlabel(r"相因子 $\theta$"); ax[3].set_ylim(0, 1.25)
ax[3].set_title("(d) 来源唯一性\n只有四分之一圈翻号")
ax[3].legend(fontsize=9)

for a in ax:
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("\n图已存：%s" % OUT)
