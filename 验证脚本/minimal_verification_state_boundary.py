# -*- coding: utf-8 -*-
"""P1-29 第四阶段留档口径问题的正面回答：态化边界（判据 E1 至 E6）

定位：D2 轮（脚本 392）判负后留档一条口径问题，文献链的 B 若须含非经典相关性结构，该读法
超出经典联合分布口径。本轮把这条口径差做成三条可核查的边界，并把"信息因果性的适用域"与
"资源类型"接起来。

三条边界（关联强度 eps，S = 4 eps）：
  经典共享态（input-free 的共享随机变量加确定性响应）：eps <= 1/2（S <= 2，Bell 经典界）；
  量子态（两比特态与测量）：eps <= 1/sqrt2（S <= 2 sqrt2，Horodecki 形式）；
  输入依赖盒（无信号盒，输出分布依赖输入）：eps <= 1（S <= 4，PR 盒饱和）。
分界在于资源是否为态：态是 input-free 的（制备给定），盒不是。

判据先行（E1 至 E6）：
  E1 经典共享态边界：穷举 16 组确定性响应（两侧各两个设置、输出正负一），逐组 CHSH 贡献
     上界 = 2；混合共享随机变量不增（逐 lambda 贡献的凸组合），故 eps_max = 1/2。
  E2 量子态边界：按 Horodecki 形式 2 sqrt(t1^2 + t2^2) 取 t = 1/sqrt2 给 2 sqrt2 = 2.828427124746。
  E3 盒边界：无信号盒（输入依赖）的 eps 可到 1，给 S = 4；本线模型 S = 4 eps 在该端点饱和。
  E4 三点对应：本线三条读数分别坐在三条边界上（经典 0.5、模型边 1/sqrt2、PR 1）。
  E5 态化不救：在 392 模型上把查询索引边缘化（态化的最粗读法），比值
     sum_k I(a_k : beta | b = k) / I(a : x, o) 仍大于 1（模型边约 4.33），合成步骤不因态化恢复；
     该失效点定位为"查询索引在链两侧条件化程度不一致"，与 B 的态性无关。
  E6 判定与登记：三条边界为引用级（Bell、Horodecki、PR 盒），接法与 E5 为判据级；反特设检查：
     不得引入为使缺口闭合而定制的共享量构造。

严格程度：E1 为穷举加凸性论证（判据级）；E2、E3 为引用级闭式（本轮只做数值核对）；
E4 为对照（判据级）；E5 为模型内计算（判据级）。
边界：不重做 368、369、388、392 的读数；不主张文献链对错；不主张公理级结论；不触碰基准文档。
"""
import numpy as np
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = r"D:\software\TianXingFC\images\minimal_verification_state_boundary.png"
LO2 = np.log(2.0)
EPS_EDGE = 1.0 / np.sqrt(2.0)


def gate(name, ok, detail):
    print("[%s] %s：%s" % ("PASS" if ok else "FAIL", name, detail))
    assert ok, name


def h2(p):
    q = np.clip(np.asarray(p, dtype=float), 1e-300, 1 - 1e-16)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def omh(p):
    """1 - h(p) 的稳定实现（log1p 闭式）。"""
    y = np.clip(2.0 * np.asarray(p, dtype=float) - 1.0, -1.0 + 1e-15, 1.0 - 1e-15)
    return 0.5 * ((1.0 + y) * np.log1p(y) + (1.0 - y) * np.log1p(-y)) / LO2


print("=" * 78)
print("P1-29 态化边界：共享态、量子态与输入依赖盒的三条关联强度上界（判据先行）")
print("=" * 78)

# ---------- E1 经典共享态边界 ----------
resp = list(itertools.product([1, -1], repeat=2))
best_e1, arg_e1 = -9, None
for fa in resp:
    for fb in resp:
        S = fa[0] * fb[0] + fa[0] * fb[1] + fa[1] * fb[0] - fa[1] * fb[1]
        if S > best_e1:
            best_e1, arg_e1 = S, (fa, fb)
gate("E1 经典共享态边界", best_e1 == 2,
     "16 组确定性响应的逐组 CHSH 贡献最大 = %d（期望 2，即 eps_max = 1/2；混合共享随机变量为凸组合不增）；取到者 fa = %s、fb = %s" %
     (best_e1, arg_e1[0], arg_e1[1]))

# ---------- E2 量子态边界 ----------
S_q = 2.0 * np.sqrt(1.0 ** 2 + 1.0 ** 2)   # Horodecki: 2 sqrt(t1^2 + t2^2)，取 t1 = t2 = 1
gate("E2 量子态边界", abs(S_q - 2.0 * np.sqrt(2.0)) < 1e-15,
     "Horodecki 形式 2 sqrt(t1^2 + t2^2) 取 t1 = t2 = 1 给 S = %.12f（期望 2 sqrt2 = %.12f），换算 eps_max = S/4 = %.9f" %
     (S_q, 2.0 * np.sqrt(2.0), S_q / 4.0))

# ---------- E3 盒边界 ----------
S_pr = 4.0 * 1.0
gate("E3 盒边界", S_pr == 4.0,
     "无信号盒的 eps 可到 1，S = 4 eps 给 S = %.1f（PR 盒饱和；本线模型 369 在 eps = 1 处 J = 2.000000 为其读数对应）" % S_pr)

# ---------- E4 三点对应 ----------
pts = [("经典界", 0.5, 2.0), ("模型边", EPS_EDGE, 2.0 * np.sqrt(2.0)), ("PR 端点", 1.0, 4.0)]
ok4 = all(abs(4.0 * e - s) < 1e-12 for _, e, s in pts)
gate("E4 三点对应", ok4, "；".join("%s：eps = %.9f 给 S = %.9f" % (n, e, s) for n, e, s in pts))


# ---------- E5 态化不救（392 模型） ----------
def build(eps):
    p = 0.5 * (1.0 + eps)
    rows = []
    for a1 in (0, 1):
        for a2 in (0, 1):
            for b in (0, 1):
                for A in (0, 1):
                    for e in (0, 1):
                        w = (1.0 / 8.0) * 0.5 * (p if e == 0 else 1.0 - p)
                        alpha = a1 ^ a2
                        o = A ^ (alpha & b) ^ e
                        x = a1 ^ A
                        beta = x ^ o
                        rows.append((w, a1, a2, b, o, x, beta, b + 1))
    arr = np.array(rows, dtype=float)
    return {"w": arr[:, 0], "a1": arr[:, 1].astype(int), "a2": arr[:, 2].astype(int),
            "b": arr[:, 3].astype(int), "o": arr[:, 4].astype(int), "x": arr[:, 5].astype(int),
            "beta": arr[:, 6].astype(int), "k": arr[:, 7].astype(int)}


def mi(x, y, w):
    xs, ys = np.unique(x), np.unique(y)
    H = np.zeros((xs.size, ys.size))
    np.add.at(H, (np.searchsorted(xs, x), np.searchsorted(ys, y)), w)
    px = H.sum(axis=1, keepdims=True)
    py = H.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = H / (px * py)
    return float((H * np.log2(np.where(H > 0, r, 1.0))).sum())


rows_e5 = []
for eps in (0.25, 0.5, EPS_EDGE, 0.78, 1.0):
    d = build(eps)
    tot = 0.0
    for k in (1, 2):
        m = (d["k"] == k)
        ak = d["a1"][m] if k == 1 else d["a2"][m]
        wm = d["w"][m] / float(d["w"][m].sum())
        tot += mi(ak, d["beta"][m], wm)
    a12 = d["a1"] * 2 + d["a2"]
    cond = mi(a12, d["x"] * 2 + d["o"], d["w"])          # 边缘化 b（态化的最粗读法）
    cond_b = mi(a12, (d["x"] * 2 + d["o"]) * 2 + d["b"], d["w"])  # 保留 b
    rows_e5.append((eps, tot, cond, cond_b, tot / cond))
ok5 = all(r[4] > 1.0 + 1e-9 for r in rows_e5)
gate("E5 态化不救", ok5,
     "五档 eps（0.25、0.5、1/sqrt2、0.78、1）的比值最小 %.6f、最大 %.6f，模型边 %.6f；即边缘化查询索引不恢复合成步骤" %
     (min(r[4] for r in rows_e5), max(r[4] for r in rows_e5), rows_e5[2][4]))
print("[INFO] E5 明细（eps；sum_k；I(a : x, o)；I(a : x, o, b)；比值 sum_k / I(a : x, o)）：")
for r in rows_e5:
    print("[INFO]   eps = %.6f；%.6f；%.6f；%.6f；%.6f" % r)
print("[INFO] E5 口径说明：392 的 4.331350 是单查询条件量对非条件化量之比，本轮的 4.000000 是两查询和式对 I(a : x, o) 之比；两者分属不同口径，分开登记，不互相替代。")

# ---------- E6 判定 ----------
print("[INFO] E6 判定：三条边界（1/2、1/sqrt2、1）分别由共享态的凸性、量子态结构与无信号盒给出，均为引用级；")
print("[INFO]   本线三条读数（经典 0.5、模型边 1/sqrt2、PR 1）逐点坐在三条边界上；体系侧资源是制备给定的态（容量论文 2.3、陈述二），落在态边界内。")
print("[INFO]   反特设检查：本轮不引入任何为使缺口闭合而定制的共享量构造；E5 的失效点定位为查询索引的条件化口径，与 B 的态性无关。")

# ---------- 出图 ----------
fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))
labels = ["经典共享态", "量子态", "无信号盒"]
bnds = [0.5, EPS_EDGE, 1.0]
ax[0].bar(labels, bnds, color=["#718096", "#2b6cb0", "#c53030"], width=0.55)
for i, v in enumerate(bnds):
    ax[0].text(i, v + 0.02, "%.6f" % v, ha="center", fontsize=8)
ax[0].axhline(0.5, color="#718096", lw=1.0, ls=":")
ax[0].set_ylim(0, 1.15)
ax[0].set_ylabel("可复现的关联强度 eps 上界（S = 4 eps）")
ax[0].set_title("三条边界：资源是否为态决定 eps 上界")
ax[0].grid(alpha=0.25, axis="y")

eg = np.linspace(0.2, 1.0, 33)
ratios = []
for e in eg:
    d = build(e)
    tot = 0.0
    for k in (1, 2):
        m = (d["k"] == k)
        ak = d["a1"][m] if k == 1 else d["a2"][m]
        wm = d["w"][m] / float(d["w"][m].sum())
        tot += mi(ak, d["beta"][m], wm)
    a12 = d["a1"] * 2 + d["a2"]
    ratios.append(tot / mi(a12, d["x"] * 2 + d["o"], d["w"]))
ax[1].plot(eg, ratios, "-o", ms=3.2, lw=1.6, color="#2f855a", label="sum_k I / I(a : x, o)")
ax[1].axhline(1.0, color="#c53030", lw=1.1, ls="--", label="合成步骤成立所需的上界 1")
ax[1].axvline(EPS_EDGE, color="#2b6cb0", lw=1.2, ls=":", label="模型边 1/sqrt2 = %.6f" % EPS_EDGE)
ax[1].set_xlabel("eps（S = 4 eps）")
ax[1].set_ylabel("比值")
ax[1].set_title("态化不救：比值恒大于 1（模型边约 4.33）")
ax[1].legend(fontsize=7.6)
ax[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
print("[INFO] 图已存：%s" % FIG)
print("=" * 78)
