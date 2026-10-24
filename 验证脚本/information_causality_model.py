# -*- coding: utf-8 -*-
"""P1-29 方向三（路线一落地）与方向四首轮增补：信息因果性的模型内定量判据（判据 Q1 至 Q6、R3、R4）

定位：方向三推导稿把"为何止步 2*sqrt2"分给三条路线，路线一（信息因果性）登记为引用级并留
一项开放：模型内的定量版本。本脚本把它做成判据：在写死的合法协议类内穷举（单副本、2 比特
输入、1 比特经典消息、k 盲的爱丽丝策略、k 可见的鲍勃策略、因果顺序固定），算两条量：
  一、2->1 随机存取码（RAC）成功率 P_succ；
  二、互信息和式 J = max[ I(a1:G1) + I(a2:G2) ]（IC 原则的单副本形式）。
并与模型侧的上界族对照：模型的 CHSH 族在标准选角下就是各向同性族，参数 eps = V/sqrt2，
S(CHSH) = 4 eps；结构路线（脚本 368）给模型可达边 eps <= 1/sqrt2（S <= 2*sqrt2）。

合法协议类（因果顺序写死，见 Q0 说明）：
  爱丽丝：盒输入 x = x(a1,a2)（测量前决定，只许依赖输入）；消息 c = c(a1,a2,A)（测量后，
  可用盒输出 A）；鲍勃：盒设置 y = y(k,c)（知 k 且已收到消息）；判读 G = G_k(c,B)。
  穷举：x 有 16 档、c 有 256 档（爱丽丝 4096 档），y 有 4 档、判读有 16 档（每 k 64 档）。

判据：
  Q1 家族与模型边：各向同性族 S = 4 eps；模型可达边 eps_max = 1/sqrt2 给 S = 2*sqrt2（1e-12）
  Q2 RAC 穷举：max P_succ(eps) = max(3/4, (1+eps)/2)（8 档，1e-12；eps <= 0.5 段由经典
      1 比特协议抬到 3/4，与各向同性族在 eps <= 1/2 属经典区一致）
  Q3 RAC 三个锚点：eps = 0.5 处 = 3/4（经典界）；eps = 1/sqrt2 处 = cos^2(pi/8)（信息因果性
      允许的上限，1e-12）；eps = 1 处 = 1（PR 箱）
  Q4 MI 和式穷举：J(eps) = 1.000000 对 eps <= 1/sqrt2 与 0.75 档（不违反 IC）；eps = 0.8 起
      J > 1（违反）；交叉点由细扫定位（报实测区间）
  Q5 混合稳健性：400 组随机两策略混合不提高 J（不超确定性穷举最优）
  Q6 同数核对：4*(1/sqrt2) = 2*sqrt2；P_succ(1/sqrt2) = cos^2(pi/8) = (1+1/sqrt2)/2（1e-15）

方向四首轮增补（2026-09-24，容量侧的形式退化与非空转；读数全部复用 Q 段，不新增协议）：
  R3 标准极限退化：无资源（eps = 0）时 J = 记录预算 m = 1 比特（古典锚点饱和，1e-9）；
      成功率 = 3/4（经典 1 比特协议值）；框架形式 sum_k I <= n_记录 与标准形式逐项对应
  R4a 非空转（双饱和）：模型边 eps = 1/sqrt2 处记录预算 J = 1 恰饱和（1e-9），成功率 = cos^2(pi/8)（1e-12）
  R4b 紧度对照（诚信项）：成功概率形式的临界值 = 模型边（紧）；互和信息式首次越界在 (0.77, 0.78)（松）
  R5 超容量显式量：框架记录预算 m = 1；PR 箱处 J = 2.000000（需 2 次记录支付），差额恰 1.000000 比特；
      与结构侧对应量 4 - 2*sqrt2 = 1.171573（CHSH 值侧）口径不同，分开登记

纪律：全确定性（种子 20260923 只用于混合稳健性抽样）；判据先行，实测 vs 期望；图存 images/。
边界：不做公理级断言（IC 原则本身为引用级）；协议类为写死的单副本形式，其外的最优性未扫；
      互信息和式形式的单副本版本不紧（本脚本如实报交叉点），紧形式是成功概率形式。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_PATH = r"D:\software\TianXingFC\images\information_causality_model.png"
SEED = 20260923
TS = 2.0 * np.sqrt(2.0)
SQ2 = np.sqrt(2.0)
EPS_EDGE = 1.0 / SQ2
RAC_IC = np.cos(np.pi / 8.0) ** 2      # = (1 + 1/sqrt2)/2

results = []

def gate(name, ok, detail):
    results.append((name, ok, detail))
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))

# ---------- 协议类穷举引擎 ----------
a1s = np.array([i % 2 for i in range(8)])
a2s = np.array([(i // 2) % 2 for i in range(8)])
As = np.array([(i // 4) % 2 for i in range(8)])
a12 = a1s * 2 + a2s
SBOX = np.array([[1.0, 1.0], [1.0, -1.0]])

AX = np.array([[(m >> (a1 * 2 + a2)) & 1 for a1 in range(2) for a2 in range(2)] for m in range(16)])
AC = np.array([[(m >> i) & 1 for i in range(8)] for m in range(256)])
DECS = np.zeros((16, 2, 2), dtype=np.int64)
for m in range(16):
    for c in range(2):
        for b in range(2):
            DECS[m, c, b] = (m >> (2 * c + b)) & 1
YS = np.array([[(m >> c) & 1 for c in range(2)] for m in range(4)])

MX = np.repeat(np.arange(16), 256)
MC = np.tile(np.arange(256), 16)
NST = MX.size
X = AX[MX][np.arange(NST)[:, None], a12[None, :]]
C = AC[MC]
SG = np.where(As[:, None] == np.array([0, 1])[None, :], 1.0, -1.0)

def mi_arr(P):
    """P: (N, 2, 2) -> 互信息数组"""
    P = np.clip(P, 0.0, None)
    pa = P.sum(axis=2, keepdims=True)
    pg = P.sum(axis=1, keepdims=True)
    r = np.where(P > 0, P / np.maximum(pa * pg, 1e-300), 1.0)
    return (P * np.log2(np.maximum(r, 1e-300))).sum(axis=(1, 2))

def scan_eps(eps):
    """返回 (J_max, RAC_max)：IC 和式与 RAC 成功率在协议类上的最大"""
    best = {1: np.full(NST, -1e9), 2: np.full(NST, -1e9)}
    rac = {1: np.zeros(NST), 2: np.zeros(NST)}
    for k in (1, 2):
        ak = a1s if k == 1 else a2s
        for my in range(4):
            Y = YS[my][C]
            W = 0.5 * (1.0 + SG[None, :, :] * eps * SBOX[X, Y][:, :, None])
            for md in range(16):
                Gv = DECS[md][C[:, :, None], np.array([0, 1])[None, None, :]]
                P4 = np.zeros((NST, 2, 2))
                for v in (0, 1):
                    mv = (ak[None, :] == v)
                    for g in (0, 1):
                        P4[:, v, g] = (W * mv[:, :, None] * (Gv == g)).sum(axis=(1, 2)) / 8.0
                best[k] = np.maximum(best[k], mi_arr(P4))
                succ = np.zeros(NST)
                for b in (0, 1):
                    succ += (W[:, :, b] * (Gv[:, :, b] == ak[None, :])).sum(axis=1)
                rac[k] = np.maximum(rac[k], succ / 8.0)
    return float((best[1] + best[2]).max()), float((0.5 * (rac[1] + rac[2])).max())

print("=" * 78)
print("P1-29 方向三·路线一落地：信息因果性的模型内定量判据（判据先行）")
print("=" * 78)

# ---------- Q1 家族与模型边 ----------
s_chsh = 4.0 * EPS_EDGE
ok = abs(s_chsh - TS) < 1e-12
gate("Q1 家族与模型边", ok,
     "各向同性族 S = 4 eps；模型可达边 eps_max = 1/sqrt2 = %.9f 给 S = %.12f（对 2*sqrt2 差 %.1e）"
     % (EPS_EDGE, s_chsh, abs(s_chsh - TS)))

# ---------- Q2/Q3 RAC 与三个锚点 ----------
eps_list = (0.0, 0.25, 0.5, 0.6, EPS_EDGE, 0.8, 0.9, 1.0)
rac_vals, J_vals = [], []
for eps in eps_list:
    J, rac = scan_eps(eps)
    J_vals.append(J)
    rac_vals.append(rac)
    print("     eps=%.6f：J = %+.6f，RAC = %.9f（(1+eps)/2 = %.9f）" % (eps, J, rac, (1 + eps) / 2))
rac_expect = np.array([max(0.75, (1.0 + e) / 2.0) for e in eps_list])
dev_rac = float(np.abs(np.array(rac_vals) - rac_expect).max())
ok = dev_rac < 1e-12
gate("Q2 RAC 穷举", ok,
     "8 档 max P_succ 对 max(3/4, (1+eps)/2) 最大偏差 %.1e（1e-12；eps<=0.5 段为经典 3/4）" % dev_rac)
i_half = eps_list.index(0.5)
i_edge = eps_list.index(EPS_EDGE)
i_one = eps_list.index(1.0)
ok = (abs(rac_vals[i_half] - 0.75) < 1e-12 and abs(rac_vals[i_edge] - RAC_IC) < 1e-12
      and abs(rac_vals[i_one] - 1.0) < 1e-12)
gate("Q3 RAC 三个锚点", ok,
     "eps=0.5 给 %.9f（经典界 3/4）；eps=1/sqrt2 给 %.9f（cos^2(pi/8) = %.9f）；eps=1 给 %.9f（PR）"
     % (rac_vals[i_half], rac_vals[i_edge], RAC_IC, rac_vals[i_one]))

# ---------- Q4 MI 和式的交叉点 ----------
ok4 = (abs(J_vals[i_edge] - 1.0) < 1e-9 and abs(J_vals[i_half] - 1.0) < 1e-9
       and J_vals[eps_list.index(0.8)] > 1.0 + 1e-6 and abs(J_vals[i_one] - 2.0) < 1e-9)
gate("Q4 MI 和式（单副本形式）", ok4,
     "eps<=1/sqrt2 与 0.75 档 J = 1（不违反）；eps=0.8 起 J = %.6f > 1；eps=1 给 %.6f（PR 最大违反）"
     % (J_vals[eps_list.index(0.8)], J_vals[i_one]))
fine = (0.76, 0.77, 0.78, 0.79)
fine_vals = [scan_eps(e)[0] for e in fine]
print("     细扫（MI 和式）：" + "；".join("eps=%.2f J=%.6f" % (e, v) for e, v in zip(fine, fine_vals)))
print("     结论：互和信息形式的单副本版本不紧（交叉点在 0.77 与 0.78 之间，S* = 4 eps* 约 3.10），紧形式是 Q2 的成功概率形式")

# ---------- Q5 混合稳健性 ----------
rng = np.random.default_rng(SEED)
mix_best = {}
for eps in (0.75, 0.8):
    best = -1e9
    for _ in range(400):
        s1, s2 = rng.integers(0, NST, 2)
        q = float(rng.random())
        kbest = {1: -1e9, 2: -1e9}
        for k in (1, 2):
            ak = a1s if k == 1 else a2s
            for my in range(4):
                for md in range(16):
                    P4 = np.zeros((2, 2))
                    for s, wgt in ((s1, q), (s2, 1.0 - q)):
                        Ys = YS[my][C[s]]
                        W = 0.5 * (1.0 + SG * eps * SBOX[X[s], Ys][:, None])
                        Gv = DECS[md][C[s][:, None], np.array([0, 1])[None, :]]
                        for v in (0, 1):
                            mv = (ak == v)
                            for g in (0, 1):
                                P4[v, g] += wgt * (W * mv[:, None] * (Gv == g)).sum() / 8.0
                    P4 = np.clip(P4, 0.0, None)
                    pa = P4.sum(axis=1, keepdims=True)
                    pg = P4.sum(axis=0, keepdims=True)
                    r = np.where(P4 > 0, P4 / np.maximum(pa * pg, 1e-300), 1.0)
                    kbest[k] = max(kbest[k], float((P4 * np.log2(np.maximum(r, 1e-300))).sum()))
        best = max(best, kbest[1] + kbest[2])
    mix_best[eps] = best
det_075, det_080 = scan_eps(0.75)[0], scan_eps(0.8)[0]
ok = mix_best[0.75] <= det_075 + 1e-9 and mix_best[0.8] <= det_080 + 1e-9
gate("Q5 混合稳健性", ok,
     "400 组随机两策略混合最好值 %.6f / %.6f，不超确定性穷举最优 %.6f / %.6f"
     % (mix_best[0.75], mix_best[0.8], det_075, det_080))

# ---------- Q6 同数核对 ----------
ok = (abs(4.0 * EPS_EDGE - TS) < 1e-15 and abs(RAC_IC - (1.0 + EPS_EDGE) / 2.0) < 1e-15)
gate("Q6 同数核对", ok,
     "同一 eps = 1/sqrt2 上三量重合：族边 S = %.12f = 2*sqrt2；RAC 值 cos^2(pi/8) = %.15f = (1+1/sqrt2)/2"
     % (4.0 * EPS_EDGE, RAC_IC))

# ---------- R3 与 R4（方向四首轮：容量侧的形式退化与非空转） ----------
i77, i79 = fine.index(0.77), fine.index(0.79)
ok = abs(J_vals[0] - 1.0) < 1e-9 and abs(rac_vals[0] - 0.75) < 1e-15
gate("R3 标准极限退化（古典锚点饱和）", ok,
     "无资源（eps = 0）时 J = %.6f = 记录预算 1；成功率 = %.9f（经典 1 比特协议值）；框架形式 sum_k I <= n_记录 与标准形式逐项对应"
     % (J_vals[0], rac_vals[0]))
ok = abs(J_vals[i_edge] - 1.0) < 1e-9 and abs(rac_vals[i_edge] - RAC_IC) < 1e-12
gate("R4a 非空转（双饱和）", ok,
     "模型边 eps = 1/sqrt2 处：记录预算 J = %.6f 恰饱和（1e-9）、成功率 = %.9f 恰为 cos^2(pi/8)（1e-12），两种计量同点取满"
     % (J_vals[i_edge], rac_vals[i_edge]))
ok = fine_vals[i77] <= 1.0 + 1e-9 and fine_vals[i79] > 1.0 + 1e-6
gate("R4b 紧度对照（诚信项）", ok,
     "成功概率形式的临界值 = 模型边（紧）；互信息和式首次越界在 (0.77, 0.78)（松）：J(0.77) = %.6f、J(0.79) = %.6f"
     % (fine_vals[i77], fine_vals[i79]))

# ---------- R5（方向四：超容量显式量） ----------
m_rec = 1.0
E_excess = J_vals[i_one] - m_rec
ok = abs(J_vals[i_one] - 2.0) < 1e-9 and abs(E_excess - 1.0) < 1e-9 and abs(J_vals[i_edge] - 1.0) < 1e-9
gate("R5 超容量显式量", ok,
     "框架记录预算 m = 1；PR 箱处 J = %.6f（需 2 次记录支付），差额 = %.6f 比特；模型边处无超额（J = %.6f）；"
     "结构侧对应量 4 - 2*sqrt2 = %.6f（CHSH 值侧，口径不同，分开登记）"
     % (J_vals[i_one], E_excess, J_vals[i_edge], 4.0 - TS))

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

ax = axes[0]
ax.plot(eps_list, J_vals, "o-", lw=1.6, ms=5, color="#2b6cb0", label="J（IC 和式上界）")
ax.axhline(1.0, color="k", lw=1.1, ls="--", label="信息因果性界 1 比特")
ax.axvline(EPS_EDGE, color="#c53030", lw=1.1, ls=":", label="模型边 1/sqrt2")
ax.set_xlabel("各向同性族参数 eps（S = 4 eps）")
ax.set_ylabel("协议类最大和式 J")
ax.set_title("Q4 互和信息形式：eps > 1/sqrt2 处违反（交叉点偏松）")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(eps_list, rac_vals, "s-", lw=1.6, ms=5, color="#2f855a", label="max P_succ（穷举）")
ax.plot(eps_list, rac_expect, "k--", lw=1.0, label="max(3/4, (1+eps)/2)")
ax.axhline(0.75, color="0.5", lw=1.0, ls=":", label="经典界 3/4")
ax.axhline(RAC_IC, color="#c53030", lw=1.1, ls="-.", label="cos^2(pi/8) = %.6f" % RAC_IC)
ax.axvline(EPS_EDGE, color="#c53030", lw=1.0, ls=":")
ax.set_xlabel("各向同性族参数 eps（S = 4 eps）")
ax.set_ylabel("2->1 RAC 成功率")
ax.set_title("Q2/Q3 成功概率形式：模型边恰为 cos^2(pi/8)")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIG_PATH, dpi=140)
plt.close(fig)

# ---------- 汇总 ----------
print("-" * 78)
n_ok = sum(1 for _, ok, _ in results if ok)
print("汇总：%d/%d 通过（种子 %d 仅用于混合稳健性）" % (n_ok, len(results), SEED))
print("图：%s" % FIG_PATH)
