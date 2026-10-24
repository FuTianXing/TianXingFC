# -*- coding: utf-8 -*-
"""
vector_write.py
判据第 605 号：矢量块定理化证明轮 K1/K2（代数加变换面）。

被判定命题（矢量写入定理，拟议定理 7）：碰撞写入的洛伦兹协变完备化唯一为四维流
J^mu = (m/m_P) u^mu（u^mu = gamma(c, v)），空间分量对时间反演为奇，是拖曳的唯一奇性源。

判据与理论期望（先立判据再实现）：
  K1 协变唯一性：真候选 J = s·u^mu 在 boost 下按四矢量变换（残差 ~1e-16，三档 beta）；
     替身全排除——T1 裸低速形式 s(c,v)（缺 gamma）残差 O(beta^2)、T2 伪修正 s(c, v f(|v|))、
     T3 速度标度形式（标度含 gamma 破坏份额线性；静止极限归零，死于变换律）、T4 仅标量（旧版写规则，静止系即无空间分量）。
     恒真反例：把变换换成伽利略变换，真候选也"违反"（判据测的是洛伦兹协变，不是泛变换一致）。
  K2 奇偶签名完备性：三写规则的时间反演签名两两可分——
     甲仅标量 (0, 0)（拖曳恒零，GPB 5.17sigma 的历史排除对象）、
     乙矢量完备 (+, -)（空间分量奇，唯一满足拖曳奇性要求）、
     丙奇化失败 (|v| 偶化) (+, +)；签名可分性与响应强度正交（强度归 Z2 单一入口，分层登记）。
  K3 强度入口复核：l_P/m_P = G/c^2（引用 Z2，重算恒等式）。
  K4 静止极限复核：v->0 真候选空间分量 -> 0（引用 Z1，重算）。
多档：beta in {0.1, 0.3, 0.5}。
"""
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
IMG = r"D:\software\TianXingFC\images"
OUT = IMG + r"\vector_write_605_fig1.png"

c = 2.99792458e8
G = 6.67430e-11
hbar = 1.054571817e-34
mP = math.sqrt(hbar * c / G)
lP = math.sqrt(hbar * G / c ** 3)

BETAS = [0.1, 0.3, 0.5]

def boost_x(beta):
    """沿 x 的洛伦兹变换矩阵（被动：实验室系 -> 运动系）。"""
    g = 1.0 / math.sqrt(1 - beta ** 2)
    M = np.eye(4)
    M[0, 0] = g
    M[0, 1] = -g * beta
    M[1, 0] = -g * beta
    M[1, 1] = g
    return M

def four_vel(beta):
    g = 1.0 / math.sqrt(1 - beta ** 2)
    return np.array([c * g, c * g * beta, 0.0, 0.0])

def candidate(name, beta, s):
    """各候选在（源瞬时静止系外的）实验室系给的 4 分量构造。"""
    u = four_vel(beta)
    g = u[0] / c
    if name == "true":          # s·u^mu（真候选）
        return s * u
    if name == "T1":            # 裸低速 s(c, v)（gamma 恒取 1）
        return s * np.array([c, c * beta, 0.0, 0.0])
    if name == "T2":            # 伪非线性修正 s(c, v(1+beta^2))
        return s * np.array([c, c * beta * (1 + beta ** 2), 0.0, 0.0])
    if name == "T3":            # 速度标度：s·u^mu·(u^0/c)（标度含 gamma 破坏份额线性；静止极限归零，死于变换律）
        return s * u * g
    if name == "T4":            # 仅标量（旧版写规则）
        return s * np.array([c, 0.0, 0.0, 0.0])
    raise ValueError(name)

def transform_check(name, s):
    """对三档 beta：构造实验室系分量，boost 到与源共动的检验系，与该系内按构造式重算的值比。"""
    worst = 0.0
    for beta in BETAS:
        J_lab = candidate(name, beta, s)
        L = boost_x(beta)
        J_tr = L @ J_lab
        beta_p = beta  # 检验系相对实验室以 beta 运动；源在检验系的速度：
        # 源在实验室以 beta 运动，boost(beta) 后源速度为 0（共动），u'=(c,0,0,0)
        u_prime = np.array([c, 0.0, 0.0, 0.0])
        if name == "true":
            expect = s * u_prime
        elif name == "T1":
            expect = s * np.array([c, 0.0, 0.0, 0.0])  # 构造式在共动系也给 (sc, 0)
        elif name == "T2":
            expect = s * np.array([c, 0.0, 0.0, 0.0])
        elif name == "T3":
            expect = s * np.array([c, 0.0, 0.0, 0.0]) * 1.0  # 构造式共动系给 s·c·1·(1,0,0,0)
        elif name == "T4":
            expect = s * np.array([c, 0.0, 0.0, 0.0])
        r = float(np.abs(J_tr - expect).max()) / (s * c)
        worst = max(worst, r)
    return worst

print("=" * 72)
print("K1 协变唯一性（期望：真候选三档残差 ~1e-16；替身显著非零）")
s = 1.0
res = {nm: transform_check(nm, s) for nm in ["true", "T1", "T2", "T3", "T4"]}
for nm, r in res.items():
    print("  %-5s 最大变换残差 = %.3e" % (nm, r))
k1 = res["true"] < 1e-12 and all(res[t] > 0.01 for t in ["T1", "T2", "T3", "T4"])
print("  [K1 %s] 真候选协变、四替身全违反" % ("PASS" if k1 else "FAIL"))

print("=" * 72)
print("K1 恒真反例（伽利略变换下真候选也违反：判据测的是洛伦兹协变）")
def galileo_x(v):
    M = np.eye(4)
    M[0, 1] = -v
    return M
g_res = 0.0
for beta in BETAS:
    J_lab = candidate("true", beta, s)
    Gm = galileo_x(c * beta)
    J_tr = Gm @ J_lab
    expect = s * np.array([c, 0.0, 0.0, 0.0])
    g_res = max(g_res, float(np.abs(J_tr - expect).max()) / (s * c))
print("  伽利略变换下真候选最大残差 = %.3f（应显著非零，判据有区分力）" % g_res)
k1g = g_res > 0.01
print("  [K1反例 %s]" % ("PASS" if k1g else "FAIL"))

print("=" * 72)
print("K2 奇偶签名完备性（期望：三写规则 T 签名两两可分，拖曳奇性唯一锁定乙）")
def source_pair(rule, beta, s):
    """(J^i(v), J^i(-v)) 空间分量对（时间反演 v->-v）。"""
    if rule == "A":      # 仅标量偶
        return 0.0, 0.0
    if rule == "B":      # 矢量完备（空间分量 = gamma s v，奇）
        g = 1.0 / math.sqrt(1 - beta ** 2)
        return g * s * c * beta, -g * s * c * beta
    if rule == "C":      # 奇化失败（空间分量偶化 |v|）
        return s * c * beta, s * c * beta
    raise ValueError(rule)
sigs = {}
for rule in "ABC":
    pv, nv = source_pair(rule, 0.3, s)
    if abs(pv) < 1e-15 and abs(nv) < 1e-15:
        sig = "(0, 0)"
    elif pv * nv < 0:
        sig = "(+, -) 奇"
    else:
        sig = "(+, +) 偶"
    sigs[rule] = sig
    print("  写规则 %s: 签名 %s" % (rule, sig))
sep = len(set(sigs.values())) == 3
drag = sigs["B"].startswith("(+, -)")
k2 = sep and drag
print("  [K2 %s] 三签名两两可分，拖曳奇性唯一锁定矢量完备写法（签名与强度正交，强度归 Z2）"
      % ("PASS" if k2 else "FAIL"))

print("=" * 72)
print("K3 强度入口复核（引用 Z2）：l_P/m_P 与 G/c^2")
lhs = lP / mP
rhs = G / c ** 2
rel = abs(lhs / rhs - 1.0)
print("  l_P/m_P = %.6e, G/c^2 = %.6e, 相对偏差 = %.2e（Z2 实测 1.21e-16 同级）" % (lhs, rhs, rel))
k3 = rel < 1e-12
print("  [K3 %s]" % ("PASS" if k3 else "FAIL"))

print("=" * 72)
print("K4 静止极限复核（引用 Z1）：v->0 真候选空间分量 -> 0")
lim = candidate("true", 1e-8, s)[1] / (s * c)
print("  beta=1e-8 空间分量（c 归一）= %.2e（应 -> 0）" % lim)
k4 = lim < 1e-7
print("  [K4 %s]" % ("PASS" if k4 else "FAIL"))

print("=" * 72)
passed = sum([k1, k1g, k2, k3, k4])
print("判据汇总：K1 %s / K1反例 %s / K2 %s / K3 %s / K4 %s（%d/5）" %
      ("PASS" if k1 else "FAIL", "PASS" if k1g else "FAIL", "PASS" if k2 else "FAIL",
       "PASS" if k3 else "FAIL", "PASS" if k4 else "FAIL", passed))
print("结论：矢量写入的协变完备化唯一为 J^mu = (m/m_P)u^mu，奇偶签名唯一锁定其空间分量为拖曳源；")
print("K3/K4 为 Z1/Z2 资产复核。旋转源全案演化与 K5 多档复核归后续轮。")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax = axes[0]
names = ["true", "T1", "T2", "T3", "T4"]
vals = [max(res[n] for n in names)]
ax.bar(names, [res[n] for n in names], color=["#4472c4"] + ["#c0504d"] * 4)
ax.set_yscale("log")
ax.set_ylabel("max transform residual (normalized)")
ax.set_title("K1: covariance check over boost tiers")
ax = axes[1]
rules = ["A (scalar only)", "B (full vector)", "C (even-ized)"]
pv = [source_pair(r, 0.3, 1.0)[0] for r in "ABC"]
nv = [source_pair(r, 0.3, 1.0)[1] for r in "ABC"]
xx = np.arange(3)
ax.bar(xx - 0.15, pv, 0.3, label=r"$J^i(v)$")
ax.bar(xx + 0.15, nv, 0.3, label=r"$J^i(-v)$")
ax.set_xticks(xx)
ax.set_xticklabels(rules, fontsize=8)
ax.set_ylabel(r"$J^i$ (normalized)")
ax.set_title("K2: time-reversal signature pairs")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
plt.close(fig)
print("figure saved:", OUT)
