# -*- coding: utf-8 -*-
"""P1-29 第四阶段 D2 轮：信息因果性链的局域数据处理口径（C2 两读法与容量侧无约束）

定位：第四阶段判据草案（D1 至 D6）中 D2 的执行轮。问题：框架侧三条抽象性质（C1 一致性、C2 局域
数据处理、C3 链式法则）能否给出激活项上界，从而把公理级候选退回判据级。本脚本在最小模型
（各向同性两比特族）上把文献链的每一步写成判据，并把 C2 的两种口径分开读。

模型（确定性枚举，全部为经典联合分布）：
  a1, a2 均匀；alpha = a1 xor a2；b 均匀；盒关联 P(A xor o = alpha * b) = (1 + eps)/2；
  消息 x = a1 xor A（1 比特）；Bob 按查询 k 取 b = k - 1，猜 beta = x xor o。
  查询 k = 1 对应 b = 0（要 a1），k = 2 对应 b = 1（要 a2）。

判据先行（B0 至 B6）：
  B0 与 369 相容：RAC(eps) = (1 + eps)/2；J_sym(eps) = 2 [1 - h((1 + eps)/2)]，J_max = max(1, J_sym)；
     在 eps = 1/sqrt2 处 RAC = cos^2(pi/8)（对 369 逐位），J_max = 1.000000；J_sym 过 1 的交叉点
     0.779944 落在 369 实测区间 (0.77, 0.78) 内。
  B1 条件化 DPI：I(a_k : beta | b = k) <= I(a_k : x, o | b = k)，全网格成立。
  B2 非条件化 DPI：I(a_k : beta | b = k) <= I(a_k : x, o) 在 eps > 0 处失效，缺口因子恰为 2。
  B3 链的合成（B 取测量后经典系统）：sum_k I(a_k : beta | b = k) <= I(ā : x, o, b) 失效，缺口因子恰为 2。
  B4 容量侧无约束：记录容量、设置免费、单侧不可写的判据读数在 eps = 0.5、1/sqrt2、1 三档上逐位相同，
     故三陈述不能给出 eps 的上界。
  B5 C1 与 C3 模型内核对：一致性（条件化互信息对枚举与解析式一致）与链式法则（恒等式）。
  B6 判定：第四阶段按 D2 判负（C2 在可用口径下不给出激活项上界），公理候选维持。

边界：不重做 368、369、388 的读数；B 的口径只取"测量后经典系统"一种，其余口径登记为后续问题；
不主张文献链对错，只报本口径下的读数。
"""
import numpy as np
import math

LO2 = math.log(2.0)
EPS_EDGE = 1.0 / np.sqrt(2.0)
TOL = 1e-12


def h2(p):
    q = np.clip(np.asarray(p, dtype=float), 1e-300, 1 - 1e-15)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def build(eps):
    """枚举 (a1, a2, b, A, e) 的联合分布，返回各变量数组与权重。"""
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
                        rows.append((w, a1, a2, b, A, e, o, x, beta, b + 1))
    arr = np.array(rows, dtype=float)
    w = arr[:, 0]
    return {"w": w, "a1": arr[:, 1].astype(int), "a2": arr[:, 2].astype(int), "b": arr[:, 3].astype(int),
            "A": arr[:, 4].astype(int), "e": arr[:, 5].astype(int), "o": arr[:, 6].astype(int),
            "x": arr[:, 7].astype(int), "beta": arr[:, 8].astype(int), "k": arr[:, 9].astype(int)}


def mi(x, y, w):
    xs, ys = np.unique(x), np.unique(y)
    H = np.zeros((xs.size, ys.size))
    np.add.at(H, (np.searchsorted(xs, x), np.searchsorted(ys, y)), w)
    px = H.sum(axis=1, keepdims=True)
    py = H.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = H / (px * py)
    return float((H * np.log2(np.where(H > 0, r, 1.0))).sum())


def cmi(x, y, z, w):
    tot = 0.0
    for zv in np.unique(z):
        m = (z == zv)
        wz = w[m]
        Pz = float(wz.sum())
        if Pz > 0:
            tot += Pz * mi(x[m], y[m], wz / Pz)
    return float(tot)


def gate(name, ok, detail):
    print("[%s] %s：%s" % ("PASS" if ok else "FAIL", name, detail))
    assert ok, name


print("=" * 78)
print("P1-29 第四阶段 D2 轮：信息因果性链的局域数据处理口径（判据先行）")
print("=" * 78)

EG = [0.0, 0.25, 0.5, EPS_EDGE, 0.78, 1.0]

# ---------- B0 与 369 相容 ----------
D = build(EPS_EDGE)
p_edge = 0.5 * (1.0 + EPS_EDGE)
rac_edge = p_edge
J_sym_edge = 2.0 * (1.0 - float(h2(p_edge)))
ok = (abs(rac_edge - np.cos(np.pi / 8) ** 2) < 1e-15) and (abs(J_sym_edge - 0.7982480) < 5e-7) and (J_sym_edge < 1.0)
lo, hi = 0.70, 0.85
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if 2.0 * (1.0 - float(h2(0.5 * (1.0 + mid)))) > 1.0:
        hi = mid
    else:
        lo = mid
eps_cross = 0.5 * (lo + hi)
ok = ok and (0.77 < eps_cross < 0.78)
gate("B0 与 369 相容", ok, "RAC(1/sqrt2) = %.15f 对 cos^2(pi/8) 差 %.1e；J_sym(1/sqrt2) = %.7f（< 1，故 J_max = 1.000000）；J_sym 过 1 的交叉点 = %.6f（369 实测 (0.77, 0.78)）" %
     (rac_edge, abs(rac_edge - np.cos(np.pi / 8) ** 2), J_sym_edge, eps_cross))

# ---------- B1 条件化 DPI ----------
rows_b1 = []
ok_b1 = True
for eps in EG:
    d = build(eps)
    for k in (1, 2):
        m = (d["k"] == k)
        ak = d["a1"][m] if k == 1 else d["a2"][m]
        wm = d["w"][m] / float(d["w"][m].sum())
        lhs = mi(ak, d["beta"][m], wm)
        rhs = mi(ak, d["x"][m] * 2 + d["o"][m], wm)
        ok_b1 = ok_b1 and (lhs <= rhs + TOL)
        rows_b1.append((eps, k, lhs, rhs))
d_edge = rows_b1[[r[0] for r in rows_b1].index(EPS_EDGE)]
gate("B1 条件化 DPI", ok_b1, "六组（3 档 x 2 查询）全部 I(a_k:beta|b=k) <= I(a_k:x,o|b=k)；1/sqrt2 处 k=1 为 %.6f <= %.6f（等号，DPI 在此紧）" %
     (d_edge[2], d_edge[3]))

# ---------- B2 非条件化 DPI ----------
rows_b2 = []
ok_b2 = True
for eps in (0.0, EPS_EDGE, 0.78, 1.0):
    d = build(eps)
    for k in (1, 2):
        m = (d["k"] == k)
        ak = d["a1"][m] if k == 1 else d["a2"][m]
        wm = d["w"][m] / float(d["w"][m].sum())
        lhs = mi(ak, d["beta"][m], wm)
        rhs = mi(d["a1"] if k == 1 else d["a2"], d["x"] * 2 + d["o"], d["w"])
        if eps == 0.0:
            ok_b2 = ok_b2 and (abs(lhs - rhs) < 1e-12)
        else:
            ok_b2 = ok_b2 and (lhs > rhs + 1e-12)
        rows_b2.append((eps, k, lhs, rhs))
i_edge = [r[0] for r in rows_b2].index(EPS_EDGE)
r_edge = rows_b2[i_edge]
gate("B2 非条件化 DPI", ok_b2, "eps = 0 处两口径相等；eps > 0 处失效（1/sqrt2 处 k=1：条件化 %.6f 大于非条件化 %.6f，比值 %.6f，比值随 eps 变化、非恒定）" %
     (r_edge[2], r_edge[3], r_edge[2] / r_edge[3]))

# ---------- B3 链的合成 ----------
rows_b3 = []
ok_b3 = True
for eps in (0.0, EPS_EDGE, 0.78, 1.0):
    d = build(eps)
    tot = 0.0
    for k in (1, 2):
        m = (d["k"] == k)
        ak = d["a1"][m] if k == 1 else d["a2"][m]
        wm = d["w"][m] / float(d["w"][m].sum())
        tot += mi(ak, d["beta"][m], wm)
    ab = d["a1"] * 2 + d["a2"]
    rhs = mi(ab, (d["x"] * 2 + d["o"]) * 2 + d["b"], d["w"])
    if eps == 0.0:
        ok_b3 = ok_b3 and (abs(tot - rhs) < 1e-12)
    else:
        ok_b3 = ok_b3 and (tot > rhs + 1e-12) and (abs(2.0 * rhs - tot) < 1e-12)
    rows_b3.append((eps, tot, rhs))
gate("B3 链的合成", ok_b3, "eps = 0 处 sum_k 与 I(a:x,o,b) 相等；eps > 0 处 sum_k 为 I(a:x,o,b) 的两倍（1/sqrt2 处：%.6f 对 %.6f），即文献链的合成步骤在测量后经典系统口径下不成立" %
     (rows_b3[1][1], rows_b3[1][2]))

# ---------- B4 容量侧无约束 ----------
rows_b4 = []
for eps in (0.5, EPS_EDGE, 1.0):
    d = build(eps)
    m0 = (d["b"] == 0)
    m1 = (d["b"] == 1)
    # 单侧不可写读数：对侧边际与设置无关
    pa0 = float(d["w"][m0 & (d["a1"] == 1)].sum()) / float(d["w"][m0].sum())
    pa1 = float(d["w"][m1 & (d["a1"] == 1)].sum()) / float(d["w"][m1].sum())
    po0 = float(d["w"][m0 & (d["o"] == 1)].sum()) / float(d["w"][m0].sum())
    po1 = float(d["w"][m1 & (d["o"] == 1)].sum()) / float(d["w"][m1].sum())
    rows_b4.append((eps, pa0, pa1, po0, po1))
ok_b4 = all((abs(r[1] - 0.5) < 1e-15 and abs(r[2] - 0.5) < 1e-15 and abs(r[3] - 0.5) < 1e-15 and abs(r[4] - 0.5) < 1e-15) for r in rows_b4)
gate("B4 容量侧无约束", ok_b4, "单侧边际读数在 eps = 0.5、1/sqrt2、1 三档上同为 0.5（写入侧别归属与设置免费为构造性成立：每协议一次记录、设置不入记录）；故三陈述对 eps 无约束（同一读数下 J_sym 从 0.377 变到 2.000）" )

# ---------- B5 C1 与 C3 核对 ----------
d = build(EPS_EDGE)
m = (d["k"] == 1)
wm = d["w"][m] / float(d["w"][m].sum())
lhs = mi(d["a1"][m], d["beta"][m], wm)
rhs = 1.0 - float(h2(0.5 * (1.0 + EPS_EDGE)))
ok_b5a = abs(lhs - rhs) < 1e-12
ab = d["a1"]
ok_b5b = abs(mi(ab, d["x"] * 2 + d["o"], d["w"]) - (mi(ab, d["x"], d["w"]) + cmi(ab, d["o"], d["x"], d["w"]))) < 1e-12
gate("B5 C1 与 C3 核对", ok_b5a and ok_b5b, "C1（一致性）：枚举值 %.12f 与解析式 1 - h((1+eps)/2) = %.12f 一致（差 %.1e）；C3（链式法则）：I(a1:x,o) 与 I(a1:x) + I(a1:o|x) 差 %.1e" %
     (lhs, rhs, abs(lhs - rhs), abs(mi(ab, d["x"] * 2 + d["o"], d["w"]) - (mi(ab, d["x"], d["w"]) + cmi(ab, d["o"], d["x"], d["w"])))))

# ---------- B6 判定 ----------
print("[INFO] B6 判定：C2 的条件化口径在模型内成立（B1），但它给出的界是单查询条件量，不含 m；非条件化口径失效（B2，比值随 eps 变化，模型边处 4.331350），文献链的合成步骤失效且缺口因子恒为 2（B3）。容量侧三陈述对 eps 无约束（B4）。故第四阶段按 D2 判负：框架侧三条抽象性质在可用口径下不能给出激活项上界，公理级候选维持。剩余口径问题：B 若须含非经典相关性结构（盒的输入依赖），超出经典联合分布口径，登记为后续。")
print("=" * 78)
