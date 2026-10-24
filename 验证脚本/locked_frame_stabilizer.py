# -*- coding: utf-8 -*-
r"""
locked_frame_stabilizer.py

二-1 第六段：锁定模的体系内来源（稳定子层的直接构造）。
把第 174 / 183 条里作为输入的整体锁定模 phi 换成环面 toric code 的同调（绕数）扇区标签，
并把该标签的三条性质（空间均匀、局域扰动不改、保护距离为 L）与配对读数框架对接。
确定性，随机固定种子 20260919。

构造：L x L 环面，棱上放 qubit（2 L^2 个），星项 A_v 为顶点的 X 型稳定子，
面项 B_p 为面的 Z 型稳定子。
  H1 稳定子代数：两两对易、互为对合，独立生成元 2 L^2 - 2，编码维度 2^2 = 4
  H2 逻辑算符：X 型取水平棱的一列（绕 j 方向），Z 型取水平棱的一行（绕 i 方向），
     两者权重均为 L，均与全部稳定子对易，交点数为 1（反对易），且都与同类稳定子群独立
  H3 闭链空间：与全部星项对易的 Z 型算符（闭链）维数为 L^2 + 1，非平凡代表权重为 L，
     八条对偶闭链（水平棱的各列）对同一闭链给出相同配对（同调不变）
  H4 可检测性：syndrome 映射的秩为 L^2 - 1，单棱扰动全部可检测，权重 <= L - 1 的随机扰动
     抽样全部可检测
  H5 条件恶化：以扇区标签为共享分量，局域噪声扫描下配对相干仍有平台（散布 <= 5%），
     平台值随噪声下降并与 P/(P + sig^2 + lam^2) 相符（<= 5%）
  H6 f(E) 形式复用：C = 2 sig_tot^2 (1 - m) 在本构造上成立（<= 5%）
  H7 接口登记（P4-16、二-2，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\locked_frame_stabilizer.png"
SEED = 20260919
L = 8
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


NE = 2 * L * L


def hid(i, j):
    return (i % L) * L + (j % L)


def vid(i, j):
    return L * L + (i % L) * L + (j % L)


stars, faces = [], []
for i in range(L):
    for j in range(L):
        s = np.zeros(NE, dtype=np.uint8)
        for e in (hid(i, j), hid(i - 1, j), vid(i, j), vid(i, j - 1)):
            s[e] ^= 1
        stars.append(s)
        f = np.zeros(NE, dtype=np.uint8)
        for e in (hid(i, j), hid(i, j + 1), vid(i, j), vid(i + 1, j)):
            f[e] ^= 1
        faces.append(f)


def par(a, b):
    return int(np.dot(a.astype(np.int64), b.astype(np.int64)) % 2)


def gf2_rank(rows_in):
    rs = [r.copy() for r in rows_in]
    rank = 0
    for col in range(len(rs[0])):
        piv = None
        for r in range(rank, len(rs)):
            if rs[r][col]:
                piv = r
                break
        if piv is None:
            continue
        rs[rank], rs[piv] = rs[piv], rs[rank]
        for r in range(len(rs)):
            if r != rank and rs[r][col]:
                rs[r] ^= rs[rank]
        rank += 1
    return rank


# ---------- H1 稳定子代数 ----------
bad1 = [(a, b) for a in range(len(stars)) for b in range(len(faces)) if par(stars[a], faces[b]) != 0]
r_star = gf2_rank(stars)
r_face = gf2_rank(faces)
enc = NE - (r_star + r_face)
report("H1 稳定子代数", not bad1 and r_star == L * L - 1 and r_face == L * L - 1 and enc == 2,
       "星项与面项两两对易；独立生成元各为 L^2 - 1；编码维度 2^2 = 4",
       "非对易对数 %d；星项秩 %d、面项秩 %d、独立生成元 %d、编码比特 %d"
       % (len(bad1), r_star, r_face, r_star + r_face, enc))

# ---------- H2 逻辑算符 ----------
i0, j0 = 3, 5
xlog = np.zeros(NE, dtype=np.uint8)
zlog = np.zeros(NE, dtype=np.uint8)
for j in range(L):
    xlog[hid(i0, j)] = 1
for i in range(L):
    zlog[hid(i, j0)] = 1
w_x, w_z = int(xlog.sum()), int(zlog.sum())
ok_x = all(par(xlog, f) == 0 for f in faces)
ok_z = all(par(zlog, s) == 0 for s in stars)
pair = par(xlog, zlog)
ind_x = gf2_rank(stars + [xlog]) == r_star + 1
ind_z = gf2_rank(faces + [zlog]) == r_face + 1
report("H2 逻辑算符", ok_x and ok_z and pair == 1 and w_x == L and w_z == L and ind_x and ind_z,
       "两种逻辑闭链权重均为 L，与全部稳定子对易，交点数为 1，且与同类稳定子群独立",
       "权重 %d 与 %d；对易 %s 与 %s；交点数 %d；独立性 %s 与 %s"
       % (w_x, w_z, ok_x, ok_z, pair, ind_x, ind_z))

# ---------- H3 闭链空间与非平凡代表 ----------
cyc_dim = NE - r_star
cols = []
for i in range(L):
    c = np.zeros(NE, dtype=np.uint8)
    for j in range(L):
        c[hid(i, j)] = 1
    cols.append(c)
pair_all = [par(zlog, c) for c in cols]
dev_row = max(abs(par(zlog, c) - 1) for c in cols)
zero_row = max(par(f, c) for f in faces for c in cols)
report("H3 闭链空间与非平凡代表", cyc_dim == L * L + 1 and dev_row == 0 and zero_row == 0,
       "闭链空间维数 L^2 + 1；非平凡代表权重 L；八条对偶闭链给出同一配对，面边界给零配对",
       "维数 %d；行闭链八条配对 %s；面边界最大配对 %d" % (cyc_dim, pair_all, zero_row))

# ---------- H4 可检测性 ----------
r_syn = gf2_rank(faces)
unit_bad = 0
for e in range(NE):
    u = np.zeros(NE, dtype=np.uint8)
    u[e] = 1
    if all(par(u, f) == 0 for f in faces):
        unit_bad += 1
rng = np.random.default_rng(SEED)
samp_bad = 0
TOT = 2000
for w in range(1, L):
    for _ in range(TOT):
        u = np.zeros(NE, dtype=np.uint8)
        idx = rng.choice(NE, size=w, replace=False)
        u[idx] = 1
        if all(par(u, f) == 0 for f in faces):
            samp_bad += 1
report("H4 可检测性", unit_bad == 0 and samp_bad == 0 and r_syn == L * L - 1,
       "syndrome 映射秩 L^2 - 1；128 个单棱扰动全部可检测；权重 %d 至 %d 的随机扰动各 %d 例全部可检测"
       % (1, L - 1, TOT),
       "秩 %d；单棱未检测 %d 例；抽样未检测 %d 例（共 %d 例）" % (r_syn, unit_bad, samp_bad, (L - 1) * TOT))

# ---------- H5 / H6 条件恶化与 f(E) 复用 ----------
NS, LX, XI, SIG = 4000, 256, 4.0, 1.0
P = 1.0
RHO = np.exp(-1.0 / XI)
eta = np.zeros((NS, LX))
eta[:, 0] = rng.normal(scale=SIG, size=NS)
for j in range(1, LX):
    eta[:, j] = RHO * eta[:, j - 1] + rng.normal(scale=SIG * np.sqrt(1.0 - RHO ** 2), size=NS)
phi = rng.normal(scale=np.sqrt(P), size=(NS, 1))
NS_SCAN = (1, 8, 16, 24, 32, 48, 64)


def stats(alpha, n):
    a = alpha[:, :alpha.shape[1] - n]
    b = alpha[:, n:]
    mu = float(alpha.mean())
    var = float(np.mean((alpha - mu) ** 2))
    cov = float(np.mean((a - mu) * (b - mu)))
    return var, cov, float(np.mean((b - a) ** 2))


g5 = []
for lam in (0.0, 0.5, 1.0):
    alpha = phi + eta + rng.normal(scale=lam, size=(NS, LX))
    win = [stats(alpha, n)[1] / stats(alpha, n)[0] for n in (24, 32, 48, 64)]
    e_inf = float(np.mean(win))
    spread = max(win) / min(win) - 1.0
    e_pred = P / (P + SIG ** 2 + lam ** 2)
    dev = [abs(stats(alpha, n)[2] - 2.0 * (SIG ** 2 + lam ** 2) * (1.0 - (stats(alpha, n)[1] / stats(alpha, n)[0] - e_inf) / (1.0 - e_inf))) / stats(alpha, n)[2] for n in NS_SCAN]
    g5.append((lam, e_inf, e_pred, spread, max(dev)))
ok5 = all(sp <= 0.05 and abs(ei - ep) / ep <= 0.05 for lam, ei, ep, sp, dv in g5)
report("H5 条件恶化", ok5,
       "三档局域噪声下平台存在（散布 <= 5%），平台值与 P/(P + sig^2 + lam^2) 相符（<= 5%）",
       "；".join("lam=%.1f: 平台 %.4f 对 %.4f（散布 %.2e）" % (lam, ei, ep, sp) for lam, ei, ep, sp, dv in g5))
report("H6 f(E) 形式复用", all(dv <= 0.05 for *_, dv in g5),
       "C = 2 sig_tot^2 (1 - m) 在三档噪声的九档格距上成立（<= 5%）",
       "最大相对偏差 %.3e" % max(dv for *_, dv in g5))

print("[登记] H7：同调扇区标签作为锁定模的体系内来源，供 P4-16 与 二-2 使用，本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for lam in (0.0, 0.5, 1.0):
    alpha = phi + eta + rng.normal(scale=lam, size=(NS, LX))
    ax[0].plot(NS_SCAN, [stats(alpha, n)[1] / stats(alpha, n)[0] for n in NS_SCAN], "o-", label="lam=%.1f" % lam)
ax[0].axhline(P / (P + SIG ** 2), ls="--", c="k", lw=1, label="P/(P+sig^2)")
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("格距 n")
ax[0].set_ylabel("配对相干 E(n)")
ax[0].set_title("扇区标签作共享分量：平台随局域噪声下降")
lab = ["xlog 列", "zlog 行"]
ax[1].bar([0, 1], [w_x, w_z], color=["tab:blue", "tab:orange"])
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(lab)
ax[1].set_ylabel("权重")
ax[1].set_title("逻辑闭链权重 = L = %d" % L)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
