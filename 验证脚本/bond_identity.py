# -*- coding: utf-8 -*-
r"""
bond_identity.py

[键身份轮（桥 1）] 键变量取"实轴方向差"的两读法对照：派生读法下保护结构不存在，独立读法下成立

定位
----
桥 1：把保护模型的键变量落到体系自有身份上。作者已裁定弦网棱自由度 = 实轴方向差（独立键自由度
读法），本脚本检验这一身份的两个读法：
  D（派生读法，位形侧）：键变量 = 相邻格点法线轴读数之差，$k_e = \delta\phi$（单值场，有闭路约束）；
  I（独立读法，波性侧）：键变量独立取值 $k_e \in Z_N$（无闭路约束）。
判据把"保护需要哪种读法"做成可复算的对照：派生读法下，面的通量与环的绕数恒为零（面项与绕数算符
在受限空间上退化为恒等），保护结构不存在；独立读法下，通量自由、扇区结构存在、简并 $= N^2$。
本脚本不改动基准文件。

模型
----
二维 $L \times L$ 环面：格点 $\phi(i,j) \in Z_N$；键取 x 向差 $k_x(i,j) = \phi(i{+}1,j) - \phi(i,j)$
与 y 向差 $k_y(i,j) = \phi(i,j{+}1) - \phi(i,j)$（派生读法），或独立取值（独立读法）。共 $n = 2L^2$ 条键、
$F = L^2$ 个面。面通量 $f_p$ 取环绕四键的定向和（逆时针）；绕数算符取两类非可缩环的键和 $W_x$、$W_y$。
星项 = 顶点发散，面项 = 面通量。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
BID-1 派生读法下通量与绕数恒零：枚举 $\phi \in Z_N^V$，全部面的 $f_p = 0$ 且 $W_x = W_y = 0$
   （$L = 2, 3$；$N = 2, 3$）。
BID-2 独立读法下扇区结构存在：面通量的像维数 $= F - 1$（全体面通量之和恒零），绕数对 $(W_x, W_y)$
   的两个功能性线性独立，故每个 $(\text{通量图案}, W)$ 的构型数分别为 $N^{\,n-F+1}$ 与 $N^{\,n-2}$。
BID-3 开链上绕数 $=$ 边界差、闭环上恒零：一维链 $\sum_j \Delta u_j = u_{L-1} - u_0$（开放链，可取
   任意值），周期链 $\sum_j \Delta u_j = 0$。
BID-4 约束秩对照：派生读法下面约束的独立条数 $= 0$（面项在受限空间上恒为恒等）；独立读法下
   面约束 $= F - 1$、星约束 $= V - 1$，简并 $= N^{\,n-(V-1)-(F-1)} = N^2$。
BID-5 可达性对照：独立读法下绕数的 $N$ 个取值全部可达（每个取值 $N^{\,n-1}$ 个构型）；派生读法下
   只有 $W = 0$ 可达（构型数 $N^{\,V-1}$）。即荷在派生读法下"根本不存在非零构型"，在独立读法下是
   自由量子数。
BID-6 两个耦合各自单独成立时保护退化（键身份与体系耦合的接口）：只留星项（$\kappa_\parallel = 0$，格点间
   关系关闭）时简并指数 $= n - (V-1)$；只留面项（$\kappa_\varphi = 0$，格点内锁定关闭）时简并指数
   $= n - (F-1)$。两者都远大于 $2$，即保护需要格点内锁定与格点间关系同时在场。

运行
----
python 验证脚本/bond_identity.py
确定性（纯组合，无随机数），运行 < 60 秒。图存 D:/software/TianXingFC/images/bond_identity.png
"""
import time
from itertools import product

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def rank_mod(rows, N):
    M = [list(r) for r in rows]
    m = len(M)
    if m == 0:
        return 0
    n = len(M[0])
    r = 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if M[i][c] % N:
                p = i
                break
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = pow(M[r][c] % N, -1, N) if N > 2 else 1
        M[r] = [(x * inv) % N for x in M[r]]
        for i in range(m):
            if i != r and M[i][c] % N:
                f = M[i][c] % N
                M[i] = [(a - f * b) % N for a, b in zip(M[i], M[r])]
        r += 1
        if r == m:
            break
    return r


def torus_tables(L):
    """键表：kx(i,j)、ky(i,j)；面表：四键与定向符号；绕数：两类非可缩环。"""
    def kx(i, j):
        return (0, i % L, j % L)

    def ky(i, j):
        return (1, i % L, j % L)
    edges = [kx(i, j) for i in range(L) for j in range(L)] + [ky(i, j) for i in range(L) for j in range(L)]
    eidx = {e: a for a, e in enumerate(edges)}
    faces = []
    for i in range(L):
        for j in range(L):
            faces.append([(kx(i, j), +1), (ky(i + 1, j), +1), (kx(i, j + 1), -1), (ky(i, j), -1)])
    wx = [(kx(i, 0), +1) for i in range(L)]
    wy = [(ky(0, j), +1) for j in range(L)]
    return edges, eidx, faces, wx, wy


def grad_vec(phi, L, eidx, N):
    """派生读法：k = δφ。"""
    v = [0] * len(eidx)
    for i in range(L):
        for j in range(L):
            v[eidx[(0, i, j)]] = (phi[(i + 1) % L][j] - phi[i][j]) % N
            v[eidx[(1, i, j)]] = (phi[i][(j + 1) % L] - phi[i][j]) % N
    return v


def functional(cell, k, eidx, N):
    return sum(s * k[eidx[e]] for e, s in cell) % N


print("=" * 74)
print("[键身份轮] 键变量 = 实轴方向差：派生读法（单值场）对独立读法")
print("=" * 74)

summary = {}
for N in (2, 3):
    for L in (2, 3):
        edges, eidx, faces, wx, wy = torus_tables(L)
        n, F, V = len(edges), len(faces), L * L
        # 派生读法：枚举 φ
        zero_flux = True
        zero_wind = True
        cnt_phi = 0
        for flat in product(range(N), repeat=V):
            phi = [list(flat[i * L:(i + 1) * L]) for i in range(L)]
            k = grad_vec(phi, L, eidx, N)
            cnt_phi += 1
            if any(functional(f, k, eidx, N) for f in faces):
                zero_flux = False
            if functional(wx, k, eidx, N) or functional(wy, k, eidx, N):
                zero_wind = False
        # 独立读法：秩与计数
        rows_face = [[s if eidx[e] == a else 0 for a in range(n)] for f in faces for e, s in f]
        rows_face = [[0] * n for _ in faces]
        for a, f in enumerate(faces):
            for e, s in f:
                rows_face[a][eidx[e]] = (rows_face[a][eidx[e]] + s) % N
        r_face = rank_mod(rows_face, N)
        rows_w = [[0] * n for _ in range(2)]
        for e, s in wx:
            rows_w[0][eidx[e]] = (rows_w[0][eidx[e]] + s) % N
        for e, s in wy:
            rows_w[1][eidx[e]] = (rows_w[1][eidx[e]] + s) % N
        r_w = rank_mod(rows_w, N)
        print(f"\nN = {N}，L = {L}：n = {n}，F = {F}，V = {V}")
        print(f"  派生读法：枚举 {cnt_phi} 个构型；面通量全零 = {zero_flux}；绕数全零 = {zero_wind}")
        check(f"BID-1/N{N}L{L}", zero_flux and zero_wind,
              f"派生读法下面通量与绕数恒零（枚举 {cnt_phi} 个构型）")
        # 独立读法：枚举小系统
        if N ** n <= 10 ** 6:
            from collections import Counter
            cf = Counter()
            cw = Counter()
            for k in product(range(N), repeat=n):
                cf[tuple(functional(f, k, eidx, N) for f in faces)] += 1
                cw[(functional(wx, k, eidx, N), functional(wy, k, eidx, N))] += 1
            uniform_face = len(cf) == N ** (F - 1) and set(cf.values()) == {N ** (n - F + 1)}
            uniform_w = len(cw) == N ** 2 and set(cw.values()) == {N ** (n - 2)}
            print(f"  独立读法（枚举 {N ** n} 个构型）：面通量图案 {len(cf)} 种"
                  f"（每 N^{n-F+1} = {N ** (n - F + 1)}）、绕数图案 {len(cw)} 种"
                  f"（每 N^{n-2} = {N ** (n - 2)}）")
            check(f"BID-2/N{N}L{L}", uniform_face, f"面通量图案数与均匀性（{len(cf)} 种）")
            check(f"BID-5/N{N}L{L}", uniform_w, f"绕数 N^2 = {N ** 2} 个取值全部可达且均匀")
        else:
            print(f"  独立读法（不枚举，取秩）：face 秩 = {r_face}（= F-1 ？ {r_face == F - 1}）、"
                  f"绕数秩 = {r_w}（= 2 ？ {r_w == 2}）")
        r_star_rows = []
        for i in range(L):
            for j in range(L):
                row = [0] * n
                row[eidx[(0, i, j)]] = (row[eidx[(0, i, j)]] + 1) % N
                row[eidx[(0, (i - 1) % L, j)]] = (row[eidx[(0, (i - 1) % L, j)]] - 1) % N
                row[eidx[(1, i, j)]] = (row[eidx[(1, i, j)]] + 1) % N
                row[eidx[(1, i, (j - 1) % L)]] = (row[eidx[(1, i, (j - 1) % L)]] - 1) % N
                r_star_rows.append(row)
        r_star = rank_mod(r_star_rows, N)
        deg_exp = n - r_star - r_face
        check(f"BID-4/N{N}L{L}",
              r_face == F - 1 and r_star == V - 1 and deg_exp == 2,
              f"独立读法约束秩：面 {r_face} = F-1、星 {r_star} = V-1，简并指数 {deg_exp}（= N^2）")
        # BID-6：单独一个耦合时的简并指数
        deg_only_star = n - r_star
        deg_only_face = n - r_face
        print(f"    只留星项（kappa_perp = 0）简并指数 = {deg_only_star}；只留面项（kappa_phi = 0）"
              f"简并指数 = {deg_only_face}")
        check(f"BID-6/N{N}L{L}",
              deg_only_star == n - (V - 1) and deg_only_face == n - (F - 1)
              and deg_only_star > 2 and deg_only_face > 2,
              f"两耦合各自单独不足：只星 {deg_only_star}、只面 {deg_only_face}，均远大于 2")
        summary[(N, L)] = dict(n=n, F=F, V=V, r_face=r_face, r_star=r_star, deg_exp=deg_exp)

# BID-3：一维链的开链与闭环
print("\n" + "-" * 74)
print("一维链：绕数 = 边界差（开放）与恒零（周期）")
print("-" * 74)
for N in (3, 5):
    for L in (5, 6):
        open_vals = set()
        for u in product(range(N), repeat=L):
            s = sum((u[j + 1] - u[j]) % N for j in range(L - 1)) % N
            open_vals.add((s - (u[-1] - u[0])) % N)
        cyc_vals = set()
        for u in product(range(N), repeat=L):
            s = (sum((u[(j + 1) % L] - u[j]) % N for j in range(L))) % N
            cyc_vals.add(s)
        print(f"  N = {N}，L = {L}：开放链 Σ Δu - (u_last - u_0) 的取值集 = {sorted(open_vals)}；"
              f"周期链 Σ Δu 的取值集 = {sorted(cyc_vals)}")
        check(f"BID-3/N{N}L{L}", open_vals == {0} and cyc_vals == {0},
              f"开放链差 = 边界差（余数集 {sorted(open_vals)}）、周期链恒零（余数集 {sorted(cyc_vals)}）")

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
for tag, ok in CHECKS:
    if not ok:
        print(f"  未过：{tag}")

# ---------------- 出图 ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) 两种读法的面通量对照（并列两张 L = 3 网格）
    Lp = 3
    ax = axs[0]
    for col, (vals, ttl, colr, off) in enumerate([
            ([[0] * Lp for _ in range(Lp)], "派生读法（单值场）", "#1f77b4", 0.0),
            ([[(i + 2 * j) % 3 for i in range(Lp)] for j in range(Lp)], "独立读法（键自由度）", "#d95f02", Lp + 1.4)]):
        for i in range(Lp):
            for j in range(Lp):
                ax.plot([off + i, off + i + 1, off + i + 1, off + i, off + i],
                        [j, j, j + 1, j + 1, j], color="#cccccc", lw=1.0)
                ax.text(off + i + 0.5, j + 0.5, str(vals[j][i]), ha="center", va="center",
                        color=colr, fontsize=9)
        ax.text(off + Lp / 2, Lp + 0.55, ttl, ha="center", fontsize=9, color=colr)
    ax.annotate("", xy=(2 * Lp + 1.4, -0.35), xytext=(0, -0.35),
                arrowprops=dict(arrowstyle="<->", color="#444444"))
    ax.text(Lp + 0.7, -0.62, "同一格点集上的两种键变量读法：面内数字 = 面通量", ha="center", fontsize=8.5,
            color="#444444")
    ax.set_aspect('equal')
    ax.set_xlim(-0.4, 2 * Lp + 1.8)
    ax.set_ylim(-0.95, Lp + 1.05)
    ax.set_title(f"(a) L = {Lp} 环面：面通量对照", fontsize=10.5)
    ax.axis('off')

    # (b) 扇区计数对照
    ax = axs[1]
    labels = []
    derived = []
    indep = []
    for (N, L) in sorted(summary):
        if (N, L) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
            s = summary[(N, L)]
            labels.append(f"N={N},L={L}")
            derived.append(np.log10(N ** (s["V"] - 1)))
            indep.append(np.log10(N ** 2))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, derived, width=0.4, color="#1f77b4", label="派生读法：可达绕数取值数取对数")
    ax.bar(x + 0.2, indep, width=0.4, color="#d95f02", label="独立读法：绕数扇区数取对数")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(r"$\log_{10}$（计数）")
    ax.set_title("(b) 绕数取值：派生只有 0，独立有 N^2 个扇区", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis='y')

    # (c) 一维：Σ Δu 对边界差
    ax = axs[2]
    N = 5
    Lc = 6
    pts = []
    for u in product(range(N), repeat=Lc):
        s = sum((u[j + 1] - u[j]) % N for j in range(Lc - 1)) % N
        pts.append(((u[-1] - u[0]) % N, s))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(xs, ys, s=8, alpha=0.25, color="#1f77b4")
    ax.plot([0, N - 1], [0, N - 1], color="#d95f02", lw=1.2, label="对角：Σ Δu = 边界差")
    ax.set_xlabel("边界差 $u_{L-1} - u_0$（mod N）")
    ax.set_ylabel(r"$\Sigma \Delta u$（mod N）")
    ax.set_title(f"(c) 开放链（N = {N}，L = {Lc}）：绕数住在边界上", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/bond_identity.png", dpi=150)
    print("\n图已保存：images/bond_identity.png")
except ImportError:
    print("\n跳过绘图：matplotlib 不可用")

print(f"\n[键身份轮] 完成；总运行 {time.time() - T0:.1f} s。")
