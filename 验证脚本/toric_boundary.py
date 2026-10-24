# -*- coding: utf-8 -*-
r"""
toric_boundary.py

[边界模式轮] 缠绕守恒边界的量子层形态：边界项着装、总通量锁定与低重量泄漏

定位
----
桥 3：把保护从环面搬回"有限、有边界"的晶格。经典层（FK 链）用缠绕守恒边界
（两端钉住，$u_{N-1}=u_0+N$）固定绕数；本脚本检验量子层的对应物。
同一 $(L+1)\times(L+1)$ 顶点方形盘面、同一手续（定向口径），四种保留模式：
  P（着装边界）：全部顶点星项（角 2 边、边 3 边、内 4 边）+ 全部 $L^2$ 面项；
  R（半着装，只锁通量）：只保留内部星项 + 全部面项；
  S（半着装，只锁荷）：全部星项 + 只保留内部面项；
  Q（朴素开放）：只保留内部星项 + 内部面项（贴边项全部丢弃）。
判据全为精确组合（稳定子秩公式 + 稳定子判据全枚举 + 轨道计数），不依赖数值对角化。
本脚本不改动基准文件。

模型（与既有轮次"定向口径"一致）
----
键：$h(i,j)$ 为 $x=i$ 处 $y=j$ 到 $j+1$ 的竖键，$v(i,j)$ 为 $y=j$ 处 $x=i$ 到 $i+1$
的横键，共 $n=2L(L+1)$ 条。星项 $A_v$ 取顶点发散（出边 $+1$、入边 $-1$），边界顶点
少边；面项 $B_p$ 取逆时针环绕 $\{v(i,j){+}1,\ h(i+1,j){+}1,\ v(i,j+1){-}1,\ h(i,j){-}1\}$。
几何贴边键指外圈 $4L$ 条键（只被一个完整面项覆盖的键）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
BND-0 对易代数：全部（星，面）对的共享边符号积之和 $\equiv 0\ (\mathrm{mod}\ N)$
   （$N=2,3$；$L=2,3,4$）。
BND-1 着装盘面满秩：$r_A=(L+1)^2-1$、$r_B=L^2$、$r_A+r_B=n$，故 $\deg=N^{n-r_A-r_B}=1$。
BND-2 绕数（总通量）锁定：面项全保留时，全体面项之积 $=$ 带定向的边界回路
   （支撑 $=$ 几何贴边键集），落进稳定子群，本征值被固定；面项只留内部时，该回路
   不在稳定子群中且与全部保留项对易，是重量 $4L$ 的非平凡逻辑算符。
BND-3 重量 1 泄漏（精确集合相等）：Z 型非平凡单边集 $=$ 不被任何保留星项覆盖的键集
   （星项全保留给空集；只留内部给恰 $4L$ 条几何贴边键）；X 型非平凡单边集 $=$ 不被
   任何保留面项覆盖的键集（面项全保留给空集；只留内部给 $8L-4$ 条）。
BND-4 简并标度：$\deg=N^{n-r_A-r_B}$；P 恒给 $1$，Q 给 $8L-5$、R 给 $4L-1$、
   S 给 $4L-4$（$L=2,3,4$，$N=2,3$）。
BND-5 局域平凡性分层：P 的体区（中央块）重量 $\le 2$ 非平凡计数为 $0$；未着装模型的
   体区读数允许非零（如实报告，见过程记录三）。
BND-6 独立核验：$N=2,3$、$L=2$ 时，用轨道计数 $|V_0|/|G_A|$（$V_0$ = 面约束解空间、
   $G_A$ = 星项群，均按集合闭包精确生成）给出的基态维度 $=$ 秩公式值。

运行
----
python 验证脚本/toric_boundary.py
确定性（纯组合，无随机数），运行约 10 秒。图存 D:/software/TianXingFC/images/toric_boundary.png
"""
import time
from itertools import combinations, product

T_START = time.time()
CHECKS = []
RESULTS = {}


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def rank_mod(M, N):
    M = [row[:] for row in M]
    m = len(M)
    n = len(M[0]) if m else 0
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


def in_span(vec, basis, N):
    if not basis:
        return all(v % N == 0 for v in vec)
    r0 = rank_mod([b[:] for b in basis], N)
    return rank_mod([b[:] for b in basis] + [vec[:]], N) == r0


def patch(L, smode, fmode):
    """smode/fmode 取 'all'（全部保留）或 'int'（只保留不贴边的项）。"""

    def h(i, j):
        return ('h', i, j)

    def v(i, j):
        return ('v', i, j)

    edges = []
    for i in range(L + 1):
        for j in range(L):
            edges.append(h(i, j))
    for i in range(L):
        for j in range(L + 1):
            edges.append(v(i, j))
    eidx = {e: a for a, e in enumerate(edges)}
    stars = {}
    for i in range(L + 1):
        for j in range(L + 1):
            if smode == 'int' and not (0 < i < L and 0 < j < L):
                continue
            lst = []
            if i < L:
                lst.append((v(i, j), +1))
            if i > 0:
                lst.append((v(i - 1, j), -1))
            if j < L:
                lst.append((h(i, j), +1))
            if j > 0:
                lst.append((h(i, j - 1), -1))
            stars[(i, j)] = lst
    faces = {}
    for i in range(L):
        for j in range(L):
            if fmode == 'int' and not (0 < i < L - 1 and 0 < j < L - 1):
                continue
            faces[(i, j)] = [(v(i, j), +1), (h(i + 1, j), +1), (v(i, j + 1), -1), (h(i, j), -1)]
    # 几何贴边键：外圈 4L 条
    bnd = set()
    for j in range(L):
        bnd.add(h(0, j))
        bnd.add(h(L, j))
    for i in range(L):
        bnd.add(v(i, 0))
        bnd.add(v(i, L))
    return edges, eidx, stars, faces, bnd


def vec_from_pairs(pairs, eidx, n, N):
    vec = [0] * n
    for e, s in pairs:
        vec[eidx[e]] = (vec[eidx[e]] + s) % N
    return vec


def matrix_of(cells, eidx, n, N):
    M = [[0] * n for _ in cells]
    for a, cell in enumerate(cells):
        for e, s in cell:
            M[a][eidx[e]] = (M[a][eidx[e]] + s) % N
    return M


def analyse(N, L):
    def h(i, j):
        return ('h', i, j)

    def v(i, j):
        return ('v', i, j)

    out = {}
    print("\n" + "=" * 72)
    print(f"[边界模式轮] N = {N}，L = {L}，边 n = {2 * L * (L + 1)}")
    print("=" * 72)
    for tag, smode, fmode in [('P（全/全，着装）', 'all', 'all'),
                              ('R（内星/全面）', 'int', 'all'),
                              ('S（全星/内面）', 'all', 'int'),
                              ('Q（内/内，朴素开放）', 'int', 'int')]:
        edges, eidx, stars, faces, bnd_geo = patch(L, smode, fmode)
        n = len(edges)
        Ms = matrix_of(list(stars.values()), eidx, n, N)
        Mf = matrix_of(list(faces.values()), eidx, n, N)
        rA = rank_mod(Ms, N) if Ms else 0
        rB = rank_mod(Mf, N) if Mf else 0
        deg_exp = n - rA - rB
        deg = N ** deg_exp
        worst = 0
        for cell_s in stars.values():
            for cell_f in faces.values():
                tot = 0
                for e1, s1 in cell_s:
                    for e2, s2 in cell_f:
                        if e1 == e2:
                            tot += s1 * s2
                q = tot % N
                worst = max(worst, min(q, N - q) if q else 0)
        # 总通量算符：全体保留面项之积，与带定向的几何边界回路比较
        face_sum = [0] * n
        for cell in faces.values():
            for e, t in cell:
                face_sum[eidx[e]] = (face_sum[eidx[e]] + t) % N
        loop_pairs = []
        for j in range(L):
            loop_pairs.append((h(0, j), -1))
        for i in range(L):
            loop_pairs.append((v(i, 0), +1))
        for j in range(L):
            loop_pairs.append((h(L, j), +1))
        for i in range(L):
            loop_pairs.append((v(i, L), -1))
        vec_loop = vec_from_pairs(loop_pairs, eidx, n, N)
        flux_eq = (face_sum == vec_loop) if faces else None
        support = all(face_sum[eidx[e]] % N != 0 for e in bnd_geo) if faces else None
        # 回路算符为 Z 型：它是稳定子当且仅当它的边集落在面项张成内
        # （X 型分量必须为零；落在星项张成内的是另一个 X 型算符，不改变 Z 型地位）
        flux_in_stab = in_span(vec_loop, Mf, N) if Mf else (not any(vec_loop))
        flux_xpart = in_span(vec_loop, Ms, N) if Ms else (not any(vec_loop))
        flux_commutes = all(sum(ss * vec_loop[eidx[ee]] for ee, ss in cell) % N == 0
                            for cell in stars.values())
        # 重量 1 泄漏（精确集合）
        Xset, Zset = set(), set()
        for e in edges:
            S = [e]
            fe = all(sum(t for ee, t in cell if ee in S) % N == 0 for cell in faces.values())
            sv = all(sum(s for ee, s in cell if ee in S) % N == 0 for cell in stars.values())
            vec = vec_from_pairs(S and [(e, +1)], eidx, n, N)
            if fe and not in_span(vec, Ms, N):
                Xset.add(e)
            if sv and not in_span(vec, Mf, N):
                Zset.add(e)
        covered_by_faces = set()
        for cell in faces.values():
            for e, t in cell:
                covered_by_faces.add(e)
        covered_by_stars = set()
        for cell in stars.values():
            for e, s in cell:
                covered_by_stars.add(e)
        pred_X = set(edges) - covered_by_faces
        pred_Z = set(edges) - covered_by_stars
        # 体区（中央块）重量 <= 2 非平凡计数
        bulk_edges = []
        for (i, j) in faces:
            if 0 < i < L - 1 and 0 < j < L - 1:
                bulk_edges += [e for e, t in faces[(i, j)]]
        bulk_edges = sorted(set(bulk_edges))
        bulk_nontriv = 0
        bulk_w = {}
        for k in (1, 2):
            cnt = 0
            for sub in combinations(bulk_edges, k):
                S = list(sub)
                fe = all(sum(t for ee, t in cell if ee in S) % N == 0 for cell in faces.values())
                sv = all(sum(s for ee, s in cell if ee in S) % N == 0 for cell in stars.values())
                vec = vec_from_pairs([(e, +1) for e in S], eidx, n, N)
                if fe and not in_span(vec, Ms, N):
                    cnt += 1
                if sv and not in_span(vec, Mf, N):
                    cnt += 1
            bulk_w[k] = cnt
            bulk_nontriv += cnt
        print(f"  {tag}：星 {len(stars)}、面 {len(faces)}、r_A = {rA}、r_B = {rB}、"
              f"简并指数 = {deg_exp}、deg = {deg}")
        print(f"      对易余数最大 = {worst}；几何贴边键 {len(bnd_geo)} 条（4L = {4 * L}）；"
              f"重量 1 泄漏 X = {len(Xset)}、Z = {len(Zset)}")
        print(f"      总通量：全体面项之积 = 定向边界回路？{flux_eq}；支撑覆盖全部贴边键？{support}；"
              f"回路为稳定子？{flux_in_stab}；回路与星项对易？{flux_commutes}；回路边集落在星项张成内？{flux_xpart}")
        print(f"      体区重量 1 非平凡 = {bulk_w[1]}、重量 2 非平凡 = {bulk_w[2]}")
        out[(smode, fmode)] = dict(n=n, rA=rA, rB=rB, deg_exp=deg_exp, deg=deg, worst=worst,
                                   bnd=len(bnd_geo), X=len(Xset), Z=len(Zset),
                                   X_ok=(Xset == pred_X), Z_ok=(Zset == pred_Z),
                                   Z_is_bnd=(Zset == (set(edges) - covered_by_stars)),
                                   flux_eq=flux_eq, flux_in=flux_in_stab, support=support,
                                   flux_commutes=flux_commutes, flux_xpart=flux_xpart,
                                   bulk1=bulk_w[1], bulk2=bulk_w[2],
                                   nstar=len(stars), nface=len(faces))
    return out


for N in (2, 3):
    for L in (2, 3, 4):
        res = analyse(N, L)
        RESULTS[(N, L)] = res
        P, R, S, Q = res[('all', 'all')], res[('int', 'all')], res[('all', 'int')], res[('int', 'int')]
        check(f"BND-0/N{N}L{L}",
              all(res[k]['worst'] == 0 for k in res),
              f"全部（星，面）对的对易余数 = 0（四模型；最大 {max(res[k]['worst'] for k in res)}）")
        if L == 3:
            check(f"BND-1/N{N}",
                  P['rA'] == (L + 1) ** 2 - 1 and P['rB'] == L ** 2 and P['rA'] + P['rB'] == P['n']
                  and P['deg'] == 1,
                  f"着装盘面满秩：r_A = {P['rA']}、r_B = {P['rB']}（n = {P['n']}）；deg = {P['deg']} = 1")
            check(f"BND-2/N{N}",
                  P['flux_eq'] and P['support'] and P['flux_in'] and R['flux_eq'] and R['flux_in']
                  and (S['flux_in'] is False) and (Q['flux_in'] is False)
                  and S['flux_commutes'] and Q['flux_commutes'],
                  f"总通量：面项全保留时全体面项之积 = 定向边界回路（{P['flux_eq']}，支撑覆盖 "
                  f"{P['support']}）且为稳定子；只留内部面项时回路不为稳定子且与全部星项对易"
                  f"（S 稳定子 {S['flux_in']}/对易 {S['flux_commutes']}；Q 稳定子 {Q['flux_in']}/"
                  f"对易 {Q['flux_commutes']}）→ 重量 4L 的非平凡逻辑算符")
            check(f"BND-3/N{N}",
                  (P['X'] == 0 and P['Z'] == 0) and P['X_ok'] and P['Z_ok']
                  and R['Z'] == 4 * L and R['Z_ok'] and S['X'] == 8 * L - 4 and S['X_ok']
                  and Q['X'] == 8 * L - 4 and Q['Z'] == 4 * L and Q['X_ok'] and Q['Z_ok'],
                  f"重量 1 泄漏：P 给 (X {P['X']}, Z {P['Z']})；R 给 Z = {R['Z']} = 4L；"
                  f"S 给 X = {S['X']} = 8L-4；Q 给 (X {Q['X']}, Z {Q['Z']})；集合相等全部成立")
            check(f"BND-4/N{N}",
                  P['deg'] == 1 and Q['deg_exp'] == 8 * L - 5 and R['deg_exp'] == 4 * L - 1
                  and S['deg_exp'] == 4 * L - 4,
                  f"简并指数：P {P['deg_exp']}、Q {Q['deg_exp']}、R {R['deg_exp']}、S {S['deg_exp']}")
        if L >= 3:
            check(f"BND-5/N{N}L{L}",
                  P['bulk1'] == 0 and P['bulk2'] == 0,
                  f"局域平凡性分层：P 体区（重量 1 {P['bulk1']}、重量 2 {P['bulk2']}）= 0；"
                  f"未着装模型如实读数 R({R['bulk1']},{R['bulk2']})、S({S['bulk1']},{S['bulk2']})、"
                  f"Q({Q['bulk1']},{Q['bulk2']})")

for N in (2, 3):
    kk = (('all', 'all'), ('int', 'all'), ('all', 'int'), ('int', 'int'))
    exps = {k: [RESULTS[(N, L)][k]['deg_exp'] for L in (2, 3, 4)] for k in kk}
    check(f"BND-4b/N{N}",
          exps[('all', 'all')] == [0, 0, 0] and exps[('int', 'all')] == [7, 11, 15]
          and exps[('all', 'int')] == [4, 8, 12] and exps[('int', 'int')] == [11, 19, 27],
          f"L = 2,3,4 简并指数：P {exps[('all', 'all')]}、R {exps[('int', 'all')]}、"
          f"S {exps[('all', 'int')]}、Q {exps[('int', 'int')]}（逐点等于 0 / 4L-1 / 4L-4 / 8L-5）")

# ---------------- BND-6：轨道计数（精确、无对角化）----------------
print("\n" + "=" * 72)
print("[BND-6] 轨道计数核验：|V_0| / |G_A|（N = 2, 3；L = 2）")
print("=" * 72)


def orbit_dim(N, L, smode, fmode):
    """dim(共同 +1 本征空间) = |V_0| / |G_A|，按集合精确计数。"""
    edges, eidx, stars, faces, _ = patch(L, smode, fmode)
    n = len(edges)
    # V_0：满足全部面项约束的基态（Z 型对角约束）
    face_rows = list(faces.values())
    V0 = 0
    for s in product(range(N), repeat=n):
        ok = True
        for cell in face_rows:
            acc = 0
            for e, t in cell:
                acc += t * s[eidx[e]]
            if acc % N:
                ok = False
                break
        if ok:
            V0 += 1
    # G_A：星项向量生成的群（按集合闭包）
    gen = []
    for cell in stars.values():
        vec = vec_from_pairs(cell, eidx, n, N)
        gen.append(tuple(vec))
    group = {tuple([0] * n)}
    frontier = [tuple([0] * n)]
    while frontier:
        nxt = []
        for g in frontier:
            for v in gen:
                h = tuple((a + b) % N for a, b in zip(g, v))
                if h not in group:
                    group.add(h)
                    nxt.append(h)
                    if len(group) > 400000:
                        return None, V0, len(group)
        frontier = nxt
    return V0 // len(group), V0, len(group)


ok6 = True
for N in (2, 3):
    for tag, smode, fmode in [('P', 'all', 'all'), ('R', 'int', 'all'),
                              ('S', 'all', 'int'), ('Q', 'int', 'int')]:
        dim, V0, G = orbit_dim(N, L=2, smode=smode, fmode=fmode)
        rA = RESULTS[(N, 2)][(smode, fmode)]['rA']
        rB = RESULTS[(N, 2)][(smode, fmode)]['rB']
        n = 2 * 2 * 3
        deg_rank = N ** (n - rA - rB)
        good = (dim == deg_rank)
        ok6 = ok6 and good
        print(f"  N = {N}，{tag}：|V_0| = {V0}、|G_A| = {G}、维度 {dim}；秩公式 {deg_rank}；"
              f"{'一致' if good else '不一致'}")
check("BND-6", ok6, "轨道计数维度 = 秩公式值（N = 2, 3；L = 2；四模型）")

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

    Lg = 3
    ax = axs[0]
    edges, eidx, stars, faces, bnd = patch(Lg, 'all', 'all')
    covered = set()
    for cell in faces.values():
        for e, t in cell:
            covered.add(e)
    central = [e for e, t in faces[(1, 1)]]
    for e in edges:
        if e[0] == 'h':
            x0, y0, x1, y1 = e[1], e[2], e[1], e[2] + 1
        else:
            x0, y0, x1, y1 = e[1], e[2], e[1] + 1, e[2]
        if e in bnd:
            ax.plot([x0, x1], [y0, y1], color="#d95f02", lw=3.2, solid_capstyle='round', zorder=3)
        elif e in central:
            ax.plot([x0, x1], [y0, y1], color="#1b1b1b", lw=3.2, solid_capstyle='round', zorder=4)
        else:
            ax.plot([x0, x1], [y0, y1], color="#1f77b4", lw=1.6, ls=(0, (4, 2)), zorder=2)
    for i in range(Lg + 1):
        for j in range(Lg + 1):
            ax.plot([i], [j], marker='o', ms=4, color="#444444", zorder=5)
    ax.set_aspect('equal')
    ax.set_xlim(-0.6, Lg + 0.6)
    ax.set_ylim(-0.75, Lg + 0.75)
    ax.set_title(f"(a) L = {Lg} 盘面：贴边键与暴露集", fontsize=11)
    ax.text(-0.55, Lg + 0.45, "橙（粗）：几何贴边键 4L = 12，Q 的 Z 型暴露", color="#d95f02", fontsize=8.5)
    ax.text(-0.55, Lg + 0.15, "蓝（虚线）：面项未覆盖键，Q 的 X 型暴露 8L-4 = 20", color="#1f77b4", fontsize=8.5)
    ax.text(-0.55, -0.55, "黑（粗）：内部面项支撑\nP 保留全部项：暴露集全空", color="#1b1b1b", fontsize=8.5)
    ax.set_xticks(range(Lg + 1))
    ax.set_yticks(range(Lg + 1))

    ax = axs[1]
    Ls = [2, 3, 4]
    for key, lab, col in [(('all', 'all'), "P 全/全（着装）", "#1b1b1b"),
                          (('int', 'all'), "R 内星/全面", "#d95f02"),
                          (('all', 'int'), "S 全星/内面", "#7570b3"),
                          (('int', 'int'), "Q 内/内（开放）", "#1f77b4")]:
        ys = [RESULTS[(2, L)][key]['deg_exp'] for L in Ls]
        ax.plot(Ls, ys, marker='o', color=col, label=lab)
        for x, y in zip(Ls, ys):
            ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xlabel("L")
    ax.set_ylabel(r"简并指数 $\log_N \deg = n - r_A - r_B$")
    ax.set_title("(b) 简并指数：只有着装边界给 0", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)

    ax = axs[2]
    labs = ["P 全/全", "R 内星/全面", "S 全星/内面", "Q 内/内"]
    keys = [('all', 'all'), ('int', 'all'), ('all', 'int'), ('int', 'int')]
    xv = list(range(len(labs)))
    xs_ = [RESULTS[(2, 3)][k]['X'] for k in keys]
    zs_ = [RESULTS[(2, 3)][k]['Z'] for k in keys]
    ax.bar([x - 0.19 for x in xv], xs_, width=0.36, color="#1f77b4", label="X 型（面项着装控制）")
    ax.bar([x + 0.19 for x in xv], zs_, width=0.36, color="#d95f02", label="Z 型（星项着装控制）")
    for x, y in zip(xv, xs_):
        ax.annotate(str(y), (x - 0.19, y), textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8.5)
    for x, y in zip(xv, zs_):
        ax.annotate(str(y), (x + 0.19, y), textcoords="offset points", xytext=(0, 3), ha='center', fontsize=8.5)
    ax.set_xticks(xv)
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel("重量 1 非平凡算符数（L = 3）")
    ax.set_title("(c) 低重量泄漏：着装项对应型归零", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/toric_boundary.png", dpi=150)
    print("\n图已保存：images/toric_boundary.png")
except ImportError:
    print("\n跳过绘图：matplotlib 不可用")

print(f"\n[边界模式轮] 完成；总运行 {time.time() - T_START:.1f} s。")