# -*- coding: utf-8 -*-
r"""
toric_boundary_3d.py

[边界模式轮（三维）] 三维带边界方格上边界项着装、闭合面关系与低重量泄漏

定位
----
桥 3 的三维级：把二维盘面的三分结果（着装给唯一基态、朴素开放给距离 1、半着装分工）
搬到三维立方体样本上；三维特有的新结构是"面项的闭合面关系"（每个初等立方体的六个
面项之积为恒等，即对偶高斯律），以及"边界面上的周界回路"作为绕数算符的三维形态。
本脚本不改动基准文件。

模型（与既有轮次"定向口径"一致）
----
顶点 $(i,j,k)$，$i,j,k \in 0..L$；键 $e^x(i,j,k)$ 连 $(i,j,k)$ 到 $(i+1,j,k)$（$i \in 0..L-1$），
$e^y$、$e^z$ 类推，共 $n = 3L(L+1)^2$ 条。星项取顶点发散（出边 $+1$、入边 $-1$），边界顶点少边。
面项取三族逆时针环绕（$xy$、$xz$、$yz$），四边符号 $(+1, +1, -1, -1)$。初等立方体 $L^3$ 个。
四档保留模式：P 全保留（着装）、R 只留内部星项 + 全部面项、S 全部星项 + 只留内部面项、
Q 两者都只留内部（朴素开放）。"内部"指相关坐标全部落在 $1..L-1$。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
B3D-0 对易代数：全部（星，面）对的共享边符号积之和 $\equiv 0\ (\mathrm{mod}\ N)$。
B3D-1 着装满秩：$r_A = (L+1)^3 - 1$、$r_B = F - L^3$（$F = 3L^2(L+1)$），$r_A + r_B = n$，
   故 $\deg = N^{n-r_A-r_B} = 1$（等价于 Euler 恒等式 $V - E + F = C + 1$，$C = L^3$）。
B3D-2 闭合面与周界回路（三维的绕数形态）：
   (a) 每个初等立方体的六个面项之和为零向量（对偶高斯律），共 $L^3$ 条且相互独立；
       只留内部面项时，落在保留面项内的关系只剩 $(L-2)^3$ 条。
   (b) 六个边界面各自的 $L^2$ 个面项之和等于该面的周界回路（支撑恰 $4L$ 条键、系数 $\pm 1$）；
       面项全保留时它是稳定子（通量锁定）；只留内部面项时它不是稳定子、仍与全部保留星项
       对易，是重量 $4L$ 的非平凡逻辑算符（绕数自由）。
B3D-3 重量 1 泄漏（精确集合相等）：Z 型非平凡单边集 $=$ 不被任何保留星项覆盖的键集；
   X 型 $=$ 不被任何保留面项覆盖的键集（四档模式）。
B3D-4 简并标度：$\deg = N^{n-r_A-r_B}$；P 恒给 $1$，其余三档给具体指数（$L = 1, 2, 3$）。
B3D-5 体区读数：内部块键集上重量 $\le 2$ 的非平凡计数，P 为零，其余如实报告。
B3D-6 独立核验：$L = 1$（单立方体，$n = 12$）用轨道计数 $|V_0|/|G_A|$ 核验秩公式值。

运行
----
python 验证脚本/toric_boundary_3d.py
确定性（纯组合，无随机数），运行约 60 秒。图存 D:/software/TianXingFC/images/toric_boundary_3d.png
"""
import time
from itertools import product, combinations

T_START = time.time()
CHECKS = []
RESULTS = {}


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


class Span:
    """GF(N) 上的张成：一次建基，多次判含。"""

    def __init__(self, rows, N):
        self.N = N
        self.basis = {}
        for r in rows:
            self._add(r)

    def _add(self, row):
        N = self.N
        v = [x % N for x in row]
        for piv in sorted(self.basis):
            if v[piv]:
                f = v[piv]
                v = [(a - f * b) % N for a, b in zip(v, self.basis[piv])]
        piv = next((c for c in range(len(v)) if v[c]), None)
        if piv is None:
            return
        inv = pow(v[piv], -1, N) if N > 2 else 1
        v = [(x * inv) % N for x in v]
        for other in list(self.basis):
            if self.basis[other][piv]:
                f = self.basis[other][piv]
                self.basis[other] = [(a - f * b) % N for a, b in zip(self.basis[other], v)]
        self.basis[piv] = v

    @property
    def rank(self):
        return len(self.basis)

    def contains(self, vec):
        N = self.N
        v = [x % N for x in vec]
        for piv in sorted(self.basis):
            if v[piv]:
                f = v[piv]
                v = [(a - f * b) % N for a, b in zip(v, self.basis[piv])]
        return not any(v)


def patch3d(L, smode, fmode):
    def ex(i, j, k):
        return ('x', i, j, k)

    def ey(i, j, k):
        return ('y', i, j, k)

    def ez(i, j, k):
        return ('z', i, j, k)

    def interior(i, j, k):
        return 0 < i < L and 0 < j < L and 0 < k < L

    verts = list(product(range(L + 1), repeat=3))
    edges = []
    for (i, j, k) in verts:
        if i < L:
            edges.append(ex(i, j, k))
        if j < L:
            edges.append(ey(i, j, k))
        if k < L:
            edges.append(ez(i, j, k))
    eidx = {e: a for a, e in enumerate(edges)}
    stars = {}
    for (i, j, k) in verts:
        if smode == 'int' and not interior(i, j, k):
            continue
        lst = []
        if i < L:
            lst.append((ex(i, j, k), +1))
        if i > 0:
            lst.append((ex(i - 1, j, k), -1))
        if j < L:
            lst.append((ey(i, j, k), +1))
        if j > 0:
            lst.append((ey(i, j - 1, k), -1))
        if k < L:
            lst.append((ez(i, j, k), +1))
        if k > 0:
            lst.append((ez(i, j, k - 1), -1))
        stars[(i, j, k)] = lst
    faces = {}
    for k in range(L + 1):
        for i in range(L):
            for j in range(L):
                if fmode == 'int' and not (interior(i, j, k) and interior(i + 1, j, k)
                                           and interior(i, j + 1, k) and interior(i + 1, j + 1, k)):
                    continue
                faces[('xy', i, j, k)] = [(ex(i, j, k), +1), (ey(i + 1, j, k), +1),
                                          (ex(i, j + 1, k), -1), (ey(i, j, k), -1)]
    for j in range(L + 1):
        for i in range(L):
            for k in range(L):
                if fmode == 'int' and not (interior(i, j, k) and interior(i + 1, j, k)
                                           and interior(i, j, k + 1) and interior(i + 1, j, k + 1)):
                    continue
                faces[('xz', i, j, k)] = [(ex(i, j, k), +1), (ez(i + 1, j, k), +1),
                                          (ex(i, j, k + 1), -1), (ez(i, j, k), -1)]
    for i in range(L + 1):
        for j in range(L):
            for k in range(L):
                if fmode == 'int' and not (interior(i, j, k) and interior(i, j + 1, k)
                                           and interior(i, j, k + 1) and interior(i, j + 1, k + 1)):
                    continue
                faces[('yz', i, j, k)] = [(ey(i, j, k), +1), (ez(i, j + 1, k), +1),
                                          (ey(i, j, k + 1), -1), (ez(i, j, k), -1)]
    cubes = {}
    for i in range(L):
        for j in range(L):
            for k in range(L):
                cubes[(i, j, k)] = [('xy', i, j, k), ('xy', i, j, k + 1), ('xz', i, j, k),
                                    ('xz', i, j + 1, k), ('yz', i, j, k), ('yz', i + 1, j, k)]
    bsurf = {}
    for k in (0, L):
        bsurf[('xy', k)] = [('xy', i, j, k) for i in range(L) for j in range(L)]
    for j in (0, L):
        bsurf[('xz', j)] = [('xz', i, j, k) for i in range(L) for k in range(L)]
    for i in (0, L):
        bsurf[('yz', i)] = [('yz', i, j, k) for j in range(L) for k in range(L)]
    faces_all = {}
    for k in range(L + 1):
        for i in range(L):
            for j in range(L):
                faces_all[('xy', i, j, k)] = [(ex(i, j, k), +1), (ey(i + 1, j, k), +1),
                                              (ex(i, j + 1, k), -1), (ey(i, j, k), -1)]
    for j in range(L + 1):
        for i in range(L):
            for k in range(L):
                faces_all[('xz', i, j, k)] = [(ex(i, j, k), +1), (ez(i + 1, j, k), +1),
                                              (ex(i, j, k + 1), -1), (ez(i, j, k), -1)]
    for i in range(L + 1):
        for j in range(L):
            for k in range(L):
                faces_all[('yz', i, j, k)] = [(ey(i, j, k), +1), (ez(i, j + 1, k), +1),
                                              (ey(i, j, k + 1), -1), (ez(i, j, k), -1)]
    return edges, eidx, stars, faces, faces_all, cubes, bsurf


def vec_of(pairs, eidx, n, N):
    v = [0] * n
    for e, s in pairs:
        v[eidx[e]] = (v[eidx[e]] + s) % N
    return v


def cells_to_vec(cells, eidx, n, N, sign=+1):
    v = [0] * n
    for e, s in cells:
        v[eidx[e]] = (v[eidx[e]] + sign * s) % N
    return v


row_cache = {}


def analyse(N, L):
    CUBE = {}
    def ex(i, j, k):
        return ('x', i, j, k)

    def ey(i, j, k):
        return ('y', i, j, k)

    def ez(i, j, k):
        return ('z', i, j, k)

    out = {}
    print("\n" + "=" * 74)
    print(f"[边界模式轮三维] N = {N}，L = {L}，边 n = {3 * L * (L + 1) ** 2}，立方体 {L ** 3}")
    print("=" * 74)
    for tag, smode, fmode in [('P（全/全，着装）', 'all', 'all'),
                              ('R（内星/全面）', 'int', 'all'),
                              ('S（全星/内面）', 'all', 'int'),
                              ('Q（内/内，朴素开放）', 'int', 'int')]:
        edges, eidx, stars, faces, faces_all, cubes, bsurf = patch3d(L, smode, fmode)
        n = len(edges)
        star_rows = [cells_to_vec(c, eidx, n, N) for c in stars.values()]
        face_rows = [cells_to_vec(c, eidx, n, N) for c in faces.values()]
        SA = Span(star_rows, N)
        SB = Span(face_rows, N)
        rA, rB = SA.rank, SB.rank
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
        # 闭合面关系：每个立方体六面之和
        rel_total = L ** 3
        rel_supported = 0
        rel_ok = True
        for key, fkey in cubes.items():
            i0, j0, k0 = key
            v = [0] * n
            miss = False
            for fk in fkey:
                plane, a, b, c = fk
                if plane == 'xy':
                    sg = +1 if c == k0 + 1 else -1
                elif plane == 'xz':
                    sg = +1 if b == j0 else -1
                else:
                    sg = +1 if a == i0 + 1 else -1
                if fk in faces:
                    for e, s in faces[fk]:
                        v[eidx[e]] = (v[eidx[e]] + sg * s) % N
                else:
                    miss = True
            if not miss:
                rel_supported += 1
                if any(v):
                    rel_ok = False
        # 周界回路（六个边界面）
        perim_ok = True
        perim_in = {}
        perim_logic = {}
        for (plane, c), flist in bsurf.items():
            v = [0] * n
            for fk in flist:
                for e, s in faces_all[fk]:
                    v[eidx[e]] = (v[eidx[e]] + s) % N
            supp = [e for e in edges if v[eidx[e]]]
            coeffs_ok = all(v[eidx[e]] == 1 or v[eidx[e]] == N - 1 for e in supp)
            if len(supp) != 4 * L or not coeffs_ok:
                perim_ok = False
            perim_in[(plane, c)] = SB.contains(v)
            # 与全部保留星项对易（Z 型条件）
            perim_logic[(plane, c)] = all(sum(sv * v[eidx[e]] for e, sv in cell) % N == 0
                                          for cell in stars.values())
        # 重量 1 泄漏
        Xset, Zset = set(), set()
        for e in edges:
            one = vec_of([(e, +1)], eidx, n, N)
            fe = all(sum(t * one[eidx[ee]] for ee, t in cell) % N == 0 for cell in faces.values())
            sv = all(sum(s * one[eidx[ee]] for ee, s in cell) % N == 0 for cell in stars.values())
            if fe and not SA.contains(one):
                Xset.add(e)
            if sv and not SB.contains(one):
                Zset.add(e)
        cov_faces = set()
        for cell in faces.values():
            for e, t in cell:
                cov_faces.add(e)
        cov_stars = set()
        for cell in stars.values():
            for e, s in cell:
                cov_stars.add(e)
        pred_X = set(edges) - cov_faces
        pred_Z = set(edges) - cov_stars
        # 体区块（内部键）重量 <= 2
        def deep(v):
            return all(0 < c < L for c in v)

        bulk = []
        for e in edges:
            t, a, b, c = e
            if t == 'x':
                p0, p1 = (a, b, c), (a + 1, b, c)
            elif t == 'y':
                p0, p1 = (a, b, c), (a, b + 1, c)
            else:
                p0, p1 = (a, b, c), (a, b, c + 1)
            if deep(p0) and deep(p1):
                bulk.append(e)
        bulk_nontriv = {}
        for w in (1, 2):
            cnt = 0
            for sub in combinations(bulk, w):
                S = [(e, +1) for e in sub]
                v = vec_of(S, eidx, n, N)
                fe = all(sum(t * v[eidx[ee]] for ee, t in cell) % N == 0 for cell in faces.values())
                sv = all(sum(s * v[eidx[ee]] for ee, s in cell) % N == 0 for cell in stars.values())
                if fe and not SA.contains(v):
                    cnt += 1
                if sv and not SB.contains(v):
                    cnt += 1
            bulk_nontriv[w] = cnt
        print(f"  {tag}：星 {len(stars)}、面 {len(faces)}、r_A = {rA}、r_B = {rB}、"
              f"简并指数 = {deg_exp}、deg = {deg}")
        print(f"      对易余数最大 {worst}；闭合面关系 {rel_supported}/{rel_total} 落在保留面项内，"
              f"六面之和为零：{rel_ok}")
        print(f"      周界回路：支撑与系数正确 {perim_ok}；在面张成内 {sorted(set(perim_in.values()))}；"
              f"与星项对易 {sorted(set(perim_logic.values()))}")
        print(f"      重量 1 泄漏 X = {len(Xset)}、Z = {len(Zset)}（集合相等：X {Xset == pred_X}、"
              f"Z {Zset == pred_Z}）；体区重量 1/2 非平凡 = {bulk_nontriv[1]}/{bulk_nontriv[2]}")
        out[(smode, fmode)] = dict(n=n, rA=rA, rB=rB, deg=deg, deg_exp=deg_exp, worst=worst,
                                   rel_total=rel_total, rel_supported=rel_supported, rel_ok=rel_ok,
                                   perim_ok=perim_ok, perim_in=all(perim_in.values()),
                                   perim_logic=all(perim_logic.values()),
                                   X=len(Xset), Z=len(Zset), X_ok=(Xset == pred_X), Z_ok=(Zset == pred_Z),
                                   bulk1=bulk_nontriv[1], bulk2=bulk_nontriv[2],
                                   nstar=len(stars), nface=len(faces))
    return out


for N in (2, 3):
    for L in (1, 2, 3):
        res = analyse(N, L)
        RESULTS[(N, L)] = res
        P, R, S, Q = res[('all', 'all')], res[('int', 'all')], res[('all', 'int')], res[('int', 'int')]
        check(f"B3D-0/N{N}L{L}", all(res[k]['worst'] == 0 for k in res),
              f"全部（星，面）对易余数 = 0（四模型，最大 {max(res[k]['worst'] for k in res)}）")
        V = (L + 1) ** 3
        F = 3 * L * L * (L + 1)
        check(f"B3D-1/N{N}L{L}",
              P['rA'] == V - 1 and P['rB'] == F - L ** 3 and P['rA'] + P['rB'] == P['n'] and P['deg'] == 1,
              f"着装满秩：r_A = {P['rA']} = V-1、r_B = {P['rB']} = F-L^3、和 = n = {P['n']}；deg = 1")
        check(f"B3D-2a/N{N}L{L}",
              P['rel_ok'] and P['rel_supported'] == L ** 3 and S['rel_supported'] == max(L - 2, 0) ** 3,
              f"闭合面：六面之和为零 {P['rel_ok']}；着装保留 {P['rel_supported']} = L^3 条；"
              f"只留内部面项保留 {S['rel_supported']} 条 = (L-2)^3")
        check(f"B3D-2b/N{N}L{L}",
              P['perim_ok'] and P['perim_in'] and R['perim_in'] and (not S['perim_in'])
              and (not Q['perim_in']) and S['perim_logic'] and Q['perim_logic'],
              f"周界回路：支撑正确 {P['perim_ok']}；面全保留为稳定子 {P['perim_in']}；"
              f"只留内部面项时非稳定子（S {S['perim_in']}、Q {Q['perim_in']}）且与星项对易"
              f"（S {S['perim_logic']}、Q {Q['perim_logic']}）")
        check(f"B3D-3/N{N}L{L}",
              P['X'] == 0 and P['Z'] == 0 and P['X_ok'] and P['Z_ok'] and R['Z_ok'] and R['X'] == 0
              and S['X_ok'] and S['Z'] == 0 and Q['X_ok'] and Q['Z_ok'],
              f"重量 1 泄漏：P 给 (X {P['X']}, Z {P['Z']})；R 给 (X {R['X']}, Z {R['Z']})；"
              f"S 给 (X {S['X']}, Z {S['Z']})；Q 给 (X {Q['X']}, Z {Q['Z']})；集合相等全部成立")
        if L == 3:
            check(f"B3D-5/N{N}L{L}", P['bulk1'] == 0 and P['bulk2'] == 0,
                  f"体区：P 重量 1/2 非平凡 = {P['bulk1']}/{P['bulk2']}；"
                  f"未着装如实读数 R({R['bulk1']},{R['bulk2']}) S({S['bulk1']},{S['bulk2']}) "
                  f"Q({Q['bulk1']},{Q['bulk2']})")

for N in (2, 3):
    r_exp = {L: RESULTS[(N, L)][('int', 'all')]['deg_exp'] for L in (1, 2, 3)}
    s_exp = {L: RESULTS[(N, L)][('all', 'int')]['deg_exp'] for L in (1, 2, 3)}
    q_exp = {L: RESULTS[(N, L)][('int', 'int')]['deg_exp'] for L in (1, 2, 3)}
    check(f"B3D-4b/N{N}",
          all(r_exp[L] == 6 * L * L + 1 for L in (1, 2, 3))
          and all(s_exp[L] == (5 if L == 1 else 12 * L * L - 12 * L + 4) for L in (1, 2, 3))
          and all(q_exp[L] == (12 if L == 1 else 18 * L * L - 12 * L + 5) for L in (1, 2, 3)),
          f"闭式（L >= 2）：R = 6L^2+1、S = 12L^2-12L+4、Q = 18L^2-12L+5；"
          f"实测 R {[r_exp[L] for L in (1, 2, 3)]}、S {[s_exp[L] for L in (1, 2, 3)]}、"
          f"Q {[q_exp[L] for L in (1, 2, 3)]}（L = 1 档 S/Q 无保留面项，各为 5/12）")

for N in (2, 3):
    exps = {k: [RESULTS[(N, L)][k]['deg_exp'] for L in (1, 2, 3)]
            for k in (('all', 'all'), ('int', 'all'), ('all', 'int'), ('int', 'int'))}
    print(f"\n[N = {N}] L = 1,2,3 简并指数：P {exps[('all', 'all')]}、R {exps[('int', 'all')]}、"
          f"S {exps[('all', 'int')]}、Q {exps[('int', 'int')]}")
    check(f"B3D-4/N{N}",
          exps[('all', 'all')] == [0, 0, 0] and all(x > 0 for k in (('int', 'all'), ('all', 'int'), ('int', 'int'))
                                                    for x in exps[k]),
          f"简并指数：P 恒 0；其余三档随 L 增长（R {exps[('int', 'all')]}、S {exps[('all', 'int')]}、"
          f"Q {exps[('int', 'int')]}）")

# ---------------- B3D-6：轨道计数（L = 1，单立方体，n = 12）----------------
print("\n" + "=" * 74)
print("[B3D-6] 轨道计数核验：L = 1（单立方体，n = 12）")
print("=" * 74)


def orbit_dim(N, L, smode, fmode):
    edges, eidx, stars, faces, _, _, _ = patch3d(L, smode, fmode)
    n = len(edges)
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
    group = {tuple([0] * n)}
    frontier = [tuple([0] * n)]
    gen = [tuple(cells_to_vec(c, eidx, n, N)) for c in stars.values()]
    while frontier:
        nxt = []
        for g in frontier:
            for v in gen:
                h = tuple((a + b) % N for a, b in zip(g, v))
                if h not in group:
                    group.add(h)
                    nxt.append(h)
                    if len(group) > 300000:
                        return None, V0, len(group)
        frontier = nxt
    return V0 // len(group), V0, len(group)


ok6 = True
for N in (2, 3):
    for tag, smode, fmode in [('P', 'all', 'all'), ('R', 'int', 'all'), ('S', 'all', 'int'), ('Q', 'int', 'int')]:
        dim, V0, G = orbit_dim(N, 1, smode, fmode)
        r = RESULTS[(N, 1)][(smode, fmode)]
        deg_rank = N ** r['deg_exp']
        good = (dim == deg_rank)
        ok6 = ok6 and good
        print(f"  N = {N}，{tag}：|V_0| = {V0}、|G_A| = {G}、维度 {dim}；秩公式 {deg_rank}；"
              f"{'一致' if good else '不一致'}")
check("B3D-6", ok6, "轨道计数维度 = 秩公式值（N = 2, 3；L = 1；四模型）")

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
for tag, ok in CHECKS:
    if not ok:
        print(f"  未过：{tag}")

# ---------------- 出图 ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(16.5, 4.8))

    Lg = 3
    edges, eidx, stars, faces, faces_all, cubes, bsurf = patch3d(Lg, 'int', 'int')
    cov_stars = set()
    for cell in stars.values():
        for e, s in cell:
            cov_stars.add(e)
    cov_faces = set()
    for cell in faces.values():
        for e, t in cell:
            cov_faces.add(e)
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    for e in edges:
        t, a, b, c = e
        if t == 'x':
            p0, p1 = (a, b, c), (a + 1, b, c)
        elif t == 'y':
            p0, p1 = (a, b, c), (a, b + 1, c)
        else:
            p0, p1 = (a, b, c), (a, b, c + 1)
        xs, ys, zs = [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]]
        if e in cov_stars and e in cov_faces:
            ax.plot(xs, ys, zs, color="#1b1b1b", lw=1.2)
        elif e in cov_stars:
            ax.plot(xs, ys, zs, color="#1f77b4", lw=2.4)
        else:
            ax.plot(xs, ys, zs, color="#d95f02", lw=2.4)
    ax.set_title(f"(a) L = {Lg} 三维样本：暴露集分层", fontsize=10)
    ax.text2D(0.02, 0.33, "橙：不在任何内部星中\n（Z 型暴露）", transform=ax.transAxes,
              color="#d95f02", fontsize=8)
    ax.text2D(0.02, 0.20, "蓝：被星覆盖、不被\n内部面覆盖（X 型暴露）", transform=ax.transAxes,
              color="#1f77b4", fontsize=8)
    ax.text2D(0.02, 0.07, "黑：内部面与内部星都覆盖", transform=ax.transAxes, color="#1b1b1b", fontsize=8)
    ax.set_xticks(range(Lg + 1))
    ax.set_yticks(range(Lg + 1))
    ax.set_zticks(range(Lg + 1))
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(1, 3, 2)
    Ls = [1, 2, 3]
    for key, lab, col in [(('all', 'all'), "P 全/全（着装）", "#1b1b1b"),
                          (('int', 'all'), "R 内星/全面", "#d95f02"),
                          (('all', 'int'), "S 全星/内面", "#7570b3"),
                          (('int', 'int'), "Q 内/内（开放）", "#1f77b4")]:
        ys = [RESULTS[(2, L)][key]['deg_exp'] for L in Ls]
        ax.plot(Ls, ys, marker='o', color=col, label=lab)
        for x, y in zip(Ls, ys):
            ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xlabel("L")
    ax.set_ylabel(r"简并指数 $\log_N \deg$")
    ax.set_title("(b) 三维简并指数：着装恒 0", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)

    ax = fig.add_subplot(1, 3, 3)
    labs = ["P 全/全", "R 内星/全面", "S 全星/内面", "Q 内/内"]
    keys = [('all', 'all'), ('int', 'all'), ('all', 'int'), ('int', 'int')]
    xv = list(range(4))
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
    ax.set_title("(c) 低重量泄漏：着装项对应型归零", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/toric_boundary_3d.png", dpi=150)
    print("\n图已保存：images/toric_boundary_3d.png")
except ImportError:
    print("\n跳过绘图：matplotlib 不可用")

print(f"\n[边界模式轮三维] 完成；总运行 {time.time() - T_START:.1f} s。")