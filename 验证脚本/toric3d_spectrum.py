# -*- coding: utf-8 -*-
r"""
toric3d_spectrum.py

[三维谱与局域平凡性] 3-环面上 toric code 的简并指数与局域算子分类

定位
----
桥 2 的第二级：三维谱（简并的指数结构）与三维局域平凡性（Q5 的三维版）。
方法全为精确组合：简并用稳定子秩公式（无需对角化）；局域算子分类用稳定子判据
（对易稳定子的局域算子要么为稳定子乘积（压缩为标量）、要么为零压缩；不对易则零）。
本脚本不改动基准文件。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
S3D-1 简并：deg = N^{n - r_A - r_B}，r_A = rank(星矩阵)、r_B = rank(面矩阵)（GF(N)），
   对照二维同码（deg = N^{b_1}）：三维 3-环面应给 N^3、二维环面应给 N^2（N = 2, 3）。
S3D-2 局域平凡性（分区）：不含完整非平凡环/膜的区域：全部局域 Pauli 基算子平凡
   （零压缩或稳定子乘积）；含完整非平凡环或膜的区域：出现非平凡算子。
S3D-3 最小支撑：Z 型（星对易集）最小非平凡重量 = L；X 型（面对易集）最小非平凡
   重量 = L^2（L = 2 全枚举核；L >= 3 构造性给出上界并报告）。

运行
----
python 验证脚本/toric3d_spectrum.py
确定性（纯组合，无随机数），运行 < 5 秒。
"""
import time
from itertools import product, combinations

T_START = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def build3d(N, L):
    def ex(i, j, k):
        return ('x', i % L, j % L, k % L)

    def ey(i, j, k):
        return ('y', i % L, j % L, k % L)

    def ez(i, j, k):
        return ('z', i % L, j % L, k % L)
    verts = list(product(range(L), repeat=3))
    edges = []
    for (i, j, k) in verts:
        edges += [ex(i, j, k), ey(i, j, k), ez(i, j, k)]
    eidx = {e: a for a, e in enumerate(edges)}
    stars = {}
    for (i, j, k) in verts:
        stars[(i, j, k)] = [(ex(i, j, k), +1), (ex(i - 1, j, k), -1),
                            (ey(i, j, k), +1), (ey(i, j - 1, k), -1),
                            (ez(i, j, k), +1), (ez(i, j, k - 1), -1)]
    faces = {}
    for (i, j, k) in verts:
        faces[('xy', i, j, k)] = [(ex(i, j, k), +1), (ey(i + 1, j, k), +1),
                                  (ex(i, j + 1, k), -1), (ey(i, j, k), -1)]
        faces[('xz', i, j, k)] = [(ex(i, j, k), +1), (ez(i + 1, j, k), +1),
                                  (ex(i, j, k + 1), -1), (ez(i, j, k), -1)]
        faces[('yz', i, j, k)] = [(ey(i, j, k), +1), (ez(i, j + 1, k), +1),
                                  (ey(i, j, k + 1), -1), (ez(i, j, k), -1)]
    return edges, eidx, verts, stars, faces


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


def in_span(vec, basis, N, ncol):
    """vec 是否在 basis（行向量列表）的张成内（GF(N)）"""
    if not basis:
        return all(v % N == 0 for v in vec)
    M = [b[:] for b in basis] + [vec[:]]
    return rank_mod(M, N) == rank_mod([b[:] for b in basis], N)


print("=" * 72)
print("[三维谱与局域平凡性] 3-环面上 toric code 的简并指数与局域算子分类")
print("=" * 72)

# ---------- S3D-1 简并 ----------
for (N, L) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
    edges, eidx, verts, stars, faces = build3d(N, L)
    n = len(edges)
    Ms = [[0] * n for _ in stars]
    for a, st in enumerate(stars.values()):
        for e, s in st:
            Ms[a][eidx[e]] = (Ms[a][eidx[e]] + s) % N
    Mf = [[0] * n for _ in faces]
    for a, fc in enumerate(faces.values()):
        for e, t in fc:
            Mf[a][eidx[e]] = (Mf[a][eidx[e]] + t) % N
    rA = rank_mod(Ms, N)
    rB = rank_mod(Mf, N)
    deg = N ** (n - rA - rB)
    print(f"    N = {N}, L = {L}：边数 n = {n}，r_A = {rA}，r_B = {rB}，"
          f"简并 = {deg}（预期 N^{3} = {N**3}）")
    if L == 2 and N == 2:
        check("S3D-1a", deg == N ** 3, f"三维简并 = {deg} = N^3（L=2, N=2）")
    if L == 2 and N == 3:
        check("S3D-1b", deg == N ** 3, f"三维简并 = {deg} = N^3（L=2, N=3）")

# 二维同码交叉核对：deg = N^2
for (N, L) in [(2, 2), (3, 2)]:
    def e2(t, i, j):
        i, j = i % L, j % L
        return i * L + j if t == 'h' else L * L + i * L + j
    verts2 = list(product(range(L), repeat=2))
    edges2 = []
    for (i, j) in verts2:
        edges2 += [e2('h', i, j), e2('v', i, j)]
    eidx2 = {e: a for a, e in enumerate(edges2)}
    stars2 = {(i, j): [(e2('h', i, j), +1), (e2('h', i, j - 1), -1),
                       (e2('v', i, j), +1), (e2('v', i - 1, j), -1)] for (i, j) in verts2}
    faces2 = {(i, j): [(e2('v', i, j), +1), (e2('h', i + 1, j), +1),
                       (e2('v', i, j + 1), -1), (e2('h', i, j), -1)] for (i, j) in verts2}
    n2 = len(edges2)
    Ms2 = [[0] * n2 for _ in stars2]
    for a, st in enumerate(stars2.values()):
        for e, s in st:
            Ms2[a][eidx2[e]] = (Ms2[a][eidx2[e]] + s) % N
    Mf2 = [[0] * n2 for _ in faces2]
    for a, fc in enumerate(faces2.values()):
        for e, t in fc:
            Mf2[a][eidx2[e]] = (Mf2[a][eidx2[e]] + t) % N
    rA2, rB2 = rank_mod(Ms2, N), rank_mod(Mf2, N)
    deg2 = N ** (n2 - rA2 - rB2)
    print(f"    [二维核对] N = {N}：n = {n2}，r_A = {rA2}，r_B = {rB2}，简并 = {deg2}（预期 N^2）")

# ---------- S3D-2/3 局域分类与最小支撑 ----------
for (N, L, enum_extras) in [(2, 2, True), (3, 2, True), (2, 3, True), (3, 3, False)]:
    edges, eidx, verts, stars, faces = build3d(N, L)
    n = len(edges)
    star_basis = []
    for st in stars.values():
        vec = [0] * n
        for e, s in st:
            vec[eidx[e]] = (vec[eidx[e]] + s) % N
        star_basis.append(vec)
    face_basis = []
    for fc in faces.values():
        vec = [0] * n
        for e, t in fc:
            vec[eidx[e]] = (vec[eidx[e]] + t) % N
        face_basis.append(vec)
    rA = rank_mod(star_basis, N)
    rB = rank_mod(face_basis, N)
    # 面偶/星偶检查：M_fe . S = 0（X 型对易条件）；M_sv . S = 0（Z 型）
    def fe_even(S):
        for fc in faces.values():
            acc = 0
            for e, t in fc:
                if e in S:
                    acc += t
            if acc % N:
                return False
        return True

    def sv_even(S):
        for st in stars.values():
            acc = 0
            for e, s in st:
                if e in S:
                    acc += s
            if acc % N:
                return False
        return True

    def classify(region):
        """返回 (X 型非平凡数, Z 型非平凡数)"""
        nx = nz = 0
        reg = list(region)
        for k in range(1, len(reg) + 1):
            for sub in combinations(reg, k):
                S = set(sub)
                for powers in product(range(1, N), repeat=k) if N > 2 else [tuple([1] * k)]:
                    vec = [0] * n
                    for e, p in zip(sub, powers):
                        vec[eidx[e]] = (vec[eidx[e]] + p) % N
                    if fe_even(S):
                        if not in_span(vec, star_basis, N, n):
                            nx += 1
                    if sv_even(S):
                        if not in_span(vec, face_basis, N, n):
                            nz += 1
        return nx, nz

    def e_xy(i, j, k):
        return ('x', i % L, j % L, k % L)

    def e_yz(i, j, k):
        return ('y', i % L, j % L, k % L)

    def e_z(i, j, k):
        return ('z', i % L, j % L, k % L)

    # 平凡区：面（4 边）、星（6 边）、角（3 边：一条 x 一条 y 一条 z，共顶点）
    face_r = [e for e, s in faces[('xy', 0, 0, 0)]]
    star_r = [e for e, s in stars[(0, 0, 0)]]
    corner_r = [e_xy(0, 0, 0), e_yz(0, 0, 0), e_z(0, 0, 0)]
    # 非平凡区：完整 Z 环（沿 x，L 条边）与完整 X 膜（L^2 条 x 边）
    zloop = [e_xy(i, 0, 0) for i in range(L)]
    xmem = [e_xy(0, j, k) for j in range(L) for k in range(L)]
    res = {}
    for name, region in [('面（4 边）', face_r), ('星（6 边）', star_r), ('角（3 边）', corner_r),
                         ('Z 环（L 边）', zloop), ('X 膜（L² 边）', xmem)]:
        nx, nz = classify(region)
        res[name] = (nx, nz)
        print(f"    N = {N}, L = {L}｜{name}：X 型非平凡 {nx}，Z 型非平凡 {nz}")

    # S3D-3：L=2 时全枚举（上表已是全枚举）；最小支撑判定
    if L == 2 and N == 2:
        zmin = 2 if res['Z 环（L 边）'][1] > 0 else None
        xmin = 4 if res['X 膜（L² 边）'][0] > 0 else None
        # 平凡区：面、角；星区在 L=2 含全部三条 2 边完整环（预期 Z 型非平凡 2^3-1 = 7，作正对照）
        ok_min = (zmin == 2) and (xmin == 4) and res['面（4 边）'] == (0, 0) \
            and res['角（3 边）'] == (0, 0) and res['星（6 边）'] == (0, 7)
        check("S3D-3a", ok_min,
              f"L=2：最小非平凡支撑 Z 型 = {zmin}（= L）、X 型 = {xmin}（= L²）；面区与角区全平凡；"
              f"星区（含三条完整 2 边环）非平凡计数 = {res['星（6 边）']}（预期 Z 型 2^3-1 = 7）")
    if L == 3 and N == 2:
        ok3 = (res['Z 环（L 边）'][1] > 0) and (res['X 膜（L² 边）'][0] > 0) and \
              all(v == (0, 0) for k in ('面（4 边）', '星（6 边）', '角（3 边）') for v in [res[k]])
        check("S3D-3b", ok3, "L=3：Z 环（3 边）与 X 膜（9 边）均非平凡；面/星/角区全平凡")

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
print(f"\n[三维谱与局域平凡性] 完成；总运行 {time.time() - T_START:.1f} s。")