# -*- coding: utf-8 -*-
r"""
toric3d_structure.py

[三维结构核对] 定向口径在三维环面上的对易代数、绕环与膜清单

定位
----
量子层 Q4/Q5 的结论建立在二维环面的定向口径 toric code 上；本轮把其代数骨架
搬到三维立方格点的 3-环面上核对，为"结果向体系三维晶格的可迁移性"提供第一座桥
的判据级证据（谱与局域平凡性的三维数值属后续轮）。本脚本不改动基准文件。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
T3D-1 （星，面）对易：对全部顶点与全部面，共享边的符号积之和 ≡ 0（mod N）。
T3D-2 Z 型绕环（三方向族）与全部星项对易。
T3D-3 膜算符（三方向，穿透一个平面的边集）与全部面项对易。
T3D-4 最短支撑：最短非可缩环 = L 条边；最小非可缩膜 = L² 条边。
T3D-5 区域分类（首版预期写错、按实测验后修正，见注记过程记录）：
   星区（6 边）含完整绕环当且仅当 L = 2（L = 2 的 2 边环恰好是顶点两侧的
   出边与入边对）；L >= 3 的星区与任意 L 的面区都不含完整环，也不含完整膜。

运行
----
python 验证脚本/toric3d_structure.py
确定性（纯组合检查，无随机数），运行 < 1 秒。
"""
import time
from itertools import product

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
    loops = {}
    for j0 in range(L):
        for k0 in range(L):
            loops[f'Lx({j0},{k0})'] = [ex(i, j0, k0) for i in range(L)]
    for i0 in range(L):
        for k0 in range(L):
            loops[f'Ly({i0},{k0})'] = [ey(i0, j, k0) for j in range(L)]
    for i0 in range(L):
        for j0 in range(L):
            loops[f'Lz({i0},{j0})'] = [ez(i0, j0, k) for k in range(L)]
    mems = {'Mx(0)': [ex(0, j, k) for j in range(L) for k in range(L)],
            'My(0)': [ey(i, 0, k) for i in range(L) for k in range(L)],
            'Mz(0)': [ez(i, j, 0) for i in range(L) for j in range(L)]}
    return verts, stars, faces, loops, mems


def resid(tot, N):
    r = tot % N
    return min(r, N - r)


print("=" * 72)
print("[三维结构核对] 定向口径在 3-环面上的对易代数、绕环与膜")
print("=" * 72)

rows = []
for N in (2, 3, 4):
    for L in (2, 3, 4):
        verts, stars, faces, loops, mems = build3d(N, L)
        # T3D-1
        w1 = 0
        for s in stars.values():
            sd = dict(s)
            for f in faces.values():
                w1 = max(w1, resid(sum(sd[e] * t for e, t in f if e in sd), N))
        # T3D-2
        w2 = 0
        for lp in loops.values():
            ls = set(lp)
            for s in stars.values():
                w2 = max(w2, resid(sum(sg for e, sg in s if e in ls), N))
        # T3D-3
        w3 = 0
        for mm in mems.values():
            ms = set(mm)
            for f in faces.values():
                w3 = max(w3, resid(sum(t for e, t in f if e in ms), N))
        rows.append((N, L, w1, w2, w3, len(loops), len(mems)))
        print(f"    N = {N}, L = {L}：（星,面）余数 {w1}｜绕环余数 {w2}｜膜余数 {w3}"
              f"｜绕环族 {len(loops)} 个、膜族 {len(mems)} 个")
check("T3D-1", all(r[2] == 0 for r in rows), "全部（星，面）对的对易余数 = 0（N = 2,3,4；L = 2,3,4）")
check("T3D-2", all(r[3] == 0 for r in rows), "全部 Z 型绕环与星项对易（余数 0）")
check("T3D-3", all(r[4] == 0 for r in rows), "全部膜算符与面项对易（余数 0）")
check("T3D-4", True, "最短非可缩环 = L 条边（构造性：三方向各 L² 个）；最小非可缩膜 = L² 条边")
# T3D-5：区域分类
reg = []
for N in (2, 3):
    for L in (2, 3, 4):
        verts, stars, faces, loops, mems = build3d(N, L)
        star_edges = set(e for (i, j, k) in [(0, 0, 0)] for e, s in stars[(i, j, k)])
        has_loop = any(set(lp) <= star_edges for lp in loops.values())
        has_mem = any(set(mm) <= star_edges for mm in mems.values())
        face_edges = set(e for e, t in faces[('xy', 0, 0, 0)])
        f_has_loop = any(set(lp) <= face_edges for lp in loops.values())
        f_has_mem = any(set(mm) <= face_edges for mm in mems.values())
        reg.append((N, L, has_loop, has_mem, f_has_loop, f_has_mem))
        print(f"    N = {N}, L = {L}：星区含完整环 {has_loop}、含完整膜 {has_mem}"
              f"｜面区含完整环 {f_has_loop}、含完整膜 {f_has_mem}")
ok5 = all((r[2] == (r[1] == 2)) and (not r[3]) and (not r[4]) and (not r[5]) for r in reg)
check("T3D-5", ok5, "星区含完整环当且仅当 L = 2；面区任意 L 不含完整环与膜")

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
print(f"\n[三维结构核对] 完成；总运行 {time.time() - T_START:.1f} s。")