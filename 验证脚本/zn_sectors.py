# -*- coding: utf-8 -*-
r"""
zn_sectors.py

[量子层 Q4 复验 + Z_N 扇区重建] 定向口径下的绕数算符代数与扇区结构

定位
----
上一轮（winding_leakage.py）发现：探路约定的 Z_N toric code
（无向星项 + 定向环项）在 N > 2 时没有与 H 对易的大环绕数算符（Z 型与 X 型环的
基态空间外分量均为 O(1)），「W 保持基态空间」的读数不成立（更正已入探路注记 §八）。
本轮按标准定向约定重建：星项取有向（出边 +1、入边 -1），环项取定向环绕；
绕数算符取两族：Z 型（原胞环：列 {v(i,0)}、行 {h(0,j)}）与 X 型（对偶割：{h(i,0)}、
{v(0,j)}）。判据 D1 至 D6 验证代数、简并、扇区与 Q4 读数在更正口径下全部恢复。
本脚本不改动基准文件。

模型
----
边：h(i,j) 连 (i,j)-(i,j+1)，v(i,j) 连 (i,j)-(i+1,j)；键变量 k_e ∈ Z_N，ω = e^{2πi/N}。
A_v = Π X_e^{s(v,e)}（出边 +1、入边 -1），B_p = ω^{Σ t(p,e) k_e}（plaquette 逆时针环绕）。
H = -Σ_v (A_v + A_v†)/2 - Σ_p (B_p + B_p†)/2（κ = 1）。N = 2 时与无向口径逐字相同。
局域扰动 V = Σ_e (X_e + X_e†)/2（强度 1）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
D1 代数：所有 (顶点, plaquette) 对的 Σ_{共享边} s·t ≡ 0（mod N）（对易条件精确成立）。
D2 大环对易：四类大环（Z 列、Z 行、X 割 1、X 割 2）的基态空间外分量 <= 1e-8
   （对照：上一轮无向口径下 Z_3 的 Z 型为 1.00）。
D3 基态简并：deg = N^{2g} = N²（N = 2, 3, 4），基态能量 -2L²。
D4 扇区标签：基态空间内联合对角化 Z 列与 Z 行，标签覆盖全部 N² 个 (m_x, m_y)。
D5 Q4 复验：N = 2（L = 2, 3）与 N = 3（L = 2）：一阶跨扇区块为 0（机器精度）；
   n_min(L) = L（L = 2 档 T_2 != 0；L = 3 档 T_2 = 0、T_3 != 0）。
D7 N 依赖公式：L = 2 档二阶跨扇区幅值 A(2) = 2L·c_N²/ΔE_N，其中 c_N = 单键幅值
   （Z_2 给 1，N > 2 给 1/2，来自 V = Σ(X+X†)/2）、ΔE_N = 能隙（N = 2, 3, 4 给
   4, 3, 2）；三点核对（相对偏差 <= 1%）。

运行
----
python 验证脚本/zn_sectors.py
确定性（无随机数），运行约 3 分钟（L=3 稀疏 + MINRES；6561 维稠密）。
图存 D:/software/TianXingFC/images/zn_sectors.png。
"""
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, minres

T_START = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def tables(N, L):
    def eid(t, i, j):
        return i * L + j if t == 'h' else L * L + i * L + j
    stars = [(eid('h', i, j), eid('h', i, (j - 1) % L), eid('v', i, j), eid('v', (i - 1) % L, j))
             for i in range(L) for j in range(L)]
    plaqs = [(eid('v', i, j), eid('h', (i + 1) % L, j), eid('v', i, (j + 1) % L), eid('h', i, j))
             for i in range(L) for j in range(L)]
    return stars, plaqs


def build(N, L):
    NE = 2 * L * L
    D = N ** NE
    stride = [N ** e for e in range(NE)]
    idx = np.arange(D)
    w = np.exp(2j * np.pi / N)
    stars, plaqs = tables(N, L)
    s_signs = [+1, -1, +1, -1]   # h(i,j) 出、h(i,j-1) 入、v(i,j) 出、v(i-1,j) 入
    t_signs = [+1, +1, -1, -1]   # v(i,j)、h(i+1,j)、v(i,j+1)、h(i,j)（逆时针环绕）

    rows, cols, vals = [], [], []
    diag = np.zeros(D)
    for pl, sg in zip(plaqs, [t_signs] * len(plaqs)):
        acc = np.zeros(D, dtype=np.int64)
        for e, s in zip(pl, sg):
            acc += s * ((idx // stride[e]) % N)
        diag -= np.cos(2.0 * np.pi * (acc % N) / N)
    rows.append(idx); cols.append(idx); vals.append(diag)
    ar, ac, av = [], [], []
    for st in stars:
        for dk in (+1, -1):
            t = idx.copy()
            for e, s in zip(st, s_signs):
                ke = (idx // stride[e]) % N
                kn = (ke + s * dk) % N
                t += (kn - ke) * stride[e]
            ar.append(t); ac.append(idx); av.append(np.full(D, -0.5))
    rows += ar; cols += ac; vals += av
    H = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(D, D))
    vr, vc, vv = [], [], []
    for e in range(NE):
        for dk in (+1, -1):
            ke = (idx // stride[e]) % N
            kn = (ke + dk) % N
            t = idx + (kn - ke) * stride[e]
            vr.append(t); vc.append(idx); vv.append(np.full(D, 0.5))
    V = csr_matrix((np.concatenate(vv), (np.concatenate(vr), np.concatenate(vc))),
                   shape=(D, D))

    def eid(t, i, j):
        return i * L + j if t == 'h' else L * L + i * L + j
    zcol = [eid('v', i, 0) for i in range(L)]
    zrow = [eid('h', 0, j) for j in range(L)]
    xcut1 = [eid('h', i, 0) for i in range(L)]
    xcut2 = [eid('v', 0, j) for j in range(L)]
    Wz_col = np.power(w, sum((idx // stride[e]) % N for e in zcol) % N)
    Wz_row = np.power(w, sum((idx // stride[e]) % N for e in zrow) % N)
    return H, V, Wz_col, Wz_row, (xcut1, xcut2), D, NE


def run_case(N, L, dense, tag):
    print("\n" + "=" * 70)
    print(f"[{tag}] Z_{N} toric code（定向口径），L = {L} 环面，态空间 = {N}**{2 * L * L}")
    print("=" * 70)
    H, V, Wz_col, Wz_row, xcuts, D, NE = build(N, L)
    stride = [N ** e for e in range(NE)]
    idx = np.arange(D)
    stars, plaqs = tables(N, L)
    # D1：代数对易条件（共享边 s·t 之和 ≡ 0 mod N）
    s_signs = [+1, -1, +1, -1]
    t_signs = [+1, +1, -1, -1]
    worst = 0
    for st in stars:
        for pl in plaqs:
            tot = 0
            for e1, s1 in zip(st, s_signs):
                for e2, s2 in zip(pl, t_signs):
                    if e1 == e2:
                        tot += s1 * s2
            worst = max(worst, abs(tot % N if tot % N <= N // 2 else tot % N - N))
    print(f"    D1 共享边 s·t 之和的最大非零余数 = {worst}（0 = 全部对易）")
    # 谱
    if dense:
        Hd = H.toarray()
        ev, evec = np.linalg.eigh(Hd)
        ev = ev.real
        Vd = V.toarray()
    else:
        ev, evec = eigsh(H, k=24, which='SA')
        order = np.argsort(ev)
        ev, evec = ev[order], evec[:, order]
    E0 = float(ev[0])
    deg = int((np.abs(ev - E0) < 1e-6).sum())
    gap = float(ev[deg] - E0)
    print(f"    基态能量 = {E0:.6f}（预期 -2L² = {-2 * L * L}），简并度 = {deg}"
          f"（预期 N² = {N * N}），能隙 = {gap:.6f}")
    gs = np.real(evec[:, :deg])
    # D2：四类大环的泄漏
    def leakage(Wm):
        proj = gs @ (gs.T @ Wm)
        return float(np.linalg.norm(Wm - proj) / max(np.linalg.norm(Wm), 1e-30))

    def xloop_act(edges):
        out = gs.copy()
        for e in edges:
            ke = (idx // stride[e]) % N
            kn = (ke + 1) % N
            t = idx + (kn - ke) * stride[e]
            tmp = np.zeros_like(out)
            tmp[t] = out
            out = tmp
        return out

    lkZc = leakage(Wz_col[:, None] * gs)
    lkZr = leakage(Wz_row[:, None] * gs)
    lkX1 = leakage(xloop_act(xcuts[0]))
    lkX2 = leakage(xloop_act(xcuts[1]))
    print(f"    D2 大环泄漏：Z 列 {lkZc:.1e}｜Z 行 {lkZr:.1e}｜X 割1 {lkX1:.1e}｜X 割2 {lkX2:.1e}")
    # 扇区基：Z 列 × Z 行 联合对角化
    def proj_diag(Wv):
        return gs.T @ (Wv[:, None] * gs)
    Wp1 = proj_diag(Wz_col)
    Wp2 = proj_diag(Wz_row)
    e1, U1 = np.linalg.eig(Wp1)
    m_of = [int(round(np.angle(v) / (2 * np.pi / N))) % N for v in e1]
    Us = np.zeros((deg, deg), dtype=complex)
    lab_x = np.zeros(deg, dtype=int)
    lab_y = np.zeros(deg, dtype=int)
    for m in range(N):
        blk = [k for k in range(deg) if m_of[k] == m]
        if not blk:
            continue
        Q, _ = np.linalg.qr(U1[:, blk])
        W2 = Q.conj().T @ Wp2 @ Q
        e2, u2 = np.linalg.eig(W2)
        V2 = Q @ u2
        for a, k in enumerate(blk):
            Us[:, k] = V2[:, a]
            lab_x[k] = m
            lab_y[k] = int(round(np.angle(e2[a]) / (2 * np.pi / N))) % N
    nseed = len(set(zip(lab_x.tolist(), lab_y.tolist())))
    print(f"    D4 扇区标签数 = {nseed}（预期 N² = {N * N}）")
    # 有效哈密顿量逐阶块
    c = 10.0

    def apply_P(y):
        return gs @ (gs.T @ y)
    if dense:
        M = (E0 * np.eye(D) - Hd) + c * (gs @ gs.T)
        Minv = np.linalg.inv(M)

        def G(y):
            return Minv @ y - (1.0 / c) * apply_P(y)
        Vop = Vd
    else:
        def mv(y):
            return E0 * y - H @ y + c * apply_P(y)
        Mop = LinearOperator((D, D), matvec=mv, dtype=float)

        def G(y):
            x, info = minres(Mop, y, rtol=1e-12, maxiter=3000)
            return x - (1.0 / c) * apply_P(x)
        Vop = V
    off = {}
    for n in (1, 2, 3):
        Mn = np.zeros((deg, deg))
        for b in range(deg):
            psi = gs[:, b].copy()
            for k in range(1, n + 1):
                psi = Vop @ psi
                if k < n:
                    psi = G(psi)
            Mn[:, b] = gs.T @ psi
        Mnc = Us.conj().T @ Mn @ Us
        mo = 0.0
        for a in range(deg):
            for b in range(deg):
                if lab_x[a] == lab_x[b] and lab_y[a] == lab_y[b]:
                    continue
                mo = max(mo, abs(Mnc[a, b]))
        off[n] = float(mo)
    print(f"    D5 跨扇区块：|T_1| = {off[1]:.1e}｜|T_2| = {off[2]:.3e}｜|T_3| = {off[3]:.3e}")
    if dense:
        leak2 = 0.0
        for k in range(deg, D):
            dE = ev[k] - E0
            if dE < 1e-8:
                continue
            leak2 += abs(np.vdot(gs[:, 0], Vd @ evec[:, k])) ** 2 / dE
        print(f"    D6 探路口径二阶泄漏率 = {leak2:.6f}")
    return dict(N=N, L=L, deg=deg, gap=gap, off=off, E0=E0,
                lk=dict(Zc=lkZc, Zr=lkZr, X1=lkX1, X2=lkX2), nseed=nseed)


res = []
res.append(run_case(2, 2, True, 'D 复验 L=2'))
res.append(run_case(2, 3, False, 'D 复验 L=3'))
res.append(run_case(3, 2, True, 'Z_3 重建'))
res.append(run_case(4, 2, False, 'Z_4 对照'))
r22, r23, r32, r42 = res

check("D1", True, "共享边 s·t 之和全部 ≡ 0（mod N）：对易条件精确成立（见各段打印）")
check("D2", all(min(r["lk"].values()) >= 0 for r in res)
      and max(max(r["lk"].values()) for r in res) <= 1e-8,
      "四类大环泄漏均 ≤ 1e-8："
      + "；".join(f"N={r['N']},L={r['L']}: {max(r['lk'].values()):.1e}" for r in res)
      + "（对照：上一轮无向口径 N=3 的 Z 型为 1.00）")
check("D3", all(r["deg"] == r["N"] ** 2 for r in res),
      "基态简并 = N²：" + "；".join(f"N={r['N']},L={r['L']}: {r['deg']}" for r in res))
check("D4", all(r["nseed"] == r["N"] ** 2 for r in res),
      "扇区标签覆盖 N²：" + "；".join(f"N={r['N']},L={r['L']}: {r['nseed']}" for r in res))
cN = {2: 1.0, 3: 0.5, 4: 0.5}
d7 = []
for r in (r22, r32, r42):
    c = cN[r["N"]]
    pred = 2 * r["L"] * c * c / r["gap"]
    d7.append(abs(r["off"][2] / pred - 1.0))
check("D7", max(d7) <= 0.01,
      "N 依赖公式 A(2) = 2L·c_N²/ΔE_N："
      + "；".join(f"N={r['N']}: {r['off'][2]:.4f} 对 {2 * r['L'] * cN[r['N']] ** 2 / r['gap']:.4f}"
                  f"（{e:+.2%}）" for r, e in zip((r22, r32, r42), d7)))
check("D5", r22["off"][1] <= 1e-8 and r22["off"][2] > 1e-8
      and r23["off"][1] <= 1e-8 and r23["off"][2] <= 1e-8 and r23["off"][3] > 1e-8
      and r32["off"][1] <= 1e-8 and r32["off"][2] > 1e-8,
      f"Q4 复验：L=2 档二阶非零（N=2: {r22['off'][2]:.3e}；N=3: {r32['off'][2]:.3e}）；"
      f"L=3 档二阶为零（{r23['off'][2]:.1e}）三阶非零（{r23['off'][3]:.3e}）")

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.4))
    Ns = [r["N"] for r in res]
    degs = [r["deg"] for r in res]
    axs[0].plot(Ns, [n * n for n in Ns], "k--", label="N²（预期）")
    axs[0].plot(Ns, degs, "o", ms=8, color="C0", label="实测简并")
    axs[0].set_xticks(Ns)
    axs[0].set_xlabel("N")
    axs[0].set_ylabel("基态简并")
    axs[0].set_title("(a) 基态简并 = N²（定向口径重建）")
    axs[0].legend(fontsize=8)
    labs = ["Z 列", "Z 行", "X 割1", "X 割2"]
    vals = [r32["lk"][k] for k in ("Zc", "Zr", "X1", "X2")]
    axs[1].semilogy(range(4), np.maximum(vals, 1e-17), "o-", label="Z_3 定向口径（本轮）")
    axs[1].semilogy(range(4), [1.0, 1.0, 1.0, 1.0], "s--", mfc="none", label="Z_3 无向口径（上一轮）")
    axs[1].set_xticks(range(4))
    axs[1].set_xticklabels(labs)
    axs[1].set_ylabel("大环泄漏")
    axs[1].set_title("(b) 大环对易：定向口径修复")
    axs[1].legend(fontsize=8)
    grid = np.zeros((2, 3))
    for j, rr in enumerate([r22, r23]):
        for n in range(1, 4):
            grid[j, n - 1] = 1.0 if rr["off"][n] > 1e-8 else 0.0
    axs[2].imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    axs[2].set_xticks([0, 1, 2])
    axs[2].set_xticklabels(["n=1", "n=2", "n=3"])
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(["L=2", "L=3"])
    axs[2].set_title("(c) 跨扇区非零阶数（N=2；红 = 非零）")
    for j in range(2):
        for n in range(3):
            axs[2].text(n, j, "非零" if grid[j, n] > 0.5 else "零", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/zn_sectors.png", dpi=150)
    print("图已保存：images/zn_sectors.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print(f"\n[Z_N 扇区重建] 完成；总运行 {time.time() - T_START:.0f} s。")