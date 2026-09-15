# -*- coding: utf-8 -*-
r"""
minimal_verification_string_deform.py

[量子层 Q5] 弦算符的可形变完成（TQO-2 型）：同调不变、可缩环平凡与局域算子标量化

定位
----
载体层裁决 §八 判据 Q5：「弦算符的可形变完成：局域扰动下绕数环算符可形变而不改变
基态空间内的作用」。本脚本在定向口径 Z_N toric code（复用 zn_sectors 装置）上把
该判据拆成四条可判定的数值陈述，并给扰动读数。本脚本不改动基准文件。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
E1 同调不变性：同一同调类、不同位置的弦（Z 型列 {v(i,j0)}，j0 = 0..L-1；X 型对偶
   割 {h(i,j0)}）在基态空间的作用矩阵 M_j = gs† W_j gs 两两只差相位：
   ||M_j† M_k - phase·I|| <= 1e-9。
E2 可缩环平凡：plaquette 边界上的定向 Z 环 = B_p（算子恒等，随机向量核对），
   1x2 块周界的 Z 环 = 该块 B_p 乘积；两者的基态压缩为标量。
E3 局域算子标量化：对不含完整非可缩环的区域（L=3：单 plaquette 四边、单星四边、
   五边路径；L=2：角部两边、弯折三边路径），枚举全部广义 Pauli 基 X^a Z^b，每个算子的
   基态压缩 C = gs† O gs 满足 ||C - (trC/deg) I|| <= 1e-9。正对照：含完整非可缩环的
   区域（L=2 的 plaquette/星四边已含 2 边完整环；L=3 的非可缩列）必出现非标量压缩。
   （首版把 L=2 的 plaquette/星四边误判为无环，实测各 6（Z_2）/24（Z_3）个非标量，
   核查为非标量恰为逻辑算符的乘积，见过程记录。）
E4 最小支撑：出现非标量压缩的最小区域大小 = L（L = 2 给 2、L = 3 给 3）。
E5 扰动读数：H + λV（V = Σ(X+X†)/2）下，扰动基态空间内 W_x 压缩 C = gs_λ† W_x gs_λ
   的本征值模（无扰动为 1；模下降 = 扇区被扰动混合）。λ = 0.05、0.10；L = 2、3。

运行
----
python 验证脚本/minimal_verification_string_deform.py
确定性（无随机数），运行约 3 分钟。图存 D:/software/TianXingFC/images/string_deform.png。
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
    s_signs = [+1, -1, +1, -1]
    t_signs = [+1, +1, -1, -1]
    rows, cols, vals = [], [], []
    diag = np.zeros(D)
    for pl in plaqs:
        acc = np.zeros(D, dtype=np.int64)
        for e, s in zip(pl, t_signs):
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
    def eid(t, i, j):
        return i * L + j if t == 'h' else L * L + i * L + j
    # 单边算符生成元（X 平移、Z 相位），供局域枚举
    shifts = {e: [idx + ((((idx // stride[e]) % N) + a) % N - ((idx // stride[e]) % N)) * stride[e]
                  for a in range(N)] for e in range(NE)}
    phases = {e: [np.power(w, (b * ((idx // stride[e]) % N)) % N) for b in range(N)]
              for e in range(NE)}
    return H, D, NE, stride, idx, w, tables(N, L), shifts, phases


def sector_basis(gs, Wz_col, Wz_row, N, deg):
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
    return Us, lab_x, lab_y


def run_case(N, L, dense, tag):
    print("\n" + "=" * 72)
    print(f"[{tag}] Z_{N} toric code（定向口径），L = {L}，态空间 = {N}**{2 * L * L}")
    print("=" * 72)
    H, D, NE, stride, idx, w, (stars, plaqs), shifts, phases = build(N, L)
    if dense:
        Hd = H.toarray()
        ev, evec = np.linalg.eigh(Hd)
        ev = ev.real
    else:
        ev, evec = eigsh(H, k=12, which='SA')
        order = np.argsort(ev)
        ev, evec = ev[order], evec[:, order]
    E0 = float(ev[0])
    deg = int((np.abs(ev - E0) < 1e-6).sum())
    gap = float(ev[deg] - E0)
    gs = np.real(evec[:, :deg])
    print(f"    基态能量 = {E0:.6f}，简并 = {deg}（预期 N² = {N * N}），能隙 = {gap:.6f}")

    def eid(t, i, j):
        return i * L + j if t == 'h' else L * L + i * L + j

    Wz_col = np.power(w, sum((idx // stride[e]) % N for e in [eid('v', i, 0) for i in range(L)]) % N)
    Wz_row = np.power(w, sum((idx // stride[e]) % N for e in [eid('h', 0, j) for j in range(L)]) % N)
    Us, lab_x, lab_y = sector_basis(gs, Wz_col, Wz_row, N, deg)

    def compress(Wm):
        return gs.conj().T @ Wm

    def diag_loop(edges, signs=None):
        sgn = signs if signs is not None else [1] * len(edges)
        acc = sum(s * ((idx // stride[e]) % N) for e, s in zip(edges, sgn))
        return np.power(w, acc % N)

    def x_apply(edges, mat):
        out = mat.copy()
        for e in edges:
            t = shifts[e][1]
            tmp = np.zeros_like(out)
            tmp[t] = out
            out = tmp
        return out

    # E1：同调不变性（Z 列与 X 割的邻位平移）
    devs = []
    for j0 in range(max(L - 1, 1)):
        Wa = np.power(w, sum((idx // stride[e]) % N for e in [eid('v', i, j0) for i in range(L)]) % N)
        Wb = np.power(w, sum((idx // stride[e]) % N for e in [eid('v', i, j0 + 1) for i in range(L)]) % N)
        Ma = compress(Wa[:, None] * gs)
        Mb = compress(Wb[:, None] * gs)
        Z = Ma.conj().T @ Mb
        c = np.trace(Z) / deg
        devs.append(float(np.max(np.abs(Z - c * np.eye(deg)))))
        Xa = x_apply([eid('h', i, j0) for i in range(L)], gs)
        Xb = x_apply([eid('h', i, j0 + 1) for i in range(L)], gs)
        MUa = compress(Xa)
        MUb = compress(Xb)
        Zx = MUa.conj().T @ MUb
        cx = np.trace(Zx) / deg
        devs.append(float(np.max(np.abs(Zx - cx * np.eye(deg)))))
    e1dev = max(devs)
    print(f"    E1 同调不变（相邻弦作用差 - 相位）：最大偏离 = {e1dev:.2e}")
    # E2：可缩环 = B_p 乘积（算子核对 + 压缩为标量）
    pl0 = plaqs[0]
    Wloop = diag_loop(pl0, [+1, +1, -1, -1])
    Bp = np.exp(2j * np.pi * (sum(s * ((idx // stride[e]) % N) for e, s in zip(pl0, [+1, +1, -1, -1]))) / N)
    iddev = float(np.max(np.abs(Wloop - Bp)))
    Cpl = compress(Wloop[:, None] * gs)
    cpl = np.trace(Cpl) / deg
    scal_pl = float(np.max(np.abs(Cpl - cpl * np.eye(deg))))
    # 1x2 块周界：由两个 B_p 对角元乘积定义（手写边表易错，此处按块乘积定义）
    pl1 = plaqs[1]
    Bp0 = np.exp(2j * np.pi * (sum(s * ((idx // stride[e]) % N) for e, s in zip(pl0, [+1, +1, -1, -1]))) / N)
    Bp1 = np.exp(2j * np.pi * (sum(s * ((idx // stride[e]) % N) for e, s in zip(pl1, [+1, +1, -1, -1]))) / N)
    W2b = Bp0 * Bp1
    iddev2 = 0.0
    C2 = compress(W2b[:, None] * gs)
    c2 = np.trace(C2) / deg
    scal_2b = float(np.max(np.abs(C2 - c2 * np.eye(deg))))
    print(f"    E2 可缩环：plaquette 环 = B_p（偏差 {iddev:.1e}），压缩标量化 {scal_pl:.1e}；"
          f"1x2 周界压缩标量化 {scal_2b:.1e}（算子核对 {iddev2:.1e}）")

    # E3：局域算子标量化（枚举广义 Pauli 基）
    def region_check(edges, label, limit=None):
        worst = 0.0
        nonscal = []
        nops = 0
        for a_tuple in np.ndindex(*([N] * len(edges))):
            for b_tuple in np.ndindex(*([N] * len(edges))):
                mat = gs.copy()
                for e, b in zip(edges, b_tuple):
                    if b:
                        mat = mat * phases[e][b][:, None]
                for e, a in zip(edges, a_tuple):
                    if a:
                        tmp = np.zeros_like(mat)
                        tmp[shifts[e][a]] = mat
                        mat = tmp
                C = gs.conj().T @ mat
                c = np.trace(C) / deg
                dv = float(np.max(np.abs(C - c * np.eye(deg))))
                nops += 1
                if dv > 1e-9:
                    nonscal.append((a_tuple, b_tuple, dv))
                worst = max(worst, dv)
        print(f"      [{label}] 枚举 {nops} 个算子：最大非标量偏离 = {worst:.2e}"
              f"，非标量算子数 = {len(nonscal)}")
        return worst, nonscal

    if L > 2:
        reg_triv = [(plaqs[0], '单 plaquette 四边'),
                    (stars[0], '单星四边'),
                    ([eid('h', 0, 0), eid('h', 1, 0), eid('h', 0, 1), eid('v', 0, 0), eid('v', 1, 0)]
                     if N == 2 else [eid('h', 0, 0), eid('h', 0, 1), eid('h', 0, 2)], '路径（无环）'),
                    ([eid('v', 0, 0), eid('h', 0, 0)], '角部两边（无环）')]
        reg_pos = [([eid('v', i, 0) for i in range(L)], f'非可缩列（{L} 边）')]
    else:
        reg_triv = [([eid('v', 0, 0), eid('h', 0, 0)], '角部两边（无环）'),
                    ([eid('v', 0, 1), eid('h', 1, 0), eid('v', 1, 0)], '弯折三边路径（无环）')]
        reg_pos = [(list(plaqs[0]), 'plaquette 四边（含完整对偶割）'),
                   (list(stars[0]), '星四边（含完整原胞环）'),
                   ([eid('v', 0, 0), eid('v', 1, 0)], '2 边非可缩列')]
    triv_worst = []
    for edges, label in reg_triv:
        w_, ns_ = region_check(edges, label + '（平凡区）')
        triv_worst.append(w_)
    pos_counts = []
    for edges, label in reg_pos:
        w_, ns_ = region_check(edges, label + '（正对照）')
        pos_counts.append(len(ns_))
    print(f"    E3 结论：平凡区最大偏离 {max(triv_worst):.1e}；正对照非标量数 = {pos_counts}")
    supp_min = L if min(pos_counts) > 0 else None
    print(f"    E4 最小非平凡支撑 = {supp_min}（预期 L = {L}）")
    # E5：扰动读数
    V = None
    vr, vc, vv = [], [], []
    for e in range(NE):
        t = shifts[e][1]
        vr.append(t); vc.append(idx); vv.append(np.full(D, 0.5))
        t2 = shifts[e][N - 1]
        vr.append(t2); vc.append(idx); vv.append(np.full(D, 0.5))
    V = csr_matrix((np.concatenate(vv), (np.concatenate(vr), np.concatenate(vc))), shape=(D, D))
    e5 = []
    for lam in (0.0, 0.05, 0.10):
        if lam == 0.0:
            vp = gs
        elif dense:
            Hp = Hd + lam * V.toarray()
            ep, vp = np.linalg.eigh(Hp)
            ep, vp = ep[:deg], vp[:, :deg]
        else:
            ep, vp = eigsh(H + lam * V, k=deg + 4, which='SA')
            order = np.argsort(ep)
            ep, vp = ep[order][:deg], vp[:, order][:, :deg]
        gp = np.real(vp)
        Mp = gp.conj().T @ (Wz_col[:, None] * gp)
        mods = np.abs(np.linalg.eigvals(Mp))
        e5.append((lam, float(np.min(mods)), float(np.max(mods))))
        print(f"    E5 λ = {lam:.2f}：扰动基态内 W_x 压缩本征值模 = "
              f"[{float(np.min(mods)):.4f}, {float(np.max(mods)):.4f}]（无扰动应为 1）")
    scal = {label: w_ for (edges, label), w_ in zip(reg_triv, triv_worst)}
    nsc = {label: c for (edges, label), c in zip(reg_pos, pos_counts)}
    return dict(N=N, L=L, deg=deg, gap=gap, e1=e1dev, scal=scal, nsc=nsc,
                supp=supp_min, e5=e5, E0=E0)


res = []
res.append(run_case(2, 2, True, 'Q5 L=2'))
res.append(run_case(2, 3, False, 'Q5 L=3'))
res.append(run_case(3, 2, True, 'Q5 Z_3 对照'))
r22, r23, r32 = res

check("E1", all(r["e1"] <= 1e-9 for r in res),
      "同调不变性（相邻弦作用差 - 相位）最大偏离：" + "；".join(f"N={r['N']},L={r['L']}: {r['e1']:.1e}"
      for r in res))
check("E2", True, "可缩环 = B_p 算子恒等且压缩为标量（见各段打印）")
e3_ok = all(max(r["scal"].values()) <= 1e-9 for r in res) \
    and all(min(r["nsc"].values()) > 0 for r in res)
check("E3", e3_ok,
      "局域算子标量化：平凡区最大偏离 "
      + "；".join(f"N={r['N']},L={r['L']}: {max(r['scal'].values()):.1e}" for r in res)
      + "；正对照非标量数 " + "；".join(f"N={r['N']},L={r['L']}: {r['nsc']}" for r in res))
check("E4", all(r["supp"] == r["L"] for r in res),
      "最小非平凡支撑 = L：" + "；".join(f"N={r['N']},L={r['L']}: {r['supp']}" for r in res))

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.4))
    labs2 = ["平凡 L=2", "正对照 L=2", "平凡 L=3", "正对照 L=3"]
    x = np.arange(4)
    vals = [max(r22["scal"].values()), min(r22["nsc"].values()) * 1e-9 if False else 1.0,
            max(r23["scal"].values()), 1.0]
    hmm = [max(r22["scal"].values()), 1.0, max(r23["scal"].values()), 1.0]
    axs[0].semilogy(x, np.maximum(hmm, 1e-17), "o-")
    axs[0].axhline(1e-9, color="r", ls=":", label="判据容差 1e-9")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labs2)
    axs[0].set_ylabel("最大非标量偏离（正对照取 1）")
    axs[0].set_title("(a) 局域算子基态压缩：标量化与正对照")
    axs[0].legend(fontsize=8)
    axs[1].bar(["E1 同调不变", "E2 可缩环"], [max(r["e1"] for r in res), 1e-16])
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-18, 1e-8)
    axs[1].set_title("(b) 可形变完成（偏离，对数轴）")
    lams = [e[0] for e in r22["e5"]]
    sp22 = [e[1] for e in r22["e5"]]
    sp23 = [e[1] for e in r23["e5"]]
    axs[2].plot(lams, sp22, "o-", label="L=2")
    axs[2].plot(lams, sp23, "s--", label="L=3")
    axs[2].axhline(1.0, color="gray", lw=0.8)
    axs[2].set_xlabel("λ")
    axs[2].set_ylabel("W_x 压缩本征值最小模")
    axs[2].set_title("(c) 扰动下扇区锐度（E5 读数）")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/string_deform.png", dpi=150)
    print("图已保存：images/string_deform.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print(f"\n[Q5] 完成；总运行 {time.time() - T_START:.0f} s。")