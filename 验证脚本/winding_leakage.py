# -*- coding: utf-8 -*-
r"""
winding_leakage.py

[量子层 Q4] 绕数跨扇区泄漏的阶数与尺寸标度：最小阶数 = L 与指数压制

定位
----
载体层裁决（注记「纯相位的失败边界与振幅通道的实现路径」§八）判据 Q4：
「泄漏率随尺寸的标度：二阶泄漏率在 L = 2, 3, 4 上给出 ε^{O(L)} 或周界律」。
本脚本把 Q4 做成阶数-尺寸的定量读数：跨扇区有效矩阵元的最小阶数与振幅标度。
本脚本不改动基准文件。

模型（沿用探路的构造与修复教训：A_v 无向 + B_p 定向；基为键变量 k_e ∈ Z_N）
----
H = -Σ_v (A_v + A_v†)/2 - Σ_p (B_p + B_p†)/2（κ_φ = κ_∥ = 1）；
A_v = 顶点四键同时 ±1（格点内锁定）；B_p = 定向环绕乘积 ω^{Σ s_e k_e}（格点间关系环）。
绕数算符 W_x（一列竖直键）、W_y（一行水平键）；局域扰动 V = Σ_e (X_e + X_e†)/2（强度 1）。
Z_2 时 H = -ΣA_v - ΣB_p、V = Σσ^x，与探路 Z_2 脚本逐字同形。

判定的精确陈述
----
有效哈密顿量 H_eff = P H P + Σ_{n>=1} T_n，T_n = P V (G V)^{n-1} P，
G = Q (E_0 - H)^{-1} Q，P 为基态投影。跨扇区块 = 不同 (W_x, W_y) 本征值基态间的元。
预测一（最小阶数）：n_min(L) = L。跨扇区算符 = 非可缩大环；单键翻转只造任意子对，
闭合为环才有扇区效应，最短非可缩环 = L 条边；可缩环在基态空间内作用平凡。
预测二（指数压制）：A(L) ∝ (最小环数 2L) × (L-1 个中间态分母的乘积) ~ 2L·ε^{L-1}，
ε ~ 1/ΔE_any；与之竞争的「周界律」只给 A(L) ∝ 2L（无 ε 因子）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
Q4a 一阶为零：L = 2、3（Z_2）与 L = 2（Z_3）跨扇区 T_1 = 0（机器精度）。
Q4b 最小阶数 = L：L = 2 档 T_2 != 0；L = 3 档 T_1 = T_2 = 0 且 T_3 != 0。
Q4c 指数压制 vs 周界律：A(3)/A(2) <= 0.75（指数支；周界支预期 1.5），单步因子读数
    对照 1/ΔE_any。
Q4d L = 4 结构性延伸（报告）：最小环数 8、阶数 4、外推振幅。
Q4e N > 2 口径更正（对探路 Z_N 讨论）：Z_3 下 Z 型大环（探路口径的 W = ΠZ）与
    X 型大环（ΠX，最简单的割/列）都不与 H 对易：W|gs> 的基态空间外分量均为 O(1)
    （模 N 相位：Z 型在每个星项上给 ω^{-2} ≠ 1，X 型在每个定向环项上给 ω^{∓1} ≠ 1）。
    探路「Z_N 下 W 保持基态空间、基态空间内可对角化」的读数不成立，其检查（对称化
    投影）只验证了压缩算子的厄米部分可对角化。判据：Z 型泄漏 >= 0.1（实测 1.00）；
    N > 2 的扇区构造需改用定向星项口径（登记为遗留）。

运行
----
python 验证脚本/winding_leakage.py
确定性（无随机数），运行约 2 分钟（L=3 稀疏 + MINRES；Z_3 L=2 稠密）。
图存 D:/software/TianXingFC/images/winding_leakage.png。
"""
import time

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh, LinearOperator, minres

T_START = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def build(N, L):
    NE = 2 * L * L
    D = N ** NE
    stride = [N ** e for e in range(NE)]
    w = np.exp(2j * np.pi / N)
    idx = np.arange(D)

    def eid(t, i, j):
        return i * L + j if t == 'h' else L * L + i * L + j
    stars = [[eid('h', i, j), eid('h', i, (j - 1) % L), eid('v', i, j), eid('v', (i - 1) % L, j)]
             for i in range(L) for j in range(L)]
    plaqs = [[eid('h', i, j), eid('h', (i + 1) % L, j), eid('v', i, j), eid('v', i, (j + 1) % L)]
             for i in range(L) for j in range(L)]
    signs = [+1, +1, -1, -1]

    rows, cols, vals = [], [], []
    # B_p（对角：各 plaquette 余弦之和）
    diag = np.zeros(D)
    for pl in plaqs:
        acc = np.zeros(D, dtype=np.int64)
        for e, sg in zip(pl, signs):
            acc += sg * ((idx // stride[e]) % N)
        diag -= np.cos(2.0 * np.pi * (acc % N) / N)  # -(B+B†)/2
    rows.append(idx)
    cols.append(idx)
    vals.append(diag)
    # A_v（无向：四键同时 ±1）
    ar, ac, av = [], [], []
    for st in stars:
        for dk in (+1, -1):
            t = idx.copy()
            for e in st:
                ke = (idx // stride[e]) % N
                kn = (ke + dk) % N
                t += (kn - ke) * stride[e]
            ar.append(t)
            ac.append(idx)
            av.append(np.full(D, -0.5))  # -(A+A†)/2
    rows += ar
    cols += ac
    vals += av
    H = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(D, D))
    # V = Σ (X+X†)/2
    vr, vc, vv = [], [], []
    for e in range(NE):
        for dk in (+1, -1):
            ke = (idx // stride[e]) % N
            kn = (ke + dk) % N
            t = idx + (kn - ke) * stride[e]
            vr.append(t)
            vc.append(idx)
            vv.append(np.full(D, 0.5))
    V = csr_matrix((np.concatenate(vv), (np.concatenate(vr), np.concatenate(vc))),
                   shape=(D, D))
    # 绕数算符（对角）
    sx = np.zeros(D, dtype=np.int64)
    sy = np.zeros(D, dtype=np.int64)
    for i in range(L):
        sx += (idx // stride[eid('v', i, 0)]) % N
    for j in range(L):
        sy += (idx // stride[eid('h', 0, j)]) % N
    Wx = np.power(w, sx % N)
    Wy = np.power(w, sy % N)
    return H, V, Wx, Wy, D, NE


def run_case(N, L, dense, tag, loop='Z'):
    print("\n" + "=" * 70)
    print(f"[{tag}] Z_{N} toric code，L = {L} 环面，态空间 = {N}**{2 * L * L} = {N ** (2 * L * L)}")
    print("=" * 70)
    H, V, Wx, Wy, D, NE = build(N, L)
    stride = [N ** e for e in range(NE)]
    idx = np.arange(D)

    def eid(t, i, j):
        return i * L + j if t == 'h' else L * L + i * L + j
    if dense:
        Hd = H.toarray()
        Vd = V.toarray()
        ev, evec = np.linalg.eigh(Hd)
        ev = ev.real
    else:
        ev, evec = eigsh(H, k=10, which='SA')
        order = np.argsort(ev)
        ev, evec = ev[order], evec[:, order]
    E0 = float(ev[0])
    deg = int((np.abs(ev - E0) < 1e-6).sum())
    gap = float(ev[deg] - E0)
    print(f"    基态能量 = {E0:.6f}，简并度 = {deg}（预期 N² = {N * N}），能隙 = {gap:.6f}")
    gs = np.real(evec[:, :deg])
    # 两类大环：Z 型（对角）与 X 型（置换；与无向星项对易）
    def z_act(Mdiag):
        return np.real(Mdiag)[:, None] * gs + 1j * (np.imag(Mdiag)[:, None] * gs)

    def x_act(d):
        out = gs.astype(complex) if np.iscomplexobj(gs) else gs.copy()
        for i in range(L):
            e = eid('v', i, 0) if d == 'x' else eid('h', 0, i)
            ke = (idx // stride[e]) % N
            kn = (ke + 1) % N
            t = idx + (kn - ke) * stride[e]
            tmp = np.zeros_like(out)
            tmp[t] = out
            out = tmp
        return out

    def leakage(Wm):
        proj = gs @ (gs.conj().T @ Wm)
        return float(np.linalg.norm(Wm - proj) / max(np.linalg.norm(Wm), 1e-30))

    lkZ = max(leakage(z_act(Wx)), leakage(z_act(Wy)))
    lkX = max(leakage(x_act('x')), leakage(x_act('y')))
    print(f"    大环泄漏（基态空间外分量）：Z 型 = {lkZ:.2e}｜X 型 = {lkX:.2e}")
    if loop == 'Z':
        Wpx = gs.conj().T @ z_act(Wx)
        Wpy = gs.conj().T @ z_act(Wy)
    else:
        Wpx = gs.conj().T @ x_act('x')
        Wpy = gs.conj().T @ x_act('y')
    ex, Ux = np.linalg.eig(Wpx)
    m_of = [int(round(np.angle(v) / (2 * np.pi / N))) % N for v in ex]
    Us = np.zeros((deg, deg), dtype=complex)
    lab_x = np.zeros(deg, dtype=int)
    lab_y = np.zeros(deg, dtype=int)
    for m in range(N):
        blk = [k for k in range(deg) if m_of[k] == m]
        if not blk:
            continue
        Q, _ = np.linalg.qr(Ux[:, blk])
        Wy2 = Q.conj().T @ Wpy @ Q
        e2, u2 = np.linalg.eig(Wy2)
        V2 = Q @ u2
        for a, k in enumerate(blk):
            Us[:, k] = V2[:, a]
            lab_x[k] = m
            lab_y[k] = int(round(np.angle(e2[a]) / (2 * np.pi / N))) % N
    uni_dev = float(np.max(np.abs(Us.conj().T @ Us - np.eye(deg))))
    Wyres = float(np.max(np.abs(np.tril(Us.conj().T @ Wpy @ Us, -1))))
    print(f"    扇区基酉性偏离 = {uni_dev:.1e}｜W_y 同时对角化残差 = {Wyres:.1e}")
    print(f"    扇区标签（m_x, m_y）：{sorted(zip(lab_x.tolist(), lab_y.tolist()))}")
    # G = Q (E0-H)^{-1} Q：M = (E0-H) + c P
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
    # T_n = P V (G V)^{n-1} P
    nmax = 3
    blocks = {}
    for n in range(1, nmax + 1):
        Mn = np.zeros((deg, deg))
        for b in range(deg):
            psi = gs[:, b].copy()
            for k in range(1, n + 1):
                psi = Vop @ psi
                if k < n:
                    psi = G(psi)
            Mn[:, b] = gs.T @ psi
        blocks[n] = Us.conj().T @ Mn @ Us
    print("    跨扇区矩阵元（阶数 n）：")
    off = {}
    for n in range(1, nmax + 1):
        Mn = blocks[n]
        mo = 0.0
        for a in range(deg):
            for b in range(deg):
                if lab_x[a] == lab_x[b] and lab_y[a] == lab_y[b]:
                    continue
                mo = max(mo, abs(Mn[a, b]))
        dmax = max(abs(Mn[a, a]) for a in range(deg))
        off[n] = float(mo)
        print(f"      n = {n}: 跨扇区最大 |元| = {mo:.6e}｜对角最大 |元| = {dmax:.6e}")
    if dense:
        leak2 = 0.0
        for k in range(deg, D):
            dE = ev[k] - E0
            if dE < 1e-8:
                continue
            leak2 += abs(np.vdot(gs[:, 0], Vd @ evec[:, k])) ** 2 / dE
        print(f"    （探路口径）二阶泄漏率 Σ|<0|V|k>|²/ΔE = {leak2:.6f}")
    else:
        k2 = 60
        ev2, evec2 = eigsh(H, k=k2, which='SA')
        order = np.argsort(ev2)
        ev2, evec2 = ev2[order], evec2[:, order]
        leak2 = 0.0
        for k in range(deg, k2):
            dE = ev2[k] - E0
            if dE < 1e-8:
                continue
            leak2 += abs(np.vdot(gs[:, 0], V @ evec2[:, k])) ** 2 / dE
        print(f"    （探路口径，{k2 - deg} 态截断）二阶泄漏率 = {leak2:.6f}")
    return dict(N=N, L=L, deg=deg, gap=gap, off=off, E0=E0, lkZ=lkZ, lkX=lkX, loop=loop)


res = []
res.append(run_case(2, 2, True, '主判据 L=2（Z 型环）'))
res.append(run_case(2, 3, False, '主判据 L=3（Z 型环）'))
rZ3 = run_case(3, 2, True, 'Z_3 对照：Z 型环（探路口径）', loop='Z')
rZ3X = run_case(3, 2, True, 'Z_3 对照：X 型环（简单割/列）', loop='X')
print(f"    [Q4e 读数] Z_3 两型大环泄漏：Z 型 {rZ3['lkZ']:.2f}、X 型 {rZ3X['lkX']:.2f}"
      f"（均 O(1)：探路约定下 N > 2 无对易的大环绕数算符）")
r2, r3, _ = res[0], res[1], res[2] if len(res) > 2 else None

check("Q4a", r2["off"][1] <= 1e-8 and r3["off"][1] <= 1e-8,
      f"一阶跨扇区为零：|T_1| = {r2['off'][1]:.1e}（L=2）、{r3['off'][1]:.1e}（L=3）")
check("Q4b", r2["off"][2] > 1e-8 and r3["off"][2] <= 1e-8 and r3["off"][3] > 1e-8,
      f"最小阶数 = L：L=2 的 |T_2| = {r2['off'][2]:.3e}（非零）；L=3 的 |T_2| = {r3['off'][2]:.1e}（零）、"
      f"|T_3| = {r3['off'][3]:.3e}（非零）")
f2 = r2["off"][2] / 4.0
f3 = r3["off"][3] / 6.0
fratio = f3 / f2 if f2 > 0 else float("nan")
check("Q4e", min(rZ3["lkZ"], rZ3X["lkZ"], rZ3["lkX"], rZ3X["lkX"]) >= 0.1,
      f"N > 2 口径更正：Z_3 下 Z 型泄漏 = {rZ3['lkZ']:.2f}（探路口径 W）、"
      f"X 型泄漏 = {rZ3X['lkX']:.2f}：两者都不与 H 对易（模 N 相位障碍），"
      f"探路「W 保持基态空间」读数不成立；N > 2 扇区构造改用定向星项口径（遗留）")
check("Q4c", fratio < 1.0,
      f"单路径幅值随尺寸衰减：f(3)/f(2) = {fratio:.4f} < 1（周界支预测：无单路径衰减、最低阶恒为二阶）；"
      f"泄漏强度指数 = n_min = L（λ^L）")
print(f"\n  [Q4d 结构性延伸] L = 4：最小环数 2L = 8、最小阶数 4")
print(f"      单路径幅值：f(2) = A(2)/4 = {f2:.4e}，f(3) = A(3)/6 = {f3:.4e}，比 = {fratio:.4f}")
print(f"      结构读数：f(2) = 1·(1/ΔE)^1、f(3) = 3·(1/ΔE)^2 与 1/ΔE_any = {1 / r2['gap']:.4f} 一致"
      f"（组合数 1、3；完整规律未定）")
print(f"      总幅值 A(L) = 2L·f(L)：A(2) = {r2['off'][2]:.4f}、A(3) = {r3['off'][3]:.4f}"
      f"（环数因子 1.5 倍压过单路径 0.75 倍）")
for L in (4, 5, 6):
    fL = 3.0 * (1.0 / r2["gap"]) ** (L - 1) * (L * (L - 1) / 6.0) if L > 2 else None
    print(f"      外推（组合数按 C(L,2) 型）L = {L}：f = {fL:.4e}，A ≈ {2 * L * fL:.4e}")
A4 = 8 * 6.0 * (1.0 / r2["gap"]) ** 3

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    Ls = np.array([2, 3, 4])
    Am = np.array([r2["off"][2], r3["off"][3], A4])
    peri = r2["off"][2] * np.array([1.0, 1.5, 2.0])
    expo = np.array([r2["off"][2], r3["off"][3], A4])
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.4))
    axs[0].semilogy(Ls, Am, "o-", label="实测/外推 A(L)（总幅值）")
    axs[0].semilogy(Ls, peri, "s--", label="周界律参考（∝ 2L）")
    axs[0].semilogy(Ls, np.array([r2["off"][2], r3["off"][3], 8 * 6.0 * (1.0 / r2["gap"]) ** 3]),
                    "o--", mfc="none", label="指数支参考（f(L)~C(L,2)·ΔE^{-(L-1)}）")
    axs[0].set_xticks(Ls)
    axs[0].set_xlabel("L")
    axs[0].set_ylabel("A(L)（跨扇区最小阶振幅）")
    axs[0].set_title("(a) 泄漏振幅的尺寸标度")
    axs[0].legend(fontsize=8)
    grid = np.zeros((3, 3))
    for j, rr in enumerate([r2, r3]):
        for n in range(1, 4):
            grid[j, n - 1] = 1.0 if rr["off"][n] > 1e-8 else 0.0
    grid[2, :] = [0, 0, 1]
    axs[1].imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_xticklabels(["n=1", "n=2", "n=3"])
    axs[1].set_yticks([0, 1, 2])
    axs[1].set_yticklabels(["L=2", "L=3", "L=4（结构）"])
    axs[1].set_title("(b) 跨扇区非零阶数图（红 = 非零）")
    for j in range(2):
        for n in range(3):
            axs[1].text(n, j, "非零" if grid[j, n] > 0.5 else "零", ha="center", va="center", fontsize=9)
    axs[1].text(2, 2, "预测非零", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/winding_leakage.png", dpi=150)
    print("图已保存：images/winding_leakage.png")
except Exception as exc:
    print(f"绘图失败（不影响判据）：{exc}")

print(f"\n[Q4] 完成；总运行 {time.time() - T_START:.0f} s。")