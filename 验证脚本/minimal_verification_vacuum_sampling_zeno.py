# -*- coding: utf-8 -*-
"""真空采样对传播份额的无扰动判定：三映射判决（P4-46，README 第 523 条）。

模型层设定（声明）：二维方格上的单份激发振幅 psi，紧邻 hopping 的一阶幺正动力学
作为"份额 U 过程"的最小实现；双缝屏障给干涉读数；逐拍 dt = 1 对应摆程节拍。
判据检验的是三种逐拍映射对相干份额的效应，映射结构（幺正搬运 / 复制 / 投影）
与维数无关，二维双缝只为条纹读数。体系对应：真空逐拍"采样"若为映射 2 或 3，
真实光传播（干涉 + 能量守恒 + 光速传播）不可能同时成立。

判据
----
C1a（幺正搬运 - 能量守恒）：exp(-iH dt) 逐拍，全程 sum|psi|^2 偏差 < 1e-10
    （屏障与开放边界均为哈密顿内部结构，反射不破坏幺正性，故无需吸收层）。
C1b（幺正搬运 - 条纹存活）：三档波长下屏上条纹对比度 V > 0.25
    （等幅双缝理论 V >= 0.6，修正裕度 2.4 倍）。
C1c（幺正搬运 - 因果前沿）：概率质量 > 1e-12 的最远格点不超过
    初始前沿 + c_m * t + 2（c_m = 1 格/拍，与判据 522 的群速上限对应）。
C2（复制映射反例，FAIL 预期）：每拍 psi <- (I + eps*A) psi，T 拍后能量
    > 10 倍初值（份额分裂 = 能量爆炸，方案注记第一节硬伤 2 的机器化）。
C3（投影映射反例，FAIL 预期）：每拍先幺正演化再按 |psi|^2 采样单格点坍缩，
    到达屏份额 < 1e-4 且小于幺正对照的 5%（扩散化压制：sqrt(D*T) 约 7 格
    << 缝屏距 45 格，正态尾 exp(-(45/6.8)^2/2) 约 3e-10，预算裕度 7 个数量级）。

C2 与 C3 兼作 EP5 恒真检验的齿：改掉映射 1（换成 2 或 3）判据即 FAIL。
容差来源：C1a 为 expm 幺正机器精度；C1b 阈值从等幅双缝理论对比度推导；
C1c 从紧邻谱支撑的 Bessel 超指数衰减推导；C2/C3 阈值见上，裕度均 > 2 倍。
判据可逆性：判据稳健（映射类别是结构性质，换网格尺寸/波长档不翻）。
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

N = 100
BARRIER_X = 40
SLITS = [(32, 33), (46, 47)]
X0, Y0, SIG = 20, 40, 3.0
X_SCREEN = 85
Y_WIN = (28, 52)
ABSORB = 2
EPS_COPY = 0.4
N_TRIALS = 20
SEED = 20261001
K0_LIST = [2.0 * np.pi / 3.5, np.pi / 2.0, 2.0 * np.pi / 5.0]
J = 1.0 / (2.0 * np.sqrt(2.0))
C_M = 2.0 * J * np.sqrt(2.0)


def build_sites():
    slit_mask = np.zeros(N, dtype=bool)
    for a, b in SLITS:
        slit_mask[a:b + 1] = True
    sites = [(x, y) for x in range(N) for y in range(N)
             if not (x == BARRIER_X and not slit_mask[y])]
    idx = {s: i for i, s in enumerate(sites)}
    return sites, idx


def build_A(sites, idx):
    rows, cols = [], []
    for (x, y), i in idx.items():
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = idx.get((x + dx, y + dy))
            if j is not None:
                rows.append(i)
                cols.append(j)
    n = len(sites)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return A


def psi_initial(sites, idx, k0x):
    xs = np.array([s[0] for s in sites], dtype=float)
    ys = np.array([s[1] for s in sites], dtype=float)
    g = np.exp(-((xs - X0) ** 2 + (ys - Y0) ** 2) / (2.0 * SIG ** 2))
    phase = np.exp(1j * k0x * xs)
    psi = (g * phase).astype(complex)
    psi /= np.linalg.norm(psi)
    return psi


def absorb_mask(sites):
    m = np.ones(len(sites))
    for i, (x, y) in enumerate(sites):
        if x < ABSORB or x >= N - ABSORB or y < ABSORB or y >= N - ABSORB:
            m[i] = 0.65
    return m


def screen_stats(psi, sites):
    sel = [i for i, s in enumerate(sites) if s[0] == X_SCREEN]
    inten = np.abs(psi[sel]) ** 2
    ys = np.array([sites[i][1] for i in sel])
    win = (ys >= Y_WIN[0]) & (ys <= Y_WIN[1])
    arrived = inten[win].sum()
    if arrived < 1e-12:
        return arrived, 0.0
    iw = inten[win]
    vmax, vmin = iw.max(), iw.min()
    return arrived, (vmax - vmin) / (vmax + vmin)


def front_x(psi, sites, thresh):
    xs = np.array([s[0] for s in sites])
    p = np.abs(psi) ** 2
    hit = xs[p > thresh]
    return hit.max() if len(hit) else 0


def run_unitary(A, H, sites, idx, k0x, T):
    psi = psi_initial(sites, idx, k0x)
    e0 = np.linalg.norm(psi) ** 2
    max_dev = 0.0
    front_ok = True
    for t in range(1, T + 1):
        psi = spl.expm_multiply((-1j) * H, psi)
        fx = front_x(psi, sites, 1e-12)
        if fx > FRONT0 + C_M * t + 2:
            front_ok = False
        dev = abs(np.linalg.norm(psi) ** 2 - e0)
        max_dev = max(max_dev, dev)
    arrived, V = screen_stats(psi, sites)
    return max_dev, front_ok, arrived, V


def run_copy(A, sites, idx, k0x, T):
    B = (sp.eye(A.shape[0]) + EPS_COPY * A).tocsr()
    psi = psi_initial(sites, idx, k0x)
    e0 = np.linalg.norm(psi) ** 2
    for _ in range(T):
        psi = B @ psi
    eT = np.linalg.norm(psi) ** 2
    return eT / e0


def run_projective(A, H, sites, idx, k0x, T, rng):
    psi = psi_initial(sites, idx, k0x)
    p = np.abs(psi) ** 2
    p = p / p.sum()
    for _ in range(T):
        psi = spl.expm_multiply((-1j) * H, psi)
        p = np.abs(psi) ** 2
        s = p.sum()
        if s <= 0:
            return 0.0
        n_star = rng.choice(len(psi), p=p / s)
        psi = np.zeros_like(psi)
        psi[n_star] = 1.0
    sel = [i for i, st in enumerate(sites) if st[0] == X_SCREEN]
    return float((np.abs(psi[sel]) ** 2).sum())


def main():
    global FRONT0
    print("=== 冒烟检查 ===")
    sites, idx = build_sites()
    A = build_A(sites, idx)
    H = (-J * A).tocsr()
    print(f"格点数（含双缝屏障删除）= {len(sites)} / {N * N}")
    print(f"模型光速 c_m = 2*J*sqrt(2) = {C_M:.6f}（每拍最大位移一格）")
    psi = psi_initial(sites, idx, np.pi / 2)
    FRONT0 = front_x(psi, sites, 1e-12)
    print(f"初态范数 = {np.linalg.norm(psi):.12f} | 初始前沿 x = {FRONT0:.0f}（p>1e-12）")

    print("\n=== C1a/C1b/C1c：幺正搬运（三档波长） ===")
    c1a_pass, c1b_pass, c1c_pass = True, True, True
    for k0 in K0_LIST:
        vg = 2.0 * J * np.sin(k0)
        T = int(round((X_SCREEN - X0) / vg)) + 8
        max_dev, front_ok, arrived, V = run_unitary(A, H, sites, idx, k0, T)
        ok_a = max_dev < 1e-10
        ok_b = (arrived > 1e-3) and (V > 0.25)
        ok_c = front_ok
        c1a_pass &= ok_a
        c1b_pass &= ok_b
        c1c_pass &= ok_c
        print(f"k0={k0:.4f} T={T}: 能量偏差实测 {max_dev:.2e} (<1e-10 "
              f"{'PASS' if ok_a else 'FAIL'}) | 到达份额 {arrived:.4f} | "
              f"V 实测 {V:.4f} (>0.25 {'PASS' if ok_b else 'FAIL'}) | "
              f"前沿 {'PASS' if ok_c else 'FAIL'}")
    print(f"C1a（能量守恒）: {'PASS' if c1a_pass else 'FAIL'}")
    print(f"C1b（条纹存活）: {'PASS' if c1b_pass else 'FAIL'}")
    print(f"C1c（因果前沿不超 c_m）: {'PASS' if c1c_pass else 'FAIL'}")

    print("\n=== C2：复制映射反例（FAIL 预期） ===")
    k0 = np.pi / 2
    vg = 2.0 * J * np.sin(k0)
    T = int(round((X_SCREEN - X0) / vg)) + 8
    ratio = run_copy(A, sites, idx, k0, T)
    ok2 = ratio > 10.0
    print(f"T={T} 拍后能量/初值 实测 = {ratio:.3e} | 期望 > 10（份额分裂=能量爆炸）"
          f" | {'PASS（反例成立）' if ok2 else 'FAIL'}")
    print(f"C2: {'PASS' if ok2 else 'FAIL'}")

    print("\n=== C3：投影映射反例（FAIL 预期） ===")
    rng = np.random.default_rng(SEED)
    arr_p = [run_projective(A, H, sites, idx, k0, T, rng)
             for _ in range(N_TRIALS)]
    arr_u = run_unitary(A, H, sites, idx, k0, T)[2]
    arr_proj = float(np.mean(arr_p))
    ok3 = (arr_proj < 1e-4) and (arr_proj < 0.05 * arr_u) and (arr_u > 1e-3)
    print(f"投影映射到达份额（{N_TRIALS} 次平均）实测 = {arr_proj:.3e} | "
          f"期望 < 1e-4 且 < 5% 幺正对照（扩散化压制）")
    print(f"幺正映射到达份额对照 = {arr_u:.4f} | 期望 > 1e-3")
    print(f"C3: {'PASS' if ok3 else 'FAIL'}（投影式逐拍读出与光传播矛盾，反例有齿）")

    n_pass = sum([c1a_pass, c1b_pass, c1c_pass, ok2, ok3])
    print(f"\n判据 {n_pass}/5（C1a/C1b/C1c 加 C2/C3 两项 EP5 齿）")
    print("边界声明：本验证是模型层结构判决，不重构体系的完整真空动力学；")
    print("不读真实光速，不检验缺陷-真空接口的份额转移细节（属 P1-50 通道）。")


if __name__ == "__main__":
    main()
