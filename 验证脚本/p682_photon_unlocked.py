# -*- coding: utf-8 -*-
r"""判据第 682 号：光子侧未锁定的模态侧表示与层级嵌入检验（P1-58 支路三）。

背景：
  - 支路三定位（2026-10-17）：光子侧载体（实轴方向场与残余 U(1) 共同旋转）住横向
    相位通道、内部空间，层级为复一维（U(1) 层）；复结构 J 在复二维（C^2）。故支路三
    是同侧跨层级问题，既不能用支路一的同层占比模板，也不能用支路二的跨侧桥。
  - 判据 672 号已建层级嵌入规则：C^2 的子空间要作为复平面层的合法像，须为 J_0 的
    本征子空间（本征复直线）；对角 U(1) 在该嵌入下保持两支投影。672 号明标未判
    「该对应是否就是光子侧的物理对应」，那需要未锁定构型在模态侧的表示。

本轮问题：
  「光子侧未锁定」此前是状态描述（残余 U(1) 共同旋转未被锁定）。本轮先把它转成
  模态侧的显式表示，再判该表示落在哪一层、与 J_0 分支是什么关系。

类型转换（判据设计第一条纪律第四步）：
  「残余 U(1) 共同旋转」-> 模态侧相位作用 U(phi) = exp(-i phi * I_2 / 2)（共同相位，
    即对角 U(1) 的那一支；对比锁定档的 R(phi) = exp(-i phi sigma_z/2) 只转一支）。
  「未锁定」-> 相位可取任意连续值（不受势阱约束），即构型对连续扰动透明。

判据设计：
  其一，对象类型前置：被问对象是「未锁定的表示」这一构型，形式取「该表示落在哪一层」
    与「该表示对 J_0 分支的作用」，不取对易子作为唯一判据。
  其二，恒真检验：须有改掉它就 FAIL 的反例。反例取「锁定档的 R(phi)」——若判据对
    共同相位与单支相位给同一答案，则判据无法区分锁定与未锁定，失效。
  其三，多档扫描：相位档与嵌入子空间档各取多档。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

J0 = 1j * SZ                       # 模态侧复结构，J^2 = -I
Pp = 0.5 * (I2 - 1j * J0)          # +i 支投影
Pm = 0.5 * (I2 + 1j * J0)          # -i 支投影

LOG = []


def rep(tag, cond, txt):
    LOG.append("[%s] %s: %s" % ("PASS" if cond else "FAIL", tag, txt))
    print(LOG[-1])
    return cond


def U_common(phi):
    """未锁定档：共同相位（对角 U(1)），两支同转。"""
    return np.exp(-1j * phi / 2.0) * I2


def R_branch(phi):
    """锁定档：单支相位（只转 z 支），对照用。"""
    return np.cos(phi / 2.0) * I2 - 1j * np.sin(phi / 2.0) * SZ


def branch_readout(M):
    """块分解：返回 (保持度, 交换度)。"""
    if np.linalg.norm(M) < 1e-14:
        return 0.0, 0.0
    keep = np.linalg.norm(Pp @ M @ Pp + Pm @ M @ Pm)
    swap = np.linalg.norm(Pp @ M @ Pm + Pm @ M @ Pp)
    tot = np.sqrt(keep ** 2 + swap ** 2)
    return float(keep / tot), float(swap / tot)


def layer_of(M):
    """判 M 落在哪一层：看它是否只作用在一条本征复直线上。

    返回 (单支度, 共同度)：
      单支度 = 1 表示 M 只转一支（R 型，住 C^2 层）
      共同度 = 1 表示 M 两支同转（U 型，可落到复平面层）
    """
    # 单支判据：M 与 J0 对易且不是 I 的倍数
    comm = np.linalg.norm(M @ J0 - J0 @ M)
    # 共同相位判据：M 正比于 I
    scalar = np.trace(M) / 2.0
    proj_err = np.linalg.norm(M - scalar * I2)
    is_common = proj_err < 1e-12
    is_single = (comm < 1e-12) and (proj_err > 1e-6)
    return float(is_single), float(is_common)


def branch_phase_gap(M):
    """两支上作用的相位差：读 M 在 e1 与 e2 上的相位。"""
    def phase_on(M, v):
        w = M @ v
        c = np.vdot(v, w) / np.vdot(v, v)
        return np.angle(c)
    e1 = np.array([1.0, 0.0], dtype=complex)
    e2 = np.array([0.0, 1.0], dtype=complex)
    d = phase_on(M, e1) - phase_on(M, e2)
    return abs((d + np.pi) % (2 * np.pi) - np.pi)


def main():
    print("=== 判据第 682 号：光子侧未锁定的模态侧表示与层级嵌入 ===")

    # ---------------- M1 冒烟：两种相位作用的代数性质 ----------------
    print("\n--- M1 冒烟：共同相位 U 与单支相位 R 的代数性质 ---")
    phi = 0.7137
    U = U_common(phi)
    R = R_branch(phi)
    rep("M1a", abs(abs(np.linalg.det(U)) - 1) < 1e-12,
        "U(phi) 行列式模为 1（%.12f）" % abs(np.linalg.det(U)))
    # U 与 J0 对易（共同相位不区分两支）
    c_u = np.linalg.norm(U @ J0 - J0 @ U)
    rep("M1b", c_u < 1e-12, "U(phi) 与 J0 对易（范数 %.2e）" % c_u)
    # U 是 I 的倍数（共同相位）
    sc = np.trace(U) / 2
    rep("M1c", np.linalg.norm(U - sc * I2) < 1e-12,
        "U(phi) 正比于 I（共同相位，两支同转）")
    # R 不是 I 的倍数（单支）
    sc_r = np.trace(R) / 2
    rep("M1d", np.linalg.norm(R - sc_r * I2) > 1e-6,
        "R(phi) 不正比于 I（单支相位，只转一支；偏离 %.4f）"
        % np.linalg.norm(R - sc_r * I2))

    # ---------------- M2 核心：层级归属 ----------------
    print("\n--- M2 核心：两种作用落在哪一层 ---")
    phis = [0.13, 0.71, 1.27, 2.09]
    rows = []
    for p_ in phis:
        s_u, c_u2 = layer_of(U_common(p_))
        s_r, c_r = layer_of(R_branch(p_))
        rows.append((p_, s_u, c_u2, s_r, c_r))
        print("    phi=%.2f  U: 单支度 %.1f 共同度 %.1f | R: 单支度 %.1f 共同度 %.1f"
              % (p_, s_u, c_u2, s_r, c_r))
    rep("M2a", all(r[2] > 0.99 for r in rows),
        "U 在四档相位下均为共同相位（共同度全 1），可落到复平面层")
    rep("M2b", all(r[3] > 0.99 for r in rows),
        "R 在四档相位下均为单支作用（单支度全 1），住 C^2 层")
    rep("M2c", all(r[1] < 0.01 for r in rows) and all(r[4] < 0.01 for r in rows),
        "两类作用在层级判据上互斥（U 的单支度全 0、R 的共同度全 0）")

    # ---------------- M3 嵌入检验（672 号规则） ----------------
    print("\n--- M3 套用 672 号嵌入规则：本征复直线 ---")
    # 本征复直线：e1（+i 支）、e2（-i 支）
    e1 = np.array([1.0, 0.0], dtype=complex)
    e2 = np.array([0.0, 1.0], dtype=complex)
    # 一般复直线（两支混合）
    gen = (e1 + e2) / np.sqrt(2.0)
    for name, v in [("e1(+i 支)", e1), ("e2(-i 支)", e2), ("一般复直线", gen)]:
        p_plus = np.linalg.norm(Pp @ v)
        p_minus = np.linalg.norm(Pm @ v)
        purity = abs(p_plus ** 2 - p_minus ** 2)
        print("    %s: |P+|^2=%.4f |P-|^2=%.4f 纯净度=%.4f" % (name, p_plus ** 2, p_minus ** 2, purity))
    pur_e1 = abs(np.linalg.norm(Pp @ e1) ** 2 - np.linalg.norm(Pm @ e1) ** 2)
    pur_gen = abs(np.linalg.norm(Pp @ gen) ** 2 - np.linalg.norm(Pm @ gen) ** 2)
    rep("M3a", pur_e1 > 0.99, "本征复直线分支纯净（e1 纯净度 %.6f）" % pur_e1)
    rep("M3b", pur_gen < 0.01, "一般复直线两支混合（纯净度 %.6f），不可作合法像" % pur_gen)

    # ---------------- M4 未锁定的判据：相位连续 ----------------
    print("\n--- M4 未锁定的判据：相位对连续扰动透明 ---")
    eps = 1e-3
    # 未锁定：phi 自由，扰动原样传递
    def response(fn, phi, eps):
        a = fn(phi)
        b = fn(phi + eps)
        # 相位响应：矩阵差范数对相位差之比，归一化到 U(0)=I 的尺度
        return np.linalg.norm(b - a) / eps
    r_unlocked = response(U_common, 0.71, eps)
    # 锁定：phi 被锁到格点，残余按 1/(1+kappa)
    def U_frozen(phi, kappa=100.0, n_fold=4):
        period = 2 * np.pi / n_fold
        phi0 = np.round(phi / period) * period
        return U_common(phi0 + (phi - phi0) / (1.0 + kappa))
    r_locked = response(U_frozen, 0.71, eps)
    print("    未锁定档响应 %.6f，锁定档响应 %.6f" % (r_unlocked, r_locked))
    # U(phi) = e^{-i phi/2} I，故 ||U(phi+eps)-U(phi)|| / eps = sqrt(2) * eps/2 / eps = 1/sqrt2
    rep("M4a", abs(r_unlocked - 1.0 / np.sqrt(2)) < 1e-6,
        "未锁定档响应 %.6f = 1/sqrt2（解析值；U=e^{-i phi/2}I 的矩阵差按 Frobenius 范数为 sqrt2*eps/2）"
        % r_unlocked)
    rep("M4b", r_locked < 1e-2,
        "锁定档响应压到 %.2e（未锁定的判据可分辨）" % r_locked)

    # ---------------- S1 支路三判语：未锁定与分支的关系 ----------------
    print("\n--- S1 支路三：未锁定的表示与 J0 分支的关系 ---")
    k_u, s_u = branch_readout(U_common(0.71))
    print("    U 的块分解：保持度 %.6f，交换度 %.6f" % (k_u, s_u))
    # 块分解对 U 与 R 同答（都判保持），故它不能用于层级归属；
    # 层级归属由 M2 的 layer_of 判（U 共同度 1、R 单支度 1），S1 改判「分支信息量」：
    # U 在每一支上的作用都是相位 e^{-i phi/2}，两支相位相同；
    # R 在 +i 支上是 e^{-i phi/2}、-i 支上是 e^{+i phi/2}，两支相位相反。
    # 故 S1 的判据取「两支相位差」：U 给 0，R 给 phi。
    gap_u = branch_phase_gap(U_common(0.71))
    gap_r = branch_phase_gap(R_branch(0.71))
    print("    两支相位差：U %.6f，R %.6f（phi=0.71）" % (gap_u, gap_r))
    rep("S1a", gap_u < 1e-12,
        "共同相位在两支上的相位差为零（%.2e）：U 不携带分支信息" % gap_u)
    rep("S1b", abs(gap_r - 0.71) < 1e-12,
        "单支相位在两支上的相位差等于 phi（%.6f）：R 携带分支信息" % gap_r)
    rep("S1c", k_u > 0.99 and k_u == s_u * 0 + 1.0,
        "块分解对 U 给保持度 %.6f，与 R 同答；故块分解不能作层级判据，"
        "层级归属须用 M2 的 layer_of（本条为判据设计边界的如实登记）" % k_u)

    # ---------------- EP5 恒真检验 ----------------
    print("\n--- EP5 恒真检验（改掉它就 FAIL） ---")
    # 反例一：若把 U 换成 R，层级判据必须给出不同答案
    s_u_ep, c_u_ep = layer_of(U_common(0.71))
    s_r_ep, c_r_ep = layer_of(R_branch(0.71))
    rep("EP5-1", (s_u_ep, c_u_ep) != (s_r_ep, c_r_ep),
        "层级判据对 U 与 R 给不同答案（U 共同、R 单支），故判据有分辨力")
    # 反例二：恒等算符也应判为共同相位（U(0)=I）
    s_id, c_id = layer_of(np.eye(2, dtype=complex))
    rep("EP5-2", c_id > 0.99 and s_id < 0.01,
        "恒等算符判为共同相位（共同度 %.1f），与 U(0)=I 一致" % c_id)
    # 反例三：若用 S_x 冒充共同相位，层级判据必须拒绝
    s_sx, c_sx = layer_of(SX)
    rep("EP5-3", c_sx < 0.01,
        "S_x 不被判为共同相位（共同度 %.1f），判据不把任意算符认作 U" % c_sx)
    # 反例四：嵌入规则的反例——非本征复直线必须被判否
    rep("EP5-4", pur_gen < 0.01 and pur_e1 > 0.99,
        "嵌入规则对一般复直线判否（%.4f）而本征复直线判是（%.4f），非恒真"
        % (pur_gen, pur_e1))

    # ---------------- 汇总 ----------------
    n_pass = sum(1 for x in LOG if x.startswith("[PASS]"))
    n_all = len(LOG)
    print("\n=== 汇总：%d/%d PASS ===" % (n_pass, n_all))

    make_plot(phis, rows, r_unlocked, r_locked, k_u, s_u, pur_e1, pur_gen)
    return n_pass, n_all


def make_plot(phis, rows, r_unlocked, r_locked, k_u, s_u, pur_e1, pur_gen):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[2] for r in rows], 0.4, label="U 共同度", color="steelblue")
    ax.bar(x + 0.2, [r[3] for r in rows], 0.4, label="R 单支度", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(["phi=%.2f" % r[0] for r in rows])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("层级判据读数")
    ax.set_title("M2: 两种作用落在不同层")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(["未锁定", "锁定"], [r_unlocked, r_locked], color=["green", "red"], alpha=0.7)
    ax.set_ylabel("相位对扰动的响应")
    ax.set_title("M4: 未锁定判据")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)

    ax = axes[2]
    phis_s1 = [0.13, 0.71, 1.27, 2.09]
    gaps_u = [branch_phase_gap(U_common(p)) for p in phis_s1]
    gaps_r = [branch_phase_gap(R_branch(p)) for p in phis_s1]
    x2 = np.arange(len(phis_s1))
    ax.bar(x2 - 0.2, gaps_u, 0.4, label="U 两支相位差", color="steelblue")
    ax.bar(x2 + 0.2, gaps_r, 0.4, label="R 两支相位差", color="salmon")
    ax.plot(x2, phis_s1, "k--o", ms=3, label="phi（基准）")
    ax.set_xticks(x2)
    ax.set_xticklabels(["phi=%.2f" % p for p in phis_s1])
    ax.set_ylabel("两支相位差")
    ax.set_title("S1: U 不携带分支信息，R 携带")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = r"D:\software\TianXingFC\images\p682_photon_unlocked.png"
    fig.savefig(out, dpi=150)
    print("图已保存: %s" % out)


if __name__ == "__main__":
    main()
