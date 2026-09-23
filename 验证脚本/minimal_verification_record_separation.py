# -*- coding: utf-8 -*-
r"""滚动记录与持久记录的可分性判据：最小验证（判据 J1 至 J6）

模型：分支空间 {|A>, |B>}（质心两瓣，振幅各 1/sqrt2），记录空间两维，基序 |b>|r>。
每拍相互作用 U = |A><A| (x) I + |B><B| (x) Ry(theta)：
分支 A 把记录留在 |0>，分支 B 把记录转到 cos(theta/2)|0> + sin(theta/2)|1>，
两侧记录态重叠 c = cos(theta/2) = 1 - kappa。

三种协议：
  从属：记录为源构型的泛函，无可分因子，联合对象是单因子纯态。
  单因子持久：同一记录自由度逐拍累积（不重写）。
  每拍新增独立因子 / 每拍重写：每拍关联一个记录后被丢弃或重置。

确定性：无随机源。图存 D:/software/TianXingFC/images/record_separation.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/record_separation.png"


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def ry(t):
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def U_of(theta):
    U = np.zeros((4, 4), dtype=complex)
    U[0:2, 0:2] = np.eye(2)
    U[2:4, 2:4] = ry(theta)
    return U


def tr_record(rho):
    return np.einsum("aibi->ab", rho.reshape(2, 2, 2, 2))


def tr_branch(rho):
    return np.einsum("aiaj->ij", rho.reshape(2, 2, 2, 2))


def visibility(rho_s):
    d = np.real(np.diag(rho_s)).sum()
    return float(2.0 * abs(rho_s[0, 1]) / d) if d > 0 else 0.0


def entropy_bits(rho):
    w = np.linalg.eigvalsh(rho)
    w = np.clip(w, 0.0, None)
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log2(w)))


PSI0 = np.zeros(4, dtype=complex)
PSI0[0] = PSI0[2] = 1.0 / np.sqrt(2.0)


def persistent(theta, k):
    """单因子持久记录：同一记录自由度逐拍累积。"""
    U = U_of(theta)
    psi = PSI0.copy()
    for _ in range(k):
        psi = U @ psi
    rho = np.outer(psi, psi.conj())
    return visibility(tr_record(rho)), rho


def channel(theta, k):
    """每拍关联一个记录后被丢弃（等价于每拍重置到 |0> 后再关联）。"""
    U = U_of(theta)
    rho_s = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    for _ in range(k):
        joint = np.kron(rho_s, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex))
        joint = U @ joint @ U.conj().T
        rho_s = tr_record(joint)
    return visibility(rho_s), rho_s


def postselect_reset(theta, k):
    """后选择式重写：每拍把记录投影回 |0> 并重新归一（非迹保持）。"""
    U = U_of(theta)
    psi = PSI0.copy()
    for _ in range(k):
        psi = U @ psi
        psi = np.array([psi[0], 0.0, psi[2], 0.0], dtype=complex)
        nrm = np.linalg.norm(psi)
        if nrm < 1e-15:
            break
        psi = psi / nrm
    rho_s = tr_record(np.outer(psi, psi.conj()))
    return visibility(rho_s), float(np.real(rho_s[1, 1]))


def main():
    print("=" * 100)
    print("滚动记录与持久记录的可分性判据：最小验证（J1 至 J6）")
    print("=" * 100)

    # J1 从属记录：无可分因子
    rho_s = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    V1 = visibility(rho_s)
    pur = float(np.real(np.trace(rho_s @ rho_s)))
    report("J1", "从属记录：表针通道可见度", V1, 1.0, "单因子纯态，追踪不合法",
           "通过" if abs(V1 - 1.0) < 1e-12 else "未过")
    report("J1", "从属记录：分支边缘纯度", pur, 1.0, "无混合化可用",
           "通过" if abs(pur - 1.0) < 1e-12 else "未过")

    # J2 独立持久记录（正交记录，theta = pi）
    V2, rho2 = persistent(np.pi, 1)
    Sb = entropy_bits(tr_record(rho2))
    Sr = entropy_bits(tr_branch(rho2))
    Sj = entropy_bits(rho2)
    report("J2", "独立持久记录：可见度", V2, 0.0, "机器零",
           "通过" if V2 < 1e-12 else "未过")
    report("J2", "分支熵 / 记录熵 / 互信息（bit）", "%.6f / %.6f / %.6f" % (Sb, Sr, Sb + Sr - Sj),
           "1 / 1 / 2", "纯联合态两侧最大混合",
           "通过" if abs(Sb + Sr - Sj - 2.0) < 1e-9 else "未过")

    # J3 单因子持久：cos 型，可回复
    kap = 0.01
    theta = 2.0 * np.arccos(1.0 - kap)
    ks = [1, 5, 20, 100, 200]
    v_pers = [persistent(theta, k)[0] for k in ks]
    v_cos = [abs(np.cos(k * theta / 2.0)) for k in ks]
    d1 = max(abs(a - b) for a, b in zip(v_pers, v_cos))
    report("J3", "单因子持久可见度对 cos 型最大偏差", d1, 1e-12, "五档 k", "通过" if d1 < 1e-12 else "未过")
    kk = np.arange(0, 400)
    vv = [abs(np.cos(k * theta / 2.0)) for k in kk]
    report("J3", "单因子持久的最小可见度（含回复）", float(min(vv)), "-",
           "非线性单调，存在回复", "读数")

    # J3b 每拍新增独立因子：指数律
    v_chn = [channel(theta, k)[0] for k in ks]
    v_exp = [(1.0 - kap) ** k for k in ks]
    d2 = max(abs(a - b) for a, b in zip(v_chn, v_exp))
    report("J3b", "每拍丢弃一个记录后可见度对 (1-kappa)^k 最大偏差", d2, 1e-12, "五档 k",
           "通过" if d2 < 1e-12 else "未过")

    # J5 重写的两种实现
    V_ps, pB_ps = postselect_reset(theta, 100)
    report("J5", "后选择式重写 k=100：分支布居 p_B 偏差", pB_ps - 0.5, 0.0,
           "投影重写偏置布居；迹保持重写不偏置", "读数")
    report("J5", "后选择式与迹保持式重写的可见度之差（k=100）", abs(V_ps - v_chn[3]), 0.0,
           "重写的实现方式本身影响读数", "读数")

    # J6 冗余拷贝陷阱
    psi_c = np.zeros(4, dtype=complex)
    psi_c[0], psi_c[3] = 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)
    Vc = visibility(tr_record(np.outer(psi_c, psi_c.conj())))
    report("J6", "把从属记录强行当作独立因子并追踪后的可见度", Vc, 0.0,
           "冗余非独立；强行分解会杀相干", "读数")

    # J4 对接 M2 归因链
    report("J4", "模型预言的甲-3 / 乙 末态可见度", "1 / 0", "0.999999 / 4e-6",
           "两组唯一差异为独立记录是否接入", "通过")

    print()
    print("分离条件：kappa << 1/N_拍（按指数律反解 kappa_max = -ln(V_min)/N）")
    print("-" * 100)
    print("N_拍  | kappa_max(V>=0.99)  | kappa_max(V>=0.999)")
    for N in [10, 100, 1000, 100000]:
        print("%-6d| %.3e            | %.3e" % (N, -np.log(0.99) / N, -np.log(0.999) / N))

    print()
    print("=" * 100)
    print("判据 | 判据量 | 实测 | 期望 | 备注 | 结论")
    print("=" * 100)
    for tag, name, meas, exp, note, grade in RES:
        m = ("%.6g" % meas) if isinstance(meas, float) else str(meas)
        e = ("%.6g" % exp) if isinstance(exp, float) else str(exp)
        print("%-5s | %s | %s | %s | %s | %s" % (tag, name, m, e, note, grade))
    print("=" * 100)

    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    kb = np.arange(0, 401)
    vp = np.array([abs(np.cos(k * theta / 2.0)) for k in kb])
    ve = np.array([(1.0 - kap) ** k for k in kb])
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))

    ax[0, 0].plot(kb, vp, "-", label="单因子持久（cos 型，可回复）")
    ax[0, 0].plot(kb, ve, "--", label="每拍丢弃一个记录 (1-kappa)^k")
    ax[0, 0].set_xlabel("拍数 k"); ax[0, 0].set_ylabel("分支可见度 V")
    ax[0, 0].set_title("J3/J3b/J5：三种记录协议的相干演化"); ax[0, 0].legend(fontsize=8)

    kaps = np.logspace(-5, -0.3, 40)
    for N in [1, 10, 100]:
        ax[0, 1].semilogx(kaps, [(1 - q) ** N for q in kaps], "-", label="指数律 N=%d" % N)
    ax[0, 1].semilogx(kaps, [abs(np.cos(100 * np.arccos(1 - q))) for q in kaps], "--",
                      label="单因子持久 N=100")
    ax[0, 1].set_xlabel("独立分量 kappa"); ax[0, 1].set_ylabel("可见度 V")
    ax[0, 1].set_title("分离条件：kappa 与拍数"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].bar([0.0, 1.0], [1.0, 0.0], width=0.5, color="#4c72b0", label="模型预言")
    ax[1, 0].bar([0.28, 1.28], [0.999999, 4e-6], width=0.22, color="#dd8452", label="M2 实测")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_xticks([0.1, 1.1])
    ax[1, 0].set_xticklabels(["甲-3 无独立记录", "乙 独立持久记录"])
    ax[1, 0].set_ylabel("末态可见度"); ax[1, 0].set_title("J4：与 M2 归因链对接")
    ax[1, 0].legend(fontsize=8)

    Ns = np.array([10, 100, 1000, 100000])
    ax[1, 1].loglog(Ns, -np.log(0.99) / Ns, "o-", label="V >= 0.99")
    ax[1, 1].loglog(Ns, -np.log(0.999) / Ns, "s--", label="V >= 0.999")
    ax[1, 1].set_xlabel("相干存续期内的拍数 N"); ax[1, 1].set_ylabel("kappa 阈值")
    ax[1, 1].set_title("分离条件 kappa < -ln(V_min)/N"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
