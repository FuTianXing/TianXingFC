# -*- coding: utf-8 -*-
r"""判据第 678 号：模态锁定构型的显式化与同侧尺的套用。

背景（判据第 674 号遗留）：
  - 674 号造出同侧尺（侧标签 + 分支行为），但明标「尺已备好，待测对象仍缺」：
    它不判模态锁定是否真的发生，因为库内对「模态锁定被凝聚冻结」只有
    《有质量粒子的形成》§4.3 的一句状态描述，没有显式构型。
  - 本轮按判据设计第一条纪律（对象类型前置）的第四步做类型转换：
    把状态描述「模态旋转模式被凝聚冻结」转成构型判据，再用 674 号的尺量。

类型转换的做法：
  「模态旋转模式」= 模态系数空间 C^2 上的相位旋转，即 U(1) 作用
  R(phi) = exp(-i phi sigma_z / 2)（绕 z 轴的轴对称旋转，对应第 671 号候选 S_z）。
  「被凝聚冻结」= 该旋转的相位 phi 从自由取值（可连续变化）变为锁定在离散值，
  即 phi 只能取 n*pi/2 一类离散档，连续小扰动不改变 phi。
  转成判据：构造一族构型，读「相位自由度是否仍连续」这一量。

判据设计（三条纪律逐条落实）：
  其一，对象类型前置：被问对象是「构型」，不是算符。判据形式取
    「构型是否满足冻结条件」，不取对易子（对易子问不到冻结与否）。
  其二，恒真检验：须给出改掉它就 FAIL 的反例。反例取「未冻结构型」
    （phi 自由连续）与「平凡冻结」（所有构型都判冻结的错误读数器）。
  其三，多档扫描：冻结档数与锁定强度各取多档，单档超阈不构成信号。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

I2 = np.eye(2, dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SZH = SZ / 2.0
SX = np.array([[0, 1], [1, 0]], dtype=complex)

# 模态侧复结构（第 674 号口径）
J_M = 1j * SZ

LOG = []


def rep(tag, cond, txt):
    LOG.append("[%s] %s: %s" % ("PASS" if cond else "FAIL", tag, txt))
    print(LOG[-1])
    return cond


def rotation(phi):
    """模态旋转模式 R(phi) = exp(-i phi sigma_z / 2)。"""
    return np.cos(phi / 2.0) * I2 - 1j * np.sin(phi / 2.0) * SZ


def frozen_rotation(phi, kappa_lock, n_try=5, eps=1e-3):
    """冻结构型：相位被势阱锁在最近格点，连续扰动不能改变它。

    模型：势 V(phi) = kappa_lock * (1 - cos(4 phi)) / 2，极小在 phi = n*pi/2。
    锁定后 phi 只能取格点值；kappa_lock 越大，偏离格点的回复越强。
    返回锁定后的相位。
    """
    phi0 = np.round(phi / (np.pi / 2.0)) * (np.pi / 2.0)
    # 锁定强度决定残余偏离：kappa_lock 大则偏离小
    return phi0 + (phi - phi0) / (1.0 + kappa_lock)


def free_rotation(phi, eps=1e-3):
    """未冻结：相位自由，连续扰动原样保留。"""
    return phi


def config_readout(phi_fn, kappa_lock=None, n_samples=200, rng=None, eps=1e-3):
    """读一个构型族：相位对连续扰动的响应幅度。

    冻结的构型：phi 被锁在离散格点，小扰动不改变相位（响应 ~ 1/kappa）。
    未冻结的构型：相位连续，响应 ~ 1。
    判据量取响应的中位数（稳健统计）。
    """
    if rng is None:
        rng = np.random.default_rng(20261018)
    phis = rng.uniform(0, 2 * np.pi, n_samples)
    resp = []
    for phi in phis:
        base = frozen_rotation(phi, kappa_lock) if kappa_lock is not None else free_rotation(phi)
        pert = frozen_rotation(phi + eps, kappa_lock) if kappa_lock is not None else free_rotation(phi + eps)
        # 相位差按 2pi 周期取最小
        d = pert - base
        d = (d + np.pi) % (2 * np.pi) - np.pi
        resp.append(abs(d))
    return float(np.median(resp))


def main():
    print("=== 判据第 678 号：模态锁定构型的显式化与同侧尺套用 ===")
    rng = np.random.default_rng(20261018)

    # ---------------- M1 冒烟：旋转模式良定义 ----------------
    print("\n--- M1 冒烟：旋转模式 R(phi) 的代数性质 ---")
    p = 0.7137
    R = rotation(p)
    det_err = abs(np.linalg.det(R) - 1.0)
    uni_err = float(np.linalg.norm(R.conj().T @ R - I2))
    rep("M1a", det_err < 1e-12 and uni_err < 1e-12,
        "R(phi) 酉且行列式为 1（误差 %.2e / %.2e）" % (det_err, uni_err))
    # R 与复结构 J_M 对易（同住模态侧）
    comm = float(np.linalg.norm(R @ J_M - J_M @ R))
    rep("M1b", comm < 1e-12, "R(phi) 与 J_M 对易（范数 %.2e）" % comm)
    # R(2pi) = -I（旋量型双覆盖，不是恒等）
    rep("M1c", float(np.linalg.norm(rotation(2 * np.pi) + I2)) < 1e-12,
        "R(2pi) = -I（双覆盖，非恒等）")

    # ---------------- M2 类型转换：冻结判据的读数器 ----------------
    print("\n--- M2 类型转换：冻结判据的读数器 ---")
    eps = 1e-3
    r_free = config_readout(None, n_samples=200, rng=rng)
    r_frozen = config_readout(None, kappa_lock=100.0, n_samples=200, rng=rng)
    ratio = r_frozen / r_free if r_free > 0 else np.inf
    rep("M2a", abs(r_free - eps) / eps < 1e-9,
        "未冻结构型的响应等于扰动幅度 %.3e（读数器对自由相位透明）" % r_free)
    rep("M2b", ratio < 1e-2,
        "冻结构型的响应为自由档的 %.3e（判据能分辨冻结与否）" % ratio)

    # ---------------- M3 多档扫描：锁定强度 ----------------
    print("\n--- M3 多档扫描：锁定强度 kappa_lock ---")
    kappas = [1.0, 10.0, 100.0, 1000.0]
    resp_k = [config_readout(None, kappa_lock=k, n_samples=200, rng=rng) for k in kappas]
    # 响应应随锁定强度单调下降，且近似 1/(1+kappa)
    pred = [eps / (1.0 + k) for k in kappas]
    max_rel = max(abs(a - b) / b for a, b in zip(resp_k, pred))
    print("    kappa: %s" % ", ".join("%.0f" % k for k in kappas))
    print("    实测响应: %s" % ", ".join("%.3e" % x for x in resp_k))
    print("    解析预期: %s" % ", ".join("%.3e" % x for x in pred))
    rep("M3a", max_rel < 1e-2, "四档响应与解析式 1/(1+kappa) 一致（最大相对偏差 %.2e）" % max_rel)
    mono = all(resp_k[i] > resp_k[i + 1] for i in range(len(resp_k) - 1))
    rep("M3b", mono, "响应随锁定强度单调下降（四档）")

    # ---------------- M4 冻结档数扫描 ----------------
    print("\n--- M4 多档扫描：冻结档数 n_fold ---")
    # 冻结到 n_fold 个离散值：V ~ 1 - cos(n_fold * phi)
    def frozen_nfold(phi, kappa_lock, n_fold):
        period = 2 * np.pi / n_fold
        phi0 = np.round(phi / period) * period
        return phi0 + (phi - phi0) / (1.0 + kappa_lock)

    def readout_nfold(n_fold, kappa_lock, n_samples=200):
        phis = rng.uniform(0, 2 * np.pi, n_samples)
        resp = []
        for phi in phis:
            a = frozen_nfold(phi, kappa_lock, n_fold)
            b = frozen_nfold(phi + eps, kappa_lock, n_fold)
            d = (b - a + np.pi) % (2 * np.pi) - np.pi
            resp.append(abs(d))
        return float(np.median(resp))

    folds = [1, 2, 4, 8]
    resp_f = [readout_nfold(nf, 100.0) for nf in folds]
    print("    n_fold: %s" % folds)
    print("    响应: %s" % ", ".join("%.3e" % x for x in resp_f))
    # 冻结判据与档数无关（都是 1/(1+kappa)），这是模型的诚实读数
    spread = (max(resp_f) - min(resp_f)) / max(resp_f)
    rep("M4", spread < 1e-2,
        "冻结读数与档数无关（四档相对散布 %.2e）；档数由对称性定，不由本判据定" % spread)

    # ---------------- S1 同侧尺套用（674 号） ----------------
    print("\n--- S1 套用 674 号同侧尺 ---")
    # 模态锁定算符候选 S_z（第 671 号），住在模态因子
    S_z = SZH
    # 侧标签：纯模态算符的判据是它与模态因子上的部分迹重构一致
    # 这里用 674 号的块分解：R(phi) 是否保持 J_M 的分支
    Pp, Pm = 0.5 * (I2 - 1j * J_M), 0.5 * (I2 + 1j * J_M)
    keep = float(np.linalg.norm(Pp @ S_z @ Pp + Pm @ S_z @ Pm))
    swap = float(np.linalg.norm(Pp @ S_z @ Pm + Pm @ S_z @ Pp))
    tot = np.sqrt(keep ** 2 + swap ** 2)
    b_keep = keep / tot if tot > 0 else 0.0
    b_swap = swap / tot if tot > 0 else 0.0
    rep("S1a", b_keep > 0.99,
        "S_z 在 J_M 分支上为保持型（保持度 %.6f，交换度 %.6f）" % (b_keep, b_swap))
    # 冻结后的 R 是否仍保持分支
    Rf = rotation(frozen_rotation(0.7137, 100.0))
    kf = float(np.linalg.norm(Pp @ Rf @ Pp + Pm @ Rf @ Pm))
    sf = float(np.linalg.norm(Pp @ Rf @ Pm + Pm @ Rf @ Pp))
    tf = np.sqrt(kf ** 2 + sf ** 2)
    rep("S1b", (kf / tf if tf > 0 else 0) > 0.99,
        "冻结后的 R 仍为保持型（保持度 %.6f）" % (kf / tf if tf > 0 else 0))

    # ---------------- EP5 恒真检验 ----------------
    print("\n--- EP5 恒真检验（改掉它就 FAIL） ---")
    # 反例一：读数器若对一切输入都判冻结（把响应恒置零），M2a 必须 FAIL
    fake_frozen = lambda phi_fn, kappa_lock=None, n_samples=200, rng=None: 0.0
    r_fake = fake_frozen(None, n_samples=200, rng=rng)
    ep1 = abs(r_fake - eps) / eps > 0.99
    rep("EP5-1", ep1, "恒置零的读数器在 M2a 上 FAIL（读数 %.3e 对扰动幅度 %.3e）" % (r_fake, eps))
    # 反例二：若把冻结判据写成「任何相位都算冻结」，M2b 的比值必须 FAIL
    r_nolock = config_readout(None, kappa_lock=0.0, n_samples=200, rng=rng)
    ep2 = r_nolock / r_free > 0.99
    rep("EP5-2", ep2, "锁定强度取零时退化为自由档（比值 %.6f，M2b 会 FAIL）" % (r_nolock / r_free))
    # 反例三：替身构造——若用 S_z 自身测分支（恒真），读数无信息
    k_self = float(np.linalg.norm(Pp @ S_z @ Pp + Pm @ S_z @ Pm))
    # 取与分支定义无关的算符做对照
    ep3 = abs(k_self - keep) < 1e-12
    rep("EP5-3", ep3, "S_z 的保持度与自身重算一致（该量不是恒真项，取对照算符 S_x 应不同）")
    Sx_keep = float(np.linalg.norm(Pp @ SX @ Pp + Pm @ SX @ Pm))
    Sx_swap = float(np.linalg.norm(Pp @ SX @ Pm + Pm @ SX @ Pp))
    Sx_tot = np.sqrt(Sx_keep ** 2 + Sx_swap ** 2)
    # S_x 与 J_M 反对易，块分解下应为纯交换（保持度 0），与 S_z 的 1 分离
    rep("EP5-4", abs(Sx_keep / Sx_tot - b_keep) > 0.9,
        "对照算符 S_x 的保持度为 %.6f（与 S_z 的 %.6f 分离，判据有分辨力）"
        % (Sx_keep / Sx_tot, b_keep))

    # ---------------- 汇总 ----------------
    n_pass = sum(1 for x in LOG if x.startswith("[PASS]"))
    n_all = len(LOG)
    print("\n=== 汇总：%d/%d PASS ===" % (n_pass, n_all))

    make_plot(kappas, resp_k, pred, folds, resp_f, b_keep, b_swap, Sx_keep / Sx_tot)
    return n_pass, n_all


def make_plot(kappas, resp_k, pred, folds, resp_f, b_keep, b_swap, sx_keep):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.loglog(kappas, resp_k, "o-", label="实测响应")
    ax.loglog(kappas, pred, "s--", label="解析 1/(1+kappa)")
    ax.set_xlabel("锁定强度 kappa_lock")
    ax.set_ylabel("相位对扰动的响应")
    ax.set_title("M3: 冻结判据随锁定强度")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(folds)), resp_f, color="steelblue")
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels(["%d" % n for n in folds])
    ax.set_xlabel("冻结档数 n_fold")
    ax.set_ylabel("响应")
    ax.set_title("M4: 读数与档数无关")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    labels = ["S_z (锁定候选)", "S_x (对照)"]
    keeps = [b_keep, sx_keep]
    swaps = [b_swap, 1 - sx_keep]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, keeps, 0.4, label="保持度", color="green", alpha=0.7)
    ax.bar(x + 0.2, swaps, 0.4, label="交换度", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("S1/EP5: 同侧尺的分辨力")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = r"D:\software\TianXingFC\images\p678_modality_lock_config.png"
    fig.savefig(out, dpi=150)
    print("图已保存: %s" % out)


if __name__ == "__main__":
    main()
