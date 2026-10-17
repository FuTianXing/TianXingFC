# -*- coding: utf-8 -*-
"""判据第 671 号：模态锁定旋转模式的显式形式与 663 号模板的迁移适用性。

背景：
  - P1-58 结构层终局评估（2026-10-17）判定三条支路都缺前置。支路一（W/Z 的模态锁定，
    同侧）的缺口是"冻结的模态锁定旋转模式没有显式形式"：库内只有《有质量粒子的形成》
    对应表里的一句定性（"模态锁定被凝聚冻结"），模态方案第 95 行自陈三种激发通道的
    显式构造未完成。
  - 本轮原计划给该模式一个显式候选（绕 z 的轴对称锁定 S_z = sigma_z/2），并用第 663 号
    的占比模板判定。执行中发现该模板不能迁移到 4 分量时空侧，本轮如实登记这一发现。

判据：冒烟 M1 至 M4、L1 至 L2，另附两项反例。
可逆性等级：判据稳健（全部为解析构造与投影范数读数）。
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
I8 = np.eye(8, dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)

# 第 667 号构造的 4 分量时空侧
GAMMA2 = np.kron(I2, Y2)
GAMMA5 = 1j * np.kron(Z2, I2) @ np.kron(X2, I2) @ GAMMA2
G5_FULL = np.kron(I2, GAMMA5)
GAMMA2_FULL = np.kron(I2, GAMMA2)

# 663 号模板所在的模态侧：C^2 上的 J0 = L_i，其"CR"是 G5 在 C^2 上的本征分解
J_MODAL = np.kron(Z2, X2)   # 占位，实际用下述模态侧复结构
SZ_MOD = SZ
# 663 号原始设定：CR 的 4 维载体是模态系数空间 C^2 x C^2（
# 那里 J0 = 左乘 i 的 4x4 表示 L_i = ZxX? 实际 663 号用的 J0 是 4x4 矩阵 = L_i）
J0 = np.kron(Z2, X2)   # 左乘 i 在 (1,i,j,k) 基下的矩阵是 [[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]
J0 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=complex)

def qmul(a, b):
    a0, a1, a2, a3 = a; b0, b1, b2, b3 = b
    return np.array([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2,
                     a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0])


def Lmat(a):
    a0, a1, a2, a3 = a
    return np.array([[a0,-a1,-a2,-a3],[a1, a0,-a3, a2],[a2, a3, a0,-a1],[a3,-a2, a1, a0]],
                    dtype=complex)


LOG = []


def rep(tag, cond, txt):
    LOG.append(f"[{'PASS' if cond else 'FAIL'}] {tag}: {txt}")
    print(LOG[-1])


def ratio_663(M, psi):
    """第 663 号模板的原始形式（4 维模态侧）。"""
    Jp = J0 @ psi
    Qv, _ = np.linalg.qr(np.column_stack([psi, Jp]))
    P = np.eye(4) - Qv @ Qv.T
    w, V = np.linalg.eigh(P)
    H = V[:, np.argsort(-w)[:2]]
    v = H[:, 0] - 1j * (J0 @ H[:, 0])
    wv = M @ v
    pm = 0.5 * (wv + 1j * (J0 @ wv))
    return float(np.linalg.norm(pm) / np.linalg.norm(wv))


def ratio_667(M, psi):
    """把同一模板套到第 667 号构造的 4 分量时空侧。"""
    Jp = GAMMA5 @ psi
    Qv, _ = np.linalg.qr(np.column_stack([psi, Jp]))
    P = np.eye(4) - Qv @ Qv.T
    w, V = np.linalg.eigh(P)
    H = V[:, np.argsort(-w)[:2]]
    v = H[:, 0] - 1j * (GAMMA5 @ H[:, 0])
    wv = M @ v
    pm = 0.5 * (wv + 1j * (GAMMA5 @ wv))
    pp = 0.5 * (wv - 1j * (GAMMA5 @ wv))
    return float(np.linalg.norm(pm) / (np.linalg.norm(pp) + 1e-30))


p0 = np.array([1, 0, 0, 0], dtype=complex)


def main():
    print("=== 判据第 671 号：模态锁定与模板迁移适用性 ===")

    # ---------------- M1：J0（模态侧）幂等与平方
    rep("M1", np.linalg.norm(J0 @ J0 + I4) < 1e-13,
        f"模态侧 J0 = L_i 平方 -I（{np.linalg.norm(J0@J0+I4):.1e}）")

    # ---------------- M2：模板在 663 号的原域（模态侧 4 维）复现
    r_m1, r_m2 = ratio_663(I4, p0), 0.0
    rep("M2", r_m1 < 1e-12, f"模板在模态侧原域：恒等占比 {r_m1:.2e}（期望 0）")

    # ---------------- M3：模板迁移到 667 号时空侧后全部失真
    vals = [ratio_667(M, np.array([.6, .5, .4, .3], dtype=complex) / np.linalg.norm([.6, .5, .4, .3]))
            for M in (I4, GAMMA2, GAMMA5)]
    rep("M3", all(abs(v - 1) < 1e-9 for v in vals),
        f"迁移到 667 号时空侧：恒等 {vals[0]:.6f}、Gamma_2 {vals[1]:.6f}、"
        f"Gamma_5 {vals[2]:.6f}，全部给 1（恒等也判为交换，模板失真）")

    # ---------------- M4：失真的结构原因
    # 663 号模板要求载体是 C^2 模态空间，其 J0 本征空间是两个复维 1；
    # 667 号构造的 Gamma_5 在 4 分量上本征空间是两个复维 2，本征矢本身即 CR 基，
    # 导致 H[:,0] 落在单一本征空间内，投影互补关系退化为平凡。
    rep("M4", np.allclose(GAMMA5 @ GAMMA5, I4, atol=1e-13)
        and int(round(np.trace(0.5 * (I4 + GAMMA5)).real)) == 2,
        f"Gamma_5 谱为 ±1 各二（迹 {np.trace(0.5*(I4+GAMMA5)).real:.1f}），本征空间各复维 2，"
        f"与 663 号模板要求的复维 1 不符")

    # ---------------- L1：候选模态锁定算符的构造
    S_z = 0.5 * SZ
    rep("L1", np.linalg.norm(S_z @ S_z - 0.25 * I2) < 1e-13 and abs(np.trace(S_z)) < 1e-13,
        f"S_z = sigma_z/2 平方 I/4（{np.linalg.norm(S_z@S_z-0.25*I2):.1e}）、迹 0")

    # ---------------- L2：态层张量积下该算符与手征算符对易（同侧的必要条件）
    c = np.linalg.norm(np.kron(S_z, I4) @ np.kron(I2, GAMMA5)
                       - np.kron(I2, GAMMA5) @ np.kron(S_z, I4))
    rep("L2", c < 1e-13, f"[S_z x I_4, I_2 x Gamma_5] = {c:.1e}（同侧算符各住因子，"
                          f"对易；同侧判据须另立，不能靠 663 号占比模板）")

    # ---------------- 反例 1：模板在模态侧原域对酉 j/k 型左乘给交换
    Lj = Lmat(np.array([0.0, 0, 1, 0]))
    Lk = Lmat(np.array([0.0, 0, 0, 1]))
    u_ok = float(np.linalg.norm(Lj.T.conj() @ Lj - I4))
    rep("R1", abs(ratio_663(Lj, p0) - 1) < 1e-9 and abs(ratio_663(Lk, p0) - 1) < 1e-9
        and u_ok < 1e-13,
        f"反例：酉性 {u_ok:.1e}，模板在模态侧原域对酉 j/k 型左乘给交换（各 "
        f"{ratio_663(Lj, p0):.6f} 与 {ratio_663(Lk, p0):.6f}），对照 M2 的 0，"
        f"模板原域功能正常")

    # ---------------- 反例 2：态层水平丛实维
    v8 = np.zeros(8, dtype=complex)
    v8[0] = 1 / np.sqrt(2)
    v8[6] = 1 / np.sqrt(2)
    Jp = G5_FULL @ v8
    Qv, _ = np.linalg.qr(np.column_stack([v8, Jp]))
    rk = int(round(np.trace(np.eye(8) - Qv @ Qv.T).real))
    rep("R2", rk == 6, f"态层水平丛实维 {rk}（期望 6：8 分量下模板也不合法）")

    n_fail = sum(1 for x in LOG if x.startswith("[FAIL]"))
    print(f"\n汇总：{len(LOG) - n_fail}/{len(LOG)} 通过")


if __name__ == "__main__":
    main()
