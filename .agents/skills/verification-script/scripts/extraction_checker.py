# -*- coding: utf-8 -*-
"""
EP 四段检验器：测量合法性前置检查（提取自第 498 条，可跨线复用）。

用法：
    verdict = check_extraction(O, H, B=None, name="label")
    # O 为被测算符（NxN 方阵），H 为哈密顿量，B 为目标子空间基（NxK）
    # 任一段判否则 verdict["ok"] 为 False，须停止读数并按判否含义登记。

本脚本不判定物理，只回答「提取程序是否良定义」。出处：
验证脚本/extraction_protocol.py（第 498 条）
"""
import numpy as np

TOL = 1e-10


def _comm_max(A, B):
    return float(np.max(np.abs(A @ B - B @ A)))


def _anti_max(A, B):
    return float(np.max(np.abs(A @ B + B @ A)))


def check_extraction(O, H, B=None, name="label", verbose=True):
    """四段判定：EP1 子空间正交性、EP2 提取正确性、EP3 污染量化、EP4 守恒标签自检。

    EP4 的分类顺序：先看反对易子是否为零（对易），再看对易子是否为零（反对易）。
    注意两个算符可能同时不满足任一（关系不定），此时须先定对易结构再读数。
    """
    out = {"name": name, "ep1": None, "ep2": None, "ep3": None, "ep4": None, "ok": True}

    # EP1：子空间正交性（B 不给则跳过）
    if B is not None:
        Pi = B @ B.conj().T
        out["ep1"] = bool(_comm_max(Pi, Pi.conj().T) < TOL
                          and np.max(np.abs(Pi @ Pi - Pi)) < TOL)
    # EP2：子空间内对角化的本征值须落在被测算符全谱上
    if B is not None:
        Ob = B.conj().T @ O @ B
        w = np.linalg.eigvalsh((Ob + Ob.conj().T) / 2)
        spec = np.linalg.eigvalsh((O + O.conj().T) / 2)
        out["ep2"] = bool(all(np.min(np.abs(spec - wi)) < 1e-8 for wi in w))
        out["eig_in_subspace"] = w
    # EP3：污染量化（子空间内随机组合矢的期望值极差）
    if B is not None:
        rng = np.random.default_rng(0)
        K = B.shape[1]
        seeds = [np.eye(K)[i] for i in range(min(6, K))]
        seeds += [rng.standard_normal(K) + 1j * rng.standard_normal(K) for _ in range(200)]
        exps = []
        for sd in seeds:
            v = B @ sd
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            v = v / nv
            exps.append(float(np.real(v.conj() @ (O @ v))))
        out["ep3"] = float(max(exps) - min(exps)) if exps else None

    # EP4：守恒标签自检
    cm = _comm_max(O, H)
    ac = _anti_max(O, H)
    out["ep4_comm"] = cm
    out["ep4_anti"] = ac
    if ac < TOL and cm < TOL:
        out["ep4"] = "scalar"          # 标量，两者皆零
    elif cm < TOL:
        out["ep4"] = "commuting"       # 对易子为零即对易：守恒，可在能量本征态上读期望
    elif ac < TOL:
        out["ep4"] = "anticommuting"   # 反对易子为零即反对易：非守恒，期望恒零，须报标签不读数
    else:
        out["ep4"] = "neither"         # 关系不定，须先定对易结构

    blockers = []
    if out["ep1"] is False:
        blockers.append("EP1 子空间基非正交")
    if out["ep2"] is False:
        blockers.append("EP2 算符未落在子空间内")
    if out["ep3"] is not None and out["ep3"] > 0.1:
        blockers.append(f"EP3 污染量 {out['ep3']:.3f}")
    if out["ep4"] == "anticommuting":
        blockers.append("EP4 非守恒标签，期望恒零")
    if out["ep4"] == "neither":
        blockers.append("EP4 对易关系不定")
    out["ok"] = len(blockers) == 0
    out["blockers"] = blockers

    if verbose:
        ep3 = out["ep3"]
        ep3s = "None" if ep3 is None else f"{ep3:.4f}"
        print(f"[{name}] EP1={out['ep1']} EP2={out['ep2']} EP3={ep3s} "
              f"EP4={out['ep4']}（comm {cm:.2e} / anti {ac:.2e}）")
        if blockers:
            print(f"    拦截：{'; '.join(blockers)} → 停止读数")
        else:
            print("    通过：可读数")
    return out


if __name__ == "__main__":
    # 回归自测：三对数学上确定的算符关系，用例本身须先核对对易结构与维数
    # （教训：跨轮移植或复用前必须核对前提，见 SKILL.md 第 2.6 条）
    I2 = np.eye(2)
    sz8 = np.kron(np.array([[1.0, 0], [0, -1.0]]), np.eye(4))   # 8x8
    I8 = np.eye(8)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sx8 = np.kron(sx, np.eye(4))                                # 8x8
    # 第一对：sigma_z 与 8x8 单位阵，对易（comm 0，anti 2 sigma_z ≠ 0）
    check_extraction(sz8, I8, name="正对照：sigma_z 对单位阵（对易，应通过）")
    # 第二对：sigma_z 与 sigma_x 嵌入，反对易（comm 2，anti 0）
    check_extraction(sz8, sx8, name="反对易：sigma_z 对 sigma_x 嵌入（应拦截）")
    # 第三对：sigma_z 对自身，对易（comm 0；反对易子 = 2 sigma_z ≠ 0 不影响对易判定）
    check_extraction(sz8, sz8, name="边界：sigma_z 对自身（commuting，应通过）")
