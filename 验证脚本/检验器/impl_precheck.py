# -*- coding: utf-8 -*-
"""
实现层前置检查器：拦三类重复出现的实现缺陷。

出处：库内验证注记的过程记录显示，三类实现层缺陷跨线重复出现，
且每次都在实现之后才发现，事后补救的代价高于开工前拦住。

    1. 符号约定错    解析梯度或力项的正负号写反，量级正确但符号相反
    2. 反解公式错    由读数反解参数时，根号因子或幂次弄错，量纲不自洽
    3. 有限维截断    在有限维或有限档上做的读数被当作无限维或连续极限的结论

用法：
    from impl_precheck import precheck

    r = precheck(
        # 1. 符号：给两个应同号的表达式在若干点的取值对
        sign_pairs=[("dV/dalpha", dV_dalpha_num, dV_dalpha_analytic)],
        # 2. 反解：给反解函数与其解析反函数
        inverse_pairs=[("kappa", kappa_from_data, kappa_analytic)],
        # 3. 截断：给读数随规模变化的序列，须收敛到某极限
        convergence=[("r", radii, devs, expected_order=None)],
        name="label",
    )
    if not r["ok"]:
        raise AssertionError("前置检查未过：" + "；".join(r["blockers"]))

本检查器不判定物理，只回答「实现是否良定义」。任一段拦截即须先修实现、
不得进入判据循环。
"""
import math

import numpy as np

TOL = 1e-8


def _as_arr(x):
    return np.asarray(x, dtype=float).ravel()


def _relerr(a, b):
    a = _as_arr(a)
    b = _as_arr(b)
    if a.size == 0 or b.size == 0:
        return float("inf")
    if a.size != b.size:
        return float("inf")
    scale = np.maximum(np.abs(b), 1e-30)
    return float(np.max(np.abs(a - b) / scale))


def check_sign(sign_pairs, tol=TOL):
    """检查解析量与数值量是否同号同形。

    sign_pairs: [(名称, 数值序列, 解析序列), ...]
    以「两者逐点相对偏差」为判据；符号整体反号时相对偏差为 2，判否。
    """
    out = []
    for name, num, ana in sign_pairs:
        num = _as_arr(num)
        ana = _as_arr(ana)
        if num.size == 0 or ana.size == 0:
            out.append((name, float("inf"), "空序列"))
            continue
        if num.size != ana.size:
            out.append((name, float("inf"), "长度不一致"))
            continue
        # 整体反号的判据：num 与 -ana 的相对偏差
        rel = _relerr(num, ana)
        rel_flip = _relerr(num, -ana)
        if rel <= tol:
            out.append((name, rel, "同号同形"))
        elif rel_flip <= tol:
            out.append((name, rel, "整体反号（符号约定错）"))
        else:
            out.append((name, rel, "非同号亦非反号（形状不符）"))
    return out


def check_inverse(inverse_pairs, tol=TOL):
    """检查反解公式的量纲自洽。

    inverse_pairs: [(名称, 由数据反解的值, 解析预期的值), ...]
    反解正确则两者逐点相对偏差在容差内。
    反解公式把根号因子或幂次弄错时，偏差是常数倍（如 2 或 4），判否。
    """
    out = []
    for name, got, want in inverse_pairs:
        g = _as_arr(got)
        w = _as_arr(want)
        if g.size == 0 or w.size == 0 or g.size != w.size:
            out.append((name, float("inf"), "形状不可比"))
            continue
        rel = _relerr(g, w)
        note = "自洽"
        if rel > tol:
            # 查是否为常数倍错解：比值的离散度小则为常数倍错
            ratio = g / np.where(np.abs(w) > 1e-30, w, 1e-30)
            spread = float(np.max(ratio) - np.min(ratio)) if ratio.size > 1 else 0.0
            if spread <= 1e-6:
                note = "常数倍错解（比值恒为 %.6f）" % float(np.mean(ratio))
            else:
                note = "非同倍错解"
        out.append((name, rel, note))
    return out


def check_convergence(convergence, tol=TOL, expected_order=None):
    """检查读数随规模是否收敛，不收敛即不得当作连续极限的结论。

    convergence: [(名称, 规模序列, 读数序列, 期望收敛阶或 None), ...]
    判据为最后两档的相对变化小于容差（或按期望阶 p 判定比值是否趋于常数）。
    """
    out = []
    for item in convergence:
        name, scale, val = item[0], _as_arr(item[1]), _as_arr(item[2])
        order = item[3] if len(item) > 3 else expected_order
        if scale.size < 3 or val.size != scale.size:
            out.append((name, float("inf"), "点数不足或长度不一致"))
            continue
        # 按规模排序，取最大的两档比
        idx = np.argsort(scale)
        s2, v2 = scale[idx][-2:], val[idx][-2:]
        if abs(v2[0]) < 1e-30:
            out.append((name, float("inf"), "读数近零，比值不可算"))
            continue
        ratio = float(v2[1] / v2[0])
        s_ratio = float(s2[1] / s2[0])
        if order is not None:
            # 期望读数按 s^-p 衰减，则比值应趋于 s_ratio^-p
            want = s_ratio ** (-order)
            rel = abs(ratio - want) / max(abs(want), 1e-30)
            note = "按阶 %g 判，比值相对偏差 %.3e" % (order, rel)
            ok = rel <= tol
        else:
            # 无期望阶：只判最后两档的比值是否明显偏离 1（即未收敛）
            rel = abs(ratio - 1.0)
            note = "末两档比值 %.6f" % ratio
            ok = rel <= tol
        out.append((name, rel, note, ok))
    return out


def precheck(sign_pairs=None, inverse_pairs=None, convergence=None,
             name="label", tol=TOL, verbose=True):
    """三段合一的前置检查。任一段拦截即 ok=False。"""
    sign_pairs = sign_pairs or []
    inverse_pairs = inverse_pairs or []
    convergence = convergence or []

    blockers = []

    if sign_pairs:
        for nm, rel, note in check_sign(sign_pairs, tol):
            if "同号同形" not in note:
                blockers.append("符号检查 %s：%s（相对偏差 %.3e）" % (nm, note, rel))

    if inverse_pairs:
        for nm, rel, note in check_inverse(inverse_pairs, tol):
            if note != "自洽":
                blockers.append("反解检查 %s：%s（相对偏差 %.3e）" % (nm, note, rel))

    if convergence:
        for row in check_convergence(convergence, tol):
            nm, rel, note = row[0], row[1], row[2]
            ok = row[3] if len(row) > 3 else True
            if not ok:
                blockers.append("截断检查 %s：未收敛（%s，相对偏差 %.3e）" % (nm, note, rel))

    out = {"name": name, "ok": len(blockers) == 0, "blockers": blockers}

    if verbose:
        if sign_pairs:
            for nm, rel, note in check_sign(sign_pairs, tol):
                print("  [符号] %-22s 相对偏差 %.3e  %s" % (nm, rel, note))
        if inverse_pairs:
            for nm, rel, note in check_inverse(inverse_pairs, tol):
                print("  [反解] %-22s 相对偏差 %.3e  %s" % (nm, rel, note))
        if convergence:
            for row in check_convergence(convergence, tol):
                print("  [截断] %-22s %s" % (row[0], row[2]))
        print("  前置检查：%s" % ("通过，可进入判据循环" if out["ok"]
                                else "拦截 → " + "；".join(blockers)))
    return out


if __name__ == "__main__":
    # 回归自测：三组用例，每组含一个正对照与一个应被拦截的反例。
    # 用例本身须先核对解析关系（教训：跨轮移植或复用前必须核对前提）。

    # 第一组：符号。解析梯度 g(x)=2x，数值梯度应为 g；反例给 -g（整体反号）。
    x = np.array([0.5, 1.0, 2.0, 3.0])
    g_ana = 2.0 * x
    precheck(sign_pairs=[("dV/dx", g_ana, g_ana)],
             name="正对照：符号同形（应通过）")
    precheck(sign_pairs=[("dV/dx", -g_ana, g_ana)],
             name="反例：整体反号（应拦截）")

    # 第二组：反解。正确关系 kappa = 2；反例给 4（把 sqrt{2k} 按 sqrt{k} 反解的典型错法）。
    k_true = np.array([2.0, 2.0, 2.0])
    precheck(inverse_pairs=[("kappa", k_true, k_true)],
             name="正对照：反解自洽（应通过）")
    precheck(inverse_pairs=[("kappa", 2.0 * k_true, k_true)],
             name="反例：常数倍错解（应拦截）")

    # 第三组：截断。读数按 s^-2 衰减，末两档比值应趋于 2^-2；反例给按 s^-1 的读数。
    s = np.array([10.0, 20.0, 40.0, 80.0])
    v_ok = 1.0 / s ** 2
    precheck(convergence=[("r", s, v_ok, 2)],
             name="正对照：按阶 2 收敛（应通过）")
    v_bad = 1.0 / s
    precheck(convergence=[("r", s, v_bad, 2)],
             name="反例：阶数不符（应拦截）")
