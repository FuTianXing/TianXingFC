# -*- coding: utf-8 -*-
r"""构型自检检验器：boost / 变换 / 初值构造的实现层前置断言（可跨线复用）。

出处：第 691 号脚本 docstring 声明的 boost 构型物理正确，代码实写却在三处
同时偏离（包络放大 \gamma、相位梯度差 \gamma^2 倍、阱宽收缩 1/\gamma），
这条文档与代码的差距跨过 691 至 807 号七轮读数，整线作废重做。本检验器把
可机器判定的防线沉淀为函数，绑定自身 CONFIG 的示范实现见
验证脚本/p809_contraction_corrected.py 的 construction_selfcheck，此处为通用版。

覆盖范围：
    防线一  构造双向对表            check_construction  文档口径独立复算 vs 代码实构造
    防线二  不变量冒烟 + 恒真对照    check_invariant     正对照守恒、离壳对照须被破坏
    防线三  声明与实现单源          属流程约束（公式只写一处），不做静态扫描，见验证规范 §四
    防线四  收尾比对                用 check_construction 复跑一次，结果入注记过程记录

本检验器不判定物理，只回答「构型实现是否与声明一致、检查器是否恒真」。
任一防线拦截即不得进入判据循环。断言属实现层前置检查，不计入判据数。
"""
import numpy as np


def _rel_err(doc, code, atol=1e-12):
    d = np.asarray(doc, dtype=float).ravel()
    c = np.asarray(code, dtype=float).ravel()
    scale = np.maximum(np.abs(d), np.abs(c)) + atol
    return float(np.max(np.abs(d - c) / scale))


def check_construction(pairs, name="构型对表", rel_tol=1e-6, atol=1e-12, verbose=True):
    r"""防线一 / 防线四：文档口径独立复算值与代码实构造值逐项双向对表。

    pairs: [(量名, doc_value, code_value), ...]
        doc_value  = 按 docstring 声明口径、用独立代码路径复算的构型量（标量或数组）
        code_value = 从实际构造对象读出的同一量
    逐点相对偏差超 rel_tol 即该项判否。doc 与 code 任一侧自身算错此处无从分辨，
    故两侧都须走独立实现路径（见验证规范 §四「声明与实现单源」的反面即各抄各的副本）。
    """
    out = {"name": name, "ok": True, "blockers": [], "rel": {}}
    for nm, doc, code in pairs:
        rel = _rel_err(doc, code, atol)
        out["rel"][nm] = rel
        if rel > rel_tol:
            out["blockers"].append("%s 文档与代码分叉（相对偏差 %.3e > %.0e）"
                                   % (nm, rel, rel_tol))
    out["ok"] = len(out["blockers"]) == 0
    if verbose:
        for nm, rel in out["rel"].items():
            print("  [对表] %-22s 相对偏差 %.3e  %s"
                  % (nm, rel, "一致" if rel <= rel_tol else "分叉"))
        print("  构型对表：%s" % ("通过，可进入判据循环" if out["ok"]
                                else "拦截 → " + "；".join(out["blockers"])))
    return out


def check_invariant(invariant_fn, onshell_cfg, offshell_cfg, name="不变量冒烟",
                    onshell_max=1e-6, offshell_min=1e-2, ratio_min=1e3, verbose=True):
    r"""防线二：变换不变量须正对照守恒、离壳对照被破坏，否则检查器恒真。

    invariant_fn(cfg) -> 残差标量：构型违反不变量的程度（守恒时应为 0）。
    onshell_cfg  : 应满足不变量的构型，残差须 <= onshell_max。
    offshell_cfg : 故意破坏不变量的对照构型，残差须显著大于正对照
                   （既 > offshell_min，又 >= ratio_min × 正对照残差）；
                   否则检查器只在「正确实现」上通过，属恒真形态，拦截。
    """
    on = float(invariant_fn(onshell_cfg))
    off = float(invariant_fn(offshell_cfg))
    out = {"name": name, "onshell": on, "offshell": off, "ok": True, "blockers": []}
    if on > onshell_max:
        out["blockers"].append("正对照未守恒（残差 %.3e > %.0e），构型或不变量取错"
                               % (on, onshell_max))
    if not (off > offshell_min and off >= ratio_min * max(on, 1e-300)):
        out["blockers"].append("离壳对照残差未显著大于正对照（离 %.3e / 正 %.3e），检查器恒真"
                               % (off, on))
    out["ok"] = len(out["blockers"]) == 0
    if verbose:
        print("  [不变量] 正对照残差 %.3e / 离壳残差 %.3e" % (on, off))
        print("  不变量冒烟：%s" % ("通过，可进入判据循环" if out["ok"]
                                else "拦截 → " + "；".join(out["blockers"])))
    return out


if __name__ == "__main__":
    # 回归自测：正例须通过、691 型写错构型须被拦截、恒真检查器须被拦截。
    # 用例数值关系先核对（教训：复用前核对前提，见验证规范 §八）。
    gamma = 2.0
    sigma0 = 1.0
    print("=== 防线一：构造双向对表 ===")
    check_construction([("包络宽", sigma0 / gamma, sigma0 / gamma)],
                       name="正对照：宽度按 1/gamma 收缩（应通过）")
    check_construction([
        ("包络宽", sigma0 / gamma, sigma0 * gamma),
        ("阱宽", np.ones(4), np.ones(4) / gamma),
    ], name="反例：691 型放大 + 误收缩（应拦截）")

    print("=== 防线二：不变量冒烟 ===")
    p = np.linspace(0.1, 1.5, 8)

    def inv_shell(cfg):
        return float(np.max(np.abs(cfg["omega"] ** 2 - cfg["p"] ** 2 - 1.0)))

    def inv_tautology(cfg):
        return float(np.max(np.abs(cfg["omega"] ** 2 - cfg["omega"] ** 2)))

    on_cfg = {"omega": np.sqrt(1.0 + p ** 2), "p": p}
    off_cfg = {"omega": np.sqrt(1.5 + p ** 2), "p": p}
    check_invariant(inv_shell, on_cfg, off_cfg,
                    name="正例：在壳守恒、离壳破坏（应通过）")
    check_invariant(inv_tautology, on_cfg, off_cfg,
                    name="反例：恒真不变量（应拦截）")
