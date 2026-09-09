# -*- coding: utf-8 -*-
"""一次性工具：把《相位的本质》开放问题登记到 内部草稿/开放问题总览。

规则（AGENTS.md 开放问题同步规范）：去重、标来源、保留分层、改状态不删历史、不动他人条目。
本脚本只插入一条 P4-30，不修改任何既有条目。
"""
import io
import re
import sys

PATH = r"D:\software\TianXingFC\内部草稿\开放问题总览：元相子理论待解决问题清单.md"
ANCHOR = "## 结构观察"

ENTRY = (
    "30. **相位本质的三通道分类与三重身份的形式化**（2026-09-09 登记，来源："
    "《相位的本质：从角位置到时间、整数与规范结构的同源生成》§7.2）。五项子问题："
    "① 三通道（全局差、局域梯度、闭合积分）穷尽性证明未形式化，论证前提为“比较必为两点、邻域或闭合回路之一”，"
    "若存在第四条通道（相位与模长耦合项、相位高阶导数带来的独立信息），分类需修正（本文自标条件性命题）；"
    "② 三重身份（运动学、几何、拓扑）的统一函子表述未建立，$S^1$ 分别用作底流形参数、纤维标架与映射源，"
    "是否存在统一的函子结构待给出（本文自标纲领性主张）；"
    "③ 相位零点全局不一致的动力学未给出，初始相位分布与新创生元相子相位校准的衔接待完成（承接 P4-19、P4-20）；"
    "④ 相位与测量的衔接：叠加态相位归属哪一层、测量中相位如何被采样为可计数事件（承接 P4-14、P4-6）；"
    "⑤ 通道二与通道三在格点尺度的关系：离散化可能引入新的绕数种类，"
    "与两种绕数区分（环面绕数与局域涡量）的衔接待明确。\r\n\r\n"
)


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()

    if "《相位的本质：从角位置到时间、整数与规范结构的同源生成》" in text:
        print("SKIP: entry already registered")
        return 0

    if text.count(ANCHOR) != 1:
        print("ABORT: anchor not unique (%d)" % text.count(ANCHOR))
        return 1

    new_text = text.replace(ANCHOR, ENTRY + ANCHOR, 1)

    # 自检：禁 em-dash、禁“账”字、禁“不是……而是”句式
    problems = []
    if re.search("[\u2014\u2015]", ENTRY):
        problems.append("em-dash")
    if "账" in ENTRY:
        problems.append("账字")
    if re.search("不是[^。]{0,25}而是", ENTRY):
        problems.append("禁句式")
    if problems:
        print("ABORT: style problems %s" % problems)
        return 1

    with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_text)

    print("OK: inserted P4-30; file grew %d -> %d chars" % (len(text), len(new_text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
