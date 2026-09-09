# -*- coding: utf-8 -*-
"""一次性工具：按评审修订更新《相位的本质》在开放问题总览中的 P4-30 条目描述。

规则（AGENTS.md 开放问题同步规范）：来源已存在则只更新状态/描述，保留历史可追溯，
不改动其他条目，不修改理论总纲基准。
"""
import io
import re
import sys

PATH = r"D:\software\TianXingFC\内部草稿\开放问题总览：元相子理论待解决问题清单.md"
MARK = "30. **相位本质的"

NEW = (
    "30. **相位本质的信息二分与三重身份的形式化**（2026-09-09 登记，2026-09-09 评审后更新描述，"
    "来源：《相位的本质：从角位置到时间、整数与规范结构的同源生成》§7.2）。五项子问题："
    "① 相位信息二分（局域微分信息与整体拓扑信息）的完备性尚未形式化，"
    "论证依赖“$S^1$ 值映射的可观测内容由微分与同调类决定”这一标准结论，"
    "若存在两类之外的信息（相位与模长耦合项）则二分需修正（本文自标条件性命题；"
    "原“三通道”表述经评审 A-1 改为二分，原论证含循环，且通道一实为通道二的路径积分）；"
    "② 三重身份（运动学、几何、拓扑）的统一函子表述未建立，"
    "$S^1$ 分别用作时间圆的源、局域标架与空间回路的源，是否存在统一的函子结构待给出（本文自标纲领性主张）；"
    "③ 相位零点全局不一致的动力学未给出，初始相位分布与新创生元相子相位校准的衔接待完成（承接 P4-19、P4-20）；"
    "④ 相位与测量的衔接：叠加态相位归属哪一层、测量中相位如何被采样为可计数事件（承接 P4-14、P4-6）；"
    "⑤ 局域信息与整体信息在格点尺度的关系：离散化可能引入新的绕数种类，"
    "与两种绕数区分（环面绕数与局域涡量）的衔接待明确。"
)


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8", newline="") as fh:
        lines = fh.read().split("\r\n")

    hits = [i for i, s in enumerate(lines) if s.startswith(MARK)]
    if len(hits) != 1:
        print("ABORT: expected exactly one P4-30 entry, found %d" % len(hits))
        return 1

    idx = hits[0]
    old = lines[idx]
    if "信息二分" in old:
        print("SKIP: already updated")
        return 0

    lines[idx] = NEW

    for bad, label in (("\u2014", "em-dash"), ("\u2015", "horiz-bar"), ("账", "账字")):
        if bad in NEW:
            print("ABORT: style problem %s" % label)
            return 1
    if re.search("不是[^。]{0,25}而是", NEW):
        print("ABORT: banned sentence pattern")
        return 1

    with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write("\r\n".join(lines))

    print("OK: P4-30 description updated (%d chars -> %d chars)" % (len(old), len(NEW)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
