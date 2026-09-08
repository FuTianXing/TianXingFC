# -*- coding: utf-8 -*-
"""记忆同步与术语提案登记（复相干无序度定名）"""
import io, os

mp = "C:/Users/Administrator/.zcode/cli/memories/projects/tianxingfc-36104028b0c854b6/memory"
m = io.open(mp + "/temperature-carrier-adjudication-v3.md", encoding="utf-8").read()
m = m.replace("温度主式定案为 v3 复相干亏损载体",
              "温度主式定案为 v3 复相干无序度载体（定名：复相干无序度，初名“复相干度亏损”已弃用）")
m = m.replace("复相干度亏损", "复相干无序度")
io.open(mp + "/temperature-carrier-adjudication-v3.md", "w", encoding="utf-8").write(m)
mm = mp + "/MEMORY.md"
mm_t = io.open(mm, encoding="utf-8").read()
mm_t = mm_t.replace("v3 复相干亏损定案", "v3 复相干无序度定案（定名复相干无序度）")
io.open(mm, "w", encoding="utf-8").write(mm_t)
print("memory updated")

tp = "内部草稿/术语提案.md"
entry = (
    "\n\n## 2026-09-08 提案：复相干无序度（作者自提案，随温度主式裁决一并定名）\n\n"
    "- **术语**：复相干无序度 $D$\n"
    "- **定义**：$D = 1 - |\\langle z\\rangle|/\\sqrt{\\langle|z|^2\\rangle}$，"
    "复振幅场 $z = Pe^{i\\Phi}$ 的一阶矩与均方根场强之比的补数，温度主式 $k_BT = E_P D$ 的载体\n"
    "- **替代的初名**：复相干度亏损（弃用：晦涩且“亏损”带会计味）；正文过渡语“非相干份额”降为定义注释\n"
    "- **词族**：相位无序度 $1-\\eta$（波侧读数，A 注记既有）、幅度无序度 $1-\\xi$（粒侧读数）、"
    "复相干无序度 $D$（主量，联合两侧）；小弥散端 $D = (1-\\eta) + \\tfrac12(1-\\xi)$，"
    "圆云团下 $1-\\xi = 2(1-\\eta)$\n"
    "- **桥梁**：$|g| = |\\langle z\\rangle|/\\sqrt{\\langle|z|^2\\rangle}$ "
    "与光学复相干度同构，标准物理对照节保留“复相干度”称谓\n"
    "- **状态**：作者 2026-09-08 提名，待术语表固化确认"
)
if os.path.exists(tp):
    t2 = io.open(tp, encoding="utf-8").read()
    assert "复相干无序度" not in t2, "已登记"
    io.open(tp, "w", encoding="utf-8").write(t2 + entry)
    print("术语提案 appended")
else:
    io.open(tp, "w", encoding="utf-8").write("# 术语提案（内部草稿，待作者确认后固化术语表）\n" + entry)
    print("术语提案 created")
