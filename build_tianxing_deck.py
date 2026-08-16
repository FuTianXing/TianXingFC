"""天行力学公理体系 — 一公理二定理 13页PPT"""
import sys, os
sys.path.insert(0, r"C:\Users\Administrator\.agents\skills\slide-maker\scripts")
import deckkit as dk
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "天行力学公理体系"
OUT_DIR.mkdir(exist_ok=True)

# 设计语言
DEEP   = dk.RGBColor(0x1B, 0x2A, 0x4A)
INK    = dk.RGBColor(0x2C, 0x3E, 0x50)
MUTE   = dk.RGBColor(0x7F, 0x8C, 0x9A)
GOLD   = dk.RGBColor(0xC9, 0x9E, 0x62)
ACCENT = dk.RGBColor(0x3A, 0x7C, 0xBD)
WHITE  = dk.WHITE
PALE   = dk.RGBColor(0xE8, 0xED, 0xF2)

dk.set_palette(deep=DEEP, blue=ACCENT, magenta=GOLD, mono=MUTE)
dk.FONT, dk.DISPLAY, dk.MONO, dk.EAFONT = "Calibri", "Georgia", "Consolas", "Microsoft YaHei"

prs = dk.blank_deck()
W, H = 10, 5.625

def content_slide(prs, title, kicker="", page=0):
    s = dk.add_slide(prs)
    dk.box(s, 0, 0, W, 1.1, fill=DEEP)
    if kicker:
        dk.text(s, 0.6, 0.15, 8.0, 0.25, [[(kicker, 11, GOLD, False, False, dk.FONT)]])
    dk.text(s, 0.6, 0.4, 8.8, 0.5, [[(title, 24, WHITE, True, False, dk.DISPLAY)]])
    dk.footer(s, "天行力学公理体系", page=page, )
    band = dk.content_band(s, top=1.15)
    return s, band

def section_slide(prs, title, subtitle="", page=0):
    s = dk.add_slide(prs)
    dk.box(s, 0, 0, W, H, fill=DEEP)
    dk.text(s, 0.8, 1.8, 8.4, 0.6, [[(title, 30, WHITE, True, False, dk.DISPLAY)]])
    if subtitle:
        dk.text(s, 0.8, 2.5, 8.4, 0.5, [[(subtitle, 16, GOLD, False, False, dk.FONT)]])
    dk.hrule(s, 0.8, 3.2, 2.0, color=GOLD, weight=0.04)
    dk.footer(s, "天行力学公理体系", page=page, )
    return s

# ---------- 第1页 · 封面 ----------
s = dk.add_slide(prs)
dk.box(s, 0, 0, W, H, fill=DEEP)
dk.text(s, 0.8, 1.0, 8.4, 0.4, [[("天行力学的公理体系", 16, GOLD, False, False, dk.FONT)]])
dk.text(s, 0.8, 1.5, 8.4, 1.0, [[("一公理二定理的\n精简地基", 34, WHITE, True, False, dk.DISPLAY)]])
dk.hrule(s, 0.8, 2.9, 2.5, color=GOLD, weight=0.04)
dk.text(s, 0.8, 3.3, 8.4, 0.5, [[("心物波粒二象 → 时空二象性定理 → 物理标度定理", 14, dk.RGBColor(0xAA,0xBB,0xCC), False, False, dk.FONT)]])
dk.text(s, 0.8, 4.5, 8.4, 0.3, [[("傅天行  ·  2026.07.05  ·  版本 1.0", 13, dk.RGBColor(0x8A,0x9A,0xB0), False, False, dk.FONT)]])

# ---------- 第2页 · 目录 ----------
s, (cx, cy, cw, ch) = content_slide(prs, "目录", page=2)
items = [
    ("01", "引言", "物理学的公理传统与精简目标"),
    ("02", "心物波粒二象公理", "最高本原 · 五重法则 · 元相子"),
    ("03", "定理一：时空二象性定理", "时空的四重结构 · 最小作用量子"),
    ("04", "定理二：物理标度定理", "h, c, G 的独立公设"),
    ("05", "公理体系完整结构", "一公理二定理的层级"),
    ("06", "结论", "理论意义与展望"),
]
for i, (num, label, desc) in enumerate(items):
    yy = cy + 0.1 + i * 0.7
    dk.text(s, cx, yy, 0.5, 0.3, [[(num, 16, ACCENT, True, False, dk.MONO)]])
    dk.text(s, cx + 0.6, yy, 2.5, 0.25, [[(label, 15, INK, True, False, dk.FONT)]])
    dk.text(s, cx + 0.6, yy + 0.28, 5.5, 0.25, [[(desc, 12, MUTE, False, False, dk.FONT)]])
    dk.hrule(s, cx + 0.6, yy + 0.6, 6.5, color=PALE, weight=0.008)

# ---------- 第3页 · 章节页：引言 ----------
section_slide(prs, "01  引言", "物理学的公理传统", page=3)

# ---------- 第4页 · 引言内容 ----------
s, (cx, cy, cw, ch) = content_slide(prs, "物理学的公理传统", kicker="01 引言", page=4)
# 上移内容避免footer冲突
traditions = [
    ("牛顿力学", "三大定律 + 万有引力"),
    ("拉格朗日·哈密顿", "最小作用量原理"),
    ("狭义相对论", "光速不变 + 相对性"),
    ("广义相对论", "等效原理 + 广义协变"),
    ("天行力学", "一公理二定理"),
]
for i, (name, desc) in enumerate(traditions):
    yy = cy + 0.1 + i * 0.85
    is_last = (i == 4)
    col = GOLD if is_last else ACCENT
    if i > 0:
        dk.box(s, cx + 0.3, cy + 0.1 + (i-1)*0.85 + 0.35, 0.03, 0.55, fill=col)
    dk.box(s, cx + 0.15, yy + 0.15, 0.3, 0.3, r=0.15, fill=col)
    dk.text(s, cx + 0.6, yy + 0.05, 2.5, 0.3, [[(name, 14, DEEP if not is_last else GOLD, True, False, dk.FONT)]])
    dk.text(s, cx + 0.6, yy + 0.35, 3.0, 0.3, [[(desc, 12, INK if not is_last else GOLD, False, False, dk.FONT)]])
    if is_last:
        dk.text(s, cx + 0.6, yy + 0.6, 5.0, 0.3, [([("精简至极致：", 12, ACCENT, True, False, dk.FONT), ("一条公理、两条定理", 12, ACCENT, True, False, dk.FONT)])])

rx = cx + 5.2
for j, (label, body, color) in enumerate([
    ("核心问题", "理论的地基究竟可以精简到何种程度？", GOLD),
    ("衡量标准", "概念经济性 + 逻辑严密性", ACCENT),
    ("天行力学的回答", "将此前被视为独立公设的命题\n在更底层的基础上推导出来", GOLD),
]):
    yy = cy + 0.1 + j * 0.9
    dk.box(s, rx, yy, 3.5, 0.7, fill=dk.tint(color, 0.12), r=0.08, line=color, line_w=1)
    dk.text(s, rx + 0.15, yy + 0.05, 3.2, 0.2, [[(label, 11, color, True, False, dk.MONO)]])
    dk.text(s, rx + 0.15, yy + 0.28, 3.2, 0.35, [[(body, 12, DEEP, False, False, dk.FONT)]])

# ---------- 第5页 · 章节页：总览 ----------
section_slide(prs, "一公理二定理", "天行力学公理体系的总览", page=5)

# ---------- 第6页 · 结构图 ----------
s, (cx, cy, cw, ch) = content_slide(prs, "一公理二定理：公理体系的层级结构", kicker="核心架构", page=6)
bx, bw = cx + 2.5, 4.0
dk.box(s, bx, cy + 0.1, bw, 0.8, fill=DEEP, r=0.08)
dk.text(s, bx, cy + 0.15, bw, 0.65, [
    [("▌公理", 13, GOLD, True, False, dk.MONO)],
    [("心物波粒二象及其五重法则", 18, WHITE, True, False, dk.DISPLAY)]
], align=dk.PP_ALIGN.CENTER)
dk.arrow(s, bx + bw/2, cy + 0.9, 0, 0.25, color=ACCENT, direction="down")
dk.box(s, bx - 0.5, cy + 1.15, bw + 1.0, 0.7, fill=dk.tint(ACCENT, 0.12), r=0.08, line=ACCENT, line_w=1)
dk.text(s, bx - 0.5, cy + 1.2, bw + 1.0, 0.6, [
    [("▌具体承载者：元相子（Onphasor）", 14, ACCENT, True, False, dk.FONT)],
    [("物侧：复平面动态单位圆  |  心侧：自主调谐能动性", 12, INK, False, False, dk.FONT)]
], align=dk.PP_ALIGN.CENTER)
dk.arrow(s, bx + bw/2 - 0.8, cy + 1.85, 0, 0.25, color=ACCENT, direction="down")
dk.arrow(s, bx + bw/2 + 0.8, cy + 1.85, 0, 0.25, color=ACCENT, direction="down")

t1x, t1w = cx + 0.5, 3.8
dk.box(s, t1x, cy + 2.1, t1w, 1.4, fill=WHITE, r=0.08, line=GOLD, line_w=1.5)
dk.text(s, t1x + 0.2, cy + 2.2, t1w - 0.4, 0.3, [[("定理一", 13, GOLD, True, False, dk.MONO)]])
dk.text(s, t1x + 0.2, cy + 2.5, t1w - 0.4, 0.3, [[("时空二象性定理", 15, DEEP, True, False, dk.FONT)]])
dk.text(s, t1x + 0.2, cy + 2.85, t1w - 0.4, 0.5, [[("数学时间 · 数学空间 · 物理时间 · 物理空间", 11, INK, False, False, dk.FONT)], [("最小作用量子的存在性", 11, INK, False, False, dk.FONT)]])

t2x = cx + 5.5
dk.box(s, t2x, cy + 2.1, t1w, 1.4, fill=WHITE, r=0.08, line=ACCENT, line_w=1.5)
dk.text(s, t2x + 0.2, cy + 2.2, t1w - 0.4, 0.3, [[("定理二", 13, ACCENT, True, False, dk.MONO)]])
dk.text(s, t2x + 0.2, cy + 2.5, t1w - 0.4, 0.3, [[("物理标度定理", 15, DEEP, True, False, dk.FONT)]])
dk.text(s, t2x + 0.2, cy + 2.85, t1w - 0.4, 0.5, [[("h  作用量标度", 11, INK, False, False, dk.FONT)], [("c  时空投影比率", 11, INK, False, False, dk.FONT)], [("G  时空生成刚度", 11, INK, False, False, dk.FONT)]])

dk.hrule(s, cx, cy + 3.8, cw, color=MUTE, weight=0.012)
dk.text(s, cx, cy + 3.9, cw, 0.3, [[("一个公理 + 两个定理  整个天行力学的理论大厦", 13, MUTE, False, False, dk.FONT)]], align=dk.PP_ALIGN.CENTER)

# ---------- 第7页 · 章节页：公理 ----------
section_slide(prs, "02  心物波粒二象公理", "最高公理 · 五重法则 · 元相子", page=7)

# ---------- 第8页 · 公理与五重法则 ----------
s, (cx, cy, cw, ch) = content_slide(prs, "心物波粒二象公理", kicker="02 最高公理", page=8)
dk.box(s, cx, cy + 0.1, 4.5, 1.7, fill=dk.tint(DEEP, 0.06), r=0.08, line=DEEP, line_w=1)
dk.text(s, cx + 0.2, cy + 0.2, 4.1, 0.2, [[("公理核心", 12, GOLD, True, False, dk.MONO)]])
dk.text(s, cx + 0.2, cy + 0.45, 4.1, 1.2, [
    [("存在者必以波粒两种方式共同演化和显现。", 12, DEEP, False, False, dk.FONT)],
    [("波为阳动，为连续，为关系，为普遍；", 12, DEEP, False, False, dk.FONT)],
    [("粒为阴静，为离散，为个体，为特殊。", 12, DEEP, False, False, dk.FONT)],
    [("任何存在者必定是心与物的统一体，二者从未分离。", 12, DEEP, True, False, dk.FONT)],
])

for j, (label, body, color) in enumerate([
    ("心", "自主进行方向校准与整合的纯粹能动性，无法被数学形式化", GOLD),
    ("物", "被动但可动态演化的结构，本质就是结构性的数学实在", ACCENT),
]):
    yy = cy + 2.0 + j * 0.65
    dk.box(s, cx, yy, 4.5, 0.55, fill=dk.tint(color, 0.1), r=0.05)
    dk.text(s, cx + 0.15, yy + 0.05, 0.5, 0.2, [[(label, 11, color, True, False, dk.MONO)]])
    dk.text(s, cx + 0.6, yy + 0.05, 3.7, 0.4, [[(body, 11, INK, False, False, dk.FONT)]])

rx = cx + 5.2
dk.text(s, rx, cy + 0.1, 3.5, 0.3, [[("五重法则", 14, GOLD, True, False, dk.DISPLAY)]])
dk.hrule(s, rx, cy + 0.4, 1.5, color=GOLD, weight=0.02)
laws = [("波粒互斥", "连续与离散的根本区分，不可彼此还原", ACCENT), ("波粒互补", "同一实体的两个不可分割侧面", dk.TEAL), ("波主粒从", "波驱动演化，粒承载持存", GOLD), ("波粒互根", "彼此以对方为存在条件", dk.STEEL), ("波粒互化", "一定条件下相互转化", dk.VIOLET)]
for i, (name, desc, color) in enumerate(laws):
    yy = cy + 0.55 + i * 0.65
    dk.box(s, rx, yy, 3.5, 0.55, fill=dk.tint(color, 0.08), r=0.05, line=color, line_w=0.8)
    dk.text(s, rx + 0.1, yy + 0.05, 1.3, 0.2, [[(f"{i+1}. {name}", 11, color, True, False, dk.FONT)]])
    dk.text(s, rx + 0.1, yy + 0.28, 2.0, 0.2, [[(desc, 10, INK, False, False, dk.FONT)]])

# ---------- 第9页 · 元相子 ----------
s, (cx, cy, cw, ch) = content_slide(prs, "元相子（Onphasor）", kicker="02 公理的具体承载者", page=9)
dk.box(s, cx, cy + 0.1, cw, 1.0, fill=dk.tint(GOLD, 0.1), r=0.08, line=GOLD, line_w=1)
dk.text(s, cx + 0.2, cy + 0.15, cw - 0.4, 0.2, [[("定义", 11, GOLD, True, False, dk.MONO)]])
dk.text(s, cx + 0.2, cy + 0.38, cw - 0.4, 0.55, [[("元相子是心物波粒二象在实在层面的第一个、也是最完美的具体承载者。它是心物合一的波粒耦合体。", 13, DEEP, False, False, dk.FONT)]])

dk.box(s, cx, cy + 1.4, 4.5, 1.5, fill=WHITE, r=0.08, line=ACCENT, line_w=1)
dk.text(s, cx + 0.2, cy + 1.5, 4.1, 0.2, [[("物侧  数学实体", 13, ACCENT, True, False, dk.FONT)]])
dk.hrule(s, cx + 0.2, cy + 1.75, 2.0, color=ACCENT, weight=0.015)
dk.text(s, cx + 0.2, cy + 1.85, 4.1, 0.9, [[("复平面动态单位圆", 14, DEEP, True, False, dk.FONT)], [("半径 R = h（不可分的普朗克标度）", 12, INK, False, False, dk.FONT)], [("相位以角频率 w 连续旋转", 12, INK, False, False, dk.FONT)], [("波性：虚轴相位推进  |  粒性：实轴半径承载", 12, INK, False, False, dk.FONT)]])

dk.box(s, cx + 5.2, cy + 1.4, 4.0, 1.5, fill=WHITE, r=0.08, line=GOLD, line_w=1)
dk.text(s, cx + 5.4, cy + 1.5, 3.6, 0.2, [[("心侧  纯粹能动性", 13, GOLD, True, False, dk.FONT)]])
dk.hrule(s, cx + 5.4, cy + 1.75, 2.0, color=GOLD, weight=0.015)
dk.text(s, cx + 5.4, cy + 1.85, 3.6, 0.9, [[("自主进行相位校准与调谐", 14, DEEP, True, False, dk.FONT)], [("即天行力本身", 12, GOLD, True, False, dk.FONT)], [("无法被几何化", 12, INK, False, False, dk.FONT)], [("若心可被几何化则受制于几何规定，与定义矛盾", 11, MUTE, False, False, dk.FONT)]])

dk.box(s, cx, cy + 3.2, cw, 0.5, fill=dk.tint(GOLD, 0.08), r=0.05)
dk.text(s, cx + 0.15, cy + 3.35, 1.5, 0.2, [[("名称释义", 11, GOLD, True, False, dk.MONO)]])
dk.text(s, cx + 0.15, cy + 3.55, 8.5, 0.2, [[("元 = 根本原始  ·  相 = 相位旋转（审视/相互）  ·  子 = 基本粒子（小而圆）", 11, INK, False, False, dk.FONT)]])

# ---------- 第10页 · 章节页：定理一 ----------
section_slide(prs, "03  定理一：时空二象性定理", "时空的四重结构 · 最小作用量子", page=10)

# ---------- 第11页 · 时空二象性定理 ----------
s, (cx, cy, cw, ch) = content_slide(prs, "时空二象性定理", kicker="03 定理一", page=11)
conclusions = [
    ("(i) 数学时间", "相位连续推进的绵延维度，连续的、可逆的、无始无终", ACCENT),
    ("(ii) 数学空间", "复平面实轴与虚轴张成的二维连续统", dk.TEAL),
    ("(iii) 物理时间", "自耦合临界事件编序，离散的、不可逆的、定向的", GOLD),
    ("(iv) 物理空间", "跃迁坐标的配置关系，离散的坐标格", dk.STEEL),
    ("(v) 最小作用量子", "波性侧累积达到阈值时触发临界事件", dk.VIOLET),
]
for i, (title, desc, color) in enumerate(conclusions):
    col_i = i % 3
    row_i = i // 3
    xx = cx + col_i * 3.2
    yy = cy + 0.1 + row_i * 1.9
    if i == 4:
        xx = cx + 0.1
    dk.box(s, xx, yy, 2.9, 1.5, fill=WHITE, r=0.08, line=color, line_w=1.5)
    dk.text(s, xx + 0.1, yy + 0.1, 2.7, 0.25, [[(title, 13, color, True, False, dk.FONT)]])
    dk.hrule(s, xx + 0.1, yy + 0.4, 1.5, color=color, weight=0.015)
    dk.text(s, xx + 0.1, yy + 0.5, 2.7, 0.8, [[(desc, 11, INK, False, False, dk.FONT)]])

dk.box(s, cx, cy + 3.9, cw, 0.5, fill=dk.tint(ACCENT, 0.08), r=0.05, line=ACCENT, line_w=0.8)
dk.text(s, cx + 0.15, cy + 3.95, 1.5, 0.2, [[("核心洞见", 11, ACCENT, True, False, dk.MONO)]])
dk.text(s, cx + 0.15, cy + 4.15, 8.5, 0.2, [[("时空不是单一的先验舞台，而是波粒二象在存在论层面的直接映射。元与象、连续与离散，时空本身是二象的。", 11, INK, False, False, dk.FONT)]])

# ---------- 第12页 · 定理二 ----------
s, (cx, cy, cw, ch) = content_slide(prs, "物理标度定理", kicker="04 定理二", page=12)
constants = [("h", "普朗克常数", "元相子物侧旋转的\n作用量标度", "E = hw", ACCENT), ("c", "光速", "波性侧连续演化与粒性侧\n离散跃迁的转译常数", "c = lP/tP", GOLD), ("G", "引力常数", "物理时空的生成刚度\n决定时空像素被压弯的难易程度", "lP = sqrt(hG/c^3)", dk.TEAL)]
for i, (symbol, name, desc, formula, color) in enumerate(constants):
    bx = cx + i * 3.2
    dk.box(s, bx, cy + 0.1, 2.9, 2.5, fill=WHITE, r=0.08, line=color, line_w=1.5)
    dk.text(s, bx, cy + 0.25, 2.9, 0.5, [[(symbol, 32, color, True, False, dk.DISPLAY)]], align=dk.PP_ALIGN.CENTER)
    dk.text(s, bx, cy + 0.75, 2.9, 0.3, [[(name, 14, DEEP, True, False, dk.FONT)]], align=dk.PP_ALIGN.CENTER)
    dk.hrule(s, bx + 0.4, cy + 1.1, 2.1, color=color, weight=0.015)
    dk.text(s, bx + 0.2, cy + 1.25, 2.5, 0.7, [[(desc, 11, INK, False, False, dk.FONT)]], align=dk.PP_ALIGN.CENTER)
    dk.text(s, bx + 0.2, cy + 2.0, 2.5, 0.3, [[(formula, 12, color, False, False, dk.MONO)]], align=dk.PP_ALIGN.CENTER)

dk.box(s, cx, cy + 2.9, cw, 1.0, fill=dk.tint(ACCENT, 0.06), r=0.08, line=ACCENT, line_w=1)
dk.text(s, cx + 0.15, cy + 3.0, 1.5, 0.2, [[("革命性结论", 11, ACCENT, True, False, dk.MONO)]])
dk.text(s, cx + 0.15, cy + 3.2, 8.5, 0.55, [[("引力本来就是量子的。G 参与物理时空量子（像素）的生成，它是量子时空的内在属性。量子性（h）和引力（G）在普朗克尺度是同一个物理实在——时空像素的两个不可分割的侧面。引力不需要被量子化，因为它从一开始就是量子的。", 11, INK, False, False, dk.FONT)]])

# ---------- 第13页 · 结论 ----------
s, (cx, cy, cw, ch) = content_slide(prs, "结论：一公理二定理的完整体系", kicker="总结", page=13)
dk.box(s, cx, cy + 0.1, cw, 0.85, fill=DEEP, r=0.08)
dk.text(s, cx + 0.2, cy + 0.15, cw - 0.4, 0.65, [[("▌公理", 14, GOLD, True, False, dk.MONO)], [("心物波粒二象及其五重法则", 17, WHITE, True, False, dk.DISPLAY)], [("形而上最高本原 · 最原初的心、物和理的统一体 · 万物的规定者", 12, dk.RGBColor(0xAA,0xBB,0xCC), False, False, dk.FONT)]])

dk.box(s, cx, cy + 1.15, cw - 0.2, 0.65, fill=dk.tint(ACCENT, 0.08), r=0.08, line=ACCENT, line_w=1)
dk.text(s, cx + 0.2, cy + 1.2, 4.0, 0.3, [[("定理一：时空二象性定理", 14, ACCENT, True, False, dk.FONT)]])
dk.text(s, cx + 0.2, cy + 1.5, 5.0, 0.25, [[("数学时间 · 数学空间 · 物理时间 · 物理空间 + 最小作用量子的存在性", 11, INK, False, False, dk.FONT)]])

dk.box(s, cx, cy + 2.0, cw - 0.2, 0.65, fill=dk.tint(GOLD, 0.08), r=0.08, line=GOLD, line_w=1)
dk.text(s, cx + 0.2, cy + 2.05, 4.0, 0.3, [[("定理二：物理标度定理", 14, GOLD, True, False, dk.FONT)]])
dk.text(s, cx + 0.2, cy + 2.35, 5.0, 0.25, [[("h  作用量标度  |  c  时空投影比率  |  G  时空生成刚度", 11, INK, False, False, dk.FONT)]])

dk.box(s, cx, cy + 3.0, cw, 0.8, fill=dk.tint(GOLD, 0.08), r=0.05)
dk.text(s, cx + 0.15, cy + 3.35, 1.5, 0.2, [[("理论意义", 11, GOLD, True, False, dk.MONO)]])
dk.text(s, cx + 0.15, cy + 3.25, 8.5, 0.35, [[("从宇宙的兴衰命运到量子力学的运动方程，从意识相位整合到历史周期律的数学根源，最终都立足于一个单一的、不可再化约的最高本原及其心物合一的承载者之上。", 11, INK, False, False, dk.FONT)]])

dk.hrule(s, cx, cy + 3.9, cw, color=GOLD, weight=0.015)
dk.text(s, cx, cy + 4.0, cw, 0.3, [[("天行力学在逻辑经济性上达到了物理学史上最精简的行列", 14, DEEP, True, False, dk.FONT)]], align=dk.PP_ALIGN.CENTER)

# ---------- 保存 ----------
output_path = OUT_DIR / "天行力学公理体系.pptx"
print(f"正在保存到: {output_path}")
try:
    dk.lint_layout(prs, strict=False)
except Exception as e:
    print(f"lint 警告: {e}")
prs.save(str(output_path))
print(f" 已保存: {output_path}")
print(f" 共 {len(prs.slides)} 页")
