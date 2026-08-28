# -*- coding: utf-8 -*-
"""
生成《螺旋阶梯几何》科普信息图：一张自绘 SVG 大图 + 极少量文字的单文件 HTML。
面向对元相子理论一无所知的读者。自绘 SVG，不依赖生图 API。

布局：自上而下 5 个步骤区块，每步含「标题 + 绘图 + 极简说明」。
第一步：会转的圆（复平面单位圆/相位指针）
第二步：转满一圈沿法线跳一格（离散跃迁）
第三步：转 + 跳 = 螺旋阶梯（连续×离散对偶）
第四步：这个螺旋是实心的（D² × ℝ 实心圆柱体，维数3）
第五步：为什么必须"一步一跳"（三个后果）
"""
import os
import math
import html

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_HTML = os.path.join(OUT_DIR, "螺旋阶梯几何_信息图.html")


# ----------------------------------------------------------------------
# 配色：深色学术
# ----------------------------------------------------------------------
BG     = "#0a0e1a"
PANEL  = "rgba(17,24,39,0.62)"
PANEL2 = "#0d1424"
LINE   = "#334155"
INK    = "#e5e7eb"
INK_D  = "#94a3b8"
ACC    = "#f5c542"   # 金：物/跳跃/重点
ACC2   = "#38bdf8"   # 青：波性/旋转
ACC3   = "#a78bfa"   # 紫：心螺旋
ACC4   = "#34d399"   # 绿：阶梯/离散
GRID   = "rgba(148,163,184,0.07)"


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def svg_text(x, y, s, size=15, color=INK, weight=700, anchor="middle",
             family="'Noto Sans SC','Microsoft YaHei',sans-serif", ls=0, opacity=1.0):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-family="{family}" font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{ls}" opacity="{opacity}">{html.escape(s)}</text>')

def svg_line(x1, y1, x2, y2, color=LINE, w=1.5, dash=None, opacity=1.0, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="{cap}"{d} opacity="{opacity}"/>')

def svg_circle(cx, cy, r, stroke=LINE, sw=1.5, fill="none", dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="{stroke}" stroke-width="{sw}" '
            f'fill="{fill}"{d} opacity="{opacity}"/>')

def svg_rect(x, y, w, h, fill=PANEL, stroke=LINE, sw=1.2, rx=16, opacity=1.0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

def arrow(x1, y1, x2, y2, color=ACC, w=2.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    ang = math.atan2(y2 - y1, x2 - x1)
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    size = 10
    pts = (f"{x2},{y2} {x2+size*math.cos(a1)},{y2+size*math.sin(a1)} "
           f"{x2+size*math.cos(a2)},{y2+size*math.sin(a2)}")
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="round"{d}/>'
            f'<polygon points="{pts}" fill="{color}"/>')

def gauss_blur(sid, std=35):
    return (f'<filter id="{sid}" x="-60%" y="-60%" width="220%" height="220%">'
            f'<feGaussianBlur stdDeviation="{std}"/></filter>')


# ----------------------------------------------------------------------
# ① 一个旋转的圆
# ----------------------------------------------------------------------
def draw_step1(cx, cy):
    s = []
    s.append(svg_circle(cx, cy, 96, stroke="rgba(56,189,248,0.30)", sw=2))
    s.append(svg_circle(cx, cy, 78, stroke="rgba(56,189,248,0.15)", sw=1, dash="4 6"))
    s.append(svg_line(cx-116, cy, cx+116, cy, color="rgba(56,189,248,0.32)", w=1.2))
    s.append(svg_line(cx, cy-116, cx, cy+116, color="rgba(56,189,248,0.32)", w=1.2))
    # 单位圆（相位轨迹）
    s.append(svg_circle(cx, cy, 90, stroke=ACC2, sw=2.5))
    # 相位指针
    ang = math.radians(-54)
    px = cx + 90*math.cos(ang)
    py = cy + 90*math.sin(ang)
    s.append(svg_line(cx, cy, px, py, color=ACC, w=3))
    s.append(svg_circle(px, py, 5, stroke=ACC, sw=2, fill=ACC))
    s.append(svg_text(px+20, py-10, "相位指针", size=13, color=ACC))
    s.append(svg_text(px+20, py+4, "(一直在转)", size=11, color=INK_D, weight=400))
    # 圆心
    s.append(svg_circle(cx, cy, 5.5, stroke="#f8fafc", sw=2, fill="#f8fafc"))
    s.append(svg_text(cx, cy-18, "圆心", size=13, color=INK))
    s.append(svg_text(cx+60, cy-58, "r = ℏ", size=14, color=INK_D))
    s.append(svg_text(cx-114, cy+6, "实轴", size=12, color=INK_D, weight=400))
    s.append(svg_text(cx+8, cy-114, "虚轴", size=12, color=INK_D, weight=400))
    return "".join(s)


# ----------------------------------------------------------------------
# ② 转满一圈 → 沿法线跳一格
# ----------------------------------------------------------------------
def draw_step2(cx, cy):
    s = []
    # 下层圆盘
    s.append(svg_circle(cx, cy+118, 64, stroke="rgba(52,211,153,0.35)", sw=2))
    s.append(svg_text(cx, cy+118+92, "旋转中…", size=12, color=INK_D, weight=400))
    # 上层圆盘
    s.append(svg_circle(cx, cy-118, 64, stroke="rgba(52,211,153,0.6)", sw=2))
    s.append(svg_text(cx, cy-118-92, "跳一格", size=13, color=ACC4))
    # 法线（竖直轴）
    s.append(svg_line(cx, cy+118, cx, cy-118, color="rgba(245,197,66,0.32)", w=2, dash="6 5"))
    # 跃迁箭头
    s.append(arrow(cx, cy+88, cx, cy-88, color=ACC, w=3))
    # 阶梯横杠
    s.append(svg_line(cx-40, cy-118, cx+40, cy-118, color=ACC4, w=4, cap="round"))
    s.append(svg_line(cx-40, cy+118, cx+40, cy+118, color=ACC4, w=4, cap="round"))
    # 标签
    s.append(svg_text(cx+108, cy+40, "法线", size=13, color=ACC))
    s.append(svg_text(cx+108, cy+58, "(竖直轴)", size=11, color=INK_D, weight=400))
    s.append(svg_text(cx+108, cy-100, "位移 l_P", size=13, color=ACC4))
    s.append(svg_text(cx, cy+158, "满周闭合触发", size=12, color=INK_D, weight=400))
    return "".join(s)


# ----------------------------------------------------------------------
# ③ 转 + 跳 = 螺旋阶梯
# ----------------------------------------------------------------------
def draw_step3(cx, cy):
    s = []
    top = cy - 150
    bot = cy + 150
    n = 15
    step = (bot - top) / n
    # 实心圆盘堆叠（强调实心 D²）
    for i in range(n):
        yy = top + i*step
        rr = 60 - abs(i - n/2)*1.3
        op = 0.22 + 0.45*abs(i/n - 0.5)
        s.append(svg_circle(cx, yy, max(rr, 30), stroke="rgba(245,197,66,0.26)", sw=1.2, opacity=op))
    # 圆心轨迹（竖直虚线）
    s.append(svg_line(cx, bot, cx, top, color="rgba(248,250,252,0.45)", w=1.6, dash="4 5"))
    # 螺旋线（相位指针沿边界）
    pts = []
    for i in range(0, 240):
        t = i/239.0
        ang = t*math.pi*5.0          # 约 2.5 圈
        yy = bot - t*(bot - top)
        px = cx + 58*math.cos(ang)
        pts.append((px, yy))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{ACC2}" stroke-width="2.3" '
             f'stroke-linecap="round" opacity="0.95"/>')
    # 离散阶梯横档
    for i in range(0, 8):
        yy = top + i*((bot - top)/7.0)
        wb = 28 + i*1.3
        s.append(svg_line(cx-wb, yy, cx+wb, yy, color=ACC4, w=3.2, cap="round", opacity=0.9))
    # 标签
    s.append(svg_text(cx, top-26, "旋转（波性 · 连续）", size=13.5, color=ACC2))
    s.append(svg_text(cx, bot+30, "前进（粒性 · 离散，一步一跳）", size=13.5, color=ACC4))
    s.append(svg_text(cx, bot+52, "二者耦合 = 螺旋阶梯", size=14, color=INK))
    return "".join(s)


# ----------------------------------------------------------------------
# ④ 实心圆柱体 D² × ℝ
# ----------------------------------------------------------------------
def draw_step4(cx, cy):
    s = []
    rx, ry = 82, 26
    top_cy = cy - 132
    bot_cy = cy + 132
    # 侧面
    s.append(f'<path d="M {cx-rx} {bot_cy} L {cx-rx} {top_cy} A {rx} {ry} 0 0 1 {cx+rx} {top_cy} '
             f'L {cx+rx} {bot_cy} A {rx} {ry} 0 0 0 {cx-rx} {bot_cy} Z" '
             f'fill="rgba(56,189,248,0.08)" stroke="none"/>')
    # 下底
    s.append(f'<ellipse cx="{cx}" cy="{bot_cy}" rx="{rx}" ry="{ry}" fill="none" '
             f'stroke="rgba(56,189,248,0.4)" stroke-width="1.5"/>')
    # 上底
    s.append(f'<ellipse cx="{cx}" cy="{top_cy}" rx="{rx}" ry="{ry}" '
             f'fill="rgba(56,189,248,0.12)" stroke="{ACC2}" stroke-width="2"/>')
    # 侧边竖线
    s.append(svg_line(cx-rx, bot_cy, cx-rx, top_cy, color="rgba(56,189,248,0.55)", w=1.6))
    s.append(svg_line(cx+rx, bot_cy, cx+rx, top_cy, color="rgba(56,189,248,0.55)", w=1.6))
    # 中心轴（法线）
    s.append(svg_line(cx, bot_cy, cx, top_cy, color="rgba(245,197,66,0.6)", w=2, dash="5 5"))
    # 内部半径（表示实心 D²）
    for a in range(0, 360, 30):
        ang = math.radians(a)
        ex = cx + rx*math.cos(ang)
        ey = top_cy + ry*math.sin(ang)
        s.append(svg_line(cx, top_cy, ex, ey, color="rgba(245,197,66,0.16)", w=1))
    # 标签
    s.append(svg_text(cx-10, top_cy-34, "底面 = 圆盘 D²", size=13.5, color=ACC2))
    s.append(svg_text(cx+rx+10, top_cy-6, "高度 = 直线 ℝ", size=12.5, color=ACC))
    s.append(svg_text(cx, bot_cy+34, "完整实体 = D² × ℝ", size=14, color=INK))
    s.append(svg_text(cx, bot_cy+56, "实心圆柱体（不是空心圆管）", size=12, color=INK_D, weight=400))
    s.append(svg_text(cx, bot_cy+78, "豪斯多夫维数 = 2 + 1 = 3", size=14, color=ACC4))
    return "".join(s)


# ----------------------------------------------------------------------
# 组装 SVG：自上而下 5 步
# ----------------------------------------------------------------------
W, H = 1680, 1900
svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="\'Noto Sans SC\',\'Microsoft YaHei\',sans-serif">')
svg.append('<defs>' + gauss_blur("bg", 40) + '</defs>')

# 背景与网格
svg.append(svg_rect(0, 0, W, H, fill=BG, stroke="none", rx=0))
svg.append(f'<circle cx="{W*0.2}" cy="{H*0.35}" r="650" fill="#12203a" opacity="0.5" filter="url(#bg)"/>')
svg.append(f'<circle cx="{W*0.85}" cy="{H*0.15}" r="550" fill="#1a1240" opacity="0.5" filter="url(#bg)"/>')
for gx in range(80, W, 80):
    svg.append(svg_line(gx, 0, gx, H, color=GRID, w=0.6, opacity=0.6))
for gy in range(80, H, 80):
    svg.append(svg_line(0, gy, W, gy, color=GRID, w=0.6, opacity=0.6))

# 大标题
svg.append(svg_text(W/2, 76, "螺旋阶梯", size=60, color="#f8fafc", weight=800, ls=8))
svg.append(svg_text(W/2, 128, "一个会转的小圆 · 转一圈就上一格", size=23, color=INK_D, weight=400, ls=3))

# ---- 总体参数：每个步骤一个横向区块 ----
MARGIN = 70
PANEL_W = W - 2*MARGIN
step_gap = 44

def panel(y, h, number, accent, title, subtitle):
    """画一块横向面板，返回下一行的 y。"""
    x = MARGIN
    svg.append(svg_rect(x, y, PANEL_W, h, fill=PANEL, stroke="rgba(148,163,184,0.22)", sw=1.3, rx=20))
    # 序号圆
    svg.append(svg_circle(x+48, y+48, 23, stroke=accent, sw=2, fill="rgba(245,197,66,0.10)"))
    svg.append(svg_text(x+48, y+55, str(number), size=22, color=accent, ls=0))
    # 标题 + 副标题（左上）
    svg.append(svg_text(x+92, y+44, title, size=24, color=INK, weight=700, anchor="start", ls=1))
    svg.append(svg_text(x+92, y+70, subtitle, size=13, color=INK_D, weight=400, anchor="start"))
    return x


# ---- 步骤 1 与 2 并排（两块小面板），步骤 3 单独大块，步骤 4-5 并排 ----
# 为保持清晰，采用三行布局：
#  行1 (h=460): 步骤1 | 步骤2
#  行2 (h=470): 步骤3 (整宽)
#  行3 (h=470): 步骤4 | 步骤5
half_w = (PANEL_W - step_gap) / 2

top_y = 172
row1_h = 470
row2_h = 470
row3_h = 470

y1 = top_y
y2 = y1 + row1_h + step_gap       # 步骤3
y3 = y2 + row2_h + step_gap       # 步骤4/5
till_bottom = y3 + row3_h

# ===== 行1：步骤1 + 步骤2 =====
# 步骤1
panel(y1, row1_h, 1, ACC2, "① 一个旋转的圆", "复平面 · 单位圆 · 相位指针")
svg.append(draw_step1(MARGIN + half_w/2, y1 + row1_h/2 + 46))
# 步骤2
px2 = MARGIN + half_w + step_gap
svg.append(svg_rect(px2, y1, half_w, row1_h, fill=PANEL, stroke="rgba(148,163,184,0.22)", sw=1.3, rx=20))
svg.append(svg_circle(px2+48, y1+48, 23, stroke=ACC4, sw=2, fill="rgba(245,197,66,0.10)"))
svg.append(svg_text(px2+48, y1+55, "2", size=22, color=ACC4, ls=0))
svg.append(svg_text(px2+92, y1+44, "② 转满一圈，跳一格", size=24, color=INK, weight=700, anchor="start", ls=1))
svg.append(svg_text(px2+92, y1+70, "连续旋转 × 离散跃迁 = 事件", size=13, color=INK_D, weight=400, anchor="start"))
svg.append(draw_step2(px2 + half_w/2, y1 + row1_h/2 + 46))

# ===== 行2：步骤3 =====
panel(y2, row2_h, 3, ACC, "③ 一边转一边上 = 螺旋阶梯", "波性(连续旋转) 与 粒性(离散跃迁) 的二象性")
svg.append(draw_step3(MARGIN + PANEL_W/2, y2 + row2_h/2 + 20))

# ===== 行3：步骤4 + 步骤5 =====
# 步骤4
panel(y3, row3_h, 4, ACC2, "④ 这个螺旋是“实心”的", "圆盘 × 直线 = 实心圆柱体")
svg.append(draw_step4(MARGIN + half_w/2, y3 + row3_h/2 + 40))
# 步骤5
px5 = MARGIN + half_w + step_gap
svg.append(svg_rect(px5, y3, half_w, row3_h, fill=PANEL, stroke="rgba(245,197,66,0.28)", sw=1.5, rx=20))
svg.append(svg_circle(px5+48, y3+48, 23, stroke=ACC, sw=2, fill="rgba(245,197,66,0.10)"))
svg.append(svg_text(px5+48, y3+55, "5", size=22, color=ACC, ls=0))
svg.append(svg_text(px5+92, y3+44, "⑤ 为什么必须“一步一跳”", size=24, color=INK, weight=700, anchor="start", ls=1))
svg.append(svg_text(px5+92, y3+70, "取消跳跃，理论会失去解释力", size=13, color=INK_D, weight=400, anchor="start"))

# 步骤5 内的三个小卡片
subs = [
    ("空间", "有最小格子", "不能“穿过中间半点”，空间有最小单位", ACC2),
    ("引力", "天生是量子的", "G 与 ℏ 的耦合真实且内禀，不是巧合", ACC),
    ("量子", "离散=周期完成", "量子现象由此获得本体论根源", ACC4),
]
ssx0 = px5 + 26
ssw = (half_w - 2*26 - 2*18) / 3
ssy = y3 + 120
ssh = row3_h - 150
for i, (t, sb, bd, col) in enumerate(subs):
    sx = ssx0 + i*(ssw + 18)
    svg.append(svg_rect(sx, ssy, ssw, ssh, fill=PANEL2, stroke="rgba(148,163,184,0.18)", sw=1.2, rx=14))
    svg.append(f'<rect x="{sx}" y="{ssy}" width="{ssw}" height="6" rx="3" fill="{col}"/>')
    svg.append(svg_text(sx+ssw/2, ssy+40, t, size=19, color=col))
    svg.append(svg_text(sx+ssw/2, ssy+66, sb, size=12.5, color=INK))
    svg.append(svg_line(sx+15, ssy+82, sx+ssw-15, ssy+82, color="rgba(148,163,184,0.2)", w=1))
    svg.append(f'<foreignObject x="{sx+13}" y="{ssy+92}" width="{ssw-26}" height="{ssh-104}">'
               f'<div xmlns="http://www.w3.org/1999/xhtml" style="color:#94a3b8;'
               f'font-family:inherit;font-size:13px;line-height:1.7;text-align:left;">'
               f'{html.escape(bd)}</div></foreignObject>')

# 底部一行总结
svg.append(svg_text(W/2, till_bottom + 60, "波性 = 连续旋转 ｜ 粒性 = 离散跃迁 ｜ 二者通过“满周闭合”耦合为一体",
                    size=16, color=INK_D, weight=400, ls=1))
svg.append(svg_text(W/2, till_bottom + 100, "依据：《论元相子的螺旋阶梯几何》 傅天行 · 天行新学研究院",
                    size=13, color="#64748b", weight=400))

svg.append("</svg>")
SVG = "\n".join(svg)


# ----------------------------------------------------------------------
# HTML 包装（极少文字，突出大图）
# ----------------------------------------------------------------------
html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>螺旋阶梯几何 · 元相子</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#060912; color:#e5e7eb;
        font-family:'Noto Sans SC','Microsoft YaHei',sans-serif; }}
  .wrap {{ max-width:1780px; margin:0 auto; padding:26px 18px 60px; }}
  .hero {{ text-align:center; padding-bottom:4px; }}
  .hero h1 {{ font-size:34px; font-weight:800; letter-spacing:5px; margin:0 0 6px; }}
  .hero p  {{ color:#94a3b8; font-size:15px; letter-spacing:2px; margin:0; }}
  svg {{ width:100%; height:auto; display:block; }}
  .legend {{ text-align:center; color:#94a3b8; font-size:14px; margin:22px 0 10px; letter-spacing:1px; }}
  .legend b {{ color:#f5c542; font-weight:600; }}
  .note {{ max-width:1100px; margin:22px auto 0; color:#94a3b8; font-size:13px;
          line-height:1.9; text-align:center; }}
  .note b {{ color:#e5e7eb; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <h1>螺旋阶梯</h1>
    <p>一个会转的小圆 · 转一圈就上一格</p>
  </div>
  {SVG}
  <div class="legend">
    五步看懂：<b>会转的圆</b> → <b>转满一圈跳一格</b> → <b>螺旋阶梯</b> → <b>实心圆柱体</b> → <b>为什么必须跳</b>
  </div>
  <div class="note">
    <b>一句话总结</b>：宇宙最底层不是一个静态粒子，而是一个<b>会旋转、又一步一跳往上爬</b>的螺旋体。
    "转"给出平滑的时间线，"跳"给出可靠的物理格子，两者咬合在一起，就成了我们看见的连续时空。
  </div>
</div>
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_doc)

print("已生成:", OUT_HTML)
print("文件大小:", os.path.getsize(OUT_HTML), "字节")
