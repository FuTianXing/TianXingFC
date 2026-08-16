const PptxGenJS = require("pptxgenjs");
const fs = require("fs");

const pptx = new PptxGenJS();
pptx.defineLayout({ name: "WIDE", width: 13.33, height: 7.5 });
pptx.layout = "WIDE";

// ── Color Palette ──
const C = {
  navy:       "1A237E",
  darkNavy:   "0D1359",
  teal:       "00897B",
  tealLight:  "B2DFDB",
  gold:       "FFB300",
  goldLight:  "FFECB3",
  white:      "FFFFFF",
  offWhite:   "F5F3EE",
  lightBg:    "F9F8F4",
  gray:       "5C6B7A",
  darkText:   "1A1A2E",
  charcoal:   "2D3436",
  cardBg:     "FFFFFF",
  cardBorder: "E0E0E0",
  sage:       "A7BEAE",
};

// ── Helper: slide number footer ──
function addFooter(slide, num, total) {
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 7.1, w: 13.33, h: 0.4,
    fill: { color: C.navy },
  });
  slide.addText(`${num} / ${total}`, {
    x: 11.8, y: 7.1, w: 1.3, h: 0.4,
    fontSize: 10, color: C.white, align: "right",
    fontFace: "Calibri", valign: "middle",
    margin: [0, 10, 0, 0],
  });
  slide.addText("复平面单位圆 · 数学基础", {
    x: 0.5, y: 7.1, w: 5, h: 0.4,
    fontSize: 10, color: C.tealLight, fontFace: "Calibri", valign: "middle",
  });
}

// ── Helper: decorative top bar ──
function addTopBar(slide, color = C.teal) {
  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: 13.33, h: 0.06, fill: { color },
  });
}

// ── Helper: section title on light slide ──
function addSlideTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.8, y: 0.3, w: 11.5, h: 0.7,
    fontSize: 28, fontFace: "Georgia", color: C.navy, bold: true,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.8, y: 0.9, w: 11.5, h: 0.45,
      fontSize: 14, fontFace: "Calibri", color: C.gray, italic: true,
    });
  }
  // subtle underline
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.8, y: 1.35, w: 2.5, h: 0.04, fill: { color: C.gold },
  });
}

// ── Helper: formula card ──
function addFormulaCard(slide, x, y, w, h, formula, desc) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, fill: { color: C.white },
    line: { color: C.cardBorder, width: 1 },
    rectRadius: 0.08,
    shadow: { type: "outer", blur: 4, offset: 2, color: "CCCCCC", opacity: 0.3 },
  });
  slide.addText(formula, {
    x: x + 0.2, y: y + 0.1, w: w - 0.4, h: h * 0.55,
    fontSize: 18, fontFace: "Georgia", color: C.navy, bold: true,
    align: "center", valign: "middle",
  });
  slide.addText(desc, {
    x: x + 0.2, y: y + h * 0.55, w: w - 0.4, h: h * 0.45,
    fontSize: 11, fontFace: "Calibri", color: C.gray, align: "center",
    valign: "top",
  });
}

// ── Helper: icon circle ──
function addIconCircle(slide, x, y, size, label, iconText, bgColor) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y, w: size, h: size,
    fill: { color: bgColor || C.navy },
  });
  slide.addText(iconText, {
    x, y, w: size, h: size,
    fontSize: size * 0.35, color: C.white, align: "center", valign: "middle",
    fontFace: "Georgia", bold: true,
  });
  slide.addText(label, {
    x: x - 0.3, y: y + size + 0.05, w: size + 0.6, h: 0.35,
    fontSize: 10, color: C.darkText, align: "center", fontFace: "Calibri",
  });
}

// ── Helper: draw unit circle on a slide ──
function drawUnitCircle(slide, cx, cy, r, opts = {}) {
  const {
    showAxes = true, showCircle = true, showAngles = false,
    showPoints = false, pointAngle = 0, pointLabel = "z",
    showCosSin = false, showTangent = false,
  } = opts;

  // Axes
  if (showAxes) {
    slide.addShape(pptx.ShapeType.line, {
      x: cx - r - 0.8, y: cy, w: (r + 0.8) * 2, h: 0,
      line: { color: C.charcoal, width: 1.2 },
    });
    slide.addShape(pptx.ShapeType.line, {
      x: cx, y: cy - r - 0.8, w: 0, h: (r + 0.8) * 2,
      line: { color: C.charcoal, width: 1.2 },
    });
    // axis labels
    slide.addText("Re", {
      x: cx + r + 0.6, y: cy - 0.2, w: 0.5, h: 0.35,
      fontSize: 11, color: C.gray, fontFace: "Calibri", italic: true,
    });
    slide.addText("Im", {
      x: cx + 0.1, y: cy - r - 0.8, w: 0.5, h: 0.35,
      fontSize: 11, color: C.gray, fontFace: "Calibri", italic: true,
    });
    // origin
    slide.addText("O", {
      x: cx - 0.3, y: cy + 0.08, w: 0.4, h: 0.3,
      fontSize: 10, color: C.gray, fontFace: "Calibri", italic: true,
    });
  }

  // Unit circle
  if (showCircle) {
    slide.addShape(pptx.ShapeType.ellipse, {
      x: cx - r, y: cy - r, w: r * 2, h: r * 2,
      line: { color: C.teal, width: 2.5 },
      fill: { color: "E8F5E9", transparency: 70 },
    });
  }

  // Radius line
  if (showPoints || showAngles) {
    const px = cx + r * Math.cos(pointAngle);
    const py = cy - r * Math.sin(pointAngle);
    slide.addShape(pptx.ShapeType.line, {
      x: cx, y: cy, w: px - cx, h: py - cy,
      line: { color: C.gold, width: 2 },
    });
    // point dot
    slide.addShape(pptx.ShapeType.ellipse, {
      x: px - 0.08, y: py - 0.08, w: 0.16, h: 0.16,
      fill: { color: C.gold },
    });
    // point label
    slide.addText(pointLabel, {
      x: px + 0.1, y: py - 0.25, w: 0.5, h: 0.3,
      fontSize: 12, color: C.gold, fontFace: "Georgia", bold: true,
    });

    // angle arc
    if (showAngles) {
      const steps = 20;
      const anglePath = [];
      for (let i = 0; i <= steps; i++) {
        const a = (pointAngle * i) / steps;
        anglePath.push({ x: cx + r * 0.25 * Math.cos(a), y: cy - r * 0.25 * Math.sin(a) });
      }
      // approximate arc with small segments
      for (let i = 0; i < anglePath.length - 1; i++) {
        slide.addShape(pptx.ShapeType.line, {
          x: anglePath[i].x, y: anglePath[i].y,
          w: anglePath[i + 1].x - anglePath[i].x,
          h: anglePath[i + 1].y - anglePath[i].y,
          line: { color: C.gold, width: 1.2 },
        });
      }
      // angle label
      const labelAngle = pointAngle / 2;
      const lx = cx + r * 0.38 * Math.cos(labelAngle);
      const ly = cy - r * 0.38 * Math.sin(labelAngle);
      slide.addText("θ", {
        x: lx - 0.15, y: ly - 0.15, w: 0.35, h: 0.3,
        fontSize: 13, color: C.gold, fontFace: "Georgia", italic: true, align: "center",
      });
    }

    // cos / sin projection lines
    if (showCosSin) {
      // cos line (to x-axis)
      slide.addShape(pptx.ShapeType.line, {
        x: px, y: py, w: 0, h: cy - py,
        line: { color: C.teal, width: 1.2, dashType: "dash" },
      });
      // sin line (to y-axis)
      slide.addShape(pptx.ShapeType.line, {
        x: px, y: cy, w: cx - px, h: 0,
        line: { color: C.gold, width: 1.2, dashType: "dash" },
      });
      slide.addText("cos θ", {
        x: px + 0.05, y: cy + 0.05, w: 0.7, h: 0.25,
        fontSize: 9, color: C.teal, fontFace: "Calibri", italic: true,
      });
      slide.addText("sin θ", {
        x: cx + 0.05, y: Math.min(py, cy - 0.3), w: 0.6, h: 0.25,
        fontSize: 9, color: C.gold, fontFace: "Calibri", italic: true,
      });
    }
  }
}

// ══════════════════════════════════════════════════════════
// SLIDE 1 - 封面
// ══════════════════════════════════════════════════════════
const s1 = pptx.addSlide();
s1.background = { fill: C.navy };
// large decorative circle
s1.addShape(pptx.ShapeType.ellipse, {
  x: 8.5, y: -1.5, w: 7, h: 7,
  fill: { color: C.teal, transparency: 85 },
});
s1.addShape(pptx.ShapeType.ellipse, {
  x: -1.5, y: 4, w: 5, h: 5,
  fill: { color: C.gold, transparency: 88 },
});
// title
s1.addText("复平面单位圆", {
  x: 1, y: 1.8, w: 10, h: 1.2,
  fontSize: 52, fontFace: "Georgia", color: C.white, bold: true,
});
s1.addText("Complex Plane & The Unit Circle", {
  x: 1, y: 3, w: 10, h: 0.6,
  fontSize: 20, fontFace: "Calibri", color: C.tealLight, italic: true,
});
// decorative line
s1.addShape(pptx.ShapeType.rect, {
  x: 1, y: 3.8, w: 3, h: 0.05, fill: { color: C.gold },
});
s1.addText("数学基础 · 欧拉公式 · 复分析入门", {
  x: 1, y: 4.1, w: 10, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: C.white, align: "left",
});
// mini unit circle decorative
s1.addShape(pptx.ShapeType.ellipse, {
  x: 9.5, y: 2, w: 2.8, h: 2.8,
  line: { color: C.gold, width: 1.5 },
  fill: { color: C.white, transparency: 70 },
});
s1.addShape(pptx.ShapeType.line, {
  x: 10.9, y: 3.4, w: 1.2, h: -0.6,
  line: { color: C.gold, width: 1.5 },
});
s1.addShape(pptx.ShapeType.ellipse, {
  x: 12.05, y: 2.65, w: 0.12, h: 0.12,
  fill: { color: C.gold },
});

// ══════════════════════════════════════════════════════════
// SLIDE 2 - 目录
// ══════════════════════════════════════════════════════════
const s2 = pptx.addSlide();
s2.background = { fill: C.lightBg };
addTopBar(s2);
addSlideTitle(s2, "内容概览", "13 个章节，从复数基础到单位圆的应用");
addFooter(s2, 2, 13);

const topics = [
  { num: "01", title: "复数基础", desc: "复数的定义、实部与虚部" },
  { num: "02", title: "复平面", desc: "阿甘图、坐标表示" },
  { num: "03", title: "单位圆定义", desc: "|z| = 1 的几何意义" },
  { num: "04", title: "极坐标表示", desc: "模长与辐角" },
  { num: "05", title: "欧拉公式", desc: "e^{iθ} = cos θ + i sin θ" },
  { num: "06", title: "单位圆上的点", desc: "cos θ + i sin θ 参数化" },
  { num: "07", title: "旋转与乘法", desc: "乘 e^{iθ} 即旋转" },
  { num: "08", title: "辐角与周期", desc: "2π 的旋转对称性" },
  { num: "09", title: "特殊角度", desc: "0, π/6, π/4, π/3, π/2, π" },
  { num: "10", title: "单位根", desc: "z^n = 1 的 n 个解" },
  { num: "11", title: "应用与联系", desc: "傅里叶变换、量子力学" },
];

topics.forEach((t, i) => {
  const col = i % 3;
  const row = Math.floor(i / 3);
  const x = 0.8 + col * 4.1;
  const y = 1.7 + row * 1.55;

  s2.addShape(pptx.ShapeType.roundRect, {
    x, y, w: 3.8, h: 1.3,
    fill: { color: C.white },
    line: { color: C.cardBorder, width: 0.5 },
    rectRadius: 0.06,
    shadow: { type: "outer", blur: 3, offset: 1, color: "DDDDDD", opacity: 0.25 },
  });
  // number indicator
  s2.addShape(pptx.ShapeType.roundRect, {
    x: x + 0.15, y: y + 0.2, w: 0.5, h: 0.5,
    fill: { color: i < 6 ? C.teal : C.navy },
    rectRadius: 0.05,
  });
  s2.addText(t.num, {
    x: x + 0.15, y: y + 0.2, w: 0.5, h: 0.5,
    fontSize: 14, color: C.white, align: "center", valign: "middle",
    fontFace: "Georgia", bold: true,
  });
  s2.addText(t.title, {
    x: x + 0.8, y: y + 0.15, w: 2.8, h: 0.45,
    fontSize: 15, fontFace: "Georgia", color: C.navy, bold: true, valign: "middle",
  });
  s2.addText(t.desc, {
    x: x + 0.8, y: y + 0.6, w: 2.8, h: 0.4,
    fontSize: 10, fontFace: "Calibri", color: C.gray, valign: "top",
  });
});

// ══════════════════════════════════════════════════════════
// SLIDE 3 - 复数基础
// ══════════════════════════════════════════════════════════
const s3 = pptx.addSlide();
s3.background = { fill: C.lightBg };
addTopBar(s3);
addSlideTitle(s3, "复数基础", "从实数到复数的扩展");
addFooter(s3, 3, 13);

// Definition card
addFormulaCard(s3, 0.8, 1.6, 5.5, 1.6,
  "z = x + iy",
  "其中 x = Re(z) 为实部，y = Im(z) 为虚部，i² = −1");
addFormulaCard(s3, 6.8, 1.6, 5.7, 1.6,
  "共轭复数:  z̅ = x − iy",
  "对称于实轴，zz̅ = x² + y² = |z|²");

// Key properties grid
const props = [
  { label: "加法", desc: "(x₁+x₂) + i(y₁+y₂)" },
  { label: "乘法", desc: "(x₁x₂−y₁y₂) + i(x₁y₂+x₂y₁)" },
  { label: "模长", desc: "|z| = √(x² + y²)" },
  { label: "辐角", desc: "arg(z) = arctan(y/x)" },
];
props.forEach((p, i) => {
  const x = 0.8 + i * 3.1;
  // icon circle
  s3.addShape(pptx.ShapeType.ellipse, {
    x: x + 1.0, y: 3.65, w: 0.6, h: 0.6,
    fill: { color: i < 2 ? C.teal : C.navy },
  });
  s3.addText(p.label, {
    x: x + 1.0, y: 3.65, w: 0.6, h: 0.6,
    fontSize: 10, color: C.white, align: "center", valign: "middle",
    fontFace: "Calibri", bold: true,
  });
  s3.addText(p.desc, {
    x, y: 4.4, w: 2.9, h: 0.6,
    fontSize: 13, fontFace: "Calibri", color: C.darkText,
    align: "center", valign: "top",
  });
});

// Historical note
s3.addShape(pptx.ShapeType.roundRect, {
  x: 0.8, y: 5.3, w: 11.7, h: 1.2,
  fill: { color: C.white },
  line: { color: C.tealLight, width: 1 },
  rectRadius: 0.06,
});
s3.addText("💡 历史小记", {
  x: 1.1, y: 5.4, w: 2, h: 0.35,
  fontSize: 12, fontFace: "Georgia", color: C.teal, bold: true,
});
s3.addText(
  "复数概念最早由意大利数学家 Cardano (1545) 提及，后经 Euler、Gauss 等人完善。" +
  "如今复分析是数学物理、工程、信号处理的核心工具。",
  { x: 1.1, y: 5.75, w: 11.1, h: 0.6, fontSize: 11, fontFace: "Calibri", color: C.gray, valign: "top", wrap: true }
);

// ══════════════════════════════════════════════════════════
// SLIDE 4 - 复平面
// ══════════════════════════════════════════════════════════
const s4 = pptx.addSlide();
s4.background = { fill: C.lightBg };
addTopBar(s4);
addSlideTitle(s4, "复平面（阿甘图）", "Argand Diagram — 复数的几何表示");
addFooter(s4, 4, 13);

// Unit circle on this slide
drawUnitCircle(s4, 3.8, 4.0, 2.2, {
  showAxes: true, showCircle: true,
  showPoints: true, showAngles: true,
  pointAngle: Math.PI / 3, pointLabel: "z = x + iy",
  showCosSin: true,
});

// Right side: explanation
s4.addShape(pptx.ShapeType.roundRect, {
  x: 7.2, y: 1.6, w: 5.3, h: 2.1,
  fill: { color: C.white },
  line: { color: C.cardBorder, width: 0.5 },
  rectRadius: 0.06,
});
s4.addText("坐标系统", {
  x: 7.5, y: 1.7, w: 4.5, h: 0.4,
  fontSize: 16, fontFace: "Georgia", color: C.navy, bold: true,
});
s4.addText(
  "• 横轴 (Re): 实数轴，表示复数的实部\n" +
  "• 纵轴 (Im): 虚数轴，表示复数的虚部\n" +
  "• 点 z = x + iy 对应坐标 (x, y)\n" +
  "• 原点 O 对应复数 0\n" +
  "• 到原点的距离 = 模长 |z|",
  { x: 7.5, y: 2.2, w: 4.8, h: 1.4, fontSize: 12, fontFace: "Calibri", color: C.darkText, valign: "top", lineSpacing: 22 }
);

// bottom note
s4.addShape(pptx.ShapeType.roundRect, {
  x: 7.2, y: 4.0, w: 5.3, h: 1.6,
  fill: { color: C.white },
  line: { color: C.cardBorder, width: 0.5 },
  rectRadius: 0.06,
});
s4.addText("复数的两种视图", {
  x: 7.5, y: 4.1, w: 4.5, h: 0.4,
  fontSize: 16, fontFace: "Georgia", color: C.navy, bold: true,
});
s4.addText(
  "笛卡尔坐标: z = x + iy\n" +
  "极坐标:     z = r(cos θ + i sin θ)\n\n" +
  "两种表示等价，但极坐标在描述旋转时更自然",
  { x: 7.5, y: 4.55, w: 4.8, h: 1.0, fontSize: 12, fontFace: "Calibri", color: C.darkText, valign: "top", lineSpacing: 20 }
);

// ══════════════════════════════════════════════════════════
// SLIDE 5 - 单位圆定义
// ══════════════════════════════════════════════════════════
const s5 = pptx.addSlide();
s5.background = { fill: C.lightBg };
addTopBar(s5);
addSlideTitle(s5, "单位圆的定义", "Unit Circle — 复平面上模为 1 的点的集合");
addFooter(s5, 5, 13);

// Main definition card
s5.addShape(pptx.ShapeType.roundRect, {
  x: 0.8, y: 1.7, w: 5.5, h: 2.0,
  fill: { color: C.navy },
  rectRadius: 0.08,
});
s5.addText("单位圆 = { z ∈ ℂ : |z| = 1 }", {
  x: 1.0, y: 1.8, w: 5.1, h: 0.7,
  fontSize: 22, fontFace: "Georgia", color: C.white, bold: true, valign: "middle",
});
s5.addText("即所有满足 x² + y² = 1 的复数点", {
  x: 1.0, y: 2.55, w: 5.1, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: C.tealLight, valign: "middle",
});
s5.addText("半径 r = 1，圆心在原点", {
  x: 1.0, y: 3.05, w: 5.1, h: 0.4,
  fontSize: 13, fontFace: "Calibri", color: C.goldLight, valign: "middle",
});

// Key properties
s5.addText("单位圆的重要性质", {
  x: 7.2, y: 1.7, w: 5.3, h: 0.5,
  fontSize: 16, fontFace: "Georgia", color: C.navy, bold: true,
});

const props5 = [
  "对称性：关于实轴、虚轴和原点均对称",
  "封闭性：单位圆上的点相乘仍在单位圆上",
  "紧致性：有界闭集，是复平面上的紧集",
  "连通的：单位圆是连通的闭合曲线",
  "群结构：构成 U(1) 李群",
];
props5.forEach((p, i) => {
  s5.addShape(pptx.ShapeType.ellipse, {
    x: 7.3, y: 2.35 + i * 0.5, w: 0.14, h: 0.14,
    fill: { color: i === 0 ? C.gold : C.teal },
  });
  s5.addText(p, {
    x: 7.6, y: 2.25 + i * 0.5, w: 4.8, h: 0.4,
    fontSize: 12, fontFace: "Calibri", color: C.darkText, valign: "middle",
  });
});

// Visual: unit circle
drawUnitCircle(s5, 3.3, 5.1, 1.3);

// bottom: important consequence
s5.addShape(pptx.ShapeType.roundRect, {
  x: 7.2, y: 5.0, w: 5.3, h: 1.3,
  fill: { color: C.goldLight },
  rectRadius: 0.06,
});
s5.addText("核心洞察", {
  x: 7.4, y: 5.05, w: 2, h: 0.35,
  fontSize: 12, fontFace: "Georgia", color: C.darkText, bold: true,
});
s5.addText(
  "单位圆上的复数纯粹由辐角 θ 决定——\n" +
  "它架起了复数乘法与几何旋转之间的桥梁",
  { x: 7.4, y: 5.4, w: 4.9, h: 0.8, fontSize: 12, fontFace: "Calibri", color: C.darkText, valign: "top" }
);

// ══════════════════════════════════════════════════════════
// SLIDE 6 - 极坐标表示
// ══════════════════════════════════════════════════════════
const s6 = pptx.addSlide();
s6.background = { fill: C.lightBg };
addTopBar(s6);
addSlideTitle(s6, "极坐标表示", "Polar Form — 模长与辐角");
addFooter(s6, 6, 13);

// Formula card
addFormulaCard(s6, 0.8, 1.6, 5.5, 1.8,
  "z = r (cos θ + i sin θ)",
  "r = |z| 为模长，θ = arg(z) 为辐角（弧度）");

addFormulaCard(s6, 6.8, 1.6, 5.7, 1.8,
  "单位圆上: z = cos θ + i sin θ",
  "因为 r = 1，复数的全部信息包含在 θ 中");

// Right side: conversion
s6.addShape(pptx.ShapeType.roundRect, {
  x: 0.8, y: 3.7, w: 5.5, h: 2.5,
  fill: { color: C.white },
  line: { color: C.cardBorder, width: 0.5 },
  rectRadius: 0.06,
});
s6.addText("极坐标 ↔ 笛卡尔坐标", {
  x: 1.1, y: 3.8, w: 5, h: 0.4,
  fontSize: 14, fontFace: "Georgia", color: C.navy, bold: true,
});
s6.addText(
  "从极坐标到笛卡尔:\n    x = r cos θ    y = r sin θ\n\n" +
  "从笛卡尔到极坐标:\n    r = √(x² + y²)    θ = arctan(y/x)",
  { x: 1.1, y: 4.25, w: 5, h: 1.8, fontSize: 12, fontFace: "Calibri", color: C.darkText, lineSpacing: 22 }
);

// Right column: visual
drawUnitCircle(s6, 9.8, 4.9, 1.8, {
  showAxes: true, showCircle: true,
  showPoints: true, showAngles: true,
  pointAngle: Math.PI / 4, pointLabel: "z",
  showCosSin: true,
});

s6.addText("辐角 θ 从正实轴逆时针测量", {
  x: 7.2, y: 4.8, w: 5.2, h: 0.3,
  fontSize: 10, fontFace: "Calibri", color: C.gray, align: "center", italic: true,
});

// ══════════════════════════════════════════════════════════
// SLIDE 7 - 欧拉公式 (dark slide)
// ══════════════════════════════════════════════════════════
const s7 = pptx.addSlide();
s7.background = { fill: C.navy };
// decorative circles
s7.addShape(pptx.ShapeType.ellipse, {
  x: 9, y: -2, w: 6, h: 6, fill: { color: C.teal, transparency: 88 },
});
s7.addShape(pptx.ShapeType.ellipse, {
  x: -2, y: 4, w: 4.5, h: 4.5, fill: { color: C.gold, transparency: 90 },
});
addFooter(s7, 7, 13);

// Title
s7.addText("欧拉公式", {
  x: 0.8, y: 0.5, w: 10, h: 0.8,
  fontSize: 32, fontFace: "Georgia", color: C.white, bold: true,
});
s7.addText("数学中最优美的公式之一", {
  x: 0.8, y: 1.2, w: 10, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: C.tealLight, italic: true,
});

// Huge formula
s7.addText("e^{iθ} = cos θ + i sin θ", {
  x: 0.8, y: 1.8, w: 11.5, h: 1.2,
  fontSize: 36, fontFace: "Georgia", color: C.gold, bold: true, align: "center",
});

// Special case: Euler's identity
s7.addShape(pptx.ShapeType.roundRect, {
  x: 2.5, y: 3.3, w: 8.3, h: 1.5,
  fill: { color: C.white, transparency: 10 },
  line: { color: C.gold, width: 1.5 },
  rectRadius: 0.08,
});
s7.addText("当 θ = π 时：欧拉恒等式", {
  x: 2.7, y: 3.4, w: 7.9, h: 0.4,
  fontSize: 13, fontFace: "Calibri", color: C.tealLight, align: "center",
});
s7.addText("e^{iπ} + 1 = 0", {
  x: 2.7, y: 3.8, w: 7.9, h: 0.8,
  fontSize: 30, fontFace: "Georgia", color: C.white, bold: true, align: "center",
});

// Explanation
s7.addText(
  "这个公式将指数函数、三角函数和虚数单位优雅地联系在一起，" +
  "被多次评为「数学中最优美的公式」，连接了 e、i、π、1、0 五个基本常数。",
  { x: 0.8, y: 5.1, w: 11.5, h: 0.8, fontSize: 12, fontFace: "Calibri", color: C.white, align: "center", wrap: true }
);

// Derivation hint
s7.addText(
  "理解路径：泰勒展开 → eˣ = 1 + x + x²/2! + ... → 代入 x = iθ 分别得到 cos 和 sin 的展开",
  { x: 0.8, y: 5.9, w: 11.5, h: 0.5, fontSize: 10, fontFace: "Calibri", color: C.gray, align: "center", italic: true }
);

// ══════════════════════════════════════════════════════════
// SLIDE 8 - 单位圆上的点
// ══════════════════════════════════════════════════════════
const s8 = pptx.addSlide();
s8.background = { fill: C.lightBg };
addTopBar(s8);
addSlideTitle(s8, "单位圆上的点", "参数化表示 z = cos θ + i sin θ");
addFooter(s8, 8, 13);

// Parametrization
addFormulaCard(s8, 0.8, 1.6, 5.5, 1.6,
  "z(θ) = cos θ + i sin θ",
  "θ ∈ [0, 2π) 参数化整个单位圆");

addFormulaCard(s8, 6.8, 1.6, 5.7, 1.6,
  "z(θ) = e^{iθ} = cis(θ)",
  "cis(θ) = cos θ + i sin θ 的简写");

// Visual: animated-like representation
drawUnitCircle(s8, 3.3, 5.0, 1.6, {
  showAxes: true, showCircle: true,
  showPoints: true, showAngles: true,
  pointAngle: Math.PI / 3, pointLabel: "e^{iθ}",
  showCosSin: true,
});

// Properties on right
s8.addShape(pptx.ShapeType.roundRect, {
  x: 7.0, y: 3.7, w: 5.5, h: 2.5,
  fill: { color: C.white },
  line: { color: C.cardBorder, width: 0.5 },
  rectRadius: 0.06,
});
s8.addText("参数化性质", {
  x: 7.3, y: 3.8, w: 5, h: 0.4,
  fontSize: 14, fontFace: "Georgia", color: C.navy, bold: true,
});
s8.addText(
  "• z(0) = 1（正实轴方向）\n" +
  "• z(π/2) = i（正虚轴方向）\n" +
  "• z(π) = −1（负实轴方向）\n" +
  "• z(3π/2) = −i（负虚轴方向）\n" +
  "• z(θ) 以 2π 为周期\n" +
  "• θ 增加 → 逆时针运动",
  { x: 7.3, y: 4.25, w: 5, h: 1.8, fontSize: 12, fontFace: "Calibri", color: C.darkText, lineSpacing: 21 }
);

// ══════════════════════════════════════════════════════════
// SLIDE 9 - 旋转与乘法
// ══════════════════════════════════════════════════════════
const s9 = pptx.addSlide();
s9.background = { fill: C.lightBg };
addTopBar(s9);
addSlideTitle(s9, "旋转与乘法", "乘以 e^{iθ} 相当于旋转 θ 弧度");
addFooter(s9, 9, 13);

// Key formula
addFormulaCard(s9, 0.8, 1.6, 5.5, 1.6,
  "e^{iα} · e^{iβ} = e^{i(α+β)}",
  "指数相加，旋转叠加");

addFormulaCard(s9, 6.8, 1.6, 5.7, 1.6,
  "z · e^{iθ} → 旋转 z 逆时针 θ",
  "模长不变，辐角增加 θ");

// Visual: rotation diagram
const cx9 = 2.5, cy9 = 5.0, r9 = 1.5;
drawUnitCircle(s9, cx9, cy9, r9, {
  showAxes: true, showCircle: true,
  showPoints: true, showAngles: true,
  pointAngle: Math.PI / 6, pointLabel: "z",
  showCosSin: false,
});
// second radius for rotation
const theta2 = Math.PI / 6 + Math.PI / 4;
const px2 = cx9 + r9 * Math.cos(theta2);
const py2 = cy9 - r9 * Math.sin(theta2);
s9.addShape(pptx.ShapeType.line, {
  x: cx9, y: cy9, w: px2 - cx9, h: py2 - cy9,
  line: { color: C.teal, width: 2 },
});
s9.addShape(pptx.ShapeType.ellipse, {
  x: px2 - 0.08, y: py2 - 0.08, w: 0.16, h: 0.16,
  fill: { color: C.teal },
});
s9.addText("z·e^{iθ}", {
  x: px2 + 0.1, y: py2 - 0.25, w: 0.8, h: 0.3,
  fontSize: 11, color: C.teal, fontFace: "Georgia", bold: true,
});

// Illustration explanation
s9.addShape(pptx.ShapeType.roundRect, {
  x: 6.8, y: 3.7, w: 5.7, h: 2.5,
  fill: { color: C.white },
  line: { color: C.cardBorder, width: 0.5 },
  rectRadius: 0.06,
});
s9.addText("几何解释", {
  x: 7.0, y: 3.8, w: 5, h: 0.4,
  fontSize: 14, fontFace: "Georgia", color: C.navy, bold: true,
});
s9.addText(
  "乘以 e^{iθ} 是复平面上的纯旋转：\n\n" +
  "• 保持到原点的距离不变\n" +
  "• 将点绕原点旋转 θ 弧度\n" +
  "• 旋转矩阵视角：\n" +
  "  [cos θ  −sin θ] [x]   \n" +
  "  [sin θ   cos θ] [y]\n\n" +
  "这正是 2×2 旋转矩阵！",
  { x: 7.0, y: 4.25, w: 5.3, h: 1.9, fontSize: 11, fontFace: "Calibri", color: C.darkText, lineSpacing: 18 }
);

// ══════════════════════════════════════════════════════════
// SLIDE 10 - 辐角与周期
// ══════════════════════════════════════════════════════════
const s10 = pptx.addSlide();
s10.background = { fill: C.lightBg };
addTopBar(s10);
addSlideTitle(s10, "辐角与周期性", "Argument & Periodicity — 2π 的旋转对称性");
addFooter(s10, 10, 13);

addFormulaCard(s10, 0.8, 1.6, 5.5, 1.5,
  "e^{i(θ+2π)} = e^{iθ}",
  "单位圆上每旋转 2π 回到原点");

addFormulaCard(s10, 6.8, 1.6, 5.7, 1.5,
  "arg(z₁z₂) = arg(z₁) + arg(z₂)",
  "乘积的辐角 = 辐角之和（模 2π）");

// Multi-valued nature
s10.addShape(pptx.ShapeType.roundRect, {
  x: 0.8, y: 3.4, w: 5.5, h: 2.5,
  fill: { color: C.white },
  line: { color: C.cardBorder, width: 0.5 },
  rectRadius: 0.06,
});
s10.addText("主辐角 Arg(z)", {
  x: 1.1, y: 3.5, w: 5, h: 0.4,
  fontSize: 14, fontFace: "Georgia", color: C.navy, bold: true,
});
s10.addText(
  "辐角不是唯一的——加 2π 的整数倍仍为同一角度。\n\n" +
  "主辐角 Arg(z) ∈ (−π, π] 取标准值：\n" +
  "• 第一象限: Arg(z) ∈ (0, π/2)\n" +
  "• 第二象限: Arg(z) ∈ (π/2, π)\n" +
  "• 第三象限: Arg(z) ∈ (−π, −π/2)\n" +
  "• 第四象限: Arg(z) ∈ (−π/2, 0)",
  { x: 1.1, y: 3.9, w: 5, h: 1.9, fontSize: 11, fontFace: "Calibri", color: C.darkText, lineSpacing: 18 }
);

// Right: visual showing periodicity
drawUnitCircle(s10, 10.0, 4.5, 1.8, {
  showAxes: true, showCircle: true,
  showPoints: true, showAngles: true,
  pointAngle: Math.PI / 3, pointLabel: "e^{iθ}",
  showCosSin: false,
});
s10.addText("θ 与 θ + 2π 对应同一点", {
  x: 7.2, y: 6.2, w: 5.2, h: 0.3,
  fontSize: 10, fontFace: "Calibri", color: C.gray, align: "center", italic: true,
});

// ══════════════════════════════════════════════════════════
// SLIDE 11 - 特殊角度
// ══════════════════════════════════════════════════════════
const s11 = pptx.addSlide();
s11.background = { fill: C.lightBg };
addTopBar(s11);
addSlideTitle(s11, "常见特殊角度", "单位圆上的关键辐角及对应的复数值");
addFooter(s11, 11, 13);

const angles = [
  { deg: "0°", rad: "0", val: "1", x: "1", y: "0" },
  { deg: "30°", rad: "π/6", val: "√3/2 + i·½", x: "√3/2", y: "1/2" },
  { deg: "45°", rad: "π/4", val: "√2/2 + i·√2/2", x: "√2/2", y: "√2/2" },
  { deg: "60°", rad: "π/3", val: "½ + i·√3/2", x: "1/2", y: "√3/2" },
  { deg: "90°", rad: "π/2", val: "i", x: "0", y: "1" },
  { deg: "180°", rad: "π", val: "−1", x: "−1", y: "0" },
];

// Table header
const colW = [0.8, 1.0, 1.0, 2.2, 1.0, 0.8];
const headers = ["角度", "弧度", "值", "cos θ", "sin θ", ""];
const startX = 0.8;
const startY = 1.7;
const rowH = 0.5;

// header row
headers.forEach((h, i) => {
  const x = startX + colW.slice(0, i).reduce((a, b) => a + b, 0);
  s11.addShape(pptx.ShapeType.rect, {
    x, y: startY, w: colW[i], h: rowH,
    fill: { color: C.navy },
  });
  s11.addText(h, {
    x, y: startY, w: colW[i], h: rowH,
    fontSize: 12, fontFace: "Calibri", color: C.white,
    align: "center", valign: "middle", bold: true,
  });
});

// data rows
angles.forEach((a, ri) => {
  const vals = [a.deg, a.rad, a.val, a.x, a.y, ""];
  const bgColor = ri % 2 === 0 ? C.offWhite : C.white;
  vals.forEach((v, ci) => {
    const x = startX + colW.slice(0, ci).reduce((a, b) => a + b, 0);
    s11.addShape(pptx.ShapeType.rect, {
      x, y: startY + (ri + 1) * rowH, w: colW[ci], h: rowH,
      fill: { color: bgColor },
      line: { color: C.cardBorder, width: 0.3 },
    });
    s11.addText(v, {
      x, y: startY + (ri + 1) * rowH, w: colW[ci], h: rowH,
      fontSize: 11, fontFace: v.includes("π") || v.includes("√") ? "Georgia" : "Calibri",
      color: C.darkText, align: "center", valign: "middle",
    });
  });
});

// Unit circle with all special points
const cx11 = 7.8, cy11 = 4.8, r11 = 2.0;
drawUnitCircle(s11, cx11, cy11, r11, {
  showAxes: true, showCircle: true, showPoints: false,
});

const specialAngles = [
  { angle: 0, label: "1" },
  { angle: Math.PI / 6, label: "π/6" },
  { angle: Math.PI / 4, label: "π/4" },
  { angle: Math.PI / 3, label: "π/3" },
  { angle: Math.PI / 2, label: "i" },
  { angle: Math.PI, label: "−1" },
  { angle: -Math.PI / 2, label: "−i" },
];
specialAngles.forEach(({ angle, label }) => {
  const px = cx11 + r11 * Math.cos(angle);
  const py = cy11 - r11 * Math.sin(angle);
  s11.addShape(pptx.ShapeType.ellipse, {
    x: px - 0.06, y: py - 0.06, w: 0.12, h: 0.12,
    fill: { color: C.gold },
  });
  const lx = px + (px > cx11 ? 0.12 : -0.5);
  s11.addText(label, {
    x: lx, y: py - 0.15, w: 0.5, h: 0.25,
    fontSize: 9, color: C.navy, fontFace: "Georgia", bold: true,
  });
});

// ══════════════════════════════════════════════════════════
// SLIDE 12 - 单位根
// ══════════════════════════════════════════════════════════
const s12 = pptx.addSlide();
s12.background = { fill: C.lightBg };
addTopBar(s12);
addSlideTitle(s12, "单位根", "Roots of Unity — zⁿ = 1 的 n 个复数解");
addFooter(s12, 12, 13);

addFormulaCard(s12, 0.8, 1.6, 5.5, 1.5,
  "zⁿ = 1 的解:  e^{2πik/n},  k = 0,1,…,n−1",
  "n 个解均匀分布在单位圆上");

addFormulaCard(s12, 6.8, 1.6, 5.7, 1.5,
  "ζₙ = e^{2πi/n}（原根）",
  "所有单位根 = {ζₙᵏ : k = 0,1,...,n−1}");

// Visual: n=3, n=4, n=5, n=6
const rootConfigs = [
  { n: 3, cx: 2.2, cy: 5.0, r: 1.3 },
  { n: 4, cx: 5.5, cy: 5.0, r: 1.3 },
  { n: 5, cx: 8.8, cy: 5.0, r: 1.3 },
  { n: 6, cx: 11.5, cy: 5.0, r: 1.1 },
];

rootConfigs.forEach(({ n, cx, cy, r }) => {
  // unit circle
  s12.addShape(pptx.ShapeType.ellipse, {
    x: cx - r, y: cy - r, w: r * 2, h: r * 2,
    line: { color: C.teal, width: 1.5 },
    fill: { color: "E8F5E9", transparency: 80 },
  });
  // axes
  s12.addShape(pptx.ShapeType.line, {
    x: cx - r - 0.3, y: cy, w: (r + 0.3) * 2, h: 0,
    line: { color: C.charcoal, width: 0.6 },
  });
  s12.addShape(pptx.ShapeType.line, {
    x: cx, y: cy - r - 0.3, w: 0, h: (r + 0.3) * 2,
    line: { color: C.charcoal, width: 0.6 },
  });

  // roots
  for (let k = 0; k < n; k++) {
    const angle = (2 * Math.PI * k) / n;
    const px = cx + r * Math.cos(angle);
    const py = cy - r * Math.sin(angle);
    s12.addShape(pptx.ShapeType.line, {
      x: cx, y: cy, w: px - cx, h: py - cy,
      line: { color: C.gold, width: 1.2 },
    });
    s12.addShape(pptx.ShapeType.ellipse, {
      x: px - 0.07, y: py - 0.07, w: 0.14, h: 0.14,
      fill: { color: C.navy },
    });
  }

  s12.addText(`n = ${n}`, {
    x: cx - 0.3, y: cy + r + 0.1, w: 0.6, h: 0.25,
    fontSize: 10, color: C.navy, fontFace: "Georgia", bold: true, align: "center",
  });
});

// Bottom property
s12.addShape(pptx.ShapeType.roundRect, {
  x: 0.8, y: 3.5, w: 11.7, h: 0.7,
  fill: { color: C.white },
  line: { color: C.tealLight, width: 0.5 },
  rectRadius: 0.05,
});
s12.addText(
  "单位根构成循环群 ℤₙ（阿贝尔群），在傅里叶变换 (FFT/NTT) 和编码理论中扮演核心角色",
  { x: 1.1, y: 3.6, w: 11.2, h: 0.5, fontSize: 12, fontFace: "Calibri", color: C.darkText, valign: "middle", align: "center" }
);

// ══════════════════════════════════════════════════════════
// SLIDE 13 - 总结与联系
// ══════════════════════════════════════════════════════════
const s13 = pptx.addSlide();
s13.background = { fill: C.navy };
// decorative
s13.addShape(pptx.ShapeType.ellipse, {
  x: -2, y: -2, w: 5, h: 5, fill: { color: C.teal, transparency: 88 },
});
s13.addShape(pptx.ShapeType.ellipse, {
  x: 10, y: 4, w: 5, h: 5, fill: { color: C.gold, transparency: 90 },
});
addFooter(s13, 13, 13);

s13.addText("总结与联系", {
  x: 0.8, y: 0.5, w: 10, h: 0.8,
  fontSize: 30, fontFace: "Georgia", color: C.white, bold: true,
});
s13.addText("Unit Circle: 一个概念，多个领域", {
  x: 0.8, y: 1.2, w: 10, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: C.tealLight, italic: true,
});

// Application areas
const apps = [
  { icon: "📐", title: "三角函数", desc: "cos θ, sin θ 的几何定义" },
  { icon: "🔄", title: "傅里叶变换", desc: "FFT 利用单位根分解信号" },
  { icon: "⚛️", title: "量子力学", desc: "U(1) 相位变换" },
  { icon: "🔌", title: "电路分析", desc: "交流电的相量表示" },
  { icon: "🎵", title: "信号处理", desc: "数字滤波器和频谱分析" },
  { icon: "🎨", title: "分形几何", desc: "Julia 集与 Mandelbrot 集" },
];

apps.forEach((a, i) => {
  const col = i % 3;
  const row = Math.floor(i / 3);
  const x = 0.8 + col * 4.1;
  const y = 1.8 + row * 1.5;

  s13.addShape(pptx.ShapeType.roundRect, {
    x, y, w: 3.8, h: 1.3,
    fill: { color: C.white, transparency: 12 },
    line: { color: C.white, width: 0.5, transparency: 60 },
    rectRadius: 0.06,
  });
  s13.addText(a.icon, {
    x: x + 0.15, y: y + 0.15, w: 0.45, h: 0.45,
    fontSize: 20, align: "center", valign: "middle",
  });
  s13.addText(a.title, {
    x: x + 0.7, y: y + 0.15, w: 2.9, h: 0.45,
    fontSize: 15, fontFace: "Georgia", color: C.goldLight, bold: true, valign: "middle",
  });
  s13.addText(a.desc, {
    x: x + 0.7, y: y + 0.65, w: 2.9, h: 0.5,
    fontSize: 11, fontFace: "Calibri", color: C.white, valign: "top",
  });
});

// Bottom quote
s13.addShape(pptx.ShapeType.rect, {
  x: 3, y: 5.3, w: 7.3, h: 0.03, fill: { color: C.gold },
});
s13.addText("「复平面单位圆——连接几何、代数与分析的最美桥梁」", {
  x: 1.5, y: 5.5, w: 10.3, h: 0.5,
  fontSize: 13, fontFace: "Georgia", color: C.tealLight, align: "center", italic: true,
});

// ══════════════════════════════════════════════════════════
// SAVE
// ══════════════════════════════════════════════════════════
const outPath = "D:\\software\\TianXingFC\\复平面单位圆.pptx";
pptx.writeFile({ fileName: outPath })
  .then(() => console.log("✅ Saved to: " + outPath))
  .catch((err) => console.error("❌ Error:", err));