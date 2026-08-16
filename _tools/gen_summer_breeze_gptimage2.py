# -*- coding: utf-8 -*-
"""用 gptimage2 (tu-zi.com 中转站) 生成"夏日微风"明亮色场手写体视觉图。"""
import subprocess
import sys

PROMPT = (
    'A refreshing summer poster design with the large Chinese calligraphy text '
    '"夏日微风" (Summer Breeze) as the sole focal point, handwritten style. '
    'The background is a large clean bright gradient color field from fresh '
    'light mint green to sky blue, bright and airy, with a soft breezy glow, '
    'like a wind / weather / lifestyle brand visual. '
    'The four Chinese characters are white, thin, elongated, elegant handwritten '
    'strokes with smooth flowing lines, long graceful arc tails that trail off '
    'like wind, feather-light and breezy. The characters are slightly staggered '
    'and offset vertically with airy spacing, leaving generous breathing room '
    'in the center of the composition. '
    'A few tiny English words (like "summer breeze") and thin delicate decorative '
    'lines and curved wind lines are scattered lightly among the characters, very subtle. '
    'High-saturation but clean main color background, bright white text, a few '
    'light thin lines. Clean, transparent, light and airy mood, high-end lifestyle '
    'brand aesthetic. '
    'The key is the drifting direction of the strokes and the negative space; '
    'NOT bold heavy slogan lettering, no heavy outlines. Minimalist, lots of '
    'negative space, crisp white strokes on the bright gradient, elegant and breezy. '
    '16:9 widescreen composition.'
)

cmd = [
    sys.executable,
    r"C:\Users\Administrator\.agents\skills\gptimage2\scripts\generate_image.py",
    "--prompt", PROMPT,
    "--size", "2560x1440",
    "--quality", "medium",
    "--thinking", "medium",
    "--output-dir", r"D:\software\TianXingFC\images",
]
print("Running:", " ".join(cmd[:3]), "...")
proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(proc.stdout)
if proc.stderr:
    print("STDERR:", proc.stderr[-4000:], file=sys.stderr)
sys.exit(proc.returncode)
