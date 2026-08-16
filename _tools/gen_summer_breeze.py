# -*- coding: utf-8 -*-
"""生成"夏日微风"黑底白字 16:9 手写体视觉图。通过 subprocess 调 generate_image.py 避免 shell 引号转义问题。"""
import subprocess
import sys

PROMPT = (
    'A refreshing summer poster design with the large Chinese calligraphy text '
    '"夏日微风" (Summer Breeze) as the sole focal point, handwritten style. '
    'PURE BLACK background (#000000), BRIGHT WHITE text. '
    'The four Chinese characters are thin, elongated, elegant handwritten strokes '
    'with smooth flowing lines, long graceful arc tails that trail off like wind, '
    'feather-light and breezy. The characters are slightly staggered and offset '
    'vertically with airy spacing, leaving generous breathing room in the center '
    'of the composition. A few tiny English words (like "summer breeze") and thin '
    'delicate decorative lines are scattered lightly among the characters, very subtle. '
    'The stroke directions suggest drifting wind motion, as if the letters are being '
    'gently blown. Clean, transparent, light and airy mood, high-end lifestyle brand '
    'aesthetic, NOT bold heavy slogan lettering. Minimalist, lots of negative space, '
    'crisp white strokes on deep black, elegant and breezy. 16:9 widescreen composition.'
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
