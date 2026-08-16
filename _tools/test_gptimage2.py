# -*- coding: utf-8 -*-
"""测试 gptimage2：生成一张简单测试图。"""
import subprocess
import sys

PROMPT = (
    'A clean minimal test image: a single cute orange tabby cat sitting on a '
    'windowsill in warm sunlight, soft warm tones, simple composition, high quality.'
)

cmd = [
    sys.executable,
    r"C:\Users\Administrator\.agents\skills\gptimage2\scripts\generate_image.py",
    "--prompt", PROMPT,
    "--size", "1024x1024",
    "--quality", "low",
    "--output-dir", r"D:\software\TianXingFC\images",
]
print("Running:", " ".join(cmd[:3]), "...")
proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(proc.stdout)
if proc.stderr:
    print("STDERR:", proc.stderr[-4000:], file=sys.stderr)
sys.exit(proc.returncode)
