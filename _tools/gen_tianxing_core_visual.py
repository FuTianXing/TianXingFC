# -*- coding: utf-8 -*-
"""用 gptimage2 基于天行新学核心思想生成视觉图：心物双螺旋 + 复平面单位圆 + 螺旋阶梯。"""
import subprocess
import sys

PROMPT = (
    'A majestic conceptual visualization of the core geometric ontology of '
    '"Tianxing New Learning" (天行新学) philosophy: the mind-matter dual helix. '
    'Deep cosmic dark background (near-black, subtle deep blue-black gradient). '
    'Center: a glowing golden unit circle on a translucent complex plane disc, '
    'representing the fundamental "onphasor" (元相子), a dynamic unit circle '
    'psi = e^(i*theta) with radius h-bar, rotating with luminous phase markers; '
    'the real axis and imaginary axis drawn as two faint thin cross lines with '
    'labels x (real) and y (imaginary). '
    'Emanating from the circle, TWO intertwined helical threads rise vertically '
    'along the normal axis like a DNA double helix: the REAL-axis matter helix '
    'in cool cyan-blue light, and the IMAGINARY-axis mind helix in warm gold-amber '
    'light, perfectly phase-offset by pi/2 (quarter turn), winding around an '
    'invisible solid cylinder (D^2 x R state space). '
    'Along the rising helices, discrete luminous steps appear like a spiral '
    'staircase: continuous smooth helical arcs punctuated by crisp horizontal '
    'quantum steps, visualizing the unity of continuous rotation and discrete '
    'transition. '
    'Yin-yang complementary feel: the two helices interlock in perfect complementarity, '
    'wave-particle duality made geometric. '
    'Minimal, precise, academic-rigorous aesthetic: thin glowing lines, subtle '
    'particle sparks at the phase markers, gentle depth of field, lots of dark '
    'negative space, elegant and profound, high-end scientific illustration style. '
    'NO text, NO labels other than the faint x/y axis hints. 16:9 widescreen composition.'
)

cmd = [
    sys.executable,
    r"C:\Users\Administrator\.agents\skills\gptimage2\scripts\generate_image.py",
    "--prompt", PROMPT,
    "--size", "2560x1440",
    "--quality", "high",
    "--thinking", "high",
    "--output-dir", r"D:\software\TianXingFC\images",
]
print("Running:", " ".join(cmd[:3]), "...")
proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(proc.stdout)
if proc.stderr:
    print("STDERR:", proc.stderr[-4000:], file=sys.stderr)
sys.exit(proc.returncode)
