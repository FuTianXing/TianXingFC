import io
p = r"验证脚本/minimal_verification_vortex_init_apparatus.py"
s = io.open(p, encoding="utf-8").read()
old1 = r'''    print("  符号约定：读数 = 绕数 x (平面内升序法向 · 芯线切线)。标准环在该约定下 n = -1，")
    print("  故 theta=0/pi/2 读 +1，theta=pi/3pi/2 读 -1；n = +1 的反向构造给相反模式（判据 M8）")'''
new1 = r'''    print("  符号约定：读数 = 绕数 x (平面内升序法向 · 芯线切线)。标准环（n = 1）在 theta=0 读 +1，")
    print("  n = -1 整体反号；同一环在 theta=0 与 theta=pi 两方位反号，因两交叉点的芯线切线反向")'''
assert s.count(old1) == 1
s = s.replace(old1, new1)

old2 = r'''    print("[M8] 符号控制：n = +1 的环与直线涡（与标准环反号）")
    check("M8 环 n=+1 读数（theta=0 方位）",
          winding(ring_naive(4.0, n=1), 1, IC, center=(4.0, 0.0), rad=2), -1.0)
    check("M8 直线涡 n=-1 读数",
          winding(straight(1, (0.0, 0.0), n=-1), 1, IC, center=(0.0, 0.0), rad=2), -1.0)'''
new2 = r'''    print("[M8] 符号控制：n = 1 与 n = -1 的环、直线涡")
    check("M8 环 n=+1 读数（theta=0 方位）",
          winding(ring_naive(4.0, n=1), 1, IC, center=(4.0, 0.0), rad=2), 1.0)
    check("M8 环 n=-1 读数（theta=0 方位）",
          winding(ring_naive(4.0, n=-1), 1, IC, center=(4.0, 0.0), rad=2), -1.0)
    check("M8 环 n=+1 在 theta=pi 方位反号",
          winding(ring_naive(4.0, n=1), 1, IC, center=(-4.0, 0.0), rad=2), -1.0)
    check("M8 直线涡 n=-1 读数",
          winding(straight(1, (0.0, 0.0), n=-1), 1, IC, center=(0.0, 0.0), rad=2), -1.0)'''
assert s.count(old2) == 1
s = s.replace(old2, new2)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched")
