# -*- coding: utf-8 -*-
"""把时空创生与膨胀机制的洞见提炼进两份理论总纲基准文件（2026-09-09）"""
import io

def read(p):
    with io.open(p, 'r', encoding='utf-8') as f:
        return f.read()

def write(p, s):
    with io.open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(s)

# ---------- 1. 核心洞见速览：本体与宇宙节新增第 16 条 ----------
p1 = r'理论总纲\天行新学核心洞见速览.md'
s1 = read(p1)

anchor1 = '## 二、螺旋与几何'
assert s1.count(anchor1) == 1, 'anchor1 不唯一: %d' % s1.count(anchor1)

new_insight = (
    '16. **时空前沿缺邻居即补位，宇宙正在持续凝固**：最外沿元相子到达外侧边界时没有邻居可撞，'
    '折返条件缺失，本原在它外面创生出新元相子作邻居，天行力校准其相位成反相；'
    '新元相子又需要更新的外侧邻居，此递归链条使时空晶格必然向外扩张，微观上每摆程推进一格（速度 $c$）。'
    '观测到的均匀膨胀 $v = H_0 d$ 由内部落位承担，边缘扩张是时空前沿本身：'
    '凝固区域半径 $R(t) = (R_0 + c/H_0)e^{H_0 t} - c/H_0$ 指数超光速扩张，'
    '创生边界（时空生成）与信号边界（信息传播）是两回事，空间生成不受光速限制。'
    '大爆炸由此拆解为两层：成核-结晶是一次初始事件（多核接缝），凝固作为过程持续发生，'
    '每次补位创生都是一个格点从无时空到有时空的微观凝固事件；'
    '宇宙没有完成凝固的那一天，正如潜无穷没有完成的那一天。'
    '落位概率 $p$ 由此获得几何来源：边缘表面层占全体的比率 $3l_P/R \\approx 3t_P/t \\approx 3H_0 t_P$，'
    '不再是特设的内禀概率。\n'
)
s1 = s1.replace(anchor1, new_insight + anchor1, 1)
write(p1, s1)
print('核心洞见速览：已插入第 16 条，字节数 %d -> %d' % (len(read(p1)) - len(new_insight.encode('utf-8')), len(read(p1))))

# ---------- 2. 要点总览：开放项登记与日期行 ----------
p2 = r'理论总纲\元相子理论要点总览.md'
s2 = read(p2)

old_open = '未闭合项（C 生成机制、让位链动力学、三维推广）见开放问题总览第 20 条。'
new_open = ('未闭合项（天行力反相校准机制、前沿创生与 C 生成机制的统一形式、'
            '$R(t)$ 与观测视界的数值对接、三维各向同性、无穷格基数）'
            '见开放问题总览第 19、20 条（论文第十二节登记五项）。')
assert s2.count(old_open) == 1, 'old_open 不唯一: %d' % s2.count(old_open)
s2 = s2.replace(old_open, new_open, 1)

old_date = '2026-09-09 增补 P8a 多核成案同步与粒子机制三主线闭环：复合位错粒子、转换波运动、电荷守恒承载层）'
new_date = ('2026-09-09 增补 P8a 多核成案同步与粒子机制三主线闭环：复合位错粒子、转换波运动、电荷守恒承载层；'
            '2026-09-09 边缘创生律与持续创生洞见提炼（核心洞见速览同步增补））')
assert s2.count(old_date) == 1, 'old_date 不唯一: %d' % s2.count(old_date)
s2 = s2.replace(old_date, new_date, 1)

write(p2, s2)
print('要点总览：已更新开放项与日期行，最终字节数 %d' % len(read(p2)))
