# 论元相子的螺旋阶梯几何：法线的离散跃迁与时空二象性的数学根基（Wan 视频生成优化提示词）

本文件是上一版通用提示词的 Wan（通义万相）适配优化版，针对 Wan 系列模型的生成特性改写。直接用于 Wan 2.1 / 2.2 / 2.5 文生视频。

## Wan 适配要点

- **语言策略**：Wan 支持中文与英文，英文提示词生成更稳定。本文件每个镜头给出英文主 prompt 与中文对照，实际提交优先用英文。
- **时长**：Wan 单次生成约 5 秒。10 秒成片推荐"两段 5 秒"方案：段一覆盖镜头 1-3，段二覆盖镜头 4-5，用 Wan 的续写/拼接能力衔接。若所用平台支持分镜（Storyboard / 多镜头）模式，可直接按 5 个分镜逐段输入。
- **抽象几何**：Wan 对三维线框、辉光材质响应较好，但对复杂数学符号易乱码。本版刻意弱化文字标注（仅保留极简 $l_P$、$t_P$、$c$、$D^2 \times \mathbb{R}$ 提示，且放在画面边缘淡出），主要靠几何形状本身表达含义。
- **镜头语言**：全部使用 Wan 识别的 camera 关键词（slow push-in、abrupt snap、pull back and tilt、orbit、split screen、fade out）。
- **离散跳跃**：Wan 无法生成严格的"无中间帧"离散跳变，本版用 abrupt / teleport-like instant jump 逼近，并要求不留残影的快速单步位移。
- **负面提示词**：另附 negative prompt，用于抑制几何变形、闪烁、乱码与不必要内容。

## 镜头分镜（每镜独立 prompt，逐镜生成后拼接）

### 镜头一（0-2.5s）波性旋转

**英文主 prompt：**

> A translucent golden circular disk floats in deep dark space, its glowing ring-shaped edge slowly rotating. A bright golden phase arrow revolves around the ring at constant speed, leaving a faint circular trail. A thin cyan luminous line stands perpendicular at the center of the disk, pulsing softly. The camera slowly pushes in toward the disk. Clean 3D wireframe mathematical animation, glowing emissive materials, dark navy blue background, cinematic lighting.

**中文对照：** 一个半透明金色圆盘悬浮在深色太空，发光的环形边缘缓缓旋转；一支明亮的金色相位指针绕环匀速转动，留下淡淡圆形轨迹；盘心竖立一根纤细的青白色发光法线，轻轻脉动；镜头缓慢推近圆盘。干净的 3D 线框数学动画，自发光材质，深蓝黑背景，电影感打光。

### 镜头二（2.5-4.5s）满周闭合，首次离散跃迁

**英文主 prompt：**

> After the phase arrow completes one full 360-degree rotation, a marker on the glowing ring flashes brightly. The entire golden disk abruptly snaps upward along the vertical cyan line, a teleport-like instant jump with no visible intermediate positions, leaving a dim afterimage disk behind. Clean 3D mathematical animation, golden and cyan glow, dark space background, sharp single-step motion.

**中文对照：** 相位指针转过整整 360° 后，发光环上的标记点骤然亮起；整个金色圆盘沿垂直青线骤然向上弹跳，如瞬移般的瞬间跳跃，不出现任何中间位置，原处留下暗淡残影；金色与青色辉光，深色太空背景，干脆的单步位移。

### 镜头三（4.5-7.0s）螺旋阶梯升起

**英文主 prompt：**

> The golden disk rotates and repeatedly jumps upward in discrete steps along the vertical axis, stacking many thin rotating disks one by one into a spiral staircase tower. The misaligned rotations make a continuous golden helix thread through all stacked layers. The camera slowly pulls back and tilts to a side view, revealing the full staircase tower. 3D wireframe glow animation, dark navy background, smooth cinematic camera movement.

**中文对照：** 金色圆盘一边旋转一边沿竖直轴逐级向上跳跃，许多薄圆盘层层堆叠成螺旋阶梯塔；错位的旋转相位使一条连续金色螺旋线贯穿所有层级；镜头缓缓拉远并转至侧视，露出完整阶梯塔；3D 线框辉光动画，深蓝黑背景，平滑电影感运镜。

### 镜头四（7.0-8.5s）采样对偶

**英文主 prompt：**

> Split screen comparison. The left side shows a smooth continuous golden helix line; the right side shows a discrete staircase function made of flat steps, each step corner exactly aligning with the helix at integer turns. A soft minimal annotation appears briefly at the edge of the frame. Clean abstract mathematical visualization, golden and cyan glow, dark background, no photorealistic elements.

**中文对照：** 分屏对比；左侧是一条连续光滑的金色螺旋线，右侧是由平直台阶组成的离散阶梯函数，每个台阶顶点恰好对齐螺旋线的整数圈采样点；画面边缘短暂浮现极简标注；干净的抽象数学可视化，金青辉光，深色背景，无写实元素。

### 镜头五（8.5-10.0s）定格成体

**英文主 prompt：**

> The camera orbits around the geometry as the stacked rotating disks and the continuous helix merge into one complete solid cylinder with a spiral surface, a 3D screw-like solid body with smooth glowing surface. The scene slowly fades out. Clean minimal mathematical 3D render, golden and cyan emissive glow, dark space background, cinematic final shot.

**中文对照：** 镜头环绕几何体一周，堆叠旋转的圆盘与连续螺旋线融为一体，成为一个表面带螺旋纹路的完整实心圆柱体，即三维螺杆状实心体；画面缓缓淡出；极简干净的数学 3D 渲染，金青自发光辉光，深色太空背景，收尾镜头。

## 两段式 10 秒拼接方案（推荐）

### 段一（0-5s）：旋转、满周闭合跳跃、阶梯堆叠

> A translucent golden circular disk floats in deep dark space, a bright golden phase arrow revolving around its glowing ring at constant speed, a thin cyan luminous line standing at the center. After the arrow completes one full rotation, the disk abruptly snaps upward along the vertical line in a teleport-like instant jump, then repeatedly jumps upward in discrete steps, stacking thin rotating disks into a spiral staircase tower threaded by a continuous golden helix. Camera slowly pulls back and tilts to a side view. Clean 3D wireframe mathematical animation, glowing emissive materials, dark navy background, cinematic.

### 段二（5-10s）：采样对偶、环绕定格成体

> Split screen comparison of a smooth continuous golden helix versus a discrete staircase function, with each step corner exactly aligning with the helix at integer turns and a soft minimal annotation at the frame edge. The camera then orbits as the stacked disks and helix merge into one complete solid cylinder with a spiral surface, a 3D screw-like solid body with smooth glowing surface, slowly fading out. Clean minimal mathematical 3D render, golden and cyan emissive glow, dark space background, cinematic.

## 单条完整 prompt（用于支持多镜头/长视频的模式）

> A 10-second abstract mathematical animation in deep dark space. A translucent golden disk with a glowing ring edge rotates while a golden phase arrow revolves around it and a thin cyan line stands at the disk center. After each full rotation the disk snaps upward along the cyan line in an instant discrete jump, repeatedly stacking thin rotating disks into a spiral staircase tower threaded by a continuous golden helix. A split screen then compares the continuous helix with a discrete staircase whose corners align at integer turns, with minimal soft labels. Finally the camera orbits as the structure merges into a complete solid spiral cylinder, fading out. Clean 3D wireframe glow animation, golden and cyan emissive materials, dark navy blue background, slow cinematic camera movements, no photorealistic elements.

## Negative Prompt（通用，所有镜头共用）

> blurry, distorted, deformed disk, broken geometry, morphing shapes, flickering, jittery unstable motion, garbled text, misshapen letters, watermark, low quality, oversaturated, photorealistic, human, face, limbs

## 生成参数建议

- 分辨率：720P（画质优先可选 1080P，若平台支持）
- 单段时长：5 秒；两段拼接为 10 秒
- 运动幅度：medium（过度运动易诱发几何抖动）
- 固定随机种子，便于复现与多轮挑选
- 若发现台阶顶点与螺旋线对不齐（Wan 常见），降低 camera 运动幅度并在段二 prompt 中保留"each step corner exactly aligning with the helix"

## 数学锚点核验清单（生成后对照）

- 圆盘是实心 $D^2$，不是纯圆环 $S^1$；旋转发生在边界圆环上；
- 法线经过圆心、垂直于盘面；
- 跃迁表现为瞬间跳跃，无连续滑行动画；
- 每跳幅度 $l_P$、每跳节拍 $t_P$，比例即光速 $c$；
- 终态为三维实心圆柱螺旋体，对应 $D^2 \times \mathbb{R}$，豪斯多夫维数 3。
