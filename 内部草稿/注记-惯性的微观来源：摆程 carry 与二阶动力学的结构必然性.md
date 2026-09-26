# 注记-惯性的微观来源：摆程 carry 与二阶动力学的结构必然性

- **日期**：2026-09-26 10:34:57
- **状态**：待作者裁决（一个机制提案、一组判据接口、一处备用方案）
- **定位**：回应第 434 条与规范侧转写裁决（惯性两侧同源 $I_0 = I$）之后的遗留项：惯性的微观来源。本注记先定位问题，再给机制提案与判据接口
- **依据**：公理 5 摆程节律（每摆程一次事件）；公理 4 波性侧相位推进；完美弹性定理（态变量只有提升相位、旋向与方向标签，无承载损耗的连续量）；[[天行力与规范原理的物理来源：去中心化校准作为规范对称性与相干基态的基础]] §3.1
- **数值验证**：未立（第 436 条三判据设计见 §六，期望 3/3）

## 一、现有依据是循环的

体系当前对方向惯性的依据写在 [[天行力与规范原理的物理来源：去中心化校准作为规范对称性与相干基态的基础]] §3.1：实轴方向在无偏转时处于校准稳态，不存在驱动它持续变化的势能梯度，于是在无势能梯度的前提下自由自由度保持静止或匀速。

这句话的后半截（无梯度则匀速）就是惯性定律本身。用它来解释方向惯性，等于用惯性来解释惯性。该段自己标注了这一依赖（"依赖实轴方向势能形式的确认为前提"），所以这是已知的悬置，不算疏漏。本注记要把它前推一步。

## 二、真正待解释的是二阶时间结构

第 434 条已经证明 $I$ 与 $I_0$ 在经典低能层不可观测：动能项与势能项同乘惯性系数，变分时两侧消去，等幅条件与长波光速都不含它。所以惯性的微观来源不落在"$I$ 等于几"上。

真正的问题是方向自由度为什么具有二阶时间动力学。三种结构的后果截然不同：

| 时间结构 | 连续极限 | 物理后果 |
|---|---|---|
| 零阶（约束） | 无时间导数 | 无动力学，方向被冻结 |
| 一阶（弛豫） | 扩散方程 | 偏转被瞬时摊平，光不存在 |
| 二阶（惯性） | 波动方程 | 传播，光存在 |

二阶性因此是光存在的结构前提。要解释的对象是这个二阶结构从哪里来。系数取值属于另一个问题，第 434 条已证它在低能层不可观测。

## 三、二阶时间结构在方程中的形式

二阶结构在三个层面等价出现：作用量含 $\dot\alpha^2$、运动方程最高时间导数为 2 且与空间二阶反号、色散为实数且 $\omega^2 \propto k^2$。三个层面是同一件事的三种写法。

作用量层：

$$L = \frac{I}{2}\sum_n \dot\alpha_n^2 - \frac{I}{2t_P^2}\sum_n\left[1-\cos(\Delta\alpha_n)\right], \qquad \Delta\alpha_n = \alpha_{n+1} - \alpha_n,$$

第一项即动能项，是二阶结构的载体；第 434 条已证 $I$ 在低能层被消去，此处只标记结构。

方程层（严格变分）：

$$I\ddot\alpha_n = \frac{I}{t_P^2}\left[\sin(\alpha_{n+1}-\alpha_n) - \sin(\alpha_n-\alpha_{n-1})\right],$$

线性层给出感知-校正链方程 $\ddot\alpha_n = \frac{1}{t_P^2}(\alpha_{n+1}+\alpha_{n-1}-2\alpha_n)$，连续极限为 $\partial_t^2\alpha = c^2\partial_x^2\alpha$，$c^2 = l_P^2/t_P^2$。左侧二阶时间、右侧二阶空间、两者反号，这是双曲型的标志，也是光速出现的条件。

摆程 carry 层（离散精确形式）：

$$v^{(s+1)} = v^{(s)} + \delta^{(s)}, \qquad \alpha^{(s+1)} = \alpha^{(s)} + v^{(s+1)},$$

消去 $v$ 得 $\alpha^{(s+1)} - 2\alpha^{(s)} + \alpha^{(s-1)} = \delta^{(s)}$，即二阶摆程差分等于邻居曲率，连续极限给 $\ddot\alpha = (l_P^2/t_P^2)\partial_x^2\alpha$。对照一阶写法（$\alpha^{(s+1)} = \alpha^{(s)} + \delta^{(s)}$）给出扩散方程 $\partial_t\alpha = (l_P^2/t_P)\partial_x^2\alpha$：同一套原料，去掉 carry 那一层，方程从双曲型掉成抛物型。

谱层：二阶的离散色散为 $\omega(k) = \frac{2}{t_P}\vert\sin(k l_P/2)\vert$，长波 $\omega = (l_P/t_P)\vert k\vert$；一阶给 $\omega = \frac{i\,4}{t_P}\sin^2(k l_P/2)$（纯虚频率，指数衰减，无传播）。判据只需看 $\omega$ 取实数还是纯虚。

二阶为何是唯一选择：作用量若含一次时间导数 $\dot\alpha$，运动方程即含 $\dot\alpha$，对应耗散并破坏时间反演；完美弹性裁决（元相子无组成部分、折返由方向因子 $\pm 1$ 实现、无承载损耗的连续量）禁止耗散通道，所以时间导数只能成对出现，最低阶为 $\dot\alpha^2$。再往上，含 $\dddot\alpha$ 及以上的方程带 Ostrogradsky 负模不稳定，同样被排除。二阶是有演化且无耗散的唯一稳定形式。

规范侧同型：线性化方程为 $\ddot\varphi_i = -(1/t_P^2)(\Lambda\varphi)_i$，$\Lambda_{ij}(k) = s(k)(\delta_{ij} - \hat k_i\hat k_j)$，$s(k) = \sum_a 4\sin^2(k_a l_P/2)$；横向模 $\omega^2 = s(k)/t_P^2$，纵向落在零本征值上被高斯约束冻结（第 435 条）。两侧方程同型是 $I_0 = I$ 的方程层证据。

## 四、标准物理的对照

标准物理把惯性拆成三个独立问题，逐个处理，最关键的那个被几何化。

层一，自由运动为何是匀速直线。狭义相对论里自由粒子作用量必须取 $S = -mc\int ds$：要求洛伦兹不变、只依赖世界线、局域、导数阶数最低，形式唯一，于是自由运动走直线（广义相对论里走测地线）。这是对称性与最小作用量原理的唯一性论证，与本文唯一性半环同型。它把机制换成"给定前提下唯一"，并不声称解释惯性为何存在。体系照抄这条路没有增量。

层二，惯性质量的数值从哪来。基本粒子质量来自希格斯机制（$m = y v/\sqrt{2}$）；可见物质的主体质量来自 QCD 结合能（胶子场能量与夸克动能，经手征对称破缺表达），夸克裸质量只占百分之几。更要紧的是，希格斯机制给出质量项，它与"$\ddot x = F/m$ 里的惯性"之间的等价在标准物理里没有推导。

层三，惯性质量为何等于引力质量。这是实验事实（Eötvös 到 MICROSCOPE，参数被压到 $10^{-15}$ 量级），广义相对论把它当公设，没有微观解释。马赫路线本可给出机制，但广义相对论只部分实现它：Gödel 的旋转宇宙解与真空解表明惯性结构可以脱离远距物质存在，Brans-Dicke 类型的马赫效应被太阳系实验压得很弱。

真空路线与熵力路线都不成立或未成立。Sakharov 的诱导引力与 Puthoff 的零点场惯性受到明确批评，量级与惯性质量的比例关系对不上；借 Unruh 效应解释惯性的尝试同样败在量级上。熵力路线（Jacobson 从热力学推出爱因斯坦方程、Verlinde 的熵力引力）在引力侧有真结果，在惯性侧仍是纲领。

结论：标准物理不解释惯性。层一被几何化，层二部分回答，层三留作实验事实。

## 五、提案：惯性 = 摆程 carry，载体是旋向

机制只用既有原料（摆程、公理 5、旋向、完美弹性），不添新公理。

其一，摆程内方向被锚定。每个摆程里相位从实轴方向出发扫过 $2\pi$，锚在摆程进行中被使用，可改写的时刻只在摆程边界。这是公理 5 的直接后果。

其二，边界处改写的对象是 carry，锚的位置由 carry 累加给出。设摆程间 carry $v$ 表示每摆程锚的移动量，校正事件的效果落在 $v$ 上：

$$v^{(s+1)} = v^{(s)} - k\,\delta^{(s)}, \qquad \alpha^{(s+1)} = \alpha^{(s)} + v^{(s+1)},$$

$\delta$ 为邻居偏转差。若改成校正直接改写锚（$\alpha^{(s+1)} = \alpha^{(s)} - k\delta^{(s)}$，没有 $v$ 这一层），递推为一阶，连续极限给扩散。

其三，连续极限合成二阶。$\Delta\alpha/\Delta t = v$ 与 $\Delta v/\Delta t = -k\delta$ 联立给出 $\ddot\alpha \propto \delta$（邻居二阶差分），即波动方程，系数比给出 $c = l_P/t_P$。所谓惯性系数 $I$ 就是把二阶差分写成 $I\ddot\alpha = -\partial V/\partial\alpha$ 时的组合系数，由单拍参数与 $t_P$ 给出。

carry 的载体是旋向。旋向是已声明的态变量（$\pm 1$ 标签），完美弹性使它只由碰撞翻转，运动学上不会自发改变。若每拍锚往哪边挪由旋向决定，则上一拍的移动方向被带到这一拍，这正是变率的持久性，也就是惯性。惯性由此有一个可指认的物理载体，不必作为裸资源挂在公理层。

两个推论。其一，两侧 carry 结构同型（都来自摆程边界加旋向记忆），同一个单拍结构给出同一个系数，所以 $I_0 = I$ 从约定升级为机制结果。其二，若惯性来自旋向记忆，碰撞翻转旋向时锚的漂移方向应当同时翻转，存在惯性符号翻转现象，与折返图像可对照。

## 六、判据化：第 436 条三条判据

判据一，二阶性对照。实现最小摆程-校正递推，位置更新规则与 carry 更新规则各跑一次：前者色散为虚（扩散），后者给 $\omega = c\vert k\vert$。期望结构对照成立，判据稳健（换实现不改结论）。

判据二，$I$ 的低能不可观测性。从单拍参数导出 $I$ 的组合形式，验证低能观测量（长波色散与等幅条件）对 $I$ 不敏感，与第 434 条互为交叉核对。判据敏感。

判据三，两侧同源。同一 carry 规则分别施于实轴方向场与基准取向场联络，验证两者给出同一系数；给两侧不同单拍参数时，判据给出 $I_0 \ne I$ 的定量偏离，说明同源是可检验的模型性质，可以就此证伪。判据敏感。

## 七、与体系其他部分的接口

[[内部草稿/注记-洛伦兹符号定理：作用量的动能势能分解与光速的无参数重现]] 把 $I$ 作为动能系数使用，本注记给出它为何存在。[[内部草稿/注记-洛伦兹符号的规范侧转写：取向惯性能、E 型双归属与等幅条件]] 的 $I_0$ 由本注记的同源机制承接。[[验证脚本/注记-规范侧纵向模的高斯约束消去（第435条）：约束守恒、模式计数与纵向零代价]] 给出约束与惯性的分工：约束管纵向冻结，惯性管横向传播，两者合起来是四分量筛剩两个横向的完整图像。

机制的上游接口是 [[内部草稿/注记-校准代价的实现半环：从天行力对齐度到 plaquette 作用量]]（对角度的对齐度审计）与公理 4 的余弦耦合。下游接口是层二的质量谱问题，本注记不触及。

## 八、风险、备用方案与待裁决点

关键假设是"摆程内方向不可自由改写"。若允许方向在摆程内连续演化，carry 结构消失，惯性退回公设。第二个软点是旋向为离散 $\pm 1$、$\dot\alpha$ 为连续量，需要把每拍一个小量、多拍累加成连续速率的粗粒化写清楚，否则该提案只算类比。

备用方案是承认惯性为不可还原的公理层资源，与 $l_P$、$t_P$ 并列为刻度常数。代价是放弃"为什么是二阶"的解释力，收益是不引入新假设。

待裁决点：其一，采纳 carry 路线还是备用方案。其二，单拍 carry 的定量规则（$v$ 的累加方式与旋向翻转条件）。其三，是否立第 436 条验证。

## 九、边界声明

验证什么：二阶性与 carry 结构的对应、$I$ 的低能不可观测性、两侧同源的可检验性。

不验证什么：$I$ 的精确数值（需要单拍微观模型的闭合）、层二的质量谱（体系尚未触及）、量子层修正、与引力侧的耦合（惯性质量与引力质量的等式在体系内尚无对应物）。

## 外部对照文献

- Mach, E. (1883). Die Mechanik in ihrer Entwickelung.
- Einstein, A. (1918). Prinzipielles zur allgemeinen Relativitätstheorie. Annalen der Physik, 55, 241.
- Thirring, H., & Lense, J. (1918). Über den Einfluss der Eigenrotation der Zentralkörper auf die Bewegung der Planeten und Monde. Physikalische Zeitschrift, 19, 156.
- Gödel, K. (1949). An example of a new type of cosmological solutions of Einstein's field equations of gravitation. Reviews of Modern Physics, 21, 447.
- Sciama, D. W. (1953). On the origin of inertia. Monthly Notices of the Royal Astronomical Society, 113, 34.
- Sakharov, A. D. (1967). Vacuum quantum fluctuations in curved space and the theory of gravitation. Doklady Akademii Nauk SSSR, 177, 70.
- Unruh, W. G. (1976). Notes on black-hole evaporation. Physical Review D, 14, 870.
- Puthoff, H. E. (1989). Gravity as a zero-point-fluctuation force. Physical Review A, 39, 2333.
- Jacobson, T. (1995). Thermodynamics of spacetime: The Einstein equation of state. Physical Review Letters, 75, 1260.
- Ciufolini, I., & Wheeler, J. A. (1995). Gravitation and Inertia. Princeton University Press.
- Eötvös, R. v., Pekár, D., & Fekete, E. (1922). Beiträge zum Gesetze der Proportionalität von Trägheit und Gravität. Annalen der Physik, 373, 11.
- Verlinde, E. (2011). On the origin of gravity and the laws of Newton. Journal of High Energy Physics, 2011(4), 29.
- Yang, Y.-B., et al. (2018). Proton mass decomposition from the QCD energy momentum tensor. Physical Review Letters, 121, 212001.
- Touboul, P., et al. (2022). MICROSCOPE mission: final results of the test of the equivalence principle. Physical Review Letters, 129, 121102.

## 关联

- 惯性在两侧的落点：[[内部草稿/注记-洛伦兹符号定理：作用量的动能势能分解与光速的无参数重现]]；[[内部草稿/注记-洛伦兹符号的规范侧转写：取向惯性能、E 型双归属与等幅条件]]
- 约束与惯性的分工：[[验证脚本/注记-规范侧纵向模的高斯约束消去（第435条）：约束守恒、模式计数与纵向零代价]]
- 正稿：[[规范作用量的审计必然性：从净值审计到电磁场作用量的推导与数值验证]]
- 术语：[[内部草稿/术语提案]]
- 登记：[[内部草稿/开放问题总览：元相子理论待解决问题清单]]
