import numpy as np
def mi(d): return (d + np.pi) % (2*np.pi) - np.pi
def build(N):
    c = np.arange(N) - N/2.0 + 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return c, X, Y, Z, np.sqrt(X**2+Y**2)
def E(phi):
    e = 0.6*float(np.sum(1.0-np.cos(phi)))
    for ax in range(3):
        e += 0.5*float(np.sum(1.0-np.cos(mi(np.diff(phi, axis=ax)))))
    return e
def pdens(P):
    a00,a10,a11,a01 = P[:-1,:-1],P[1:,:-1],P[1:,1:],P[:-1,1:]
    return (mi(a10-a00)+mi(a11-a10)+mi(a01-a11)+mi(a00-a01))/(2*np.pi)
def grad(phi):
    p = np.pad(phi, 1); ctr = (slice(1,-1),)*3
    s = (np.sin(p[2:,1:-1,1:-1]-p[ctr])+np.sin(p[:-2,1:-1,1:-1]-p[ctr])
         +np.sin(p[1:-1,2:,1:-1]-p[ctr])+np.sin(p[1:-1,:-2,1:-1]-p[ctr])
         +np.sin(p[1:-1,1:-1,2:]-p[ctr])+np.sin(p[1:-1,1:-1,:-2]-p[ctr]))
    return -0.5*s + 0.6*np.sin(phi)

print("标准环初态 E0 随盒尺寸（Rc = 4，半格偏移，开放边界）")
prev = None
for N in (24, 32, 40, 48, 56):
    c, X, Y, Z, RHO = build(N)
    phi = mi(np.arctan2(Z, RHO - 4.0))
    e = E(phi)
    print(f"  N={N:3d}  E0={e:10.1f}   E0/N={e/N:8.2f}  E0/N^3={e/N**3:7.4f}")
    prev = e

print()
print("梯度流轨迹：能量、交叉读出、环半径（N=48）")
N = 48
c, X, Y, Z, RHO = build(N)
IC = N//2
phi = mi(np.arctan2(Z, RHO - 4.0))
for step in (0, 50, 100, 150, 200, 300, 400, 600, 800, 1200):
    while True:
        d = pdens(np.take(phi, IC, axis=1))
        hit = np.argwhere(np.abs(d) > 0.5)
        if len(hit):
            rho = np.hypot(c[hit[0][0]]+0.5, 0.5)
            val = d[hit[0][0], hit[0][1]]
            print(f"  step {step:5d}  E={E(phi):12.4e}  交叉条数={len(hit):2d}  首个交叉 rho={rho:5.2f}  读数={val:+.1f}")
        else:
            print(f"  step {step:5d}  E={E(phi):12.4e}  无交叉（读数 0）")
        if step == 0: break
        phi = phi - 0.02*grad(phi)
        if step == 50: pass
        break
    if step < 1200:
        nxt = {0:50,50:100,100:150,150:200,200:300,300:400,400:600,600:800,800:1200}.get(step)
        if nxt:
            phi = phi.copy()
            for _ in range(nxt-step):
                phi = phi - 0.02*grad(phi)
