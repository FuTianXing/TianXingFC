import numpy as np
N = 48
c = np.arange(N) - N / 2.0 + 0.5          # half-offset coords
X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
IC = N // 2
def mi(d): return (d + np.pi) % (2*np.pi) - np.pi
def idx(v):  # coordinate value -> nearest index
    return int(np.argmin(np.abs(c - v)))
Rc = 4.0
rho = np.sqrt(X**2 + Y**2)
naive = np.arctan2(Z, rho - Rc)

def wind_x(phi, i, j0, k0, r):
    t = 0.0
    for j in range(j0-r, j0+r):
        for k in range(k0-r, k0+r):
            p00=phi[i,j,k]; p10=phi[i,j+1,k]; p11=phi[i,j+1,k+1]; p01=phi[i,j,k+1]
            t += mi(p10-p00)+mi(p11-p10)+mi(p01-p11)+mi(p00-p01)
    return t/(2*np.pi)

def wind_y(phi, j, i0, k0, r):
    t = 0.0
    for i in range(i0-r, i0+r):
        for k in range(k0-r, k0+r):
            p00=phi[i,j,k]; p10=phi[i+1,j,k]; p11=phi[i+1,j,k+1]; p01=phi[i,j,k+1]
            t += mi(p10-p00)+mi(p11-p10)+mi(p01-p11)+mi(p00-p01)
    return t/(2*np.pi)

print("coords[0]=%.1f  x=Rc at index %.1f" % (c[0], Rc + N/2 - 0.5))
i_core = idx(Rc)          # nearest site to x=Rc
print("nearest site to x=Rc: i=%d (x=%.1f)" % (i_core, c[i_core]))
print()
print("[A] fixed-x window (j,k), the composite-script reading")
for xv in [3.5, 4.5, 5.5]:
    print("   x=%.1f  (i=%d): W = %+.4f" % (xv, idx(xv), wind_x(naive, idx(xv), IC, IC, 3)))
print()
print("[B] fixed-y window (i,k) around core point (x=Rc, z=0) -- transverse plane")
for yv in [-0.5, 0.5]:
    j = idx(yv)
    for r in [1, 2, 3, 4, 6]:
        print("   y=%.1f (j=%d) rad=%d: W = %+.4f" % (yv, j, r, wind_y(naive, j, idx(Rc), idx(0.0), r)))
print()
print("[C] straight vortex along x at (y,z)=(0,0): phi=atan2(z,y)")
lin = np.arctan2(Z, Y)
print("   fixed-x window at i=%d: W = %+.4f (expect 1)" % (IC, wind_x(lin, IC, IC, IC, 3)))
print("   straight vortex along y at (x,z)=(4,0): phi=atan2(z,x-4)")
lin2 = np.arctan2(Z, X - 4.0)
print("   fixed-y window at j=%d rad=3 around (x=4,z=0): W = %+.4f (expect 1)" % (IC, wind_y(lin2, IC, idx(4.0), idx(0.0), 3)))
print()
print("[D] window center sensitivity for naive ring, transverse plane y=0.5")
for dx in [-1, 0, 1]:
    for dz in [-1, 0, 1]:
        print("   shift(di=%+d, dk=%+d): W = %+.4f" % (dx, dz, wind_y(naive, idx(0.5), idx(Rc)+dx, idx(0.0)+dz, 2)))
