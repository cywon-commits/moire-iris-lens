import numpy as np, sympy as sp
w,b,bb,m,mb,d=sp.symbols('w b bb m mb d')
w1,w2=-d/2,d/2
z=b+m/(w-w1)+m/(w-w2)
eq=sp.together(w-mb/(z-w1)-mb/(z-w2)-bb)
num=sp.numer(eq); poly=sp.Poly(sp.expand(num),w)
coef_f=sp.lambdify((b,bb,m,mb,d),poly.all_coeffs(),'numpy')
def images(beta,mass,dd,tol=1e-6):
    c=np.array(coef_f(beta,np.conj(beta),mass,np.conj(mass),dd),dtype=complex)
    while len(c) and abs(c[0])<1e-14: c=c[1:]
    out=[]
    for ww in np.roots(c):
        if min(abs(ww+dd/2),abs(ww-dd/2))<1e-8: continue
        zz=beta+mass/(ww+dd/2)+mass/(ww-dd/2)
        if abs(ww-np.conj(zz))<tol*max(1,abs(zz)):
            S=mass/(ww+dd/2)**2+mass/(ww-dd/2)**2; par=1 if 1-abs(S)**2>0 else -1
            if all(abs(zz-q[0])>1e-6 for q in out): out.append((zz,par))
    return out
def images_single(beta,mass,zc):
    # beta = z - mass/conj(z-zc)  -> shift
    bp=beta-zc; bm=abs(bp)**2; mr=mass
    # reuse quartic with a=0: z' - m/conj(z')=bp ; |z'|^2 - z' conj? solve via polar: e^{i phi}(r - m/r)=bp
    B=bm+2*mr.real; disc=B*B-4*abs(mr)**2
    if disc<0 or B<=0: return []
    out=[]
    for sg in (1,-1):
        r=np.sqrt((B+sg*np.sqrt(disc))/2); ph=np.angle(bp)-np.angle(r-mr/r); out.append((zc+r*np.exp(1j*ph),1 if r*r>abs(mr) else -1))
    return out
if __name__=="__main__":
    th=1/30.; C=4.; mass=C/(1j*th); s3=np.sqrt(3)
    a1=np.array([1.,0]); a2=np.array([.5,s3/2]); S=[]
    for n1 in range(-6,7):
        for n2 in range(-6,7):
            v=n1*a1+n2*a2; bet=(v[0]+1j*v[1])/(1j*th)
            if abs(bet)<75: S.append(bet)
    def nn(imgs,rmax=60): return sum(1 for z,p in imgs if p>0 and abs(z)<rmax), sum(1 for z,p in imgs if p<0 and abs(z)<rmax)
    import json
    U=dict((round(r[0]),r[3]) for r in json.load(open('U.json')))
    for dd in [8.,12.,16.,20.,26.,32.,40.,50.]:
        P=np.array([nn(images(bt,mass,dd)) for bt in S]).sum(0)
        L=np.array([nn(images_single(bt,mass,-dd/2)) for bt in S]).sum(0)
        R=np.array([nn(images_single(bt,mass,dd/2)) for bt in S]).sum(0)
        B0=np.array([nn([(bt,1)]) for bt in S]).sum(0)
        ex=P-(L+R-B0)
        print(f'd={dd:4.0f} (d/R={dd/np.sqrt(C/th):.2f}): rigid lens excess normal {ex[0]:+d}, mirrored {ex[1]:+d}   | relaxed U_moire={U[round(dd)]:.2f}')
