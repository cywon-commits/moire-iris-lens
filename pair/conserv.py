import numpy as np, json, sys
from scipy.ndimage import map_coordinates
from topo import frac, wind_points, cell_windings, c1
s3=np.sqrt(3.); Ac=s3/2; th=1/30.
def circ(ux,uy,x,xc,yc,r,n=6000):
    h=x[1]-x[0]; t=np.linspace(0,2*np.pi,n,endpoint=False); X=xc+r*np.cos(t); Y=yc+r*np.sin(t)
    I=(X-x[0])/h; J=(Y-x[0])/h; return map_coordinates(ux,[I,J],order=1),map_coordinates(uy,[I,J],order=1)
def Qint(ux,uy,x,xc,yc,r): return wind_points(np.stack(frac(*circ(ux,uy,x,xc,yc,r)),-1))
def Qarea(ux,uy,x,xc,yc,r):
    a,b=circ(ux,uy,x,xc,yc,r); a2=np.r_[a,a[:1]]; b2=np.r_[b,b[:1]]
    return 0.5*np.sum(a2[:-1]*b2[1:]-a2[1:]*b2[:-1])/Ac
def ub(x,alpha):
    X,Y=np.meshgrid(x,x,indexing='ij'); k=th*np.exp(1j*np.radians(alpha)); Z=X+1j*Y
    return c1[0]+(k*Z).real, c1[1]+(k*Z).imag
def profile(f,fbg,alpha,center,rs,cores,el=None,elsign=1.):
    D=np.load(f+'.npz'); B=np.load(fbg+'.npz'); x=D['x']; xc,yc=center
    s1,s2=frac(D['ux'],D['uy']); W=cell_windings(s1,s2); sb1,sb2=frac(B['ux'],B['uy']); Wb=cell_windings(sb1,sb2)
    xm=0.5*(x[1:]+x[:-1]); XC,YC=np.meshgrid(xm,xm,indexing='ij'); R_=np.hypot(XC-xc,YC-yc)
    exc=np.zeros_like(R_,bool)
    for (a,b) in cores: exc|=np.hypot(XC-a,YC-b)<3.0
    out=[]
    if el is not None:
        E=np.load(el+'.npz'); u0=ub(x,alpha); dux=elsign*(E['ux']-u0[0]); duy=elsign*(E['uy']-u0[1])
    for r in rs:
        row=dict(r=r,dQ=Qint(D['ux'],D['uy'],x,xc,yc,r)-Qint(B['ux'],B['uy'],x,xc,yc,r),
                 dQarea=Qarea(D['ux'],D['uy'],x,xc,yc,r)-Qarea(B['ux'],B['uy'],x,xc,yc,r),
                 dNplus=int((Wb[(R_<r)]>0).sum())-int((W[(R_<r)&~exc]>0).sum()),
                 q_in=int(W[(R_<r)&~exc].sum()))
        if el is not None:
            row['dQlin']=Qarea(u0[0]+dux,u0[1]+duy,x,xc,yc,r)-Qarea(u0[0],u0[1],x,xc,yc,r)
        out.append(row)
    return out

def nodes_of(f):
    D=np.load(f+'.npz'); x=D['x']; W=cell_windings(*frac(D['ux'],D['uy'])); xm=0.5*(x[1:]+x[:-1]); XC,YC=np.meshgrid(xm,xm,indexing='ij')
    return np.stack([XC[W>0],YC[W>0]],-1)
def slide(f,fbg,center,r0=22,r1=48):
    P=nodes_of(f); B=nodes_of(fbg); c=np.array(center)
    sel=P[(np.hypot(*(P-c).T)>r0)&(np.hypot(*(P-c).T)<r1)]
    dd=[]
    for p in sel:
        k=np.argmin(np.hypot(*(B-p).T)); 
        if np.hypot(*(B[k]-p))<12: dd.append(p-B[k])
    return np.mean(dd,0), len(dd)
def profile2(f,fbg,center,rs,cores):
    s,n=slide(f,fbg,center)
    D=np.load(f+'.npz'); B=np.load(fbg+'.npz'); x=D['x']; xc,yc=center; xb,yb=xc-s[0],yc-s[1]
    W=cell_windings(*frac(D['ux'],D['uy'])); Wb=cell_windings(*frac(B['ux'],B['uy']))
    xm=0.5*(x[1:]+x[:-1]); XC,YC=np.meshgrid(xm,xm,indexing='ij')
    exc=np.zeros_like(XC,bool)
    for (a,b) in cores: exc|=np.hypot(XC-a,YC-b)<3.0
    out=[]
    for r in rs:
        out.append(dict(r=r,dQ=Qint(D['ux'],D['uy'],x,xc,yc,r)-Qint(B['ux'],B['uy'],x,xb,yb,r),
             dQarea=Qarea(D['ux'],D['uy'],x,xc,yc,r)-Qarea(B['ux'],B['uy'],x,xb,yb,r),
             dNplus=int((Wb[np.hypot(XC-xb,YC-yb)<r]>0).sum())-int((W[(np.hypot(XC-xc,YC-yc)<r)&~exc]>0).sum())))
    return out,s,n

def window(r,r0=20.,r1=50.):
    return np.where(r<r0,1.,np.where(r>r1,0.,np.cos(0.5*np.pi*(r-r0)/(r1-r0))**2))
def smooth_charge(f,center,cores,r0=20.,r1=50.):
    from topo import analyse
    D=np.load(f+'.npz'); x=D['x']; W=cell_windings(*frac(D['ux'],D['uy'])); xm=0.5*(x[1:]+x[:-1]); XC,YC=np.meshgrid(xm,xm,indexing='ij')
    w=window(np.hypot(XC-center[0],YC-center[1]),r0,r1)
    if cores:
        o,_=analyse(f,cores); k=int(np.ceil(3.0/(x[1]-x[0]))); keep=np.ones_like(W,bool)
        for (a,b) in cores:
            ci,cj=np.argmin(abs(x-a)),np.argmin(abs(x-b)); keep[ci-k:ci+k,cj-k:cj+k]=False
        Qh=sum(p['Qhole'] for p in o['cores']); Np=float((w*(W>0)*keep).sum()); Nm=float((w*(W<0)*keep).sum())
        return Np-Nm+Qh, Np, Nm, Qh
    return float((w*W).sum()), float((w*(W>0)).sum()), 0., 0
