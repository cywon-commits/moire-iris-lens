import numpy as np, json
from scipy.ndimage import maximum_filter, map_coordinates
from affine import domain_field
from hexcore import Vhex, nodetype
def wrap(a): return (a+180)%360-180
def track(x,y,ux,uy,chords,Lm,rmin=1.0):
    dx=x[1]-x[0]; dy=y[1]-y[0]; V=Vhex(ux,uy,1.,np.sqrt(3.))[0]; f=domain_field(ux,uy)
    pk=(V==maximum_filter(V,size=11,mode='wrap'))&(V>8.0)
    I,J=np.nonzero(pk); radii=np.arange(0.42*Lm,rmin-1e-9,-0.25); profs=[]
    for i,j in zip(I,J):
        # subpixel centre: weighted centroid of (V-7.5)+ in window
        ii=(np.arange(i-4,i+5))%len(x); jj=(np.arange(j-4,j+5))%len(y); W=np.clip(V[np.ix_(ii,jj)]-7.5,0,None)
        if W.sum()==0: continue
        cx=x[i]+dx*((W.sum(1)*np.arange(-4,5)).sum()/W.sum()); cy=y[j]+dy*((W.sum(0)*np.arange(-4,5)).sum()/W.sum())
        prev=np.array(chords,float); prof=[]; ok=True
        for r in radii:
            th=np.linspace(0,2*np.pi,2881)[:-1]
            v=map_coordinates(f,[((cx+r*np.cos(th))/dx)%len(x),((cy+r*np.sin(th))/dy)%len(y)],order=1,mode='grid-wrap')
            s=np.sign(v); idx=np.nonzero(s!=np.roll(s,-1))[0]
            if len(idx)!=6: ok=False; break
            ang=[]
            for k in idx:
                k2=(k+1)%len(th); a1=th[k2] if k2>0 else 2*np.pi; ang.append(np.degrees(th[k]+(a1-th[k])*v[k]/(v[k]-v[k2])))
            ang=np.array(ang)%360
            new=np.array([p+wrap(ang[np.argmin(abs(wrap(ang-p)))]-p) for p in prev])
            prof.append(new-np.array(chords)); prev=new
        if ok: profs.append(np.array(prof))
    return radii,profs
