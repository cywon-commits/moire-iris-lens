import numpy as np, json
from scipy.ndimage import map_coordinates
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)])
Minv=np.linalg.inv(np.array([[1.,0.5],[0.,s3/2]]))      # columns a1,a2
def frac(ux,uy):
    s1=Minv[0,0]*(ux-c1[0])+Minv[0,1]*(uy-c1[1]); s2=Minv[1,0]*(ux-c1[0])+Minv[1,1]*(uy-c1[1]); return s1,s2
def wind_points(S):
    """S: (n,2) closed polygon (last!=first) in lattice coords; returns sum of winding numbers about all integer points"""
    S=np.vstack([S,S[:1]]); lo=np.floor(S.min(0)).astype(int); hi=np.ceil(S.max(0)).astype(int); tot=0
    P=np.stack(np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij'),-1).reshape(-1,2)
    if len(P)==0: return 0
    A=np.arctan2(S[None,:,1]-P[:,None,1],S[None,:,0]-P[:,None,0]); d=np.diff(A,axis=1); d=(d+np.pi)%(2*np.pi)-np.pi
    return int(np.round(d.sum(1)/(2*np.pi)).sum())
def cell_windings(s1,s2):
    """signed node count per plaquette (corners ij, i+1j, i+1j+1, ij+1)"""
    C1=np.stack([s1[:-1,:-1],s1[1:,:-1],s1[1:,1:],s1[:-1,1:]],-1); C2=np.stack([s2[:-1,:-1],s2[1:,:-1],s2[1:,1:],s2[:-1,1:]],-1)
    W=np.zeros(C1.shape[:2],int)
    ext=np.maximum(C1.max(-1)-C1.min(-1),C2.max(-1)-C2.min(-1))
    small=ext<0.5
    # fast path: single candidate = nearest integer to centroid
    p1=np.round(C1.mean(-1)); p2=np.round(C2.mean(-1))
    A=np.arctan2(C2-p2[...,None],C1-p1[...,None]); d=np.diff(np.concatenate([A,A[...,:1]],-1),axis=-1); d=(d+np.pi)%(2*np.pi)-np.pi
    W[small]=np.round(d.sum(-1)/(2*np.pi)).astype(int)[small]
    for i,j in zip(*np.where(~small)):
        W[i,j]=wind_points(np.stack([C1[i,j],C2[i,j]],-1))
    return W
def circle(D,xc,yc,r,n=4000):
    x=D['x']; h=x[1]-x[0]; t=np.linspace(0,2*np.pi,n,endpoint=False)
    X=xc+r*np.cos(t); Y=yc+r*np.sin(t); I=(X-x[0])/h; J=(Y-x[0])/h
    return map_coordinates(D['ux'],[I,J],order=1), map_coordinates(D['uy'],[I,J],order=1)
def labels_on(ux,uy):
    s1,s2=frac(ux,uy); f1=s1-np.floor(s1); f2=s2-np.floor(s2); return (f1+f2<1).astype(int)   # 1=up triangle (one stacking), 0=down
def zcount(D,xc,yc,r):
    ux,uy=circle(D,xc,yc,r); L=labels_on(ux,uy); return int(np.sum(L!=np.roll(L,1)))
def sq_loop(s1,s2,a0,a1,b0,b1):
    P=[(i,b0) for i in range(a0,a1+1)]+[(a1,j) for j in range(b0+1,b1+1)]+[(i,b1) for i in range(a1-1,a0-1,-1)]+[(a0,j) for j in range(b1-1,b0,-1)]
    P=np.array(P); return np.stack([s1[P[:,0],P[:,1]],s2[P[:,0],P[:,1]]],-1)
def analyse(f,cores,rq=3.0,box=70.):
    D=np.load(f+'.npz'); x=D['x']; h=x[1]-x[0]
    s1,s2=frac(D['ux'],D['uy']); W=cell_windings(s1,s2)
    b0,b1=np.searchsorted(x,-box),np.searchsorted(x,box)
    keep=np.zeros_like(W,bool); keep[b0:b1,b0:b1]=True
    k=int(np.ceil(rq/h)); blocks=[]
    for (xc,yc) in cores:
        ci,cj=np.argmin(abs(x-xc)),np.argmin(abs(x-yc)); blocks.append((ci-k,ci+k,cj-k,cj+k)); keep[ci-k:ci+k,cj-k:cj+k]=False
    out=dict(Nplus=int((W[keep]>0).sum()),Nminus=int((W[keep]<0).sum()),Nsigned=int(W[keep].sum()),Qbox=wind_points(sq_loop(s1,s2,b0,b1,b0,b1)))
    xm=0.5*(x[1:]+x[:-1]); XC,YC=np.meshgrid(xm,xm,indexing='ij')
    per=[]
    for (xc,yc),(a0,a1,c0,c1_) in zip(cores,blocks):
        Qh=wind_points(sq_loop(s1,s2,a0,a1,c0,c1_))
        reg=keep&(np.hypot(XC-xc,YC-yc)<14)
        for (x2,y2) in cores:
            if (x2,y2)!=(xc,yc): reg&=np.hypot(XC-xc,YC-yc)<=np.hypot(XC-x2,YC-y2)
        per.append(dict(core=(xc,yc),Qhole=Qh,z4=zcount(D,xc,yc,4.5),z7=zcount(D,xc,yc,7.),
                        Nminus_local=int((W[reg]<0).sum()),Nplus_local=int((W[reg]>0).sum())))
    out['cores']=per; out['gauss']=out['Nsigned']+sum(p['Qhole'] for p in per)-out['Qbox']
    return out,W
if __name__=='__main__':
    import sys
    f=sys.argv[1]; d=float(sys.argv[2]) if len(sys.argv)>2 else 0.; mode=sys.argv[3] if len(sys.argv)>3 else 'pair'
    cores={'pair':[(-d/2,0.),(d/2,0.)],'R':[(d/2,0.)],'none':[]}[mode]
    o,_=analyse(f,cores); print(json.dumps(o))
