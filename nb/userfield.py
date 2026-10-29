import numpy as np
s3=np.sqrt(3.)
A1=np.array([2*np.pi,2*np.pi/s3]); A2=np.array([0,4*np.pi/s3])     # their real lattice (G.A=2pi)
G=[np.array([1.,0]),np.array([-.5,s3/2]),np.array([-.5,-s3/2])]
def w_field(X,Y,x0,y0,k,Ds,Cf):
    """user's construction: phi_j = G_j . w ;  w = r0 + mutifactor*(r-r0)"""
    dist=np.sqrt((X-x0)**2+((Y-y0)/k)**2)+1e-10
    m=(Ds/dist)*np.maximum(1-dist/Cf,0)+np.minimum(dist/Cf,1)
    return x0+m*(X-x0), y0+m*(Y-y0)
CASES={'cell30':(0,0,1,4*np.pi,6*np.pi),
       'cell31':(0,2*s3,2,1.5*np.pi,3*np.pi),
       'cell32':(0,4*s3,3,1.5*np.pi,2.25*np.pi),
       'cell33':(0,6*s3,4,1.5*np.pi,2.25*np.pi)}
def rays(case,rad=0.05):
    x0,y0,k,Ds,Cf=CASES[case]; th=np.linspace(0,2*np.pi,20001)[:-1]
    X=x0+rad*np.cos(th); Y=y0+rad*k*np.sin(th); wx,wy=w_field(X,Y,x0,y0,k,Ds,Cf)
    n=0
    for g in G:
        ph=(g[0]*wx+g[1]*wy)/(2*np.pi); n+=int(np.sum(np.floor(ph)!=np.floor(np.roll(ph,-1))))
    return n
def enclosed(case,rad=0.05):
    x0,y0,k,Ds,Cf=CASES[case]; th=np.linspace(0,2*np.pi,4001)[:-1]
    wx,wy=w_field(x0+rad*np.cos(th),y0+rad*k*np.sin(th),x0,y0,k,Ds,Cf)
    from matplotlib.path import Path
    P=Path(np.c_[wx,wy]); pts=[n1*A1+n2*A2 for n1 in range(-6,7) for n2 in range(-6,7)]
    inside=[tuple(np.round(p,2)) for p in pts if P.contains_point(p)]
    return inside
if __name__=="__main__":
    for c in CASES:
        e=enclosed(c); print(f'{c}: walls at defect = {rays(c)}   AA nodes swallowed Q = {len(e)}  {e}')
