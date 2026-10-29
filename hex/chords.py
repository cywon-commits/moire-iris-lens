import numpy as np, json
from hexcore import s3
c1=np.array([0.5,1/(2*s3)])
def chord_gaps(A):
    A=np.array(A,float); Ai=np.linalg.inv(A)
    # node lattice: r = -Ai (c1 + R); neighbours: differences -Ai R for the 6 shortest moire vectors
    vs=[]
    for n1 in range(-3,4):
        for n2 in range(-3,4):
            R=np.array([n1+0.5*n2,n2*s3/2]); 
            if n1==0 and n2==0: continue
            vs.append(-Ai@R)
    vs=sorted(vs,key=lambda v: np.hypot(*v))
    # take 6 shortest that form the Delaunay star: pick shortest then ensure angular coverage
    cand=vs[:12]; L=[np.hypot(*v) for v in cand]
    ang=sorted(np.degrees(np.arctan2(v[1],v[0]))%360 for v in cand[:6])
    gaps=np.diff(np.concatenate([ang,[ang[0]+360]]))
    return np.round(ang,1).tolist(),np.round(gaps,1).tolist(),np.round(L[:6],2).tolist()
if __name__=="__main__":
    from angles import cases
    for k,A in cases.items():
        a,g,l=chord_gaps(A); print(f'{k:8s} chord dirs {a}  gaps {g}  lengths {l}')
