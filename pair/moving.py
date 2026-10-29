import os
import numpy as np, sys, json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','nb'))
from iris_an import nodes, node_handedness
from binlens import images_single
th=1/30.; C=4.; mass=C/(1j*th); R=np.sqrt(C/th); s3=np.sqrt(3)
S=[]
for n1 in range(-7,8):
    for n2 in range(-7,8):
        v=n1*np.array([1.,0])+n2*np.array([.5,s3/2]); b=(v[0]+1j*v[1])/(1j*th)
        if abs(b)<90: S.append(b)
S=np.array(S); win=50.0
def count_rigid(xl):
    n=m=0
    for b in S:
        for z,p in images_single(b,mass,xl):
            if abs(z)<win: n+=p>0; m+=p<0
    return n,m
xs=np.arange(0,35.01,0.1); rig=np.array([count_rigid(x) for x in xs])
rel=[]
for d in [0,8,12,16,20,26,32,40,46,50,56,66]:
    D=np.load(f'R_d{float(d)}_V0.005.npz'); x=D['x']; P=nodes(x,D['ux'],D['uy'],D['core']); P=P[np.hypot(P[:,0],P[:,1])<win]
    hd=node_handedness(x,D['ux'],D['uy'],P); rel.append((d/2,int((hd>0).sum()),int((hd<0).sum())))
    print(f'lens at x={d/2:5.1f}: relaxed normal {int((hd>0).sum())}, mirrored {int((hd<0).sum())}   | rigid normal {count_rigid(d/2)[0]}, mirrored {count_rigid(d/2)[1]}')
# events: sources crossing the swallow circle |beta - x_lens| = sqrt2 R
ev=[]
for b in S:
    if abs(b)>win: continue
    # solve |b - x|^2 = 2 R^2 for x in [0,30]
    disc=2*R**2-b.imag**2
    if disc>=0:
        for xe in (b.real-np.sqrt(disc),b.real+np.sqrt(disc)):
            if 0<=xe<=35: ev.append((float(xe),'enter' if xe<b.real else 'leave',[float(b.real),float(b.imag)]))
print('swallow events (lens x, type, node original position):',sorted([(round(e[0],1),e[1],[round(v,1) for v in e[2]]) for e in ev]))
json.dump(dict(xs=xs.tolist(),rig=rig.tolist(),rel=rel,events=ev,R=R),open('moving.json','w'))
