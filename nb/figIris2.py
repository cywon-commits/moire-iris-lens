import os
import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from affine import domain_field
from iris_an import nodes, node_handedness
plt.rcParams.update({'font.size':10}); theta=1/40; W=28
Cs=[(1.2,'well below'),(2.5,'just below'),(4.4,'just above: first shell'),(8.0,'two shells'),(15.0,'three shells')]
fig,ax=plt.subplots(2,5,figsize=(23,10))
for c,(C,lab) in enumerate(Cs):
    D=np.load(f'iris2_C{C}.npz'); x=D['x']; core=D['core']; rs=np.sqrt(C/theta); rc=np.sqrt(core.sum())*(x[1]-x[0])/np.sqrt(np.pi)
    for r,(tag,ux,uy) in enumerate([('rigid',D['ux0'],D['uy0']),('relaxed',D['ux'],D['uy'])]):
        g=domain_field(ux,uy).astype(float); g[core]=np.nan; a=ax[r,c]
        a.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1)
        a.add_patch(plt.Circle((0,0),rs,fill=False,ec='k',ls='--',lw=1.1))
        P=nodes(x,ux,uy,core); P=P[np.hypot(P[:,0],P[:,1])<W+6]
        if len(P):
            hd=node_handedness(x,ux,uy,P)
            a.scatter(P[hd>0,0],P[hd>0,1],s=60,facecolors='none',edgecolors='#1d9e75',linewidths=1.8)
            a.scatter(P[hd<0,0],P[hd<0,1],s=60,facecolors='none',edgecolors='#d4537e',linewidths=1.8)
            nm=int((hd<0).sum())
        else: nm=0
        a.set_xlim(-W,W); a.set_ylim(-W,W); a.set_xticks([]); a.set_yticks([])
        a.set_title((f'C = {C}a²  ({lab})\n' if r==0 else '')+f'{"before relaxation" if r==0 else "relaxed"}: mirrored nodes = {nm}',fontsize=10)
fig.text(0.5,0.035,'Moiré period 40a, wall width ≈ 4–6a, closed core radius 2.2a (C ≤ 4.4) or √(2C) (C = 8, 15).  Dashed: inversion radius r* = √(C/θ).  Green: normal nodes,  pink: mirrored (reversed-twist) nodes.\n'
         'Mirrored shells sit at r ≈ C/ρ with ρ = a, √3a, 2a … (distance in u-space from the closed AA); a shell survives only if it lies outside both the core and the reconstruction limit √(C·w).',ha='center',fontsize=9.5,color='#444')
fig.savefig('iris_threshold.png',dpi=95,bbox_inches='tight')
