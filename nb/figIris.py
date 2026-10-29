import os
import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from affine import domain_field
from iris_an import nodes, jac_sign, node_handedness
plt.rcParams.update({'font.size':10}); theta=1/12; W=26
fig=plt.figure(figsize=(20,10.2)); gs=fig.add_gridspec(2,4,hspace=0.25,wspace=0.22)
for r,C in enumerate([10.0,35.0]):
    D=np.load(f'iris_C{C}.npz'); x=D['x']; core=D['core']; rs=np.sqrt(C/theta)
    for c,(tag,ux,uy) in enumerate([('rigid',D['ux0'],D['uy0']),('relaxed',D['ux'],D['uy'])]):
        g=domain_field(ux,uy).astype(float); g[core]=np.nan
        a=fig.add_subplot(gs[r,c]); a.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1)
        a.add_patch(plt.Circle((0,0),rs,fill=False,ec='k',ls='--',lw=1.2))
        if tag=='relaxed':
            P=nodes(x,ux,uy,core); P=P[np.hypot(P[:,0],P[:,1])<W+4]; hd=node_handedness(x,ux,uy,P)
            a.scatter(P[hd>0,0],P[hd>0,1],s=36,facecolors='none',edgecolors='#1d9e75',linewidths=1.6)
            a.scatter(P[hd<0,0],P[hd<0,1],s=36,facecolors='none',edgecolors='#d4537e',linewidths=1.6)
        a.set_xlim(-W,W); a.set_ylim(-W,W); a.set_xticks([]); a.set_yticks([])
        a.set_title(f'({"ab"[r]}{c+1}) area deficit C={C:.0f}a², {"before relaxation" if tag=="rigid" else "relaxed"}'+('\ngreen: normal nodes, pink: mirrored nodes' if tag=='relaxed' else '\ndashed: r* = √(C/θ)'),fontsize=9.5)
    J=jac_sign(x,D['ux'],D['uy'],core,2.0); a=fig.add_subplot(gs[r,2])
    a.imshow(np.sign(J).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='PiYG',vmin=-1.5,vmax=1.5)
    a.add_patch(plt.Circle((0,0),rs,fill=False,ec='k',ls='--',lw=1.2)); a.set_xlim(-W,W); a.set_ylim(-W,W); a.set_xticks([]); a.set_yticks([])
    a.set_title(f'({"ab"[r]}3) sign of det ∇u (relaxed, smoothed 2a)\ngreen: same twist sense, pink: reversed',fontsize=9.5)
a=fig.add_subplot(gs[0,3]); rr=np.linspace(2,30,400)
for C,cl in [(4,'#85B7EB'),(10,'#2a6fb0'),(20,'#6a3d9a'),(35,'#c0392b')]:
    a.plot(rr,1-(C/(theta*rr**2))**2,color=cl,lw=2,label=f'C={C}a²'); a.axvline(np.sqrt(C/theta),color=cl,ls=':',lw=1)
a.axhline(0,color='k',lw=0.8); a.set_ylim(-3,1.2); a.set_xlabel('r (a)'); a.set_ylabel('det ∇u / θ²  = 1 − (C/θr²)²')
a.set_title('(c) twist + iris field: orientation flips inside r*',fontsize=10); a.legend(frameon=False,fontsize=8.5)
a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
a=fig.add_subplot(gs[1,3])
Cs=[4,10,20,35]; rig=[6,18,38,68]; rel=[6,16,30,58]; nor_rig=[18,12,8,12]; nor_rel=[14,10,6,10]
a.plot(Cs,rig,'s--',color='#d4537e',ms=6,label='mirrored, before relaxation'); a.plot(Cs,rel,'o-',color='#d4537e',lw=2,ms=6,label='mirrored, relaxed')
a.plot(Cs,nor_rel,'o-',color='#1d9e75',lw=2,ms=6,label='normal, relaxed (r<30a)')
a.set_xlabel('area deficit C (a²)'); a.set_ylabel('number of nodes'); a.set_title('(d) mirrored nodes collected around the closure point',fontsize=10)
a.legend(frameon=False,fontsize=8.5); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.text(0.5,0.05,'Twisted moiré (period 12a, honeycomb V) with a closed hole: the missing area pulls the top layer inward, u_r = −C/r.  White disc: closed core (no registry).  Node counts near the core are lower bounds (nodes closer than the grid resolution are missed).',ha='center',fontsize=9,color='#444')
fig.savefig('iris_closure.png',dpi=100,bbox_inches='tight')
