import os
import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','caus'))
from affine import domain_field
from iris_an import nodes, node_handedness
from quartic import images
plt.rcParams.update({'font.size':10}); theta=1/40; W=44
cases=[('sh_a0_C16.0_s0.0.npz','(a) no shear: Einstein ring'),('sh_a0_C16.0_s0.5.npz','(b) shear |a|=0.5: Einstein cross'),('sh_a0_C4.0_s0.5.npz','(c) smaller deficit, same shear')]
fig,ax=plt.subplots(2,3,figsize=(17,11.6))
for c,(f,t) in enumerate(cases):
    D=np.load(f); x=D['x']; core=D['core']; C=float(D['C']); a=float(D['shear']); R=np.sqrt(C/theta)
    for r,(tag,ux,uy) in enumerate([('rigid',D['ux0'],D['uy0']),('relaxed',D['ux'],D['uy'])]):
        g=domain_field(ux,uy).astype(float); g[core]=np.nan; A=ax[r,c]
        A.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1)
        A.add_patch(plt.Circle((0,0),R,fill=False,ec='k',ls='--',lw=1))
        if a>0:
            for z,p in images(0j,a,C/theta+0j):
                A.plot(z.real,z.imag,marker='x',color='k',ms=13,mew=2.2)
        P=nodes(x,ux,uy,core); rr=np.hypot(P[:,0],P[:,1]); P=P[(rr<W+3)]
        if len(P):
            hd=node_handedness(x,ux,uy,P)
            A.scatter(P[hd>0,0],P[hd>0,1],s=90,facecolors='none',edgecolors='#1d9e75',linewidths=2)
            A.scatter(P[hd<0,0],P[hd<0,1],s=50,facecolors='none',edgecolors='#d4537e',linewidths=1.6)
        A.set_xlim(-W,W); A.set_ylim(-W,W); A.set_xticks([]); A.set_yticks([])
        A.set_title((t+f'\nC={C:.0f}a², R={R:.0f}a\n' if r==0 else '')+('before relaxation' if r==0 else 'relaxed'),fontsize=10.5)
fig.text(0.5,0.06,'Converging lens (mismatch moiré α=0°, vacancy), deficit placed exactly on an AA node (source at β=0).  ×: predicted images of that node (Chang–Refsdal lens);  dashed: Einstein radius.\n'
         'Green ○ normal nodes, pink ○ mirrored nodes.  (a) without shear the whole circle |z|=R is AA-stacked (a ring) — non-generic, and relaxation erases it.  (b) shear splits it into a cross of 2 normal + 2 mirrored nodes,\n'
         'which survives relaxation when the images are far apart.  (c) with a small deficit the cross images are close (10–18a) and relaxation lets the normal/mirrored pairs annihilate (net charge 0).',ha='center',fontsize=9.3,color='#444')
fig.savefig('einstein_cross.png',dpi=95,bbox_inches='tight')
