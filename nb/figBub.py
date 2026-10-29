import os
import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); sys.path.insert(0,'.')
from affine import domain_field
from bscan import measure
plt.rcParams.update({'font.size':10})
rows=[[('bub_SP_1.0.npz','(a) on a wall, H=1a, footprint 2.9a'),('bub_SP_2.0.npz','(b) on a wall, H=2a, footprint 3.4a'),
       ('bubsz_SP_6.0_1.5.npz','(c) on a wall, shallow H=1.5a, footprint 5a'),('bub_SP_6.0.npz','(d) on a wall, steep H=6a')],
      [('bub_AA_4.0.npz','(e) on AA, H=4a, footprint 4a'),('bubsz_AA_9.0_2.5.npz','(f) on AA, shallow H=2.5a, footprint 8.1a'),
       ('bubsz_AA_11.0_2.5.npz','(g) on AA, shallow H=2.5a, footprint 9.9a'),('bub_AA_8.0.npz','(h) on AA, steep H=8a')]]
fig,ax=plt.subplots(2,4,figsize=(18,9.4))
for r,row in enumerate(rows):
    for c,(f,t) in enumerate(row):
        D=dict(np.load(f)); D['xc']=float(D['xc']); D['yc']=float(D['yc']); x=D['x']; h=D['h']
        M=measure(D); rl,w,Q=M['loops'][0][:3]
        g=domain_field(D['ux'],D['uy']).astype(float); g[h>0.42]=np.nan
        a=ax[r,c]; a.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1)
        lev=[l for l in (1,2,4,6) if l<h.max()]
        if lev: a.contour(x,x,h.T,levels=lev,colors='k',linewidths=0.6)
        a.set_xlim(-20,20); a.set_ylim(D['yc']-20,D['yc']+20); a.set_xticks([]); a.set_yticks([])
        a.set_title(t+f'\nwalls ending on bubble rim: {w},  nodes inside Q={Q}',fontsize=9.5)
fig.text(0.5,0.06,'Top layer is continuous (no tear): Föppl–von Kármán sheet with adhesion; a trapped particle lifts it.  Blank: delaminated (h > interlayer range);  black lines: height contours 1, 2, 4, 6a.  Moiré period 12a.',ha='center',fontsize=9.5,color='#444')
fig.savefig('bubble_nodes.png',dpi=100,bbox_inches='tight')
