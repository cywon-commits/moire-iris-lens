import os
import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from affine import domain_field
from anal import loopdata
plt.rcParams.update({'font.size':10})
E={'nb_cell31u_h1.5.npz':366.31,'nb_cell31_h1.5.npz':366.64,'nb_cell31u_h2.5.npz':364.67,
   'nb_cell30_h3.5.npz':368.46,'nb_cell30u_h3.5.npz':364.65,'nb_cell30u_h5.0.npz':363.59}
rows=[[('nb_cell31_h0.0.npz','init','(a) notebook cell 31 as built:\n10 walls at one point (Q=2)'),
       ('nb_cell31_h0.0.npz','fin','(b) relaxed, no pore:\nheals to the plain moiré'),
       ('nb_cell31_h1.5.npz','fin','(c) pore 1.5a×3a, notebook start:\n10 walls held (metastable)'),
       ('nb_cell31u_h1.5.npz','fin','(d) same pore, plain-moiré start:\npore sits on one wall (lower E)'),
       ('nb_cell31u_h2.5.npz','fin','(e) pore 2.5a×5a, plain-moiré start:\nboth nodes absorbed → 10 walls')],
      [('nb_cell30_h0.0.npz','init','(f) notebook cell 30 as built:\n18 walls at one point (Q=7)'),
       ('nb_cell30_h0.0.npz','fin','(g) relaxed, no pore:\nheals to the plain moiré'),
       ('nb_cell30_h3.5.npz','fin','(h) round pore r=3.5a, notebook start:\n18 walls held (metastable)'),
       ('nb_cell30u_h3.5.npz','fin','(i) same pore, plain-moiré start:\nonly 1 node absorbed (lower E)'),
       ('nb_cell30u_h5.0.npz','fin','(j) round pore r=5a, plain-moiré start:\nstill 1 node (neighbours 12a away)')]]
fig,ax=plt.subplots(2,5,figsize=(22,10))
for r,pan in enumerate(rows):
    for c,(f,which,t) in enumerate(pan):
        D=np.load(f); x=D['x']; Y0=float(D['Y0'])
        ux,uy=(D['ux0'],D['uy0']) if which=='init' else (D['ux'],D['uy'])
        g=domain_field(ux,uy).astype(float)
        if which=='fin': g[D['pore']]=np.nan
        a=ax[r,c]; a.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1)
        a.set_xlim(-24,24); a.set_ylim(Y0-24,Y0+24); a.set_xticks([]); a.set_yticks([])
        hole=float(f.split('_h')[1][:-4])
        if which=='fin' and hole>0:
            w,Q=loopdata(f,rad=hole+0.4)[1:]; t+=f'\nwalls ending on pore rim: {w}   E={E[f]:.2f}'
        a.set_title(t,fontsize=9.5)
fig.text(0.5,0.06,'red / blue: AB and BA stacking;  white lines: walls;  blank: pore (no top-layer atoms).  Moiré period 12a, λ=μ.  Energies are comparable only for the same pore.',ha='center',fontsize=9.5,color='#444')
fig.savefig('notebook_relaxed.png',dpi=100,bbox_inches='tight')
