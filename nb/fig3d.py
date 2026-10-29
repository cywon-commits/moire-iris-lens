import os
import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm, colors
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from affine import domain_field
plt.rcParams.update({'font.size':10})
cases=[('bub_SP_2.0.npz',4.0,2.0,'(a) on a wall, H=2a: 10 walls'),
       ('bubsz_AA_11.0_2.5.npz',11.0,2.5,'(b) on AA, wide and shallow: 18 walls'),
       ('bub_SP_6.0.npz',4.0,6.0,'(c) on a wall, steep H=6a: 10 walls, spiral')]
fig=plt.figure(figsize=(18,11)); W=16; cmap=plt.get_cmap('coolwarm')
for k,(f,rho,H0,t) in enumerate(cases):
    D=np.load(f); x=D['x']; h=D['h']; xc,yc=float(D['xc']),float(D['yc'])
    g=domain_field(D['ux'],D['uy']); lifted=h>0.42
    ix=np.where(np.abs(x-xc)<=W)[0]; iy=np.where(np.abs(x-yc)<=W)[0]; st=1
    ix=ix[::st]; iy=iy[::st]
    X,Y=np.meshgrid(x[ix],x[iy],indexing='ij'); Z=h[np.ix_(ix,iy)]
    C=cmap(0.5*(np.tanh(4*g[np.ix_(ix,iy)])+1)); L=lifted[np.ix_(ix,iy)]
    C[L]=np.array([0.86,0.86,0.84,1.0])
    ax=fig.add_subplot(2,3,k+1,projection='3d')
    ax.plot_surface(X-xc,Y-yc,Z,facecolors=C,rstride=1,cstride=1,linewidth=0,antialiased=False,shade=False)
    ax.set_zticks(np.arange(0,H0+0.01,1 if H0<4 else 2))
    ax.set_box_aspect((1,1,0.45 if H0>3 else 0.3)); ax.view_init(elev=32,azim=-58)
    ax.set_xlabel('x (a)'); ax.set_ylabel('y (a)'); ax.set_zlabel('h (a)'); ax.set_title(t,fontsize=11)
    # cross-section along y through centre (x = xc)
    i0=np.argmin(abs(x-xc)); jj=np.where(np.abs(x-yc)<=W)[0]; s=x[jj]-yc
    prof=h[i0,jj]; fob=H0*np.maximum(1-s**2/rho**2,0)
    a=fig.add_subplot(2,3,k+4)
    a.fill_between(s,0,fob,color='#BA7517',alpha=0.35,lw=0,label='trapped particle')
    a.plot(s,prof,color='#2a6fb0',lw=2.2,label='top layer h')
    a.axhline(0.42,color='#888',ls='--',lw=1,label='interlayer range (lifted above)')
    a.axhline(0,color='k',lw=1.2); a.text(s[0]+0.5,-0.25,'bottom layer',fontsize=8.5,color='#444')
    lm=s[(prof>0.42)]; 
    if len(lm): a.axvspan(lm.min(),lm.max(),color='#ddd',alpha=0.5,lw=0)
    a.set_xlim(-W,W); a.set_ylim(-0.5,max(H0,prof.max())*1.25); a.set_aspect('auto')
    a.set_xlabel('distance from bubble centre along the cut (a)'); a.set_ylabel('h (a)')
    a.set_title('cross-section'+(' through the two captured nodes' if 'SP' in f else ''),fontsize=10)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    if k==0: a.legend(frameon=False,fontsize=8.5,loc='upper right')
fig.text(0.5,0.505,'surface colour: red / blue = AB / BA stacking, light lines = walls, grey = lifted (no registry)',ha='center',fontsize=9.5,color='#444')
fig.savefig('bubble_3d.png',dpi=105,bbox_inches='tight')
