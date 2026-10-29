import os
import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from affine import domain_field
from hexcore import Vhex
from scipy.ndimage import distance_transform_edt
s3=np.sqrt(3); plt.rcParams.update({'font.size':10})
def load(f):
    D=np.load(f); x=D['x']; h=D['h']; ux,uy=D['ux'],D['uy']; d1=float(D['d1'])
    dev=h-d1*Vhex(ux,uy,1.,s3)[0]/9; g=domain_field(ux,uy)
    return D,x,h,dev,g,float(D['xc']),float(D['yc'])
maps=[('wrk_N_0.85.npz','(a) no bubble'),('wrk_B_0.85.npz','(b) bubble, stiff adhesion\n(well width 0.85a)'),
      ('wrk_B_1.20.npz','(c) bubble, softer adhesion\n(well width 1.2a)'),('wrk_SPsd.npz','(d) bubble, soft adhesion\n(well width 2a)')]
fig=plt.figure(figsize=(20,10.4)); gs=fig.add_gridspec(2,4,hspace=0.28,wspace=0.22)
for k,(f,t) in enumerate(maps):
    D,x,h,dev,g,xc,yc=load(f); a=fig.add_subplot(gs[0,k]); v=dev.copy(); v[h>0.9]=np.nan
    im=a.imshow(v.T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='RdBu_r',vmin=-0.4,vmax=0.4)
    a.contour(x,x,g.T,levels=[0],colors='k',linewidths=0.45)
    a.set_xlim(xc-16,xc+16); a.set_ylim(yc-16,yc+16); a.set_xticks([]); a.set_yticks([]); a.set_title(t,fontsize=10)
cb=fig.colorbar(im,ax=fig.axes[:4],fraction=0.012,pad=0.01); cb.set_label('h − d(u)  (a)')
for k,(f,t) in enumerate([('wrk_B_1.20.npz','(e) 3D, softer adhesion (z ×3)'),('wrk_SPsd.npz','(f) 3D, soft adhesion (z ×3)')]):
    D,x,h,dev,g,xc,yc=load(f); ax=fig.add_subplot(gs[1,k],projection='3d')
    ix=np.where(np.abs(x-xc)<=12)[0]; iy=np.where(np.abs(x-yc)<=12)[0]
    X,Y=np.meshgrid(x[ix]-xc,x[iy]-yc,indexing='ij'); Z=np.minimum(h[np.ix_(ix,iy)],1.6)
    cm=plt.get_cmap('coolwarm'); C=cm(0.5*(np.tanh(4*g[np.ix_(ix,iy)])+1)); C[h[np.ix_(ix,iy)]>0.9]=[0.86,0.86,0.84,1]
    ax.plot_surface(X,Y,Z,facecolors=C,rstride=1,cstride=1,linewidth=0,antialiased=False,shade=False)
    ax.set_box_aspect((1,1,3*1.6/24)); ax.view_init(elev=38,azim=-60); ax.set_zticks([0,1]); ax.set_title(t+'\nbubble top cut at 1.6a',fontsize=10)
a=fig.add_subplot(gs[1,2:])
for f,lab,c in [('wrk_B_0.85.npz','well 0.85a, with gap modulation d(u)','#2a6fb0'),('wrk_B0_0.85.npz','well 0.85a, no gap modulation (→0 beyond 10a)','#85B7EB'),
                ('wrk_B_1.20.npz','well 1.2a','#6a3d9a'),('wrk_SPsd.npz','well 2a','#c0392b'),('wrk_N_0.85.npz','no bubble (well 0.85a)','#888')]:
    D,x,h,dev,g,xc,yc=load(f); dist=distance_transform_edt(~(np.abs(g)<0.08))*(x[1]-x[0])
    X,Y=np.meshgrid(x-xc,x-yc,indexing='ij'); R=np.hypot(X,Y); rr=np.arange(4.5,24,1.0); on=[];off=[]
    for r0 in rr:
        m=(R>r0)&(R<r0+1)&(h<0.9); on.append(np.sqrt(np.mean(dev[m&(dist<1)]**2)) if (m&(dist<1)).any() else np.nan)
    a.plot(rr+0.5,on,'o-',color=c,lw=2,ms=4,label=lab)
a.set_yscale('log'); a.set_ylim(2e-3,1.5); a.set_xlabel('distance from bubble centre (a)   [bubble rim ≈ 5a]'); a.set_ylabel('rms of h − d(u) on walls (a)')
a.set_title('(g) wall ridges: localized at the rim for stiff adhesion, spreading out as adhesion softens',fontsize=10)
a.legend(frameon=False,fontsize=8.5,loc='upper right'); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.savefig('bubble_wrinkles.png',dpi=100,bbox_inches='tight')
