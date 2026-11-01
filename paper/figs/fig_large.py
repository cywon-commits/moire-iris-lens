import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Circle
from lat import *; from defects import radial
plt.rcParams.update({'font.size':7.5,'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif'})
th=1/30.; Lw=150; C=100.; rh=16.0
R=np.sqrt(C/th); LM=1/th
def moire(alpha):
    L1=honeycomb(Lw+6)
    if alpha==90: L1=rot(L1,-th/2)
    X=honeycomb(Lw+6); z=X[:,0]+1j*X[:,1]; k=np.abs(z)>rh; zn=radial(z[k],C); Y=np.c_[zn.real,zn.imag]
    Y=rot(Y,th/2) if alpha==90 else Y*(1+th)
    d,_=cKDTree(L1).query(Y); f=np.exp(-(d/0.18)**2)
    nb=520; H,_,_=np.histogram2d(Y[:,0],Y[:,1],bins=nb,range=[[-Lw,Lw],[-Lw,Lw]],weights=f); N,_,_=np.histogram2d(Y[:,0],Y[:,1],bins=nb,range=[[-Lw,Lw],[-Lw,Lw]])
    s=1.2; M=gaussian_filter(H,s)/np.maximum(gaussian_filter(N,s),1e-9); M[gaussian_filter(N,s)<0.3]=np.nan; return M
fig,axs=plt.subplots(1,2,figsize=(7.0,3.6),gridspec_kw=dict(left=0.01,right=0.99,top=0.9,bottom=0.12,wspace=0.04))
for ax,a,t,phm in ((axs[0],90,'(a) twist, area removed ($\\varphi_m=-90^\\circ$)',-90),(axs[1],0,'(b) mismatch, area removed ($\\varphi_m=0^\\circ$)',0)):
    M=moire(a); ax.imshow(M.T,origin='lower',extent=[-Lw,Lw,-Lw,Lw],cmap='magma',vmin=0.12,vmax=0.8,interpolation='bilinear')
    ax.add_patch(Circle((0,0),R,fill=False,ec='#7fdbff',lw=0.9,ls='--'))
    bs=2*R*abs(np.sin(np.radians(phm)/2))
    ax.set_xlim(-Lw,Lw); ax.set_ylim(-Lw,Lw); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,loc='left',fontsize=8,pad=3)
    ax.text(0.5,-0.02,f'dashed: Einstein radius $R={R:.0f}$ ($={R/LM:.1f}\\,L_M$); removed nodes $T={2*np.pi*th*C/(np.sqrt(3)/2):.0f}$',transform=ax.transAxes,ha='center',va='top',fontsize=6.8)
fig.savefig('large_defect.png',dpi=200); fig.savefig('large_defect.pdf')
print(R,R/LM)
