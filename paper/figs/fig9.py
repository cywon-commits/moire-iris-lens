import os,sys,numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
sys.path.insert(0,'/home/claude/ridge/hex'); from hexcore import Vhex
from lat import *; from defects import radial
plt.rcParams.update({'font.size':7.5,'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif','xtick.direction':'in','ytick.direction':'in','axes.linewidth':0.6})
s3=np.sqrt(3); th=1/30.; Lm=150; LIM=140
def registry_map(Y,L1):
    d,_=cKDTree(L1).query(Y); f=np.exp(-(d/0.18)**2)
    nb=420; H,_,_=np.histogram2d(Y[:,0],Y[:,1],bins=nb,range=[[-Lm,Lm],[-Lm,Lm]],weights=f); N,_,_=np.histogram2d(Y[:,0],Y[:,1],bins=nb,range=[[-Lm,Lm],[-Lm,Lm]])
    M=gaussian_filter(H,1.4)/np.maximum(gaussian_filter(N,1.4),1e-9); M[gaussian_filter(N,1.4)<0.5]=np.nan; return M
def vac_layer(C,rh):
    X=honeycomb(Lm+3); z=X[:,0]+1j*X[:,1]; k=np.abs(z)>rh; zn=radial(z[k],C); return np.c_[zn.real,zn.imag]
def grain_layer(rg,phi):
    X=honeycomb(Lm+3); z=X[:,0]+1j*X[:,1]; r=np.abs(z); ang=np.where(r<rg,phi,phi*rg**2/np.maximum(r,1e-9)**2); zn=z*np.exp(1j*ang); return np.c_[zn.real,zn.imag]
def images(m,kap):
    n=np.arange(-14,15); N1,N2=np.meshgrid(n,n,indexing='ij'); Lv=((N1+0.5*N2)+1j*(s3/2*N2)).ravel(); P=[];Q=[]
    for b in Lv/kap:
        R2=abs(m); ph=np.angle(m); sg=abs(b)**2+2*R2*np.cos(ph); dsc=sg*sg-4*R2**2
        if dsc<0 or sg<0: continue
        for rr,lst in ((np.sqrt((sg+np.sqrt(dsc))/2),P),(np.sqrt((sg-np.sqrt(dsc))/2),Q)):
            e=b/(rr-R2*np.exp(1j*ph)/rr); lst.append(rr*e/abs(e))
    P=np.array(P); Q=np.array(Q); return P[(abs(P.real)<LIM-3)&(abs(P.imag)<LIM-3)],Q
fig=plt.figure(figsize=(7.0,5.1))
ga=fig.add_gridspec(1,3,left=0.01,right=0.99,top=0.955,bottom=0.555,wspace=0.05)
gb=fig.add_gridspec(1,3,left=0.075,right=0.99,top=0.425,bottom=0.075,wspace=0.12,width_ratios=[1.25,1,1])
C=120.; R=np.sqrt(C/th); rh=np.sqrt(2*C/0.7)
L1t=rot(honeycomb(Lm+3),-th/2); L1m=honeycomb(Lm+3)
specs=[('(a) twist, vacancy cluster',lambda: rot(vac_layer(C,rh),th/2),L1t,C/(1j*th*0+1j*th),True,
        r'$R=2L_M$, $\varphi_m=-90^\circ$: arcs, swallow $b_s=\sqrt{2}R$'),
       ('(b) mismatch, vacancy cluster',lambda: vac_layer(C,rh)*(1+th),L1m,C/(th/(1+th)),True,
        r'$R=2L_M$, $\varphi_m=0$: Einstein ring, $b_s=0$'),
       ('(c) twist, counter-rotated grain',lambda: rot(grain_layer(40.,-2*th),th/2),L1t,None,False,
        r'grain $r_g=40>L_M$, $\phi=-2\theta$: ring at $R=\sqrt{2}\,r_g$')]
for j,(t,mk,L1,m,show,cap) in enumerate(specs):
    ax=fig.add_subplot(ga[j]); M=registry_map(mk(),L1)
    ax.imshow(M.T,origin='lower',extent=[-Lm,Lm,-Lm,Lm],cmap='magma',vmin=0.15,vmax=0.75,interpolation='bilinear')
    tt=np.linspace(0,2*np.pi,300)
    if show:
        kap=1j*th if j==0 else th/(1+th); P,Q=images(m,kap)
        ax.plot(P.real,P.imag,'o',ms=3.4,mfc='none',mec='#7fdbff',mew=0.6)
        q=Q[np.abs(Q)>np.sqrt(C)*1.3]; ax.plot(q.real,q.imag,'o',ms=3.4,mfc='none',mec='#2ecc71',mew=0.6)
        ax.plot(R*np.cos(tt),R*np.sin(tt),'w--',lw=0.7)
    else:
        ax.plot(40*np.cos(tt),40*np.sin(tt),':',c='#f1c40f',lw=0.9); Rg=np.sqrt(2)*40; ax.plot(Rg*np.cos(tt),Rg*np.sin(tt),'w--',lw=0.7)
    ax.set_xlim(-LIM,LIM); ax.set_ylim(-LIM,LIM); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,loc='left',fontsize=8,pad=3)
    ax.text(0.5,-0.02,cap,transform=ax.transAxes,ha='center',va='top',fontsize=6.6)
fig.legend([Line2D([],[],ls='',marker='o',mfc='none',mec='#7fdbff'),Line2D([],[],ls='',marker='o',mfc='none',mec='#2ecc71'),Line2D([],[],ls='--',c='k',lw=0.8),Line2D([],[],ls=':',c='#d4ac0d',lw=1.0)],
           ['lens image, parity $+$','lens image, parity $-$','Einstein radius $R$','grain edge'],loc='center',ncol=4,fontsize=6.6,bbox_to_anchor=(0.5,0.483))
# (d) regimes
ax=fig.add_subplot(gb[0]); tdeg=np.logspace(-1,np.log10(5),100); tr=np.radians(tdeg); Acell=s3/2
ax.loglog(tdeg,1/tr,'k-',lw=0.9); ax.loglog(tdeg,2*np.pi/(Acell*tr),'k--',lw=0.9); ax.loglog(tdeg,2*np.pi/(Acell*tr**2),'k:',lw=1.1)
ax.set_xlim(0.1,5); ax.set_ylim(3,3e6); ax.set_xlabel(r'twist angle or mismatch $\theta$ (deg)'); ax.set_ylabel(r'missing unit cells $n_{\rm miss}$')
ax.set_xticks([0.1,0.2,0.5,1,2,5]); ax.set_xticklabels(['0.1','0.2','0.5','1','2','5']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.text(0.13,5.5,'I: sub-node ($T<1$)',fontsize=6.6)
p1=ax.transData.transform((0.3,2*np.pi/(Acell*np.radians(0.3)))); p2=ax.transData.transform((1.2,2*np.pi/(Acell*np.radians(1.2)))); ang=np.degrees(np.arctan2(p2[1]-p1[1],p2[0]-p1[0]))
xm=0.35; ym=np.sqrt((1/np.radians(xm))*(2*np.pi/(Acell*np.radians(xm))))
ax.text(xm,ym,'II: few-node lens',fontsize=6.6,rotation=ang,rotation_mode='anchor',ha='left',va='center')
ax.text(0.13,1.8e4,'III: resolved lens\n(arcs, Einstein ring)',fontsize=6.6,va='center')
ax.text(1.0,5e5,'IV: core $>L_M$',fontsize=6.6)
th_deg=np.degrees(1/30.)
ax.plot(th_deg,2*np.pi*4/Acell,'s',ms=4.5,mfc='#c0392b',mec='k',mew=0.4); ax.plot(th_deg,2*np.pi*120/Acell,'o',ms=4.5,mfc='#2471a3',mec='k',mew=0.4)
ax.annotate('runs of Secs. III–VII',(th_deg,2*np.pi*4/Acell),xytext=(1.25,4.2),fontsize=6.0,ha='center',va='bottom',arrowprops=dict(arrowstyle='-',lw=0.5))
ax.annotate('panels (a), (b), (e), (f)',(th_deg,2*np.pi*120/Acell),xytext=(0.5,2.6e3),fontsize=6.0,ha='center',va='bottom',arrowprops=dict(arrowstyle='-',lw=0.5))
ax.set_title('(d) size regimes',loc='left',fontsize=8,pad=3)
# (e),(f) relaxed
def Vmap(f):
    D=np.load(f); V,_,_=Vhex(D['ux'],D['uy'],0.005,s3*0.005); V=V.copy(); V[D['core']]=np.nan; return V,D['x']
for k,(f,t) in enumerate((('biglens_a90_C120+0j_r20_L150','(e) relaxed: spiral walls'),('biglens_a0_C120+0j_r20_L150','(f) relaxed: target walls'))):
    ax=fig.add_subplot(gb[1+k]); P='/home/claude/ridge/pair/'+f
    fn=P+'.npz' if os.path.exists(P+'.npz') else (P+'_ck.npz' if os.path.exists(P+'_ck.npz') else None)
    if fn:
        V,x=Vmap(fn); Z=95; m=np.abs(x)<Z+1
        ax.imshow(np.log10(V[np.ix_(m,m)].T+1e-4),origin='lower',cmap='magma',extent=[x[m][0],x[m][-1]]*2,interpolation='bilinear',vmin=-4,vmax=np.log10(0.08))
        tt=np.linspace(0,2*np.pi,300); ax.plot(R*np.cos(tt),R*np.sin(tt),'--',c='#7fdbff',lw=0.8)
        if k==0: ax.plot(np.sqrt(2)*R*np.cos(tt),np.sqrt(2)*R*np.sin(tt),':',c='#7fdbff',lw=0.9)
    ax.set_xlim(-95,95); ax.set_ylim(-95,95); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,loc='left',fontsize=8,pad=3)
fig.savefig('fig9.pdf'); fig.savefig('fig9_preview.png',dpi=170)
