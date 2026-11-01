from style import *
from matplotlib.patches import Polygon, Circle
def lens_images(m,zd,kap,lim):
    n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); L=((N1+0.5*N2)+1j*(s3/2*N2)).ravel()
    beta=L/kap-zd; beta=beta[np.abs(beta)<lim*1.6]
    R2=abs(m); phm=np.angle(m); out_p=[];out_n=[];sw=[]
    for b in beta:
        s=abs(b)**2+2*R2*np.cos(phm); disc=s*s-4*R2**2
        if disc<0 or s<0: sw.append(b); continue
        for rr,lst in ((np.sqrt((s+np.sqrt(disc))/2),out_p),(np.sqrt((s-np.sqrt(disc))/2),out_n)):
            e=b/(rr-R2*np.exp(1j*phm)/rr); e/=abs(e); lst.append(zd+rr*e)
    return np.array(out_p),np.array(out_n),np.array(sw)+zd,beta+zd
fig=plt.figure(figsize=(W2,2.25))
gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,1],wspace=0.08,left=0.02,right=0.99,bottom=0.13,top=0.90)
# (a) registry torus: lifted line arrangement
ax=fig.add_subplot(gs[0]); a1=np.array([1,0]); a2=np.array([0.5,s3/2])
for i in range(-3,4):
    for j in range(-3,4):
        p=i*a1+j*a2; up=[p,p+a1,p+a2]; dn=[p+a1,p+a1+a2,p+a2]
        ax.add_patch(Polygon(up,closed=True,fc='#f6d7b0',ec='none')); ax.add_patch(Polygon(dn,closed=True,fc='#cfe2f3',ec='none'))
for k in range(-8,9):
    t=np.linspace(-6,6,2)
    for d,o in ((a1,a2),(a2,a1),(a2-a1,a1)):
        p0=k*o; ax.plot(p0[0]+t*d[0],p0[1]+t*d[1],'k-',lw=0.7)
pts=np.array([i*a1+j*a2 for i in range(-4,5) for j in range(-4,5)])
ax.plot(pts[:,0],pts[:,1],'o',ms=4.5,mfc='#c0392b',mec='k',mew=0.4)
cell=np.array([[0,0],a1,a1+a2,a2]); ax.add_patch(Polygon(cell,closed=True,fill=False,ec='k',lw=1.4,ls='--'))
ax.set_xlim(-0.6,2.2); ax.set_ylim(-0.55,2.25); ax.set_aspect('equal'); ax.axis('off')
ax.text(0.5,0.29,'AB',ha='center',va='center',fontsize=7.5); ax.text(1.0,0.58,'BA',ha='center',va='center',fontsize=7.5)
ax.annotate('AA',xy=(1,0),xytext=(1.55,-0.38),fontsize=7.5,arrowprops=dict(arrowstyle='-',lw=0.6),ha='center',
            bbox=dict(fc='w',ec='none',pad=0.5))
ax.annotate('wall',xy=(1.75,0.43),xytext=(2.05,1.15),fontsize=7.5,arrowprops=dict(arrowstyle='-',lw=0.6),ha='center',
            bbox=dict(fc='w',ec='none',pad=0.5))
plabel(ax,'(a)','registry torus, lifted')
# (b) simulated twist moire with area-removing defect, (c) beta plane, (d) image plane
D=np.load(P+'c_tw_def.npz'); x=D['x']; V=Vmap(D); zd=8+0j; kap=1j*TH; m=4/kap; lim=58
ax=fig.add_subplot(gs[1]); mx=np.abs(x-8)<lim+1; my=np.abs(x)<lim+1
ax.imshow(V[np.ix_(mx,my)].T,origin='lower',cmap='magma',extent=[x[mx][0],x[mx][-1],x[my][0],x[my][-1]],interpolation='bilinear')
zp,zn,sw,bet=lens_images(m,zd,kap,lim)
k=(np.abs((zp-zd).real)<lim-2)&(np.abs(zp.imag)<lim-2)
ax.plot(zp[k].real,zp[k].imag,'o',ms=5.5,mfc='none',mec='#7fdbff',mew=0.9)
ax.add_patch(Circle((8,0),RL,fill=False,ec='w',ls=':',lw=0.8))
ax.set_xlim(8-lim,8+lim); ax.set_ylim(-lim,lim); ax.set_xticks([]); ax.set_yticks([])
plabel(ax,'(b)','relaxed twist moiré')
ax.text(0.5,-0.02,'circles: lens-map prediction',transform=ax.transAxes,fontsize=6.3,va='top',ha='center')
# (c) source plane
ax=fig.add_subplot(gs[2]); bs=2*RL*abs(np.sin(np.angle(m)/2))
ax.add_patch(Circle((0,0),bs,fc='#fde2e2',ec=CT,lw=0.8))
bb=bet-zd; kk=np.abs(bb)<lim
inside=np.abs(bb)<bs
ax.plot(bb[kk&~inside].real,bb[kk&~inside].imag,'o',ms=4,mfc='#444',mec='none')
ax.plot(bb[kk&inside].real,bb[kk&inside].imag,'x',ms=5,mec=CT,mew=1.2,label='swallowed source')
ax.plot([],[],'o',ms=4,mfc='#444',mec='none',label='source with images')
H1,L1=ax.get_legend_handles_labels()
ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5,-0.02,r'disk radius $b_s=2R\,|\sin(\varphi_m/2)|$',transform=ax.transAxes,fontsize=6.3,va='top',ha='center',color=CT)
plabel(ax,'(c)',r'source plane $\beta$')
# (d) image plane
ax=fig.add_subplot(gs[3])
ax.add_patch(Circle((0,0),RL,fc='#eeeeee',ec='k',lw=0.6,ls='--'))
a=zp-zd; b=zn-zd
ax.plot(a[np.abs(a)<lim].real,a[np.abs(a)<lim].imag,'o',ms=4,mfc='#444',mec='none',label='parity $+$')
ax.plot(b.real,b.imag,'o',ms=3.6,mfc='w',mec=CM,mew=0.9,label='parity $-$ (inverted)')
ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5,-0.02,r'dashed circle: Einstein radius $R$',transform=ax.transAxes,fontsize=6.3,va='top',ha='center')
#ax.annotate('$|z|<R$',xy=(RL*0.7,RL*0.7),xytext=(24,30),fontsize=7,ha='center',arrowprops=dict(arrowstyle='-',lw=0.5),bbox=dict(fc='w',ec='none',pad=0.4))
H2,L2=ax.get_legend_handles_labels()
fig.legend(H1+H2,L1+L2,loc='lower center',bbox_to_anchor=(0.735,-0.015),ncol=4,fontsize=6.3,handletextpad=0.15,columnspacing=0.7)
plabel(ax,'(d)',r'image plane $z$')
save(fig,'fig1')
