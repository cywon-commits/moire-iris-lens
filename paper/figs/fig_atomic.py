import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from lat import *; from defects import *
plt.rcParams.update({'font.size':7.5,'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif'})
th=1/30.; Lm=66
cases=[('vacancy cluster','area removed, $C>0$',vacancy),('inserted island','area added, $C<0$',island),
       ('rotated patch','torque, ${\\rm Im}\\,C\\neq0$',rotpatch),('vacancy + island pair','$\\sum C\\approx0$ (dipole)',neutral),
       ('edge dislocation','Burgers $b\\neq0$',edge)]
verdict=['lens, $\\varphi_m=-90^\\circ$\nswallow radius $\\sqrt{2}R$','lens, $\\varphi_m=+90^\\circ$\nswallow radius $\\sqrt{2}R$','lens, $\\varphi_m=180^\\circ$\nswallow radius $2R$','no lens: local only\n(field $\\sim 1/\\bar z^{2}$)','not a lens: moiré\ndislocation $B=A^{-1}b$']
fig=plt.figure(figsize=(7.0,3.75))
gs=fig.add_gridspec(2,5,left=0.055,right=0.99,top=0.86,bottom=0.115,wspace=0.06,hspace=0.12)
H0=honeycomb(16); BD=None
L1=rot(honeycomb(Lm+4),-th/2); T1=cKDTree(L1)
for j,(nm,sub,fn) in enumerate(cases):
    # --- atomic close-up of the defect layer
    ax=fig.add_subplot(gs[0,j]); X=H0.copy()
    Y,C,_=fn(X)
    tag=_
    P=bonds(Y); seg=np.stack([Y[P[:,0]],Y[P[:,1]]],1)
    ax.add_collection(LineCollection(seg,colors='#8a8a8a',linewidths=0.45))
    cl=np.array(['#2c3e50','#c0392b','#e67e22'])[tag]
    ax.scatter(Y[:,0],Y[:,1],s=2.6 if j!=3 else 1.6,c=cl,lw=0,zorder=3)
    if j==4: ax.plot([-11,0],[0,0],ls=':',c='#c0392b',lw=0.8); ax.text(0.4,-1.4,'$\\perp$',fontsize=10,color='#c0392b',ha='center',va='top')
    ax.set_xlim(-11 if j!=3 else -17,11 if j!=3 else 17); ax.set_ylim(-11 if j!=3 else -17,11 if j!=3 else 17)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'({chr(97+j)}) {nm}\n{sub}',fontsize=7.2,pad=3,linespacing=1.2)
    # --- moire scale: atom-resolved registry of the two layers
    ax=fig.add_subplot(gs[1,j])
    X2=honeycomb(Lm+4); Y2,C,_=fn(X2); Y2=rot(Y2,th/2)
    d,_=T1.query(Y2); f=np.exp(-(d/0.18)**2)
    nb=150; H,xe,ye=np.histogram2d(Y2[:,0],Y2[:,1],bins=nb,range=[[-Lm,Lm],[-Lm,Lm]],weights=f)
    N,_,_=np.histogram2d(Y2[:,0],Y2[:,1],bins=nb,range=[[-Lm,Lm],[-Lm,Lm]])
    M=gaussian_filter(H,1.6)/np.maximum(gaussian_filter(N,1.6),1e-9); M[gaussian_filter(N,1.6)<0.5]=np.nan
    ax.imshow(M.T,origin='lower',extent=[-Lm,Lm,-Lm,Lm],cmap='magma',vmin=0.15,vmax=0.75,interpolation='bilinear')
    if j<3:
        m=C/(1j*th); n=np.arange(-8,9); N1,N2=np.meshgrid(n,n,indexing='ij'); Lv=((N1+0.5*N2)+1j*(s3/2*N2)).ravel()
        for b in Lv/(1j*th):
            R2=abs(m); ph=np.angle(m); sg=abs(b)**2+2*R2*np.cos(ph); dsc=sg*sg-4*R2**2
            if dsc<0 or sg<0: continue
            rp=np.sqrt((sg+np.sqrt(dsc))/2); e=b/(rp-R2*np.exp(1j*ph)/rp); zz=rp*e/abs(e)
            if abs(zz.real)<57 and abs(zz.imag)<57: ax.plot(zz.real,zz.imag,'o',ms=6.5,mfc='none',mec='#7fdbff',mew=0.7)
    ax.set_xlim(-60,60); ax.set_ylim(-60,60); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5,-0.03,verdict[j],transform=ax.transAxes,ha='center',va='top',fontsize=6.6,linespacing=1.15)
    print(nm,'C=',np.round(C,2),'R=',np.round(np.sqrt(abs(C)/th),1) if C!=0 else 0)
fig.text(0.008,0.66,'atomic\nlayer',rotation=90,fontsize=7.5,va='center',ha='left')
fig.text(0.008,0.30,'twisted\nmoiré',rotation=90,fontsize=7.5,va='center',ha='left')
fig.savefig('atomic_defects.pdf'); fig.savefig('atomic_defects.png',dpi=220)
