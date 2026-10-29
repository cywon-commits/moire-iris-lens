import numpy as np, json, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from hexcore import mismatch,nodetype
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(16,15)); gs=fig.add_gridspec(3,4,hspace=0.38,wspace=0.3,height_ratios=[1,1,0.95])
# row 1: crossover
for k,v in enumerate([0.0,0.06,0.075,0.0866]):
    D=np.load(f't1_V3_{v}.npz'); x,y=D['x'],D['y']; m=mismatch(D['dx'],D['dy']); t=nodetype(D['dx'],D['dy'])
    rgb=np.zeros(m.shape+(3,)); far=(m>0.3)
    rgb[...,0]=np.clip(t/2.6,0,1)*far; rgb[...,2]=np.clip(-t/2.6,0,1)*far; rgb[...,1]=np.clip(m/0.58,0,1)*0.55
    rgb=np.tile(rgb,(1,2,1))
    ax=fig.add_subplot(gs[0,k]); ax.imshow(np.transpose(rgb,(1,0,2)),origin='lower',extent=[x[0],x[-1],y[0],y[-1]+16],aspect='equal')
    ax.set_title(f'(a{k+1}) $V_3/V_0$ = {v/0.05:.2f}',fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.text(0.13,0.655,'black: stacking at lattice point (A)   blue: stacking at centroid c₂   orange: centroid c₁ (node)   green: walls',fontsize=9,color='#444')
# row 2: pinwheels
for k,(t,ttl) in enumerate([('three','(b1) 3 sectors, Q=1'),('six_twist','(b2) 6 sectors, pure twist, Q=6'),('six_mix','(b3) 6 sectors, twist+contr., Q=6')]):
    D=np.load(f't2_{t}.npz'); x=D['x']; m=mismatch(D['ux'],D['uy']); ty=nodetype(D['ux'],D['uy'])
    ax=fig.add_subplot(gs[1,k]); ax.imshow(m.T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='magma',vmin=0,vmax=0.58)
    pk=(m==maximum_filter(m,size=13))&(m>0.5); N=len(x); pk[:6]=pk[-6:]=False; pk[:,:6]=pk[:,-6:]=False
    I,J=np.nonzero(pk); ax.scatter(x[I],x[J],s=140,facecolors='none',edgecolors=['#ff6b6b' if ty[i,j]>0 else '#5ab0ff' for i,j in zip(I,J)],linewidths=2)
    ax.set_title(ttl+f'\nnodes found: {len(I)}',fontsize=10); ax.set_xticks([-12,0,12]); ax.set_yticks([-12,0,12])
# row 2 last + row 3: wall pairs
def theory(phi,psi,mu=1.,Y=3.):
    M=np.diag([mu,Y]); r=lambda a: np.array([np.cos(a),np.sin(a)])
    b1=r(np.radians(psi)); b2=r(np.radians(psi+phi)); B=lambda u,v: u@M@v
    coin=np.sqrt(4/3*(B(b1,b1)+B(b1,b2)+B(b2,b2))); sep=np.sqrt(2/3*B(b1,b1))+np.sqrt(2/3*B(b2,b2))
    return coin/sep-1
cols={0:'#c0392b',60:'#e67e22',120:'#6a3d9a',180:'#2a6fb0'}
ax=fig.add_subplot(gs[2,:])
for phi in [0,60,120,180]:
    pts=[]
    for f in glob.glob(f'w3_{phi}_*.json'):
        d=json.load(open(f)); E=np.array(d['E']); pts.append([d['psi'],(E[0,1]-E[-1,1])/E[-1,1]])
    pts=np.array(sorted(pts))
    pp=np.linspace(0,180 if phi in (60,120) else 90,181)
    ax.plot(pp,[theory(phi,p) for p in pp],'-',color=cols[phi],lw=1.5,alpha=0.8)
    if len(pts): ax.plot(pts[:,0],pts[:,1],'o',color=cols[phi],ms=7,label=f'φ = {phi}°')
ax.axhline(0,color='#999',lw=0.8,ls=':')
ax.set_xlabel('ψ: angle between $\\mathbf{b}_1$ and the wall (°)'); ax.set_ylabel('$(E_{coin}-E_{sep})/E_{sep}$')
ax.set_title('(c1) wall pair across the middle layer: dots = 1D trilayer, lines = $\\sqrt{(4/3)(B_{11}+B_{12}+B_{22})}\\,/\\,\\sum_k\\sqrt{(2/3)B_{kk}}-1$',fontsize=9.5)
ax.legend(frameon=False,ncol=2,fontsize=9,loc='upper right')
ax=fig.add_subplot(gs[1,3])
for psi,c,lab in [(120,'#6a3d9a','φ=120°, ψ=120° merged ∥ wall'),(30,'#b07cc6','φ=120°, ψ=30° merged ⊥ wall'),(0,'#c0392b','φ=0°, ψ=0°')]:
    phi=0 if lab.startswith('φ=0') else 120
    d=json.load(open(f'w3_{phi}_{psi}.json')); E=np.array(d['E']); ax.plot(E[:,0],(E[:,1]-E[-1,1])/E[-1,1],'o-',color=c,lw=2,ms=5,label=lab)
ax.axhline(0,color='#999',lw=0.8,ls=':'); ax.set_xlabel('wall separation D (a)'); ax.set_ylabel('$(E(D)-E_{sep})/E_{sep}$')
ax.set_title('(c2) E(D) for three pairs',fontsize=10); ax.legend(frameon=False,fontsize=8,loc='upper right')
fig.savefig('hex_results.png',dpi=110,bbox_inches='tight')
