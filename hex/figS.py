import numpy as np, json, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from affine import domain_field
from track import track
from spiral import delaunay_dirs
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(17,7.4)); gs=fig.add_gridspec(2,4,hspace=0.12,wspace=0.3,height_ratios=[0.8,1])
def dmap(ax,f,ttl,zoom=None):
    D=np.load(f); x,y=D['x'],D['y']; g=domain_field(D['ux'],D['uy'])
    ax.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='coolwarm',aspect='equal',vmin=-1,vmax=1)
    ax.set_title(ttl,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
for k,(f,t) in enumerate([('aff_twist.npz','(a) pure twist, χ=0°: straight'),('aff_tw+iso1.npz','(b) twist+contraction, χ=30°'),('aff_tw+iso2.npz','(c) twist+contraction, χ=60°'),('aff_iso_only.npz','(d) pure contraction, χ=90°:\nspontaneously chiral')]):
    dmap(fig.add_subplot(gs[0,k]),f,t)
dmap(fig.add_subplot(gs[1,0]),'aff_tw+uni1.npz','(e) twist + uniaxial, s/θ=0.29')
dmap(fig.add_subplot(gs[1,1]),'aff_tw+uni3.npz','(f) twist + uniaxial, s/θ=0.87')
T=json.load(open('track.json')); ax=fig.add_subplot(gs[1,2])
cols={'tw+iso1':'#2a6fb0','tw+iso2':'#6a3d9a','tw+iso4':'#b07cc6','tw+iso3':'#c0392b'}
for tag,c in cols.items():
    d=T[tag]; r=np.array(d['r']); dd=np.array(d['d']); chi=d['chi']
    ax.plot(r,-dd/chi,'-',color=c,lw=2,label=f"χ={chi:.0f}°, L$_m$={d['L']:.0f}a")
ax.axhline(1,color='#888',ls=':',lw=1); ax.set_xlim(0,10)
ax.set_xlabel('distance from AA node r (a)'); ax.set_ylabel('wall rotation  −δ / χ')
ax.set_title('(g) walls turn toward pure shear near the node',fontsize=10); ax.legend(frameon=False,fontsize=8.5)
ax=fig.add_subplot(gs[1,3]); rows=[]
for f in glob.glob('chiral_lam*.npz'):
    D=np.load(f); lam=float(f.split('lam')[1][:-4]); rows.append((lam+1,float(D['Ef']-D['Es'])))
rows=np.array(sorted(rows)); dep={0.5:4.7,0.75:41.0,1.0:53.0,1.5:64.0,2.0:69.2,4.0:76.5}
ax.plot(rows[:,0],[dep[round(k,2)] for k in rows[:,0]],'o-',color='#6a3d9a',lw=2,ms=6,label='|departure angle| (°)')
ax.set_xlabel('bulk / shear modulus  K/μ = (λ+μ)/μ'); ax.set_ylabel('|departure angle| at r = a (°)')
for k,e in rows: ax.annotate(f'ΔE={e:.2f}',(k,dep[round(k,2)]),textcoords='offset points',xytext=(6,-12),fontsize=8,color='#555')
ax.set_title('(h) pure contraction: chirality vs K/μ',fontsize=10)
fig.text(0.5,0.0,'red / blue: the two degenerate stackings (AB, BA);  walls = partial dislocations;  nodes = AA',ha='center',fontsize=9,color='#444')
fig.savefig('spiral_results.png',dpi=110,bbox_inches='tight')
