import os
import numpy as np, sys, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','nb'))
from affine import domain_field
from iris_an import nodes, node_handedness
plt.rcParams.update({'font.size':10}); R=np.sqrt(4*30.)
def clean(a): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig=plt.figure(figsize=(21,12.5)); gs=fig.add_gridspec(2,3,hspace=0.3,wspace=0.24)
U=json.load(open('Uall.json'))
a=fig.add_subplot(gs[0,0])
for k,lab,c,mk in [('90_x','twist lens, pair along x','#6a3d9a','o'),('90_y','twist lens, pair along y','#b07cc6','s'),('0_x','converging lens (mismatch moiré), along x','#2a6fb0','^')]:
    rows=U[k]; a.plot([r[0]/R for r in rows],[r[1] for r in rows],mk+'-',color=c,lw=2,ms=7,label=lab)
a.axhspan(-0.5,0.5,color='#eee',lw=0); a.axhline(0,color='k',lw=0.7); a.axvline(2*np.sqrt(2),color='#888',ls=':',lw=1)
a.text(2*np.sqrt(2)+0.05,3.6,'2√2 R',fontsize=9,color='#666'); a.text(0.1,0.6,'grey band: elastic-control noise',fontsize=8.5,color='#666')
a.set_xlabel('separation d / R'); a.set_ylabel('U_moiré (per pair)'); a.set_ylim(-1,7.8)
a.set_title('(a) pair interaction: twist lens is direction-independent and short-ranged;\nconverging lens repels at all separations studied',fontsize=10); a.legend(frameon=False,fontsize=8.5,loc='center right'); clean(a)
a=fig.add_subplot(gs[0,1])
D=np.load('pair_d26.0_V0.005_a0_x.npz'); x=D['x']; g=domain_field(D['ux'],D['uy']).astype(float); g[D['core']]=np.nan
a.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1); a.set_xlim(-50,50); a.set_ylim(-50,50); a.set_xticks([]); a.set_yticks([])
a.set_title('(b) converging pair, d = 26a: the spontaneously chiral\n(pinwheel) background is undone between the two defects',fontsize=10)
a=fig.add_subplot(gs[0,2])
D=np.load('blind.npz'); x=D['x']; g=domain_field(D['ux'],D['uy']).astype(float); g[D['core']]=np.nan
a.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1,alpha=0.55)
P=nodes(x,D['ux'],D['uy'],D['core']); P=P[np.hypot(P[:,0],P[:,1])<62]; hd=node_handedness(x,D['ux'],D['uy'],P)
a.scatter(P[hd>0,0],P[hd>0,1],s=50,facecolors='none',edgecolors='#1d9e75',linewidths=1.6,label='normal nodes (data)')
a.scatter(P[hd<0,0],P[hd<0,1],s=28,facecolors='none',edgecolors='#d4537e',linewidths=1.3,label='mirrored nodes (data)')
tr=json.load(open('blind.json'))['truth']; fitres=[r for r in json.load(open('blind_fit.json')) if r['keep']][-1]['out']
for (xt,yt,ct) in tr: a.plot(xt,yt,'kx',ms=14,mew=2.5)
for (xf,yf,cf) in fitres:
    a.add_patch(plt.Circle((xf,yf),6.0,fill=False,ec='#c0392b',lw=2.4))
    t=min(tr,key=lambda t: np.hypot(t[0]-xf,t[1]-yf)); a.text(xf+6.5,yf-9,f'C = {cf:.2f}\n(true {t[2]})',fontsize=8.5,color='#c0392b')
a.plot([],[],'kx',ms=10,mew=2,label='true deficits (hidden)'); a.plot([],[],'o',mfc='none',mec='#c0392b',mew=2,label='recovered from node positions')
a.set_xlim(-55,55); a.set_ylim(-45,55); a.set_xticks([]); a.set_yticks([])
a.set_title('(c) blind test: three hidden vacancy clusters recovered from\n37 node positions only (number chosen by BIC: 3)',fontsize=10); a.legend(frameon=False,fontsize=8,loc='lower left')
a=fig.add_subplot(gs[1,0]); r2=json.load(open('invert2.json'))
for name,c in [('all nodes','#6a3d9a'),('normal only','#888')]:
    s=[0.0,0.5,1.0,2.0,3.0]; a.plot(s,[r2[f'{name}_{v}'][0] for v in s],'o-',color=c,lw=2,ms=6,label=f'{name}: position error (a)')
    a.plot(s,[100*r2[f'{name}_{v}'][1]/10 for v in s],'s--',color=c,lw=1.3,ms=5,label=f'{name}: C error (%/10)')
a.set_yscale('log'); a.set_xlabel('noise on node positions σ (a)'); a.set_ylabel('error'); a.set_ylim(0.1,30)
a.set_title('(d) robustness (pair d = 32a): with mirrored nodes the two deficits\nare resolved; far (normal) nodes alone see only their total',fontsize=10); a.legend(frameon=False,fontsize=8,loc='center right'); clean(a)
a=fig.add_subplot(gs[1,1]); M=json.load(open('moving.json')); xs=np.array(M['xs']); rig=np.array(M['rig'])
a.step(xs,rig[:,0],where='mid',color='#1d9e75',lw=2,label='rigid lens: normal nodes in view')
rel=np.array(M['rel']); a.plot(rel[:,0],rel[:,1],'o',color='#1d9e75',ms=8,mfc='white',mew=2,label='relaxed')
for xe,typ,b in M['events']:
    a.axvline(xe,color='#c0392b' if typ=='enter' else '#2a6fb0',ls=':',lw=1.2)
a.text(15.7,8.25,'node at origin\nreleased',fontsize=8,color='#2a6fb0'); a.text(22.3,7.25,'nodes (26,±15)\nswallowed',fontsize=8,color='#c0392b'); a.text(30.1,8.25,'…released',fontsize=8,color='#2a6fb0')
a.set_ylim(3.5,8.6); a.set_xlabel('lens position x (a)   [moving along x from an AA node]'); a.set_ylabel('normal AA nodes within 50a')
a.set_title('(e) moving lens: events where nodes cross the swallow circle √2R;\nrelaxation delays release (effective radius ≈ 1.4 × √2R)',fontsize=10); a.legend(frameon=False,fontsize=8.5,loc='lower left'); clean(a)
a=fig.add_subplot(gs[1,2]); a.axis('off')
a.text(0,0.97,
"summary\n\n"
"interaction (twist lens)\n"
"  repulsive for d ≲ 2.4R, ≈ 0 beyond;\n  same along x and y\n\n"
"interaction (converging lens)\n"
"  +5…7 for 0.7R–3.6R: the pair breaks the\n  spontaneous chirality of the mismatch moiré\n\n"
"inversion (strong + weak lensing)\n"
"  each node gives u_def exactly at its\n  position: positions to ~0.2a, C to ~5 %\n"
"  far nodes only → total deficit (shell theorem)\n"
"  uniform deficit density ≡ uniform lattice\n  mismatch (mass-sheet degeneracy)\n\n"
"moving lens\n"
"  node swallow / release events follow\n  the swallow circle; relaxation adds lag",va='top',fontsize=10.5,family='monospace')
fig.savefig('next_steps.png',dpi=90,bbox_inches='tight')
