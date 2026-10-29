import os
import numpy as np, sys, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','nb'))
from affine import domain_field
from iris_an import nodes, node_handedness
from binary import crit_curves
plt.rcParams.update({'font.size':10}); R=np.sqrt(4*30.)
U=json.load(open('U.json')); UN=json.load(open('Unodes.json'))
rig={8:5,12:5,16:5,20:5,26:3,32:2,40:0,50:4}
fig=plt.figure(figsize=(20,11)); gs=fig.add_gridspec(2,4,hspace=0.3,wspace=0.28)
a=fig.add_subplot(gs[0,0:2]); d=np.array([r[0] for r in U])
a.axvspan(0,1,color='#eee',lw=0); a.axvspan(1,2*np.sqrt(2),color='#e1f0fa',lw=0); a.axvspan(2*np.sqrt(2),5,color='#fdf0e6',lw=0)
for x0,t in [(0.5,'close'),(1.9,'intermediate'),(3.9,'wide')]: a.text(x0,3.25,t,ha='center',fontsize=9,color='#555')
a.plot(d/R,[r[3] for r in U],'o-',color='#6a3d9a',lw=2.2,ms=7,label='moiré-mediated  U_moiré = U_total − U_elastic')
a.plot(d/R,[r[2] for r in U],'s--',color='#888',lw=1.4,ms=5,label='elastic control (no registry): ≈ 0 ± 0.5')
a.axhline(0,color='k',lw=0.8); a.set_xlim(0,5); a.set_ylim(-1.6,3.5)
a.set_xlabel('separation d / R   (R = √(C/θ) = 11a)'); a.set_ylabel('interaction energy (per pair)')
a.set_title('(a) two vacancy clusters (C = 4a² each) in a twisted moiré: interaction exists only through the moiré\nshading: topology of the binary-lens critical curves (3 / 1 / 2 curves)',fontsize=10)
a.legend(frameon=False,fontsize=9,loc='lower right'); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
a=fig.add_subplot(gs[0,2:4])
a.plot(d/R,[r[2] for r in UN],'o-',color='#1d9e75',lw=2,ms=7,label='relaxed: extra normal nodes of the pair')
a.plot(d/R,[rig[int(x)] for x in d],'s--',color='#1d9e75',lw=1.2,ms=6,mfc='none',label='rigid lens prediction')
a.axhline(0,color='k',lw=0.6); a.set_xlabel('d / R'); a.set_ylabel('N_pair − (N_L + N_R − N_bg)  (normal AA nodes)')
a.set_title('(b) cause: two lenses together swallow fewer nodes than each alone (deflections cancel between them)',fontsize=10)
a.legend(frameon=False,fontsize=9,loc='upper right'); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
for k,dd in enumerate([12.,32.,50.]):
    D=np.load(f'pair_d{dd}_V0.005.npz'); x=D['x']; g=domain_field(D['ux'],D['uy']).astype(float); g[D['core']]=np.nan
    A=fig.add_subplot(gs[1,k]); A.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1)
    for cc in crit_curves(dd/R): A.plot(cc.real*R,cc.imag*R,'k-',lw=1.0)
    P=nodes(x,D['ux'],D['uy'],D['core']); P=P[np.hypot(P[:,0],P[:,1])<58]; hd=node_handedness(x,D['ux'],D['uy'],P)
    A.scatter(P[hd>0,0],P[hd>0,1],s=70,facecolors='none',edgecolors='#1d9e75',linewidths=1.9); A.scatter(P[hd<0,0],P[hd<0,1],s=35,facecolors='none',edgecolors='#d4537e',linewidths=1.4)
    A.set_xlim(-52,52); A.set_ylim(-52,52); A.set_xticks([]); A.set_yticks([])
    Um=[r[3] for r in U if r[0]==dd][0]
    A.set_title(f'(c{k+1}) d = {dd:.0f}a = {dd/R:.1f}R,  U_moiré = {Um:+.2f}\nblack: binary-lens critical curves',fontsize=10)
A=fig.add_subplot(gs[1,3]); A.axis('off')
A.text(0,0.95,"• direct elastic interaction of two\n  area-deficit centres = 0 (Eshelby):\n  the control stays within ±0.5\n\n• moiré-mediated part:\n  repulsive for d ≲ 2.4R,\n  ≈ 0 for 2.4R ≲ d ≲ 3.6R\n  (near the close→wide change of the\n  critical curves at d = 2√2 R),\n  bumps further out where the pair\n  meets particular nodes of the lattice\n\n• each extra AA node ≈ +2 in energy:\n  the interaction is counted in nodes",va='top',fontsize=10.5,family='monospace')
fig.text(0.5,0.035,'Twisted moiré, period 30a; two vacancy clusters C = 4a² at x = ±d/2.  Green ○ normal nodes, pink ○ mirrored nodes.  Energies: E_pair − E_left − E_right + E_background, minus the same combination with the registry switched off.',ha='center',fontsize=9.3,color='#444')
fig.savefig('pair_interaction.png',dpi=95,bbox_inches='tight')
