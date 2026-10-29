import os
import numpy as np, sys, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from affine import domain_field
from iris_an import nodes, node_handedness
plt.rcParams.update({'font.size':10}); W=46; theta=1/40
cases=[('gl_a0_C4.0.npz','gl_a0_C0.0.npz','(a) mismatch moiré α=0°, vacancy\nφ_m=0°: converging lens'),
       ('gl_a180_C4.0.npz','gl_a180_C0.0.npz','(b) mismatch moiré α=180°, vacancy\nφ_m=180°: diverging lens'),
       ('gl_a90_C4.0.npz','lens_wall_C0.0.npz','(c) twist moiré, vacancy\nφ_m=−90°: clockwise swirl'),
       ('gl_a90_C-4.0.npz','lens_wall_C0.0.npz','(d) twist moiré, interstitial\nφ_m=+90°: counter-clockwise swirl'),
       ('gl_a45_C4.0.npz','gl_a45_C0.0.npz','(e) mixed α=45°, vacancy\nφ_m=−45°')]
fig=plt.figure(figsize=(23,10)); gs=fig.add_gridspec(2,5,hspace=0.16,wspace=0.18,height_ratios=[1,0.9])
for k,(f,fr,t) in enumerate(cases):
    D=np.load(f); x=D['x']; core=D['core']; C=float(D['C']); al=float(D['alpha']); R=np.sqrt(abs(C)/theta)
    m=C*np.exp(-1j*np.radians(al))/theta; phim=np.angle(m)
    g=domain_field(D['ux'],D['uy']).astype(float); g[core]=np.nan
    a=fig.add_subplot(gs[0,k]); a.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1)
    Dr=np.load(fr); Pr=nodes(x,Dr['ux'],Dr['uy'],Dr['core']); a.plot(Pr[:,0],Pr[:,1],'k+',ms=9,mew=1.3)
    P=nodes(x,D['ux'],D['uy'],core); P=P[np.hypot(P[:,0],P[:,1])<W+4]; hd=node_handedness(x,D['ux'],D['uy'],P)
    a.scatter(P[hd>0,0],P[hd>0,1],s=80,facecolors='none',edgecolors='#1d9e75',linewidths=2)
    a.scatter(P[hd<0,0],P[hd<0,1],s=40,facecolors='none',edgecolors='#d4537e',linewidths=1.5)
    a.add_patch(plt.Circle((0,0),R,fill=False,ec='k',ls='--',lw=1.1))
    bs=2*R*abs(np.sin(phim/2))
    if bs>0.5: a.add_patch(plt.Circle((0,0),bs,fill=False,ec='#555',ls=':',lw=1.3))
    for zr in Pr:
        zz=zr[0]+1j*zr[1]
        if abs(zz)>W: continue
        hit=[p for p,h in zip(P,hd) if h>0 and np.hypot(*(p-zr))<0.45*abs(zz)]
        if hit:
            p=min(hit,key=lambda p: np.hypot(*(p-zr))); a.annotate('',xy=p,xytext=zr,arrowprops=dict(arrowstyle='->',color='k',lw=1.3))
    a.set_xlim(-W,W); a.set_ylim(-W,W); a.set_xticks([]); a.set_yticks([]); a.set_title(t,fontsize=10)
rel=json.load(open('gl_rel.json'))
a=fig.add_subplot(gs[1,0:2]); rr=np.linspace(1.02,5,300)
cols={0:'#2a6fb0',180:'#c0392b',-90:'#6a3d9a',90:'#b07cc6',-45:'#e67e22'}
for f,A in rel.items():
    pm=int(round(A['phim'])); pm=-180 if pm==180 else pm; pmk=180 if abs(pm)==180 else pm; cl=cols[pmk]
    mm=np.exp(1j*np.radians(A['phim']))
    a.plot(rr,-np.degrees(np.angle(1-mm/rr**2)),color=cl,lw=2,label=f"φ_m = {A['phim']:.0f}°")
    for n in A['nodes']:
        if n.get('found') and not n['swallowed_pred']: a.plot(n['r']/A['R'],n['dphi'],'o',color=cl,ms=7,mfc='none',mew=1.8)
a.axhline(0,color='k',lw=0.6); a.set_xlim(1,5); a.set_ylim(-60,60); a.set_xlabel('node distance r / R  (normal nodes, r > R)'); a.set_ylabel('rotation of node about the defect (deg)')
a.set_title('(f) swirl  −arg(1 − e^{iφ_m} R²/r²): sense set by the mass phase (lines: theory, ○: relaxed runs)',fontsize=10)
a.legend(frameon=False,fontsize=8.5,ncol=2); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
a=fig.add_subplot(gs[1,2:4]); b=np.linspace(0,6,500)
for pmd,cl in [(0,'#2a6fb0'),(-45,'#e67e22'),(-90,'#6a3d9a'),(180,'#c0392b')]:
    ph=np.radians(pmd); bb=b**2+2*np.cos(ph); disc=bb**2-4; ok=(disc>=0)&(bb>0)
    rp=np.sqrt((bb[ok]+np.sqrt(disc[ok]))/2); rm=np.sqrt((bb[ok]-np.sqrt(disc[ok]))/2)
    a.plot(b[ok],rp,color=cl,lw=2,label=f'φ_m={pmd}°: swallow radius {2*abs(np.sin(ph/2)):.2f}R'); a.plot(b[ok],rm,color=cl,lw=1.2,ls='--')
for f,A in rel.items():
    pm=int(round(A['phim'])); pmk=180 if abs(pm)==180 else pm; cl=cols[pmk]
    for n in A['nodes']:
        if n.get('found') and not n['swallowed_pred']: a.plot(n['beta']/A['R'],n['r']/A['R'],'o',color=cl,ms=7,mfc='none',mew=1.8)
        if n['swallowed_pred']: a.plot(n['beta']/A['R'],0.05,'x',color=cl,ms=9,mew=2)
a.plot(b,b,color='#999',ls=':',lw=1); a.axhline(1,color='k',ls='--',lw=0.6)
a.set_xlim(0,4.2); a.set_ylim(0,4.2); a.set_xlabel('original node distance β / R'); a.set_ylabel('node distance r / R')
a.set_title('(g) radial law: solid outer image, dashed mirrored image, × swallowed (relaxed runs: ○)',fontsize=10)
a.legend(frameon=False,fontsize=8.5,loc='upper left'); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
a=fig.add_subplot(gs[1,4]); a.axis('off')
txt=("mass phase  φ_m = arg C − α\n\n"
     "C > 0: vacancy cluster in top\n        (≡ interstitials in bottom)\n"
     "C < 0: interstitials in top\n        (≡ vacancy cluster in bottom)\n\n"
     "α = 0°   top lattice larger\n"
     "α = 180° top lattice smaller\n"
     "α = ±90° pure twist\n\n"
     "φ_m = 0      converging: pushed out\n"
     "φ_m = 180°  diverging: pulled in,\n                  swallowed within 2R\n"
     "φ_m = ∓90°  swirl (cw / ccw),\n                  swallowed within √2 R")
a.text(0.0,0.98,txt,va='top',fontsize=10.5,family='monospace')
fig.text(0.5,0.035,'Moiré period 40a; |C| = 4a² placed on a wall.  + : node positions without the defect;  arrows: where each node moved;  green ○ normal, pink ○ mirrored nodes;  dashed: R = √(|C|/θ);  dotted: swallow radius 2R|sin(φ_m/2)|.',ha='center',fontsize=9.5,color='#444')
fig.savefig('lens_types.png',dpi=92,bbox_inches='tight')
