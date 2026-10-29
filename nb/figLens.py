import os
import numpy as np, sys, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from affine import domain_field
from iris_an import nodes, node_handedness
plt.rcParams.update({'font.size':10}); theta=1/40; W=48; s3=np.sqrt(3)
rows=[('wall','on a wall (edge)',[0.0,3.0,8.0]),('dom','in a domain (face)',[0.0,4.0,10.0]),('AA','on a node',[0.0,15.0,30.0])]
fig=plt.figure(figsize=(22,17)); gs=fig.add_gridspec(3,5,hspace=0.22,wspace=0.32)
c1=np.array([0.5,1/(2*s3)]); a1=np.array([1.,0]); a2=np.array([.5,s3/2])
for r,(site,name,Cs) in enumerate(rows):
    for c,C in enumerate(Cs):
        D=np.load(f'lens_{site}_C{C}.npz'); x=D['x']; core=D['core']; o=D['u0']
        g=domain_field(D['ux'],D['uy']).astype(float); g[core]=np.nan
        a=fig.add_subplot(gs[r,c]); a.imshow(np.tanh(4*g).T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='coolwarm',vmin=-1,vmax=1)
        B=[]
        for n1 in range(-6,7):
            for n2 in range(-6,7):
                q=c1+n1*a1+n2*a2-o; z=(q[0]+1j*q[1])/(1j*theta)
                if abs(z)<W+5: B.append(z)
        B=np.array(B); a.plot(B.real,B.imag,'k+',ms=7,mew=1.0)
        if C>0:
            R=np.sqrt(C/theta); a.add_patch(plt.Circle((0,0),R,fill=False,ec='k',ls='--',lw=1.2)); a.add_patch(plt.Circle((0,0),np.sqrt(2)*R,fill=False,ec='#555',ls=':',lw=1.3))
        P=nodes(x,D['ux'],D['uy'],core); P=P[np.hypot(P[:,0],P[:,1])<W+4]
        if len(P):
            hd=node_handedness(x,D['ux'],D['uy'],P)
            a.scatter(P[hd>0,0],P[hd>0,1],s=70,facecolors='none',edgecolors='#1d9e75',linewidths=1.9)
            a.scatter(P[hd<0,0],P[hd<0,1],s=40,facecolors='none',edgecolors='#d4537e',linewidths=1.5)
        a.set_xlim(-W,W); a.set_ylim(-W,W); a.set_xticks([]); a.set_yticks([])
        sw={'wall':(0.5,'2 end nodes'),'dom':(1/s3,'3 corner nodes'),'AA':(1.0,'6 neighbours')}[site]; Cth=sw[0]**2/(2*theta)
        t=f'({"abc"[r]}{c+1}) {name}, ' + ('no deficit' if C==0 else f'C={C:.0f}a² ' + ('< ' if C<Cth else '> ')+f'{Cth:.1f}a²\n'+('nearest nodes swirled, mirrored copies appear' if C<Cth else f'{sw[1]} swallowed'))
        a.set_title(t,fontsize=10)
d=json.load(open('lens_rows.json'))
a=fig.add_subplot(gs[0,3:5]); b=np.linspace(np.sqrt(2),8,400)
a.plot(b,np.sqrt((b**2+np.sqrt(b**4-4))/2),color='#1d9e75',lw=2.2,label='outer image (same handedness)')
a.plot(b,np.sqrt((b**2-np.sqrt(b**4-4))/2),color='#d4537e',lw=2.2,label='inner image (mirrored)')
a.plot(b,b,color='#999',ls=':',lw=1,label='no deficit (r = β)')
mk={'wall':'o','dom':'s','AA':'^'}
for k,rws in d.items():
    site,C=k.split('_'); C=float(C); R=np.sqrt(C/theta)
    for q in rws:
        if q['beta']/R>8: continue
        a.plot(q['beta']/R,q['r']/R,mk[site],ms=6,mfc='none',mec='#1d9e75' if q['par']>0 else '#d4537e',mew=1.4)
a.axvline(np.sqrt(2),color='#555',ls=':',lw=1); a.axhline(1,color='k',ls='--',lw=0.8)
a.text(1.45,3.0,'sources with β < √2 R:\nno image (swallowed)',fontsize=9,color='#555')
a.set_xlim(0,8); a.set_ylim(0,6); a.set_xlabel('original node distance β / R'); a.set_ylabel('node distance r / R')
a.set_title('(d) lens equation  β = z + iR²/z̄ ,  R = √(C/θ):  all runs collapse (○ wall, □ domain, △ node)',fontsize=10)
a.legend(frameon=False,fontsize=9,loc='upper left'); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
a=fig.add_subplot(gs[1,3:5]); rr=np.linspace(0.05,4,400)
a.plot(rr,-np.degrees(np.arctan(1/rr**2)),'k-',lw=2,label='−arctan(R²/r²)')
for k,rws in d.items():
    site,C=k.split('_'); C=float(C); R=np.sqrt(C/theta)
    for q in rws:
        if q['beta']/R>8: continue
        a.plot(q['r']/R,q['dphi'],mk[site],ms=6,mfc='none',mec='#1d9e75' if q['par']>0 else '#d4537e',mew=1.4)
a.set_xlim(0,4); a.set_ylim(-140,10); a.set_xlabel('node distance r / R'); a.set_ylabel('rotation of node about the deficit (deg)')
a.set_title('(e) swirl: every node is turned by −arctan(R²/r²)  (twist moiré turns the radial pull sideways)',fontsize=10)
a.legend(frameon=False,fontsize=9,loc='lower right'); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
a=fig.add_subplot(gs[2,3:5])
for f,lab,cl in [('bubsz_AA_11.0_2.5.npz','wide bubble (H=2.5a)','#c0392b'),('bub_SP_2.0.npz','small bubble (H=2a)','#2a6fb0')]:
    D=np.load(f); x=D['x']; h=D['h']; dx=x[1]-x[0]; xc,yc=float(D['xc']),float(D['yc'])
    hx,hy=np.gradient(h,dx); hxx,hxy=np.gradient(hx,dx); _,hyy=np.gradient(hy,dx); K=hxx*hyy-hxy**2
    X,Y=np.meshgrid(x-xc,x-yc,indexing='ij'); Rr=np.hypot(X,Y); bins=np.arange(0,16,0.4); kb=[K[(Rr>=b)&(Rr<b+0.4)].mean() for b in bins]
    dA=0.5*np.sum(hx**2+hy**2)*dx*dx; K2=-0.5*np.sum(K*Rr**2)*dx*dx
    a.plot(bins+0.2,kb,color=cl,lw=2,label=f'{lab}: ∫K dA = 0,  area excess {dA:.1f}a²,  −½∫K r² dA = {K2:.1f}a²')
a.axhline(0,color='k',lw=0.8); a.set_xlabel('distance from bubble centre (a)'); a.set_ylabel('Gaussian curvature K (1/a²)')
a.set_title('(f) the bubble: positive cap + negative skirt; its second moment equals the area it removes from the plane',fontsize=10)
a.legend(frameon=False,fontsize=8.5,loc='upper right'); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.text(0.31,0.085,'+ : original (undistorted) node positions;  green ○ normal nodes;  pink ○ mirrored nodes;  dashed: Einstein radius R = √(C/θ);  dotted: √2 R (originals inside are swallowed).  Moiré period 40a.',ha='center',fontsize=9.5,color='#444')
fig.savefig('lens_results.png',dpi=88,bbox_inches='tight')
