import sys,numpy as np,json
from neb import setup,string
d=float(sys.argv[1]); lam=float(sys.argv[2]); a,b,tag=sys.argv[3],sys.argv[4],sys.argv[5]
EG,P,N,x=setup(d,lam)
ld=lambda f: (lambda D: np.concatenate([D['ux'].ravel(),D['uy'].ravel()]))(np.load(f+'.npz'))
zA,zB=ld(a),ld(b); log=open(f'neb2_{tag}.log','w')
Z,E=string(EG,P,zA,zB,M=13,nit=800,dt=0.5,climb_after=10**9,log=log)
E=np.array(E); print('stage1',np.round(E-E[0],3).tolist(),file=log,flush=True)
# refine the first uphill segment: from A to the first image that is lower than A by >0.3 (or image 3)
k=next((i for i in range(1,13) if E[i]<E[0]-0.3),3); k=max(k,2)
Z2,E2=string(EG,P,zA,Z[k],M=13,nit=800,dt=0.5,climb_after=10**9,log=log)
E2=np.array(E2); print('stage2 (A..img%d)'%k,np.round(E2-E2[0],4).tolist(),file=log,flush=True)
# arc-length coordinates
s1=np.concatenate([[0],np.cumsum([np.linalg.norm(Z[i+1]-Z[i]) for i in range(12)])])
s2=np.concatenate([[0],np.cumsum([np.linalg.norm(Z2[i+1]-Z2[i]) for i in range(12)])])
json.dump(dict(d=d,lam=lam,E1=E.tolist(),s1=s1.tolist(),E2=E2.tolist(),s2=s2.tolist(),k=k),open(f'neb2_{tag}.json','w'))
i=int(np.argmax(E2)); np.savez_compressed(f'neb2_{tag}_saddle.npz',z=Z2[i].astype(np.float32),x=x)
np.savez_compressed(f'neb2_{tag}_path.npz',Z=np.array(Z,dtype=np.float32))
