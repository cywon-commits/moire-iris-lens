import sys,numpy as np,json
from neb import setup,string
d=float(sys.argv[1]); lam=float(sys.argv[2]); a,b,tag=sys.argv[3],sys.argv[4],sys.argv[5]
nit=int(sys.argv[6]) if len(sys.argv)>6 else 1500
EG,P,N,x=setup(d,lam)
ld=lambda f: (lambda D: np.concatenate([D['ux'].ravel(),D['uy'].ravel()]))(np.load(f+'.npz'))
zA,zB=ld(a),ld(b)
log=open(f'neb_{tag}.log','w')
Z,E=string(EG,P,zA,zB,M=13,nit=nit,dt=0.5,climb_after=int(nit*0.55),log=log)
json.dump(dict(d=d,lam=lam,A=a,B=b,E=E),open(f'neb_{tag}.json','w'))
np.savez_compressed(f'neb_{tag}_path.npz',Z=np.array(Z[::2],dtype=np.float32),x=x)
print('done',tag,np.round(np.array(E)-E[0],3).tolist(),file=log,flush=True)
