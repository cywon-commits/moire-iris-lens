import json,numpy as np
th=1/30.; res={}
for a in (90,0):
    D=json.load(open(f'permod2_a{a}.json'))
    d=np.array([x['Ua'][0] for x in D]); w=np.array([x['Ua'][1] for x in D])
    g1=np.array([x['Ub'][0] for x in D]); g2=np.array([x['Ub'][1] for x in D]); e=np.array([x['e'] for x in D])
    cols=dict(one=1+0*d,d=d,w=w,g1=g1,g2=g2,dd=d*d,ww=w*w,dw=d*w,gg=g1*g1+g2*g2,g1g1_m=g1*g1-g2*g2,g1g2=g1*g2,
              dg1=d*g1,dg2=d*g2,wg1=w*g1,wg2=w*g2,ddd=d**3,www=w**3,ddw=d*d*w,dww=d*w*w,dgg=d*(g1*g1+g2*g2),wgg=w*(g1*g1+g2*g2))
    M=np.c_[tuple(cols.values())]; c,*_=np.linalg.lstsq(M,e,rcond=None); C=dict(zip(cols,c))
    K=C['dd']/2; chi=2*C['ww']; X=C['dw']; mu=C['gg']/2
    H=np.array([[4*K+4*mu,X],[X,chi+4*mu]]); H0=np.array([[4*K,X],[X,chi]]); Mx=np.linalg.solve(H,H0)
    res[a]=dict(K=K/th**2,mu=mu/th**2,chi=chi/th**2,X=X/th**2,sig_d=C['d']/th**2,tau_w=C['w']/th**2,
                aniso=C['g1g1_m']/th**2,M00=Mx[0,0],M01=Mx[0,1],M10=Mx[1,0],M11=Mx[1,1],S_cauchy=K/(K+mu),
                fit_resid=float(np.abs(e-M@c).max()/th**2), e0=C['one']/th**2)
    print(a,{k:round(float(v),4) for k,v in res[a].items()})
json.dump(res,open('permod2_fit.json','w'),indent=1)
