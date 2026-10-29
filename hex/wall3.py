import numpy as np, json, sys
from scipy.optimize import minimize
from hexcore import Vhex
def energy_D(b1,b2,D,Ly=30.,dy=0.1,lam=1.,mu=1.,V0=0.05,Kp=20.,x0=None,psi=0.):
    cp,sp=np.cos(psi),np.sin(psi)
    y=np.arange(-Ly,Ly+dy/2,dy); n=len(y); Y=lam+2*mu
    ya,yb=-D/2,D/2; ia=int(np.argmin(abs(y-ya))); ib=int(np.argmin(abs(y-yb)))
    b1=np.array(b1,float); b2=np.array(b2,float); h1=b1/np.linalg.norm(b1); h2=b2/np.linalg.norm(b2)
    step=lambda y0: 0.5*(1+np.tanh((y-y0)/1.0))
    off=np.array([[0,0],b1,b1+b2])
    if x0 is None:
        U=np.zeros((3,2,n)); U[1]+=np.outer(b1,step(ya)); U[2]+=np.outer(b1,step(ya))+np.outer(b2,step(yb))
        U-=U.mean(axis=0,keepdims=True)*0  # keep
        x0=np.concatenate([U[:,:,1:-1].ravel(),np.zeros(2)])
    def unpack(z):
        U=np.zeros((3,2,n)); U[:,:,1:-1]=z[:-2].reshape(3,2,n-2); s=z[-2:]
        U[:,:,-1]=s[None,:]+off; return U
    def EG(z):
        U=unpack(z); G=np.zeros_like(U); E=0.
        for k in range(3):
            dux=np.diff(U[k,0])/dy; duy=np.diff(U[k,1])/dy
            E+=dy*np.sum(0.5*mu*dux**2+0.5*Y*duy**2)
            for c,(dd,M) in enumerate([(dux,mu),(duy,Y)]):
                g=M*dd; G[k,c,1:]+=g; G[k,c,:-1]-=g
        for k in range(2):
            ddx=U[k+1,0]-U[k,0]; ddy=U[k+1,1]-U[k,1]
            rx=cp*ddx+sp*ddy; ry=-sp*ddx+cp*ddy; V,gxr,gyr=Vhex(rx,ry,V0); E+=dy*np.sum(V)
            gx=cp*gxr-sp*gyr; gy=sp*gxr+cp*gyr
            G[k+1,0]+=dy*gx; G[k+1,1]+=dy*gy; G[k,0]-=dy*gx; G[k,1]-=dy*gy
        # pins: projection of relative displacement at wall centres = half jump
        for (k,i,h,b) in [(0,ia,h1,b1),(1,ib,h2,b2)]:
            r=(U[k+1,:,i]-U[k,:,i])@h-0.5*np.linalg.norm(b); E+=0.5*Kp*r**2
            G[k+1,:,i]+=Kp*r*h; G[k,:,i]-=Kp*r*h
        gs=G[:,:,-1].sum(axis=0)
        return E,np.concatenate([G[:,:,1:-1].ravel(),gs])
    r=minimize(EG,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=50000,gtol=1e-11,ftol=1e-16,maxcor=30))
    return r.fun,r.x,unpack(r.x),y
def rot(v,a): c,s=np.cos(a),np.sin(a); return (c*v[0]-s*v[1],s*v[0]+c*v[1])
if __name__=="__main__":
    phi=float(sys.argv[1]); psi=float(sys.argv[2])
    # b1 = a1 in lattice frame; lattice rotated by psi in lab (wall along lab x)
    b1l=(1.,0.); b2l=rot(b1l,np.radians(phi))
    b1=rot(b1l,np.radians(psi)); b2=rot(b2l,np.radians(psi))
    Ds=[0,0.5,1,1.5,2,3,4,6,8,12]; out=[]
    for D in Ds:
        E,x0,U,y=energy_D(b1,b2,D,psi=np.radians(psi)); out.append([D,E])
        if D in (0,12): np.save(f'w3U_{int(phi)}_{int(psi)}_{D}.npy',U)
    print(f'phi={phi:.0f} psi={psi:.0f} Esep={out[-1][1]:.4f}', ' '.join(f'{D}:{E-out[-1][1]:+.4f}' for D,E in out),flush=True)
    json.dump(dict(phi=phi,psi=psi,E=out),open(f'w3_{int(phi)}_{int(psi)}.json','w'))
