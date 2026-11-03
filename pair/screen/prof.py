import sys,numpy as np
P='/home/claude/cywon-commits/moire-iris-lens/pair'; sys.path.insert(0,P); sys.path.insert(0,P+'/../hex')
from hexcore import Vhex
from pscan_run import place
th=1/30.
def prof(f,bg,alpha,pl='GEN',rb=np.arange(3,40,2.)):
    D=np.load(f+'.npz'); B=np.load(bg+'.npz'); x=D['x']; h=x[1]-x[0]; z0=place(pl,alpha)
    X,Y=np.meshgrid(x,x,indexing='ij'); r=np.hypot(X-z0.real,Y-z0.imag)
    def fields(E):
        ux,uy=E['ux'],E['uy']
        a=np.gradient(ux,h,axis=0); b=np.gradient(ux,h,axis=1); c=np.gradient(uy,h,axis=0); d=np.gradient(uy,h,axis=1)
        V,_,_=Vhex(ux,uy,0.005,np.sqrt(3)*0.005)
        return 0.5*(c-b), a+d, V
    w,dv,V=fields(D); wb,dvb,Vb=fields(B); core=D['core']
    out=[]
    for r0,r1 in zip(rb[:-1],rb[1:]):
        m=(r>=r0)&(r<r1)&~core
        out.append((0.5*(r0+r1),(w[m]-wb[m]).mean()/th,(dv[m]-dvb[m]).mean()/th,(V[m]<0.1*0.005).mean(),(Vb[m]<0.1*0.005).mean()))
    return np.array(out)
if __name__=='__main__':
    cases=[('tw K0.5 V.005','d_a90_V0.005_C8_p+0_GEN_L72','bg_a90_V0.005_L72',90),
           ('tw K0.5 V.02','d_a90_V0.02_C8_p+0_GEN_L72','bg_a90_V0.02_L72',90),
           ('tw K2 V.005','d_l1_a90_V0.005_C8_p+0_GEN_L72','bg_l1_a90_V0.005_L72',90),
           ('mm K0.5 V.005','d_l-0.5_a0_V0.005_C8_p+0_GEN_L72','bg_l-0.5_a0_V0.005_L72',0),
           ('mm K0.1 V.005','d_l-0.9_a0_V0.005_C8_p+0_GEN_L72','bg_l-0.9_a0_V0.005_L72',0)]
    for n,f,b,a in cases:
        o=prof(f,b,a); print(n); print('  r     ',' '.join('%5.0f'%v for v in o[:,0]))
        print('  dω/θ  ',' '.join('%5.2f'%v for v in o[:,1])); print('  ddiv/θ',' '.join('%5.2f'%v for v in o[:,2]))
        print('  lock  ',' '.join('%5.2f'%v for v in o[:,3])); print('  lockbg',' '.join('%5.2f'%v for v in o[:,4]))
