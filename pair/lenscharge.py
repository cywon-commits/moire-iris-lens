import numpy as np, json
th=1/30.; Ac=np.sqrt(3)/2; C=4.; R=np.sqrt(C/th); m=R**2
def removed(d,sgn,L=400.,n=4001,rh=2.4):
    x=np.linspace(-L,L,n); X,Y=np.meshgrid(x,x,indexing='ij'); Zb=X-1j*Y; h=x[1]-x[0]
    fp=-m/(Zb+d/2)**2 - sgn*m/(Zb-d/2)**2
    a=np.minimum(1.,np.abs(fp)**2)
    core=(np.hypot(X+d/2,Y)<rh)|(np.hypot(X-d/2,Y)<rh); a[core]=1.   # core region: no ordinary nodes
    tail=0.
    return th**2/Ac*(a.sum()*h*h)
single=2*np.pi*th*C/Ac
ds=[0.,4.,8.,12.,16.,22.,30.,40.,56.,80.]
out={'single':single,'pm':[(d,removed(d,-1)) for d in ds],'pp':[(d,removed(d,+1)) for d in ds]}
print('single analytic',round(single,3))
for (d,a),(_,b) in zip(out['pm'],out['pp']): print(f'd={d:5.1f} d/R={d/R:4.2f}  removed(+,-)={a:5.3f}  removed(+,+)={b:5.3f}')
json.dump(out,open('lenscharge.json','w'))
