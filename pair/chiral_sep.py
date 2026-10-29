import json,glob,numpy as np
E={}
for f in glob.glob('*.json'):
    try: d=json.load(open(f))
    except Exception: continue
    if 'mode' not in d: continue
    E[(d['mode'],d['d'],d['V0'],d.get('alpha',90.),d.get('orient','x'),d.get('sym',False))]=d['E']
ds=[8.,16.,26.,40.,56.]
Ue={d:E[('pair',d,0.0,90.,'x',False)]-2*E[('R',d,0.0,90.,'x',False)] for d in ds}
bgC=E[('none',0.0,0.005,0.,'x',False)]; bgA=E[('none',0.0,0.005,0.,'x',True)]
print(f'background: chiral {bgC:.3f}, achiral {bgA:.3f}  -> chirality condensation {bgC-bgA:.3f} (box), per moire cell {((bgC-bgA)/ (144**2/(np.sqrt(3)/2*30**2))):.3f}')
rows=[]
for d in ds:
    pc=E[('pair',d,0.005,0.,'x',False)]; rc=E[('R',d,0.005,0.,'x',False)]; lc=E.get(('L',d,0.005,0.,'x',False),rc)
    pa=E[('pair',d,0.005,0.,'x',True)]; ra=E[('R',d,0.005,0.,'x',True)]
    Uc=pc-lc-rc+bgC-Ue[d]; Ua=pa-2*ra+bgA-Ue[d]; Uchi=Uc-Ua
    dE_pair=pc-pa; dE_single=rc-ra; dE_bg=bgC-bgA
    rows.append(dict(d=d,U_chiral=Uc,U_achiral=Ua,U_chirality=Uchi,dEchi_pair=dE_pair,dEchi_single=dE_single,dEchi_bg=dE_bg))
    print(f'd={d:4.0f} (d/R={d/np.sqrt(120):.2f}): U(full)={Uc:+.2f}   U(achiral, lens only)={Ua:+.2f}   chirality part={Uchi:+.2f}   |  E_chi gain: bg {dE_bg:+.2f}, single {dE_single:+.2f}, pair {dE_pair:+.2f}')
json.dump(rows,open('chiral_sep.json','w'))
