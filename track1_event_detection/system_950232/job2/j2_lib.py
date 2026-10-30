"""Job 2: does the paste extend beyond the template that matched it?

Two clips that fire the same template are aligned sample-exactly by their hit positions.
They share ONLY the pasted waveform, so the local correlation between them is high exactly
over the part of the master recording that both clips carry. If the generator cuts random
spans out of a long master, some clips carry more than the template holds and the shared
region reaches past the template edges.
"""
from __future__ import annotations
import numpy as np
import lib
SR=16000

def box(v,n):
    cs=np.concatenate([[0.0],np.cumsum(v,dtype=np.float64)])
    return cs[n:]-cs[:-n]

def coh(a,b,win_ms=24.0):
    n=int(round(win_ms*SR/1000))
    L=min(len(a),len(b))
    if L<n+2: return np.zeros(0,dtype=np.float32), n
    a=a[:L]; b=b[:L]
    num=box(a*b,n); den=np.sqrt(np.maximum(box(a*a,n)*box(b*b,n),1e-24))
    return (num/den).astype(np.float32), n

def accepted(H, clips, tau, dur):
    """clip -> [(template, onset seconds, ncc)] from the shipped greedy decode."""
    out={}
    for c in clips:
        cands=sorted(((v,p/SR,p/SR+dur[k],k) for k,p,v in H[c] if v>=tau[k]),reverse=True)
        acc=[]
        for v,on,off,k in cands:
            if all(off<=a or on>=b for _,a,b,_ in acc): acc.append((v,on,off,k))
        out[c]=sorted(((k,on,v) for v,on,off,k in acc), key=lambda r:r[1])
    return out

def pair_extent(xa, pa, xb, pb, dur, pad=1.0, thr=0.45, win_ms=24.0):
    """Shared extent of two clips aligned at their hit onsets, in seconds relative to the
    hit onset.  Returns (start, end) or None."""
    P=int(pad*SR); D=int(round(dur*SR))
    sa=int(round(pa*SR))-P; sb=int(round(pb*SR))-P
    ea=sa+2*P+D; eb=sb+2*P+D
    if sa<0 or sb<0 or ea>len(xa) or eb>len(xb): 
        lo=max(0-sa,0-sb,0); hi=min(len(xa)-sa,len(xb)-sb,2*P+D)
        if hi-lo<int(0.2*SR): return None
        a=xa[sa+lo:sa+hi]; b=xb[sb+lo:sb+hi]; off=lo
    else:
        a=xa[sa:ea]; b=xb[sb:eb]; off=0
    r,n=coh(a.astype(np.float64),b.astype(np.float64),win_ms)
    if len(r)==0: return None
    # sign from the interior
    i0=max(0,P-off+int(0.05*SR)); i1=min(len(r),P-off+D-int(0.05*SR))
    if i1-i0<5: return None
    s=np.sign(np.median(r[i0:i1])) or 1.0
    r=r*s
    good=r>=thr
    if not good[i0:i1].any(): return None
    # close short holes: a loud speech burst in one clip can dip the coherence for a moment
    hole=int(round(0.20*SR))
    g=good.copy()
    idx=np.flatnonzero(g)
    if len(idx)>1:
        d=np.diff(idx)
        for a0,gap in zip(idx[:-1],d):
            if 1<gap<=hole: g[a0:a0+gap]=True
    good=g
    # grow from the interior
    c=(i0+i1)//2
    j=c
    while j>0 and good[j-1]: j-=1
    k=c
    while k<len(good)-1 and good[k+1]: k+=1
    start=(off+j)/SR-pad; end=(off+k+n)/SR-pad
    return float(start), float(end), float(np.median(r[i0:i1]))
