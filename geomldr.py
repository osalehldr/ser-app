import sys
import numpy as np
CV={"H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"S":1.05,"CL":1.02,"BR":1.20}
def rd(p):
    L=open(p).read().splitlines(); n=int(L[0].split()[0])
    s=[l.split()[0] for l in L[2:2+n]]
    X=np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2+n]])
    return s,X
def dih(p0,p1,p2,p3):
    b0,b1,b2=p0-p1,p2-p1,p3-p2; b1=b1/np.linalg.norm(b1)
    v=b0-np.dot(b0,b1)*b1; w=b2-np.dot(b2,b1)*b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w)))
def nb(s,X,i,t=1.30):
    o=[]
    for j in range(len(s)):
        if j==i-1: continue
        r=np.linalg.norm(X[j]-X[i-1])
        rc=CV.get(s[i-1].upper(),0.76)+CV.get(s[j].upper(),0.76)
        if r<t*rc: o.append(j+1)
    return o
def pax(X,iR,n):
    B=np.array([(X[j-1]-X[iR-1])/np.linalg.norm(X[j-1]-X[iR-1]) for j in n])
    a=np.linalg.svd(B)[2][-1]
    if np.dot(a,-B.sum(axis=0))<0: a=-a
    return a/np.linalg.norm(a)
f=sys.argv[1]; iR,iC1,iA,iB=map(int,sys.argv[2:6])
s,X=rd(f); g=lambda i: X[i-1]
print("===",f,"===")
n=nb(s,X,iR)
print("radical atom",iR,s[iR-1],"neighbours",[(j,s[j-1]) for j in n])
ax=pax(X,iR,n); d=g(iR)+ax
ag=[np.degrees(np.arccos(np.clip(np.dot(ax,(g(j)-g(iR))/np.linalg.norm(g(j)-g(iR))),-1,1))) for j in n]
print("pyramidalization %5.1f deg (0=planar, 19.5=sp3)" % np.mean([abs(a-90) for a in ag]))
R=[]
for nm,ib in (("A C%d-C%d"%(iC1,iA),iA),("B C%d-C%d"%(iC1,iB),iB)):
    th=dih(d,g(iR),g(iC1),g(ib))
    dv=min(abs(abs(th)),abs(abs(th)-180.0))
    bv=g(ib)-g(iC1); bv=bv/np.linalg.norm(bv)
    di=np.degrees(np.arccos(np.clip(abs(np.dot(ax,bv)),0,1)))
    R.append((nm,th,dv,di,float(np.linalg.norm(g(ib)-g(iC1)))))
print("%-12s %9s %10s %9s %8s" % ("bond","dihedral","dev_align","ax/bond","r(C-C)"))
for r in R: print("%-12s %9.1f %10.1f %9.1f %8.3f" % r)
dl=R[0][4]-R[1][4]; dv=R[1][2]-R[0][2]; da=R[1][3]-R[0][3]
def V(x,t): return "no difference" if abs(x)<t else ("favors A" if x>0 else "favors B")
print("bond-length asym  %+.3f A   %s" % (dl,V(dl,0.005)))
print("dihedral adv      %+.1f deg %s" % (dv,V(dv,2.0)))
print("p-axis/bond adv   %+.1f deg %s" % (da,V(da,2.0)))
if abs(dv)<5 and abs(dl)<0.010 and abs(da)<5:
    print("FLAT: scaffold not differentiating these bonds. Redesign.")
else:
    print("REAL DIFFERENTIATION: TS search worthwhile.")
rA=float(np.linalg.norm(g(iA)-g(iC1)))
rB=float(np.linalg.norm(g(iB)-g(iC1)))
rC=float(np.linalg.norm(g(iA)-g(iB)))
rR=float(np.linalg.norm(g(iC1)-g(iR)))
print()
print("RING INTEGRITY CHECK")
print("  C%d-C%d %6.3f  (scission bond A)" % (iC1,iA,rA))
print("  C%d-C%d %6.3f  (scission bond B)" % (iC1,iB,rB))
print("  C%d-C%d %6.3f  (distal bond)" % (iA,iB,rC))
print("  C%d-C%d %6.3f  (radical to ring)" % (iR,iC1,rR))
if max(rA,rB,rC)>1.8:
    print("  *** RING ALREADY OPENED -- everything above is INVALID ***")
elif len(n)>3:
    print("  *** atom %d has %d heavy neighbours, not a radical -- WRONG INDEX ***" % (iR,len(nb)))
else:
    print("  INTACT -- analysis above is valid")
