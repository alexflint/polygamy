
import operator

def gcd(f, g, mod=operator.mod):
    while g != 0:
        f, g = g, mod(f, g)
    return f

def extended_gcd(f, g):
    r,rp = f,g
    s,sp = 1,0
    t,tp = 0,1
    while rp != 0:
        q = r // rp
        r,rp = rp, r - q*rp
        s,sp = sp, s - q*sp
        t,tp = tp, t - q*tp
    return r,s,t
