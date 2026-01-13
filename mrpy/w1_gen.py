import numpy as np
import scipy

# base functions

def erf(t, beta, tc, Tp):
    return scipy.special.erf(beta*2/Tp*(t - tc))

def gauss(t, beta, tc, Tp):
    return np.exp(-0.5*(beta*2/Tp*(t - tc))**2)

def sechn(t, beta, n, tc, Tp):
    return np.cosh(beta*(2/Tp*(t - tc))**n)**-1

def sechn2(t, beta, n, tc, Tp):
    return np.cosh(beta*(2/Tp*(t - tc))**n)**-2

def tanh(t, beta, tc, Tp):
    return np.tanh(beta*2/Tp*(t - tc))

# wrapper functions

def chirp_am(t, w1m):
    return w1m*np.ones(len(t))

def chirp_fm(t, A, tc, Tp):
    return 2*A/Tp*(t - tc)

def chirp_phi(t, A, tc, Tp, phi0=None, phic=None):
    e_phi = A/Tp*(t - tc)**2

    if phi0 is not None:
        phic = phi0 - e_phi[0]
        return e_phi + phic
    elif phic is not None:
        return e_phi + phic
    else:
        return e_phi

def gauss_am(t, beta, tc, Tp, w1m):
    return w1m*gauss(t=t, beta=beta, tc=tc, Tp=Tp)

def gauss_fm(t, A, beta, tc, Tp):
    return A*erf(t=t, beta=beta, tc=tc, Tp=Tp)/erf(t=tc+0.5*Tp, beta=beta, tc=tc, Tp=Tp)

def HS1_am(t, beta, tc, Tp, w1m):
    return w1m*sechn(t=t, beta=beta, n=1, tc=tc, Tp=Tp)

def HS1_fm(t, A, beta, tc, Tp):
    return A*tanh(t=t, beta=beta, tc=tc, Tp=Tp)/tanh(t=tc+0.5*Tp, beta=beta, tc=tc, Tp=Tp)

def HS1_phi(t, A, beta, tc, Tp, phi0=None, phic=None):
    # accepts either the phi offset (from 0 at tc) or an initial phi; otherwise phi=0 at tc

    e_phi = -A*Tp/(2*beta)*np.log(np.cosh(beta*2/Tp*(t - tc)))

    if phi0 is not None:
        phic = phi0 - e_phi[0]
        e_phi = e_phi + phic
        return e_phi
    elif phic is not None:
        return e_phi + phic
    else:
        return e_phi

def HSn_am(t, beta, n, tc, Tp, w1m):
    return w1m*sechn(t=t, beta=beta, n=n, tc=tc, Tp=Tp)

@np.vectorize
def HSn_fm(t, A, beta, n, tc, Tp):
    return A*scipy.integrate.quad(sechn2, a=tc, b=t, args=(beta, n, tc, Tp))[0]/scipy.integrate.quad(sechn2, a=tc, b=tc+0.5*Tp, args=(beta, n, tc, Tp))[0]

def integrate_phi(t, fm, tc, phi0=None, phic=None):
    # accepts either the phi offset (phic at tc) or an initial phi0 at t[0]; otherwise phi=0 at tc
    # requires tc to be in the sampled range; not necessary but maintains parallels to HS1 phi

    if phi0 is not None:
        return scipy.integrate.cumulative_simpson(y=-1*fm, x=t, initial=phi0)
    else:
        if (tc < t[0]) or (tc > t[-1]):
            print("tc not in the sampled domain; cannot integrate phi")
            return None
        
        i_pretc = np.where(t <= tc)
        unoffset_phi0 = -1*scipy.integrate.cumulative_simpson(y=-1*fm[i_pretc], x=t[i_pretc], initial=0.0)[-1]

        if phic is not None:
            return scipy.integrate.cumulative_simpson(y=-1*fm, x=t, initial=unoffset_phi0 + phic)
        else:
            return scipy.integrate.cumulative_simpson(y=-1*fm, x=t, initial=unoffset_phi0)
        
# complex final outputs

def chirp_w1(t, A, tc, Tp, w1m, phi0=None, phic=None):
    ch_am = chirp_am(t=t, w1m=w1m)
    ch_phi = chirp_phi(t=t, A=A, tc=tc, Tp=Tp, phi0=phi0, phic=phic)
    ch_w1 = ch_am*np.exp(1j*ch_phi)

    return ch_w1

def HS1_w1(t, A, beta, tc, Tp, w1m, phi0=None, phic=None):
    hs_am = HS1_am(t=t, beta=beta, tc=tc, Tp=Tp, w1m=w1m)
    hs_phi = HS1_phi(t=t, A=A, beta=beta, tc=tc, Tp=Tp, phi0=phi0, phic=phic)
    hs_w1 = hs_am*np.exp(1j*hs_phi)

    return hs_w1

def HSn_w1(t, A, beta, n, tc, Tp, w1m):
    hs_am = HSn_am(t=t, beta=beta, n=n, tc=tc, Tp=Tp, w1m=w1m)
    hs_fm = HSn_fm(t=t, A=A, beta=beta, n=n, tc=tc, Tp=Tp)
    hs_phi = integrate_phi(t=t, fm=hs_fm, tc=tc)
    hs_w1 = hs_am*np.exp(1j*hs_phi)

    return hs_w1