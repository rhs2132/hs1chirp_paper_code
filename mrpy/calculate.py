
import mpmath as mp
import numpy as np

# General
def calc_f_M(M):
    # M assumed to be np array; handling the full array here is awkward
    if len(M.shape) == 1:
        return mp.mpmathify((M[0] + 1j*M[1])/(np.linalg.norm(M) + M[2]))
    else:
        return [mp.mpmathify((M[i, 0] + 1j*M[i, 1])/(np.linalg.norm(M[i, :]) + M[i, 2])) for i in range(M.shape[0])]
    
def calc_m_M(M):
    # M assumed to be np array; handling the full array here is awkward
    if len(M.shape) == 1:
        return mp.mpmathify(np.linalg.norm(M))
    else:
        return [mp.mpmathify(np.linalg.norm(M[i, :])) for i in range(M.shape[0])]
    
def calc_M_fm(f, m):
    # M returned as np array
    M = np.zeros((len(f), 3))

    M[:, 0] = [m[i]*mp.re(2*f[i])/(1+mp.fabs(f[i])**2) for i in range(len(f))]
    M[:, 1] = [m[i]*mp.im(2*f[i])/(1+mp.fabs(f[i])**2) for i in range(len(f))]
    M[:, 2] = [m[i]*(1-mp.fabs(f[i])**2)/(1+mp.fabs(f[i])**2) for i in range(len(f))]

    return M


# Chirp
def chirp_calc_A(R, Tp):
    # A in rad/s
    return R*mp.pi/Tp

def chirp_calc_a(A, Tp, w1m):
    return -1-1j*Tp*w1m**2/(8*A)

def chirp_calc_cf(tx, fx, a, df, A, tc, Tp, w1m):
    px = chirp_calc_p_t(tx=tx, A=A, df=df, tc=tc, Tp=Tp)
    const = (1j - 1)*mp.sqrt(2 * A/Tp)*chirp_calc_w1_t(tx=tx, A=A, tc=tc, Tp=Tp, w1m=w1m)/w1m**2

    fA0 = mp.hermite(a, px)
    fA1 = -2*px*fA0 + 2*a*mp.hermite(a-1, px)
    fB0 = mp.hyp1f1(-0.5*a, 0.5, px**2)
    fB1 = -2*px*fB0 - 2*a*px*mp.hyp1f1(1-0.5*a, 1.5, px**2)
    
    return -(fx*fA0 + const*fA1)/(fx*fB0 + const*fB1)

def chirp_calc_f(a, cf, A, df, tc, Tp, w1m, tx):
    px = chirp_calc_p_t(tx=tx, A=A, df=df, tc=tc, Tp=Tp)

    fA0 = mp.hermite(a, px)
    fA1 = -2*px*fA0 + 2*a*mp.hermite(a-1, px)
    fB0 = mp.hyp1f1(-0.5*a, 0.5, px**2)
    fB1 = -2*px*fB0 - 2*a*px*mp.hyp1f1(1-0.5*a, 1.5, px**2)

    return (1 - 1j)*mp.sqrt(2 * A/Tp)*chirp_calc_w1_t(tx=tx, A=A, tc=tc, Tp=Tp, w1m=w1m)/w1m**2*(fA1 + cf*fB1)/(fA0 + cf*fB0)

def chirp_calc_p_t(tx, A, df, tc, Tp):
    return mp.sqrt(1j*A/Tp)*(tx - tc + Tp*df/(2*A))

def chirp_calc_w1_t(tx, A, tc, Tp, w1m, phic=0):
    phi1 = phic + A/Tp*(tx - tc)**2
    return w1m*mp.exp(1j*phi1)

def chirp_calc_w1m(target_flip, Tp):
    return np.pi*target_flip/Tp

def chirp_parameterize_f(A, df, tc, Tp, w1m, tx, fx):
    a = chirp_calc_a(A=A, Tp=Tp, w1m=w1m)
    cf = chirp_calc_cf(tx=tx, fx=fx, a=a, df=df, A=A, tc=tc, Tp=Tp, w1m=w1m)

    return {"a": a, "cf": cf, "A": A, "df": df, "tc":tc, "Tp": Tp, "w1m": w1m}

# HS1
def hs1_calc_A(beta, R, Tp):
    # A in rad/s
    return R*mp.pi/(Tp*mp.tanh(beta))

def hs1_calc_abc(A, beta, df, Tp, w1m):
    t4b = Tp/(4*beta)
    swA = mp.sqrt(w1m**2 - A**2)

    a = t4b*(1j*A + swA)
    b = t4b*(1j*A - swA)
    c = 0.5 + 1j*t4b*(df + A)

    return a, b, c

def hs1_calc_abc_zhang(A, beta, df, Tp, w1m):
    at4b = A*Tp/(4*beta)
    swA = mp.sqrt((w1m/A)**2 - 1)

    a = at4b*(1j + swA)
    b = at4b*(1j - swA)
    c = 0.5 + 1j*at4b*(df/A + 1)

    return a, b, c

def hs1_calc_cf(px, fx, a, b, c, A, beta, phic, Tp, w1m):
    const = 1j*2*beta/(Tp*w1m**2)*hs1_calc_w1_p(px=px, A=A, beta=beta, phic=phic, Tp=Tp, w1m=w1m)
    ptoc = mp.power(px, -c)
    pto1c = mp.power(px, 1-c)

    if mp.almosteq(c-b, c, rel_eps=1e-8):
        f0 = 1.0
        f1 = mp.power(1-px, -1)
        f2 = mp.power(1-px, -(a-c+1))
        f3 = mp.power(1-px, -(a-c+2))
    else:
        f0 = mp.hyp2f1(a, b, c, px, zeroprec=100)
        f1 = mp.hyp2f1(a+1, b+1, c+1, px, zeroprec=100)
        f2 = mp.hyp2f1(a-c+1, b-c+1, 2-c, px, zeroprec=100)
        f3 = mp.hyp2f1(a-c+2, b-c+2, 3-c, px, zeroprec=100)

    fA0 = f0
    fA1 = a*b/c*f1
    fB0 = pto1c*f2
    fB1 = (1-c)*ptoc*f2 + (a-c+1)*(b-c+1)/(2-c)*pto1c*f3
    
    return -(fx*fA0 + const*fA1)/(fx*fB0 + const*fB1)

def hs1_calc_cf_zhang(f0, A, beta, c, df, Tp, w1m):
    phi0 = df*Tp/(4*beta)*mp.ln(0.5*(1 - mp.tanh(beta)))

    return 1j*w1m*Tp*f0*mp.expj(phi0)/(mp.power(2, 2+1j*A*Tp/(2*beta))*(1-c)*beta)

def hs1_calc_f(a, b, c, cf, A, beta, phic, tc, Tp, w1m, px):
    ptoc = mp.power(px, -c)
    pto1c = mp.power(px, 1-c)

    if mp.almosteq(c-b, c, rel_eps=1e-8):
        f0 = 1.0
        f1 = mp.power(1-px, -1)
        f2 = mp.power(1-px, -(a-c+1))
        f3 = mp.power(1-px, -(a-c+2))
    else:
        f0 = mp.hyp2f1(a, b, c, px, zeroprec=100)
        f1 = mp.hyp2f1(a+1, b+1, c+1, px, zeroprec=100)
        f2 = mp.hyp2f1(a-c+1, b-c+1, 2-c, px, zeroprec=100)
        f3 = mp.hyp2f1(a-c+2, b-c+2, 3-c, px, zeroprec=100)
    
    return -2j*beta/(Tp*w1m**2)*hs1_calc_w1_p(px=px, A=A, beta=beta, phic=phic, Tp=Tp, w1m=w1m)*(a*b/c*f1 + cf*(1-c)*ptoc*f2 + cf*(a-c+1)*(b-c+1)/(2-c)*pto1c*f3)/(f0 + cf*pto1c*f2)

def hs1_calc_p_t(tx, beta, tc, Tp):
    return 0.5*(1 + mp.tanh(2*beta/Tp*(tx - tc)))

def hs1_calc_t_p(px, beta, tc, Tp):
    return tc + 0.5*Tp*mp.atanh(2*px - 1)/beta

def hs1_calc_w1_p(px, A, beta, phic, Tp, w1m):
    return w1m*mp.power(4*px*(1-px), 0.5 + 1j*A*Tp/(4*beta))*mp.expj(phic)

def hs1_calc_w1_t(tx, A, beta, phic, Tp, w1m):
    return w1m*mp.power(mp.sech(beta*(2*tx/Tp - 1)), 1 + 1j*A*Tp/(2*beta))*mp.expj(phic)

def hs1_parameterize_f(A, beta, df, phic, tc, Tp, w1m, px, fx):
    a, b, c = hs1_calc_abc(A=A, beta=beta, df=df, Tp=Tp, w1m=w1m)
    cf = hs1_calc_cf(px=px, fx=fx, a=a, b=b, c=c, A=A, beta=beta, phic=phic, Tp=Tp, w1m=w1m)
    
    return {"a": a, "b":b, "c":c, "cf":cf, "A":A, "beta":beta, "df":df, "tc":tc, "Tp":Tp, "w1m":w1m}

def hs1_parameterize_f_zhang(A, beta, df, Tp, w1m, f0):
    try:
        a, b, c = hs1_calc_abc_zhang(A=A, beta=beta, df=df, Tp=Tp, w1m=w1m)
    except:
        print("zhang abc failed")
        a, b, c = hs1_calc_abc(A=A, beta=beta, df=df, Tp=Tp, w1m=w1m)

    cf = hs1_calc_cf_zhang(f0=f0, A=A, beta=beta, c=c, df=df, Tp=Tp, w1m=w1m)
    return {"a": a, "b":b, "c":c, "cf":cf, "A":A, "beta":beta, "df":df, "Tp":Tp, "w1m":w1m}

# square
def square_calc_cf(df, w1m, phic, tx, fx):
    dfw1m = mp.sqrt(df**2 + w1m**2)
    cf = mp.atanh((w1m*mp.expj(phic)*fx + df)/dfw1m) - 0.5j*dfw1m*tx
    return cf

def square_calc_f(cf, df, phic, w1m, tx):
    dfw1m = mp.sqrt(df**2 + w1m**2)
    fx = mp.expj(phic)/w1m*(-df + dfw1m*mp.tanh(cf + 0.5j*dfw1m*tx))
    return fx

# utilities
def angular_error_cos(Ma, Mb):
    return np.arccos(np.nanmin([np.dot(Ma, Mb)/(np.linalg.norm(Ma)*np.linalg.norm(Mb)), 1.0]))

def angular_error_sin(Ma, Mb):
    return np.arcsin(np.linalg.norm(np.cross(Ma, Mb))/(np.linalg.norm(Ma)*np.linalg.norm(Mb)))

# abs value errors
def component_error(Ma, Mb):
    x_e, x_ne = np.abs(Ma[0]) - np.abs(Mb[0]), (np.abs(Ma[0]) - np.abs(Mb[0]))/np.abs(Ma[0])
    y_e, y_ne = np.abs(Ma[1]) - np.abs(Mb[1]), (np.abs(Ma[1]) - np.abs(Mb[1]))/np.abs(Ma[1])
    xy_e, xy_ne = np.abs(np.linalg.norm(Ma[:2])) - np.abs(np.linalg.norm(Mb[:2])), (np.abs(np.linalg.norm(Ma[:2])) - np.abs(np.linalg.norm(Mb[:2])))/np.abs(np.linalg.norm(Ma[:2]))
    z_e, z_ne = np.abs(Ma[2]) - np.abs(Mb[2]), (np.abs(Ma[2]) - np.abs(Mb[2]))/np.abs(Ma[2])
    return x_e, x_ne, y_e, y_ne, xy_e, xy_ne, z_e, z_ne