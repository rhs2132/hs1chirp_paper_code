import numpy as np

def bloch_rk_prealloc(M, M0, B, dt, nt, gamma, R1, R2):
    """
    Bloch simulation with rotation and relaxation for a given M0 and B; initial magnetization already stored in the first position

    Args:
        M (ndarray): (3,) float array of initial magnetization
        M0 (float): the equilibrium magnetization magnitude (G)
        B (ndarray): (nt, 3) float array of the external magnetic field (G)
        dt (float): time step interval (s)
        nt (int): number of time steps
        gamma (float): gyromagnetic ratio (rad/(G*s))
        R1 (float): 1/T1 constant (1/s)
        R2 (float): 1/T2 constant (1/s)
        
    Modifies in memory:
        M (ndarray): (nt, 3) float array of calculated magnetizations
    """

    for i in range(1, nt):
        k1 = dt * dMdt_full(Mt=M[i-1], Bt=B[i-1], gamma=gamma, R1=R1, R2=R2, M0=M0)
        k2 = dt * dMdt_full(Mt=M[i-1]+0.5*k1, Bt=0.5*(B[i-1]+B[i]), gamma=gamma, R1=R1, R2=R2, M0=M0)
        k3 = dt * dMdt_full(Mt=M[i-1]+0.5*k2, Bt=0.5*(B[i-1]+B[i]), gamma=gamma, R1=R1, R2=R2, M0=M0)
        k4 = dt * dMdt_full(Mt=M[i-1]+k3, Bt=B[i], gamma=gamma, R1=R1, R2=R2, M0=M0)
        M[i] = M[i-1] + (k1+2*k2+2*k3+k4)/6
        
def dMdt_full(Mt, Bt, gamma, R1, R2, M0) -> np.ndarray:
    return np.array([
        gamma * (Mt[1]*Bt[2] - Mt[2]*Bt[1]) - Mt[0]*R2,
        gamma * (Mt[2]*Bt[0] - Mt[0]*Bt[2]) - Mt[1]*R2,
        gamma * (Mt[0]*Bt[1] - Mt[1]*Bt[0]) - (Mt[2] - M0)*R1
    ])