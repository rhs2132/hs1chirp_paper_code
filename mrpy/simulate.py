
import numpy as np

from . import bloch

def prep_pm_Beff(t, df, gamma, w1) -> np.ndarray:
    """
    create a 3x len(t) array with the complex components of w1 in x and y and the fictitious field from the off resonance in z

    Args:
        t (ndarray): (nt,) float array of the times at which simulation will take place (s)
        df (float): off resonance (w0 - wc) (rad/s)
        gamma (float): gyromagnetic ratio (rad/(G*s))
        w1 (ndarray): (nt,) complex array with x=real(w1) and y=imag(w1) (rad/s)

    Returns:
        Beff (ndarray): (nt, 3) float array holding the net magnetic field in the phase modulated frame (G)
    """

    Beff = np.zeros((t.shape[0], 3))
    Beff[:, 0] = np.real(w1)/gamma
    Beff[:, 1] = np.imag(w1)/gamma
    Beff[:, 2] = df/gamma

    return Beff

def prep_pm_Msim(t, M_a) -> np.ndarray:
    """
    create a 3x len(t) array with M_0 in the first position to hold the magnetization over time from a simulation

    Args:
        t (ndarray): (nt,) float array of the times at which simulation will take place (s)
        M_0 (ndarray): (3,) float array of the initial magnetization

    Returns:
        Msim (ndarray): (nt, 3) float array holding the magnetization over time during the simulation
    """

    Msim = np.zeros((t.shape[0], 3))
    Msim[0, :] = M_a

    return Msim

def run_pm_sim(t, df, gamma, M0, M_a, R1, R2, w1):
    """
    simulate the evolution of an isochromat at times t returning magnetization at all time points

    Args:
        t (ndarray): (nt,) float array of the times at which simulation will take place (s)
        df (float): off resonance (w0 - wc) (rad/s)
        gamma (float): gyromagnetic ratio (rad/(G*s))
        M0 (float): equilibrium magnitude of magnetization (i.e. 1.0)
        M_a (ndarray): (3,) float array of the initial magnetization
        R1 (float): 1/T1 constant (1/s)
        R2 (float): 1/T2 constant (1/s)
        w1 (ndarray): (nt,) complex array with x=real(w1) and y=imag(w1) (rad/s)     

    Returns:
        Msim (ndarray): (nt, 3) float array holding the simulated magnetization in the phase modulated frame (G)
    """

    dt = t[1] - t[0]
    nt = t.shape[0]

    if not np.isclose(np.diff(t), dt).all():
        print("Error: variable dt")
        return None
    if (R1 < 0) or (R2 < 0):
        print("Error: relaxation parameters must be non-negative")
        return None
    if w1.shape[0] != nt:
        print("Error: w1 different length than t")
        return None
    
    Beff = prep_pm_Beff(t=t, w1=w1, df=df, gamma=gamma)
    Msim = prep_pm_Msim(t=t, M_a=M_a)

    bloch.bloch_rk_prealloc(M=Msim, M0=M0, B=Beff, dt=dt, nt=nt, gamma=gamma, R1=R1, R2=R2)

    return Msim
    