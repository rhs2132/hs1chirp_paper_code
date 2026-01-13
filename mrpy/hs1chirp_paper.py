

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import mpmath as mp
import numpy as np
import pandas as pd
import seaborn as sns
from . import calculate, simulate, w1_gen

# calculations

def bir4_hs1_sim_v_calcs_compM(app, target_flip, n_steps):
    # total steps is 2*n_steps
    # total pulse duration is 2*Tp
    phic = np.pi + 0.5*target_flip

    app["dt"] = app["Tp"]/n_steps
    tcs = np.array([0.0, app["Tp"], 2*app["Tp"]])
    tas = np.array([0.0, 0.5*app["Tp"], 1.5*app["Tp"]])
    tbs = np.array([0.5*app["Tp"], 1.5*app["Tp"], 2.0*app["Tp"]])

    t = np.arange(tas[0], tbs[-1] + 0.5*app["dt"], app["dt"])
    comp_i = np.arange(t.shape[0])
    t_comp = t[comp_i]

    ints_i = []
    for i in range(tcs.shape[0]):
        ints_i.append(np.where(np.logical_and(t>=tas[i]-0.5*app["dt"], t<=tbs[i]+0.5*app["dt"]))[0])
    comps_i = []
    for i in range(tcs.shape[0]):
        comps_i.append(np.where(np.logical_and(t_comp>=tas[i]-0.5*app["dt"], t_comp<=tbs[i]+0.5*app["dt"]))[0])
    
    # later pulse occupies overlap
    w1 = np.zeros(t.shape[0], dtype=complex)
    for i in range(tcs.shape[0]):
        w1[ints_i[i]] = w1_gen.HS1_w1(t=t[ints_i[i]], A=app["A"], beta=app["beta"], tc=tcs[i], Tp=app["Tp"], w1m=app["w1m"])
        if i==1:
            w1[ints_i[i]] = np.exp(1j*phic)*w1[ints_i[i]]

    w1amn = np.zeros(t.shape[0], dtype=float)
    for i in range(tcs.shape[0]):
        w1amn[ints_i[i]] = w1_gen.HS1_am(t=t[ints_i[i]], beta=app["beta"], tc=tcs[i], Tp=app["Tp"], w1m=1.0)

    w1fmn = np.zeros(t.shape[0], dtype=float)
    for i in range(tcs.shape[0]):
        w1fmn[ints_i[i]] = w1_gen.HS1_fm(t=t[ints_i[i]], A=1.0, beta=app["beta"], tc=tcs[i], Tp=app["Tp"])

    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[comp_i, :]

    # calculation over 3 intervals
    fa_boundary = calculate.calc_f_M(M=app["M_a"])
    f_full = [fa_boundary]
    for i in range(tcs.shape[0]):
        phic_i = 0.0
        if i == 1:
            phic_i = phic

        t_int = t_comp[ints_i[i]]
        p_int = [calculate.hs1_calc_p_t(tx=tx, beta=app["beta"], tc=tcs[i], Tp=app["Tp"]) for tx in t_int]

        cpr = calculate.hs1_parameterize_f(A=app["A"], beta=app["beta"], df=app["df"], phic=phic_i, tc=tcs[i], Tp=app["Tp"], w1m=app["w1m"], px=p_int[0], fx=fa_boundary)
        cpr["tc"] = tcs[i]
        fxr = [calculate.hs1_calc_f(a=cpr["a"], b=cpr["b"], c=cpr["c"], cf=cpr["cf"], A=cpr["A"], beta=cpr["beta"], phic=phic_i, tc=cpr["tc"], Tp=cpr["Tp"], w1m=cpr["w1m"], px=px) for px in p_int]
        f_full.extend(fxr[1:])
        fa_boundary = fxr[-1]
    mconst = [app["M0"] for _ in range(t_comp.shape[0])]
    M_cr = calculate.calc_M_fm(f=f_full, m=mconst)

    return pd.DataFrame({
            "t": t_comp,
            "w1amn": w1amn,
            "w1fmn": w1fmn,
        }), pd.DataFrame({
            "t": t_comp,
            "Sim": np.abs(Msim[:, 0] + 1j*Msim[:, 1]),
            "Exact": np.abs(M_cr[:, 0] + 1j*M_cr[:, 1]),
        }), pd.DataFrame({
            "t": t_comp,
            "Sim": Msim[:, 2],
            "Exact": M_cr[:, 2],
        })


def bir4_hs1_sim_v_calcs_finale(app, target_flip, n_steps):
    # total steps is 2*n_steps
    # total pulse duration is 2*Tp
    # doesn't use a global phic
    phic_mid = np.pi + 0.5*target_flip

    app["dt"] = app["Tp"]/n_steps
    tcs = np.array([0.0, app["Tp"], 2*app["Tp"]])
    tas = np.array([0.0, 0.5*app["Tp"], 1.5*app["Tp"]])
    tbs = np.array([0.5*app["Tp"], 1.5*app["Tp"], 2.0*app["Tp"]])

    t = np.arange(tas[0], tbs[-1] + 0.5*app["dt"], app["dt"])
    comp_i = np.arange(t.shape[0])
    t_comp = t[comp_i]

    ints_i = []
    for i in range(tcs.shape[0]):
        ints_i.append(np.where(np.logical_and(t>=tas[i]-0.5*app["dt"], t<=tbs[i]+0.5*app["dt"]))[0])
    comps_i = []
    for i in range(tcs.shape[0]):
        comps_i.append(np.where(np.logical_and(t_comp>=tas[i]-0.5*app["dt"], t_comp<=tbs[i]+0.5*app["dt"]))[0])
    
    # later pulse occupies overlap
    w1 = np.zeros(t.shape[0], dtype=complex)
    for i in range(tcs.shape[0]):
        if i != 1:
            phic = 0.0
        else:
            phic = phic_mid
        w1[ints_i[i]] = w1_gen.HS1_w1(t=t[ints_i[i]], A=app["A"], beta=app["beta"], tc=tcs[i], Tp=app["Tp"], w1m=app["w1m"], phic=phic)

    w1amn = np.zeros(t.shape[0], dtype=float)
    for i in range(tcs.shape[0]):
        w1amn[ints_i[i]] = w1_gen.HS1_am(t=t[ints_i[i]], beta=app["beta"], tc=tcs[i], Tp=app["Tp"], w1m=1.0)

    w1fmn = np.zeros(t.shape[0], dtype=float)
    for i in range(tcs.shape[0]):
        w1fmn[ints_i[i]] = w1_gen.HS1_fm(t=t[ints_i[i]], A=1.0, beta=app["beta"], tc=tcs[i], Tp=app["Tp"])

    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[comp_i, :]

    # calculation over 3 intervals
    fa_boundary = calculate.calc_f_M(M=app["M_a"])
    f_full = [fa_boundary]
    for i in range(tcs.shape[0]):
        phic_i = 0.0
        if i == 1:
            phic_i = phic_mid

        # select only the first and last points of each boundary
        t_int = t_comp[ints_i[i]][[0, -1]]
        p_int = [calculate.hs1_calc_p_t(tx=tx, beta=app["beta"], tc=tcs[i], Tp=app["Tp"]) for tx in t_int]

        cpr = calculate.hs1_parameterize_f(A=app["A"], beta=app["beta"], df=app["df"], phic=phic_i, tc=tcs[i], Tp=app["Tp"], w1m=app["w1m"], px=p_int[0], fx=fa_boundary)
        cpr["tc"] = tcs[i]
        fxr = [calculate.hs1_calc_f(a=cpr["a"], b=cpr["b"], c=cpr["c"], cf=cpr["cf"], A=cpr["A"], beta=cpr["beta"], phic=phic_i, tc=cpr["tc"], Tp=cpr["Tp"], w1m=cpr["w1m"], px=px) for px in p_int]
        f_full.extend(fxr[1:])
        fa_boundary = fxr[-1]
        # print(f_full)
    mconst = [app["M0"] for _ in range(t_comp.shape[0])]
    M_cr = calculate.calc_M_fm(f=f_full, m=mconst)

    return calculate.angular_error_sin(Msim[-1], M_cr[-1])


def bir4_hs1_sim_v_calcs_finale_wrapper(app, target_flip, n_step_list, parameter, values):
    dflist = []
    for value in values:
        app[parameter] = value
        app["A"] = float(calculate.hs1_calc_A(beta=app["beta"], R=app["R"], Tp=app["Tp"]))

        errs_r = []
        for n_steps in n_step_list:
            # n_steps multiplied by 0.5 so that the number from the list is the total steps
            err_r = bir4_hs1_sim_v_calcs_finale(app=app, target_flip=target_flip, n_steps=0.5*n_steps)
            errs_r.append(err_r)
        dflist.append(pd.DataFrame({
            "Simulation Steps": n_step_list,
            "Method": "Exact",
            "label": "Exact {0} {1}".format(parameter, value),
            "Error (rad)": errs_r
        }))
    return pd.concat(dflist)


def chirp_sim_v_calcs_compM(app, comp_n, n_steps):
    app["dt"] = app["Tp"]/n_steps
    app["tc"] = 0.5*app["Tp"]
    app["ta"] = app["tc"] - 0.5*app["Tp"]
    app["tb"] = app["tc"] + 0.5*app["Tp"]

    t = np.arange(app["ta"], app["tb"] + 0.5*app["dt"], app["dt"])
    comp_i = np.round(np.linspace(0, t.shape[0]-1, comp_n)).astype(int)
    t_comp = t[comp_i]

    w1 = w1_gen.chirp_w1(t=t, A=app["A"], tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"])
    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[comp_i, :]

    fa = calculate.calc_f_M(M=app["M_a"])

    cpr = calculate.chirp_parameterize_f(A=app["A"], df=app["df"], tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], tx=t_comp[0], fx=fa)
    cpr["tc"] = app["tc"]
    fxr = [calculate.chirp_calc_f(a=cpr["a"], cf=cpr["cf"], A=cpr["A"], df=cpr["df"], tc=cpr["tc"], Tp=cpr["Tp"], w1m=cpr["w1m"], tx=tx) for tx in t_comp]
    mconst = [app["M0"] for _ in range(len(fxr))]
    M_cr = calculate.calc_M_fm(f=fxr, m=mconst)

    return pd.DataFrame({
            "t": t_comp,
            "Sim": np.abs(Msim[:, 0] + 1j*Msim[:, 1]),
            "Exact": np.abs(M_cr[:, 0] + 1j*M_cr[:, 1]),
        }), pd.DataFrame({
            "t": t_comp,
            "Sim": Msim[:, 2],
            "Exact": M_cr[:, 2],
        })


def chirp_sim_v_calcs_finale(app, n_steps):
    app["dt"] = app["Tp"]/n_steps
    app["tc"] = 0.5*app["Tp"]
    app["ta"] = app["tc"] - 0.5*app["Tp"]
    app["tb"] = app["tc"] + 0.5*app["Tp"]

    t = np.arange(app["ta"], app["tb"] + 0.5*app["dt"], app["dt"])
    t_comp = np.array([t[0], t[-1]])

    w1 = w1_gen.chirp_w1(t=t, A=app["A"], tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"])
    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[-1, :]

    fa = calculate.calc_f_M(M=app["M_a"])

    cpr = calculate.chirp_parameterize_f(A=app["A"], df=app["df"], tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], tx=t_comp[0], fx=fa)
    cpr["tc"] = app["tc"]
    fxr = [calculate.chirp_calc_f(a=cpr["a"], cf=cpr["cf"], A=cpr["A"], df=cpr["df"], tc=cpr["tc"], Tp=cpr["Tp"], w1m=cpr["w1m"], tx=tx) for tx in t_comp]

    mconst = [app["M0"] for _ in range(len(fxr))]
    M_cr = calculate.calc_M_fm(f=fxr, m=mconst)[-1]
    err_r = calculate.angular_error_sin(Msim, M_cr)

    return err_r


def chirp_finale_wrapper(app, n_step_list, parameter, values):
    dflist = []
    for value in values:
        app[parameter] = value
        app["A"] = float(calculate.chirp_calc_A(R=app["R"], Tp=app["Tp"]))

        errs_r = []
        for n_steps in n_step_list:
            err_r = chirp_sim_v_calcs_finale(app=app, n_steps=n_steps)
            errs_r.append(err_r)
        dflist.append(pd.DataFrame({
            "Simulation Steps": n_step_list,
            "Method": "Exact",
            "label": "Exact {0} {1}".format(parameter, value),
            "Error (rad)": errs_r
        }))
    return pd.concat(dflist)


def hs1_sim_v_calcs_compM(app, comp_n, n_steps):
    app["dt"] = app["Tp"]/n_steps
    app["tc"] = 0.5*app["Tp"]
    app["ta"] = app["tc"] - 0.5*app["Tp"]
    app["tb"] = app["tc"] + 0.5*app["Tp"]

    t = np.arange(app["ta"], app["tb"] + 0.5*app["dt"], app["dt"])
    comp_i = np.round(np.linspace(0, t.shape[0]-1, comp_n)).astype(int)
    t_comp = t[comp_i]

    w1 = w1_gen.HS1_w1(t=t, A=app["A"], beta=app["beta"], tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"])
    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[comp_i, :]

    p_comp = [calculate.hs1_calc_p_t(tx=tx, beta=app["beta"], tc=app["tc"], Tp=app["Tp"]) for tx in t_comp]
    fa = calculate.calc_f_M(M=app["M_a"])

    cpr = calculate.hs1_parameterize_f(A=app["A"], beta=app["beta"], df=app["df"], phic=0.0, tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], px=p_comp[0], fx=fa)
    cpr["tc"] = app["tc"]
    fxr = [calculate.hs1_calc_f(a=cpr["a"], b=cpr["b"], c=cpr["c"], cf=cpr["cf"], A=cpr["A"], beta=cpr["beta"], phic=0.0, tc=cpr["tc"], Tp=cpr["Tp"], w1m=cpr["w1m"], px=px) for px in p_comp]
    mconst = [app["M0"] for _ in range(len(fxr))]
    M_cr = calculate.calc_M_fm(f=fxr, m=mconst)

    cpz = calculate.hs1_parameterize_f_zhang(A=app["A"], beta=app["beta"], df=app["df"], Tp=app["Tp"], w1m=app["w1m"], f0=fa)
    cpz["tc"] = app["tc"]
    fxz = [calculate.hs1_calc_f(a=cpz["a"], b=cpz["b"], c=cpz["c"], cf=cpz["cf"], A=cpz["A"], beta=cpz["beta"], phic=0.0, tc=cpz["tc"], Tp=cpz["Tp"], w1m=cpz["w1m"], px=px) for px in p_comp]
    mconst = [app["M0"] for _ in range(len(fxz))]
    M_cz = calculate.calc_M_fm(f=fxz, m=mconst)

    return pd.DataFrame({
            "t": t_comp,
            "Sim": np.abs(Msim[:, 0] + 1j*Msim[:, 1]),
            "Zhang": np.abs(M_cz[:, 0] + 1j*M_cz[:, 1]),
            "Exact": np.abs(M_cr[:, 0] + 1j*M_cr[:, 1]),
        }), pd.DataFrame({
            "t": t_comp,
            "Sim": Msim[:, 2],
            "Zhang": M_cz[:, 2],
            "Exact": M_cr[:, 2],
        })


def hs1_sim_v_calcs_finale(app, n_steps, component_error=False):
    app["dt"] = app["Tp"]/n_steps
    app["tc"] = 0.5*app["Tp"]
    app["ta"] = app["tc"] - 0.5*app["Tp"]
    app["tb"] = app["tc"] + 0.5*app["Tp"]

    if "phic" not in app:
        app["phic"] = 0.0

    t = np.arange(app["ta"], app["tb"] + 0.5*app["dt"], app["dt"])
    t_comp = np.array([t[0], t[-1]])

    w1 = w1_gen.HS1_w1(t=t, A=app["A"], beta=app["beta"], tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], phic=app["phic"])
    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[-1, :]

    p_comp = [calculate.hs1_calc_p_t(tx=tx, beta=app["beta"], tc=app["tc"], Tp=app["Tp"]) for tx in t_comp]
    fa = calculate.calc_f_M(M=app["M_a"])

    cpr = calculate.hs1_parameterize_f(A=app["A"], beta=app["beta"], df=app["df"], phic=app["phic"], tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], px=p_comp[0], fx=fa)
    cpr["tc"] = app["tc"]
    fxr = [calculate.hs1_calc_f(a=cpr["a"], b=cpr["b"], c=cpr["c"], cf=cpr["cf"], A=cpr["A"], beta=cpr["beta"], phic=app["phic"], tc=cpr["tc"], Tp=cpr["Tp"], w1m=cpr["w1m"], px=px) for px in p_comp]
    mconst = [app["M0"] for _ in range(len(fxr))]
    M_cr = calculate.calc_M_fm(f=fxr, m=mconst)[-1]
    err_r = calculate.angular_error_sin(Msim, M_cr)

    cpz = calculate.hs1_parameterize_f_zhang(A=app["A"], beta=app["beta"], df=app["df"], Tp=app["Tp"], w1m=app["w1m"], f0=fa)
    cpz["tc"] = app["tc"]
    fxz = [calculate.hs1_calc_f(a=cpz["a"], b=cpz["b"], c=cpz["c"], cf=cpz["cf"], A=cpz["A"], beta=cpz["beta"], phic=app["phic"], tc=cpz["tc"], Tp=cpz["Tp"], w1m=cpz["w1m"], px=px) for px in p_comp]
    mconst = [app["M0"] for _ in range(len(fxz))]
    M_cz = calculate.calc_M_fm(f=fxz, m=mconst)[-1]
    err_z = calculate.angular_error_sin(Msim, M_cz)

    if not component_error:
        return err_r, err_z
    else:
        # only return zhang results; bespoke for figure
        return (calculate.component_error(Msim, M_cz), err_z), (calculate.component_error(Msim, M_cr), err_r)


def hs1_finale_wrapper(app, n_step_list, parameter, values):
    dflist = []
    for value in values:
        app[parameter] = value
        app["A"] = float(calculate.hs1_calc_A(beta=app["beta"], R=app["R"], Tp=app["Tp"]))

        errs_r = []
        errs_z = []
        for n_steps in n_step_list:
            err_r, err_z = hs1_sim_v_calcs_finale(app=app, n_steps=n_steps)
            errs_r.append(err_r)
            errs_z.append(err_z)
        dflist.append(pd.DataFrame({
            "Simulation Steps": n_step_list,
            "Method": "Exact",
            "label": "Exact {0} {1}".format(parameter, value),
            "Error (rad)": errs_r
        }))
        dflist.append(pd.DataFrame({
            "Simulation Steps": n_step_list,
            "Method": "Zhang",
            "label": "Zhang {0} {1}".format(parameter, value),
            "Error (rad)": errs_z
        }))
    return pd.concat(dflist)

def hs1_finale2_wrapper(app, n_steps, thetas):
    zerrs_x = []
    zerrs_y = []
    zerrs_xy = []
    zerrs_z = []
    zerrsn_x = []
    zerrsn_y = []
    zerrsn_xy = []
    zerrsn_z = []
    zerrs_angle = []

    rerrs_x = []
    rerrs_y = []
    rerrs_xy = []
    rerrs_z = []
    rerrsn_x = []
    rerrsn_y = []
    rerrsn_xy = []
    rerrsn_z = []
    rerrs_angle = []

    for theta in thetas:
        app["M_a"] = np.array([0, np.sin(theta), np.cos(theta)])

        ((zerr_x, zerrn_x, zerr_y, zerrn_y, zerr_xy, zerrn_xy, zerr_z, zerrn_z), zerr_angle), ((rerr_x, rerrn_x, rerr_y, rerrn_y, rerr_xy, rerrn_xy, rerr_z, rerrn_z), rerr_angle) = hs1_sim_v_calcs_finale(app=app, n_steps=n_steps, component_error=True)
        zerrs_x.append(zerr_x)
        zerrsn_x.append(zerrn_x)
        zerrs_y.append(zerr_y)
        zerrsn_y.append(zerrn_y)
        zerrs_xy.append(zerr_xy)
        zerrsn_xy.append(zerrn_xy)
        zerrs_z.append(zerr_z)
        zerrsn_z.append(zerrn_z)
        zerrs_angle.append(zerr_angle)

        rerrs_x.append(rerr_x)
        rerrsn_x.append(rerrn_x)
        rerrs_y.append(rerr_y)
        rerrsn_y.append(rerrn_y)
        rerrs_xy.append(rerr_xy)
        rerrsn_xy.append(rerrn_xy)
        rerrs_z.append(rerr_z)
        rerrsn_z.append(rerrn_z)
        rerrs_angle.append(rerr_angle)

    return pd.DataFrame({
        "theta": thetas,
        "zx": zerrs_x,
        "zxn": zerrsn_x,
        "zy": zerrs_y,
        "zyn": zerrsn_y,
        "zxy": zerrs_xy,
        "zxyn": zerrsn_xy,
        "zz": zerrs_z,
        "zzn": zerrsn_z,
        "za": zerrs_angle,
        "rx": rerrs_x,
        "rxn": rerrsn_x,
        "ry": rerrs_y,
        "ryn": rerrsn_y,
        "rxy": rerrs_xy,
        "rxyn": rerrsn_xy,
        "rz": rerrs_z,
        "rzn": rerrsn_z,
        "ra": rerrs_angle,
    })

def hs1_SIT_compM(app, comp_n, n_steps):
    # requires df=0, A=0
    if app["df"] != 0 or app["A"] != 0:
        print("outside of test case")
        return
    
    # replace w1m
    app["w1m"] = 2*np.pi*app["beta"]/(app["Tp"]*np.arctan(np.exp(app["beta"])))

    app["dt"] = app["Tp"]/n_steps
    app["tc"] = 0.5*app["Tp"]
    app["ta"] = app["tc"] - 0.5*app["Tp"]
    app["tb"] = app["tc"] + 0.5*app["Tp"]

    if "phic" not in app:
        app["phic"] = 0.0

    t = np.arange(app["ta"], app["tb"] + 0.5*app["dt"], app["dt"])
    comp_i = np.round(np.linspace(0, t.shape[0]-1, comp_n)).astype(int)
    t_comp = t[comp_i]

    w1 = w1_gen.HS1_w1(t=t, A=app["A"], beta=app["beta"], tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], phic=app["phic"])
    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[comp_i, :]

    p_comp = [calculate.hs1_calc_p_t(tx=tx, beta=app["beta"], tc=app["tc"], Tp=app["Tp"]) for tx in t_comp]
    fa = calculate.calc_f_M(M=app["M_a"])

    cpr = calculate.hs1_parameterize_f(A=app["A"], beta=app["beta"], df=app["df"], phic=0.0, tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], px=p_comp[0], fx=fa)
    cpr["tc"] = app["tc"]
    fxr = [calculate.hs1_calc_f(a=cpr["a"], b=cpr["b"], c=cpr["c"], cf=cpr["cf"], A=cpr["A"], beta=cpr["beta"], phic=0.0, tc=cpr["tc"], Tp=cpr["Tp"], w1m=cpr["w1m"], px=px) for px in p_comp]
    mconst = [app["M0"] for _ in range(len(fxr))]
    M_cr = calculate.calc_M_fm(f=fxr, m=mconst)

    sit_a = app["Tp"]*app["w1m"]/(4*app["beta"])
    fsit = [1j*mp.tan(2*sit_a*mp.asin(mp.sqrt(px))) for px in p_comp]
    M_sit = calculate.calc_M_fm(f=fsit, m=mconst)

    return pd.DataFrame({
            "t": t_comp,
            "Sim": np.abs(Msim[:, 0] + 1j*Msim[:, 1]),
            "SIT": np.abs(M_sit[:, 0] + 1j*M_sit[:, 1]),
            "Exact": np.abs(M_cr[:, 0] + 1j*M_cr[:, 1]),
        }), pd.DataFrame({
            "t": t_comp,
            "Sim": Msim[:, 2],
            "SIT": M_sit[:, 2],
            "Exact": M_cr[:, 2],
        })

def square_sim_v_calcs_compM(app, comp_n, n_steps):
    # times set as start at ta with duration Tp (for simulation purposes)
    app["dt"] = app["Tp"]/n_steps
    app["tb"] = app["ta"] + app["Tp"]
    app["tc"] = app["ta"] + 0.5*app["Tp"]

    if "phic" not in app:
        app["phic"] = 0.0

    t = np.arange(app["ta"], app["tb"] + 0.5*app["dt"], app["dt"])
    comp_i = np.round(np.linspace(0, t.shape[0]-1, comp_n)).astype(int)
    t_comp = t[comp_i]

    w1 = w1_gen.chirp_w1(t=t, A=0, tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], phic=app["phic"])
    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[comp_i, :]
    fa = calculate.calc_f_M(M=app["M_a"])

    cf = calculate.square_calc_cf(df=app["df"], w1m=app["w1m"], phic=app["phic"], tx=app["ta"], fx=fa)
    fxr = [calculate.square_calc_f(cf=cf, df=app["df"], phic=app["phic"], w1m=app["w1m"], tx=tx) for tx in t_comp]
    mconst = [app["M0"] for _ in range(len(fxr))]
    M_cr = calculate.calc_M_fm(f=fxr, m=mconst)

    return pd.DataFrame({
            "t": t_comp,
            "Sim": np.abs(Msim[:, 0] + 1j*Msim[:, 1]),
            "Exact": np.abs(M_cr[:, 0] + 1j*M_cr[:, 1]),
        }), pd.DataFrame({
            "t": t_comp,
            "Sim": Msim[:, 2],
            "Exact": M_cr[:, 2],
        })

def square_sim_v_calcs_finale(app, n_steps):
    # times set as start at ta with duration Tp (for simulation purposes)
    app["dt"] = app["Tp"]/n_steps
    app["tb"] = app["ta"] + app["Tp"]
    app["tc"] = app["ta"] + 0.5*app["Tp"]

    if "phic" not in app:
        app["phic"] = 0.0

    t = np.arange(app["ta"], app["tb"] + 0.5*app["dt"], app["dt"])
    t_comp = np.array([t[0], t[-1]])

    w1 = w1_gen.chirp_w1(t=t, A=0, tc=app["tc"], Tp=app["Tp"], w1m=app["w1m"], phic=app["phic"])
    Msim = simulate.run_pm_sim(t=t, M0=app["M0"], M_a=app["M_a"], df=app["df"], gamma=app["gamma"], R1=app["R1"], R2=app["R2"], w1=w1)[-1, :]
    fa = calculate.calc_f_M(M=app["M_a"])

    cf = calculate.square_calc_cf(df=app["df"], w1m=app["w1m"], phic=app["phic"], tx=app["ta"], fx=fa)
    fxr = [calculate.square_calc_f(cf=cf, df=app["df"], phic=app["phic"], w1m=app["w1m"], tx=tx) for tx in t_comp]
    mconst = [app["M0"] for _ in range(len(fxr))]
    M_cr = calculate.calc_M_fm(f=fxr, m=mconst)[-1]

    err_r = calculate.angular_error_sin(Msim, M_cr)

    return err_r

def square_finale_wrapper(app, n_step_list, parameter, values):
    dflist = []
    for value in values:
        app[parameter] = value

        errs_r = []
        for n_steps in n_step_list:
            err_r = square_sim_v_calcs_finale(app=app, n_steps=n_steps)
            errs_r.append(err_r)
        dflist.append(pd.DataFrame({
            "Simulation Steps": n_step_list,
            "Method": "Exact",
            "label": "Exact {0} {1}".format(parameter, value),
            "Error (rad)": errs_r
        }))
    return pd.concat(dflist)

# plotting

def plot_hs1chirp_pulses(palette):
    legend_line_mult = 1.5

    pulse_params = None
    pulse_params = {
        "A": 1.0,
        "tc": 1.0e-3,
        "Tp": 2.0e-3,
        "w1m": 1.0
    }
    n_steps = 200
    pulse_params["dt"] = pulse_params["Tp"]/n_steps
    pulse_params["ta"] = pulse_params["tc"] - 0.5*pulse_params["Tp"]
    pulse_params["tb"] = pulse_params["tc"] + 0.5*pulse_params["Tp"]
    t = np.arange(pulse_params["ta"], pulse_params["tb"] + 0.5*pulse_params["dt"], pulse_params["dt"])

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(2, 2, hspace=0.1, wspace=0.1)
    sns.set_palette(palette)

    x_limexp = 0.05
    x_lims = [pulse_params["tc"] - (0.5 + x_limexp)*pulse_params["Tp"], pulse_params["tc"] + (0.5 + x_limexp)*pulse_params["Tp"]]
    x_ticks = [pulse_params["ta"], pulse_params["tc"], pulse_params["tb"]]
    x_ticklabels = [r"$t_c\!-\!T_p/2$", r"$t_c$", r"$t_c\!+\!T_p/2$"]

    y_limexp = 0.05
    y_lims_am = [-y_limexp, (1 + y_limexp)*pulse_params["w1m"]]
    y_ticks_am = [0.0, 1.0]
    y_ticklabels_am = [0, r"$\omega_1^m$"]
    # y_ticklabels_am = [0, r"$\frac{\omega_1^m}{2}$", r"$\omega_1^m$"]
    y_axislabel_am = r"$\omega_\text{AM}$"

    y_lims_fm = [-(1 + y_limexp)*pulse_params["A"], (1 + y_limexp)*pulse_params["A"]]
    y_ticks_fm = [-pulse_params["A"], 0.0, pulse_params["A"]]
    y_ticklabels_fm = [r"$-A$", r"$0$", r"$A$"]

    y_labelpos_a = -0.15
    y_labelpos_b = -0.15

    # 1a: HS1 AM and FM at 3 truncations
    df1a1 = pd.DataFrame({
        "t": t,
        r"$0.01$": w1_gen.HS1_am(t=t, beta=5.298, tc=pulse_params["tc"], Tp=pulse_params["Tp"], w1m=pulse_params["w1m"]),
        r"$0.10$": w1_gen.HS1_am(t=t, beta=2.993, tc=pulse_params["tc"], Tp=pulse_params["Tp"], w1m=pulse_params["w1m"]),
        r"$0.25$": w1_gen.HS1_am(t=t, beta=2.063, tc=pulse_params["tc"], Tp=pulse_params["Tp"], w1m=pulse_params["w1m"]),
    })
    ax = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=pd.melt(df1a1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_title("HS1")
    ax.set_xlabel(None)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel(y_axislabel_am)
    ax.yaxis.set_label_coords(y_labelpos_a, 0.5)
    ax.set_ylim(y_lims_am)
    ax.set_yticks(y_ticks_am)
    ax.set_yticklabels(y_ticklabels_am)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    df1a2 = pd.DataFrame({
        "t": t,
        r"$0.01$": w1_gen.HS1_fm(t=t, A=pulse_params["A"], beta=5.298, tc=pulse_params["tc"], Tp=pulse_params["Tp"]),
        r"$0.10$": w1_gen.HS1_fm(t=t, A=pulse_params["A"], beta=2.993, tc=pulse_params["tc"], Tp=pulse_params["Tp"]),
        r"$0.25$": w1_gen.HS1_fm(t=t, A=pulse_params["A"], beta=2.063, tc=pulse_params["tc"], Tp=pulse_params["Tp"]),
    })
    ax = fig.add_subplot(gs[1, 0])
    sns.lineplot(data=pd.melt(df1a2, ['t']), x="t", y="value", hue='variable', ax=ax)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    xticklabels = ax.get_xticklabels()
    xticklabels[0].set_horizontalalignment("left")
    xticklabels[2].set_horizontalalignment("right")
    ax.set_ylabel(r"$\omega_\text{FM}$")
    ax.yaxis.set_label_coords(y_labelpos_a, 0.5)
    ax.set_ylim(y_lims_fm)
    ax.set_yticks(y_ticks_fm)
    ax.set_yticklabels(y_ticklabels_fm)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    legend = ax.legend(loc="upper left", title=r"$\operatorname{sech}\beta$", frameon=False)
    for line in legend.get_lines():
        line.set_linewidth(legend_line_mult*line.get_linewidth())

    # 1b: chirp AM and FM
    df1b1 = pd.DataFrame({
        "t": t,
        r"chirp": w1_gen.chirp_am(t=t, w1m=pulse_params["w1m"]),
    })
    ax = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=pd.melt(df1b1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_title("Chirp")
    ax.set_xlabel(None)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel(None)
    ax.yaxis.set_label_coords(y_labelpos_b, 0.5)
    ax.set_ylim(y_lims_am)
    ax.set_yticks(y_ticks_am)
    ax.set_yticklabels([])
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    df1b2 = pd.DataFrame({
        "t": t,
        r"chirp": w1_gen.chirp_fm(t=t, A=pulse_params["A"], tc=pulse_params["tc"], Tp=pulse_params["Tp"]),
    })
    ax = fig.add_subplot(gs[1, 1])
    sns.lineplot(data=pd.melt(df1b2, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    xticklabels = ax.get_xticklabels()
    xticklabels[0].set_horizontalalignment("left")
    xticklabels[2].set_horizontalalignment("right")
    ax.set_ylabel(None)
    ax.yaxis.set_label_coords(y_labelpos_b, 0.5)
    ax.set_ylim(y_lims_fm)
    ax.set_yticks(y_ticks_fm)
    ax.set_yticklabels([])
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def plot_7panel_comparison(app, dfa1, dfa2, dfb1, dfb2, dfc1, dfc2, dfd, palettes, labels, legend_pos="lower center"):
    # fig 2 plotting

    fig = plt.figure(figsize=(8, 4))
    # gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.3)
    gs = gridspec.GridSpec(2, 4, hspace=0.05, wspace=0.1)

    if "ta" not in app:
        app["ta"] = 0.0

    apt = {
        "tc": app["ta"] + 0.5*app["Tp"],
        "ta": app["ta"],
        "tb": app["ta"] + app["Tp"],
        "Tp": app["Tp"]
    }

    #for a-c; d is bespoke
    x_limexp = 0.05
    x_lims = [apt["tc"] - (0.5 + x_limexp)*apt["Tp"], apt["tc"] + (0.5 + x_limexp)*apt["Tp"]]
    x_ticks = [apt["ta"], apt["tc"], apt["tb"]]
    x_ticklabels = [1000*x for x in x_ticks]
    x_axislabel = r"$t$ (ms)"

    y_limexp = 0.05
    y_lims_xy = [-y_limexp, (1 + y_limexp)]
    y_ticks_xy = [0.0, 1.0]
    y_ticklabels_xy = [0, 1]
    y_axislabel_xy = r"$|M_{xy}|/|M|$"

    y_lims_z = [-(1 + y_limexp), (1 + y_limexp)]
    y_ticks_z = [-1, 0, 1]
    y_ticklabels_z = [-1, 0, 1]
    y_axislabel_z = r"$M_{z}/|M|$"

    x_labelpos = -0.15
    y_labelpos = -0.15

    legend_line_mult = 1.5

    # panel a
    sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    ax.set_title(labels[0])
    ax.set_xlabel("")
    # ax.set_xlabel(x_axislabel)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel(y_axislabel_xy) #, labelpad=-1)
    ax.set_ylim(y_lims_xy)
    ax.set_yticks(y_ticks_xy)
    ax.set_yticklabels(y_ticklabels_xy)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.get_legend().set_title(None)
    ax.get_legend().set_frame_on(False)
    ax.get_legend().set_loc(legend_pos)
    for line in ax.get_legend().get_lines():
        line.set_linewidth(legend_line_mult*line.get_linewidth())

    ax = fig.add_subplot(gs[1, 0])
    sns.lineplot(data=pd.melt(dfa2, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_xlabel(x_axislabel)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.xaxis.set_label_coords(0.5, x_labelpos)
    ax.set_ylabel(y_axislabel_z)
    ax.set_ylim(y_lims_z)
    ax.set_yticks(y_ticks_z)
    ax.set_yticklabels(y_ticklabels_z)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)

    # panel b
    sns.set_palette(palettes[1])
    ax = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=pd.melt(dfb1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    ax.set_title(labels[1])
    ax.set_xlabel("")
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel("")
    ax.set_ylim(y_lims_xy)
    ax.set_yticks(y_ticks_xy)
    ax.set_yticklabels([])
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.get_legend().set_title(None)
    ax.get_legend().set_frame_on(False)
    ax.get_legend().set_loc(legend_pos)
    for line in ax.get_legend().get_lines():
        line.set_linewidth(legend_line_mult*line.get_linewidth())

    ax = fig.add_subplot(gs[1, 1])
    sns.lineplot(data=pd.melt(dfb2, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_xlabel(x_axislabel)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.xaxis.set_label_coords(0.5, x_labelpos)
    ax.set_ylabel("")
    ax.set_ylim(y_lims_z)
    ax.set_yticks(y_ticks_z)
    ax.set_yticklabels([])
    ax.yaxis.set_label_coords(y_labelpos, 0.5)

    # panel c
    sns.set_palette(palettes[2])
    ax = fig.add_subplot(gs[0, 2])
    sns.lineplot(data=pd.melt(dfc1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    ax.set_title(labels[2])
    ax.set_xlabel("")
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel("")
    ax.set_ylim(y_lims_xy)
    ax.set_yticks(y_ticks_xy)
    ax.set_yticklabels([])
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.get_legend().set_title(None)
    ax.get_legend().set_frame_on(False)
    ax.get_legend().set_loc(legend_pos)
    for line in ax.get_legend().get_lines():
        line.set_linewidth(legend_line_mult*line.get_linewidth())

    ax = fig.add_subplot(gs[1, 2])
    sns.lineplot(data=pd.melt(dfc2, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_xlabel(x_axislabel)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.xaxis.set_label_coords(0.5, x_labelpos)
    ax.set_ylabel("")
    ax.set_ylim(y_lims_z)
    ax.set_yticks(y_ticks_z)
    ax.set_yticklabels([])
    ax.yaxis.set_label_coords(y_labelpos, 0.5)

    if dfd is None:
        print("skipping plot d")
        return
    
    # panel d
    sns.set_palette(palettes[3])
    ax = fig.add_subplot(gs[0:2, 3])
    sns.lineplot(dfd, x="Simulation Steps", y="Error (rad)", hue="label", marker="o", markevery=[2], ax=ax, legend=False)
    ax.set_xlabel("Simulation Steps")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs=[1e3, 1e5, 1e7]))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(locs=[1e4, 1e6]))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(""))
    ax.set_xscale("log")
    ax.xaxis.set_label_coords(0.5, -0.07)
    ax.set_ylabel("Error (rad)", labelpad=-2)
    ax.yaxis.set_major_locator(ticker.FixedLocator(locs=[1e-12, 1e-9, 1e-6, 1e-3, 1e0]))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(locs=[1e-13, 1e-11, 1e-10, 1e-8, 1e-7, 1e-5, 1e-4, 1e-2, 1e-1]))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(""))
    ax.set_yscale("log")


def plot_9panel_comparison(app, dfa0, dfa1, dfa2, dfb0, dfb1, dfb2, dfc, palettes, labels):
    print(palettes)
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(6, 8, hspace=0.2, wspace=0.2)
    # gs = gridspec.GridSpec(6, 8, hspace=0.2, wspace=0.8)

    apt = {
        "tc": 0.5*app["Tp"],
        "ta": 0,
        "tb": app["Tp"],
        "Tp": app["Tp"],
    }

    x_limexp = 0.05
    x_lims = [apt["tc"] - (0.5 + x_limexp)*apt["Tp"], apt["tb"] + apt["tc"] + (0.5 + x_limexp)*apt["Tp"]]
    x_ticks = [apt["ta"], apt["tc"], apt["tb"], apt["tb"] + apt["tc"], 2*apt["tb"]]
    x_ticklabels = [0, 1, 2, 3, 4]
    x_axislabel = r"$t$ (ms)"

    y_ticklabels_w1am = [0, r"$\omega_1^m$"]
    y_axislabel_w1am = r"$\omega_\text{AM}$"

    y_ticklabels_w1fm = [r"$-A$", r"$0$", r"$A$"]
    y_axislabel_w1fm = r"$\omega_\text{FM}$"

    y_limexp = 0.05
    y_lims_xy = [-y_limexp, (1 + y_limexp)]
    y_ticks_xy = [0.0, 1.0]
    y_ticklabels_xy = [0, 1]
    y_axislabel_xy = r"$|M_{xy}|/|M|$"

    y_lims_z = [-(1 + y_limexp), (1 + y_limexp)]
    y_ticks_z = [-1, 0, 1]
    y_ticklabels_z = [-1, 0, 1]
    y_axislabel_z = r"$M_{z}/|M|$"

    x_labelpos = -0.15
    y_labelpos = -0.1 # -0.08 #-0.15

    sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[0, 0:3])
    ax.fill_between(dfa0["t"], dfa0["w1amn"])
    ax.set_title(labels[0])
    ax.set_xlabel(None)
    ax.set_xlim(x_lims)
    ax.set_xticks([])
    ax.set_xticklabels("")
    ax.set_ylabel(y_axislabel_w1am) #, labelpad=-1)
    ax.set_ylim(y_lims_xy)
    ax.set_yticks(y_ticks_xy)
    ax.set_yticklabels(y_ticklabels_w1am)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax = fig.add_subplot(gs[1, 0:3])
    ax.fill_between(dfa0["t"], dfa0["w1fmn"])
    ax.set_xlabel(None)
    ax.set_xlim(x_lims)
    ax.set_xticks([])
    ax.set_xticklabels("")
    ax.set_ylabel(y_axislabel_w1fm) #, labelpad=-1)
    ax.set_ylim(y_lims_z)
    ax.set_yticks(y_ticks_z)
    ax.set_yticklabels(y_ticklabels_w1fm)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax = fig.add_subplot(gs[2:4, 0:3])
    sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_xlabel(x_axislabel)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels("")
    ax.xaxis.set_label_coords(0.5, x_labelpos)
    ax.set_ylabel(y_axislabel_xy) #, labelpad=-1)
    ax.set_ylim(y_lims_xy)
    ax.set_yticks(y_ticks_xy)
    ax.set_yticklabels(y_ticklabels_xy)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)

    ax = fig.add_subplot(gs[4:6, 0:3])
    sns.lineplot(data=pd.melt(dfa2, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_xlabel(x_axislabel)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.xaxis.set_label_coords(0.5, x_labelpos)
    ax.set_ylabel(y_axislabel_z) #, labelpad=-1)
    ax.set_ylim(y_lims_z)
    ax.set_yticks(y_ticks_z)
    ax.set_yticklabels(y_ticklabels_z)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)

    ax = fig.add_subplot(gs[0, 3:6])
    sns.set_palette(palettes[1])
    ax.fill_between(dfb0["t"], dfb0["w1amn"])
    ax.set_title(labels[1])
    ax.set_xlabel(None)
    ax.set_xlim(x_lims)
    ax.set_xticks([])
    ax.set_xticklabels("")
    ax.set_ylabel(None) #, labelpad=-1)
    ax.set_ylim(y_lims_xy)
    ax.set_yticks(y_ticks_xy)
    ax.set_yticklabels([])
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax = fig.add_subplot(gs[1, 3:6])
    ax.fill_between(dfb0["t"], dfb0["w1fmn"])
    # ax.set_title(r"$\operatorname{sech}\beta = 0.01$")
    ax.set_xlabel(None)
    ax.set_xlim(x_lims)
    ax.set_xticks([])
    ax.set_xticklabels("")
    ax.set_ylabel(None) #, labelpad=-1)
    ax.set_ylim(y_lims_z)
    ax.set_yticks(y_ticks_z)
    ax.set_yticklabels([])
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax = fig.add_subplot(gs[2:4, 3:6])
    sns.lineplot(data=pd.melt(dfb1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_xlabel(x_axislabel)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels("")
    ax.xaxis.set_label_coords(0.5, x_labelpos)
    ax.set_ylabel(None) #, labelpad=-1)
    ax.set_ylim(y_lims_xy)
    ax.set_yticks(y_ticks_xy)
    ax.set_yticklabels([])
    ax.yaxis.set_label_coords(y_labelpos, 0.5)

    ax = fig.add_subplot(gs[4:6, 3:6])
    sns.lineplot(data=pd.melt(dfb2, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    ax.set_xlabel(x_axislabel)
    ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.xaxis.set_label_coords(0.5, x_labelpos)
    ax.set_ylabel(None) #, labelpad=-1)
    ax.set_ylim(y_lims_z)
    ax.set_yticks(y_ticks_z)
    ax.set_yticklabels([])
    ax.yaxis.set_label_coords(y_labelpos, 0.5)

    sns.set_palette(palettes[2])
    ax = fig.add_subplot(gs[0:6, 6:8])
    ax = sns.lineplot(dfc, x="Simulation Steps", y="Error (rad)", hue="label", legend=False)
    ax.set_xlabel("Simulation Steps")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs=[1e3, 1e5, 1e7]))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(locs=[1e4, 1e6]))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(""))
    ax.set_xscale("log")
    ax.xaxis.set_label_coords(0.5, -0.045)
    ax.set_ylabel("Error (rad)", labelpad=-3)
    ax.yaxis.set_major_locator(ticker.FixedLocator(locs=[1e-12, 1e-9, 1e-6, 1e-3, 1e0]))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(locs=[1e-13, 1e-11, 1e-10, 1e-8, 1e-7, 1e-5, 1e-4, 1e-2, 1e-1]))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(""))
    ax.set_yscale("log")

def plot_5panel_serror(app, dfs, palettes, titles, xticks, xticklabels):
    # fig S1; 

    fig = plt.figure(figsize=(8, 8))
    # gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.3)
    gs = gridspec.GridSpec(4, 3, hspace=0.1, wspace=0.25)

    x_ticks = xticks
    x_ticklabels = xticklabels
    x_axislabel = r"$\theta$ (rad)"

    y_label_xy = r"$|M_{xy}^{S}| - |M_{xy}^{C}|$"
    y_label_xyn = r"$(|M_{xy}^{S}| - |M_{xy}^{C}|)/|M_{xy}^{S}|$"
    y_label_z = r"$|M_{z}^{S}| - |M_{z}^{C}|$"
    y_label_zn = r"$(|M_{z}^{S}| - |M_{z}^{C}|)/|M_{z}^{S}|$"
    y_label_angle = r"$\epsilon (rad)$"
    lim_xy = 0.02
    lim_xyn = 1
    lim_z = 0.02
    lim_zn = 1
    y_lims_xy = [-lim_xy, lim_xy]
    y_lims_xyn = [-lim_xyn, lim_xyn]
    y_lims_z = [-lim_z, lim_z]
    y_lims_zn = [-lim_zn, lim_zn]

    y_labelpos = -0.22

    melted_dfs = pd.melt(dfs, ['theta'])

    sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zxy", "rxy"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    ax.set_title(titles[0])
    ax.set_xlabel("")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel(y_label_xy)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.set_ylim(y_lims_xy)

    ax = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zxyn", "rxyn"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    ax.set_title(titles[1])
    ax.set_xlabel("")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel(y_label_xyn)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.set_ylim(y_lims_xyn)

    ax = fig.add_subplot(gs[1, 0])
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zz", "rz"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(y_label_z)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.set_ylim(y_lims_z)

    ax = fig.add_subplot(gs[1, 1])
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zzn", "rzn"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(y_label_zn)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)
    ax.set_ylim(y_lims_zn)

    ax = fig.add_subplot(gs[0, 2])
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["za", "ra"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    ax.set_title("Angular Error")
    ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(y_label_angle)
    ax.yaxis.set_label_coords(y_labelpos, 0.5)

# deprecated and unoptimized
def plot_11panel_serror(app, dfs, palettes, titles, xticks, xticklabels):
    # fig S1; 

    fig = plt.figure(figsize=(8, 8))
    # gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.3)
    gs = gridspec.GridSpec(4, 3, hspace=0.1, wspace=0.25)

    # x_limexp = 0.05
    x_lims = [dfs["theta"].iloc[0], dfs["theta"].iloc[-1]]
    x_ticks = xticks
    x_ticklabels = xticklabels
    x_axislabel = r"$\theta$ (rad)"

    # y_limexp = 0.05
    # y_lims_xy = [-y_limexp, (1 + y_limexp)]
    # y_ticks_xy = [0.0, 1.0]
    # y_ticklabels_xy = [0, 1]
    # y_axislabel_xy = r"$|M_{xy}|/|M|$"

    # y_lims_z = [-(1 + y_limexp), (1 + y_limexp)]
    # y_ticks_z = [-1, 0, 1]
    # y_ticklabels_z = [-1, 0, 1]
    # y_axislabel_z = r"$M_{z}/|M|$"

    # x_labelpos = -0.15
    # y_labelpos = -0.15

    # legend_line_mult = 1.5
    melted_dfs = pd.melt(dfs, ['theta'])

    sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[0, 0])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zx", "rx"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # sns.lineplot(data=pd.melt(dfs, ['t']), x="t", y="value", hue='variable', ax=ax, legend=False)
    # ax.set_title(labels[0])
    ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel("Mx")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)
    
    # sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[0, 1])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zxn", "rxn"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel("Mxn")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)

    sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[1, 0])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zy", "ry"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel("My")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)

    # sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[1, 1])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zyn", "ryn"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel("Myn")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)

    # sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[2, 0])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zxy", "rxy"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel("Mxy")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)

    # sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[2, 1])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zxyn", "rxyn"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_ylabel("Mxyn")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)

    # sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[3, 0])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zz", "rz"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    # ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Mz")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)

    # sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[3, 1])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["zzn", "rzn"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    # ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Mzn")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)

    # sns.set_palette(palettes[0])
    ax = fig.add_subplot(gs[0, 2])
    # sns.lineplot(data=pd.melt(dfa1, ['t']), x="t", y="value", hue='variable', ax=ax, legend=True)
    sns.lineplot(data=melted_dfs[melted_dfs["variable"].isin(["za", "ra"])], x="theta", y="value", hue="variable", ax=ax, legend=False)
    # ax.set_title(labels[0])
    # ax.set_xlabel("")
    # # ax.set_xlabel(x_axislabel)
    # ax.set_xlim(x_lims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("angle")
    # ax.set_ylim(y_lims_xy)
    # ax.set_yticks(y_ticks_xy)
    # ax.set_yticklabels(y_ticklabels_xy)