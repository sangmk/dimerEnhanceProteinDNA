import numpy as np
# from sympy import nsolve, symbols
from scipy.integrate import solve_ivp
import sys
from datetime import datetime

# perform ODE solver based on the reactions used for Gillespie input
def ODE_function(current_counts, t, parms, reactions):
    propList, rxnMatrix = reactions(current_counts, parms)
    return np.matmul(rxnMatrix.T, propList)

# define the reactions for monomer forming dimers and binding to DNA
# input parameters:
#    S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PSNPN
#    kaP, kbP, kaS, kbS, kaN, kbN, gamma, V
def reversible_dimer(current_counts, kineticParms):
    
    # get parameters and currernt state
    kaP, kbP, kaS, kbS, kaN, kbN, gamma, V = kineticParms
    S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PSNPN = np.array(current_counts)
    
    C0 = 0.6022
    
    # 3D reaction (1) ---------------------
    # P + P <-> PP
    a_P_P_PP = kaP*P*P/V    # if P == 1: a_P_P_PP = 0 # no dimerization if there is only one P left
    a_PP_P_P = kbP*PP
    # 3D and 1D transitions (7) ---------------------
    # P + N <-> PN
    a_P_N_PN = kaN*P*N/V
    a_PN_P_N = kbN*PN
    # P + S <-> PS
    a_P_S_PS = kaS*P*S/V   # if S == 1: a_P_S_PS = 0
    a_PS_P_S = kbS*PS
    # P + PS <-> PPS
    a_P_PS_PPS = 2*kaP*P*PS/V
    a_PPS_P_PS = kbP*PPS
    # P + PN <-> PPN
    a_P_PN_PPN = 2*kaP*P*PN/V
    a_PPN_P_PN = kbP*PPN    
    # P + PSN <-> PPSN
    a_P_PSN_PPSN = 2*kaP*P*PSN/V
    a_PPSN_P_PSN = kbP*PPSN
    # PP + N <-> PPN
    a_PP_N_PPN = 2*kaN*PP*N/V
    a_PPN_PP_N = kbN*PPN
    # PP + S <-> PPS
    a_PP_S_PPS = 2*kaS*PP*S/V   # if S == 1: a_PP_S_PPS = 0
    a_PPS_PP_S = kbS*PPS
    # 1D dimerizations (3) ----------------------
    # PN + PN <-> PNPN (gamma)
    a_PN_PN_PNPN = gamma*kaP*PN*PN/V    # if PN == 1: a_PN_PN_PNPN = 0
    a_PNPN_PN_PN = kbP*PNPN
    # PSN + PN <-> PSNPN (gamma)
    a_PSN_PN_PSNPN = 2*gamma*kaP*PSN*PN/V
    a_PSNPN_PSN_PN = kbP*PSNPN
    # PN + PS <-> PSPN (gamma)
    a_PN_PS_PSPN = 2*gamma*kaP*PN*PS/V
    a_PSPN_PN_PS = kbP*PSPN
    # 1D nonspecific bindings (5) ----------------------
    # PS + N <-> PSN (gamma)
    a_PS_N_PSN = gamma*kaN*PS*N/V
    a_PSN_PS_N = kbN*PSN
    # PPN + N <-> PNPN (gamma)
    a_PPN_N_PNPN = gamma*kaN*PPN*N/V
    a_PNPN_PPN_N = 2*kbN*PNPN
    # PPS + N <-> PPSN (gamma)
    a_PPS_N_PPSN = gamma*kaN*PPS*N/V
    a_PPSN_PPS_N = kbN*PPSN
    # PPS + N <-> PSPN (gamma)
    a_PPS_N_PSPN = gamma*kaN*PPS*N/V
    a_PSPN_PPS_N = kbN*PSPN
    # PPSN + N <-> PSNPN (gamma)
    a_PPSN_N_PSNPN = gamma*kaN*PPSN*N/V
    a_PSNPN_PPSN_N = kbN*PSNPN
    # PSPN + N <-> PSNPN (gamma)
    a_PSPN_N_PSNPN = gamma*kaN*PSPN*N/V
    a_PSNPN_PSPN_N = kbN*PSNPN
    # 1D target bindings (4) ----------------------
    # PN + S <-> PSN (gamma)
    a_PN_S_PSN = gamma*kaS*PN*S/V   # if S == 1: a_PN_S_PSN = 0
    a_PSN_PN_S = kbS*PSN
    # PPN + S <-> PPSN (gamma)
    a_PPN_S_PPSN = gamma*kaS*PPN*S/V  # if S == 1: a_PPN_S_PPSN = 0
    a_PPSN_PPN_S = kbS*PPSN
    # PPN + S <-> PSPN (gamma)
    a_PPN_S_PSPN = gamma*kaS*PPN*S/V  # if S == 1: a_PPN_S_PSPN = 0
    a_PSPN_PPN_S = kbS*PSPN
    # PNPN + S <-> PSNPN (gamma)
    a_PNPN_S_PSNPN = 2*gamma*kaS*PNPN*S/V  # if S == 1: a_PNPN_S_PSNPN = 0
    a_PSNPN_PNPN_S = kbS*PSNPN
    
    
    propensities = np.array(
        [
            a_P_P_PP, a_PP_P_P, # P + P <-> PP
            a_P_N_PN, a_PN_P_N, # P + N <-> PN
            a_P_S_PS, a_PS_P_S, # P + S <-> PS
            a_P_PS_PPS, a_PPS_P_PS, # P + PS <-> PPS
            a_P_PN_PPN, a_PPN_P_PN, # P + PN <-> PPN
            a_P_PSN_PPSN, a_PPSN_P_PSN, # P + PSN <-> PPSN
            a_PP_N_PPN, a_PPN_PP_N, # PP + N <-> PPN
            a_PP_S_PPS, a_PPS_PP_S, # PP + S <-> PPS
            a_PN_PN_PNPN, a_PNPN_PN_PN, # PN + PN <-> PNPN
            a_PSN_PN_PSNPN, a_PSNPN_PSN_PN, # PSN + PN <-> PSNPN
            a_PN_PS_PSPN, a_PSPN_PN_PS, # PN + PS <-> PSPN
            a_PS_N_PSN, a_PSN_PS_N, # PS + N <-> PSN
            a_PPN_N_PNPN, a_PNPN_PPN_N, # PPN + N <-> PNPN
            a_PPS_N_PPSN, a_PPSN_PPS_N, # PPS + N <-> PPSN
            a_PPS_N_PSPN, a_PSPN_PPS_N, # PPS + N <-> PSPN
            a_PPSN_N_PSNPN, a_PSNPN_PPSN_N, # PPSN + N <-> PSNPN
            a_PSPN_N_PSNPN, a_PSNPN_PSPN_N, # PSPN + N <-> PSNPN
            a_PN_S_PSN, a_PSN_PN_S, # PN + S <-> PSN
            a_PPN_S_PPSN, a_PPSN_PPN_S, # PPN + S <-> PPSN
            a_PPN_S_PSPN, a_PSPN_PPN_S, # PPN + S <-> PSPN
            a_PNPN_S_PSNPN, a_PSNPN_PNPN_S, # PNPN + S <-> PSNPN
        ]
    )
    
    rxnMatrix = np.array(
        [
            # S,  N,  P, PP, PN, PS, PSN, PPN, PPS,PPSN,PNPN,PSPN,PSNPN
            [ 0,  0, -2,  1,  0,  0,   0,   0,   0,   0,   0,   0,    0], # P + P -> PP
            [ 0,  0,  2, -1,  0,  0,   0,   0,   0,   0,   0,   0,    0], # PP -> P + P
            [ 0, -1, -1,  0,  1,  0,   0,   0,   0,   0,   0,   0,    0], # P + N -> PN
            [ 0,  1,  1,  0, -1,  0,   0,   0,   0,   0,   0,   0,    0], # PN -> P + N
            [-1,  0, -1,  0,  0,  1,   0,   0,   0,   0,   0,   0,    0], # P + S -> PS
            [ 1,  0,  1,  0,  0, -1,   0,   0,   0,   0,   0,   0,    0], # PS -> P + S
            [ 0,  0, -1,  0,  0, -1,   0,   0,   1,   0,   0,   0,    0], # P + PS -> PPS
            [ 0,  0,  1,  0,  0,  1,   0,   0,  -1,   0,   0,   0,    0], # PPS -> P + PS
            [ 0,  0, -1,  0, -1,  0,   0,   1,   0,   0,   0,   0,    0], # P + PN -> PPN
            [ 0,  0,  1,  0,  1,  0,   0,  -1,   0,   0,   0,   0,    0], # PPN -> P + PN
            [ 0,  0, -1,  0,  0,  0,  -1,   0,   0,   1,   0,   0,    0], # P + PSN -> PPSN
            [ 0,  0,  1,  0,  0,  0,   1,   0,   0,  -1,   0,   0,    0], # PPSN -> P + PSN
            [ 0, -1,  0, -1,  0,  0,   0,   1,   0,   0,   0,   0,    0], # PP + N -> PPN
            [ 0,  1,  0,  1,  0,  0,   0,  -1,   0,   0,   0,   0,    0], # PPN -> PP + N
            [-1,  0,  0, -1,  0,  0,   0,   0,   1,   0,   0,   0,    0], # PP + S -> PPS
            [ 1,  0,  0,  1,  0,  0,   0,   0,  -1,   0,   0,   0,    0], # PPS -> PP + S
            [ 0,  0,  0,  0, -2,  0,   0,   0,   0,   0,   1,   0,    0], # PN + PN -> PNPN
            [ 0,  0,  0,  0,  2,  0,   0,   0,   0,   0,  -1,   0,    0], # PNPN -> PN + PN
            [ 0,  0,  0,  0, -1,  0,  -1,   0,   0,   0,   0,   0,    1], # PSN + PN -> PSNPN
            [ 0,  0,  0,  0,  1,  0,   1,   0,   0,   0,   0,   0,   -1], # PSNPN -> PSN + PN
            [ 0,  0,  0,  0, -1, -1,   0,   0,   0,   0,   0,   1,    0], # PN + PS -> PSPN
            [ 0,  0,  0,  0,  1,  1,   0,   0,   0,   0,   0,  -1,    0], # PSPN -> PN + PS
            [ 0, -1,  0,  0,  0, -1,   1,   0,   0,   0,   0,   0,    0], # PS + N -> PSN
            [ 0,  1,  0,  0,  0,  1,  -1,   0,   0,   0,   0,   0,    0], # PSN -> PS + N
            [ 0, -1,  0,  0,  0,  0,   0,  -1,   0,   0,   1,   0,    0], # PPN + N -> PNPN
            [ 0,  1,  0,  0,  0,  0,   0,   1,   0,   0,  -1,   0,    0], # PNPN -> PPN + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,  -1,   1,   0,   0,    0], # PPS + N -> PPSN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   1,  -1,   0,   0,    0], # PPSN -> PPS + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,  -1,   0,   0,   1,    0], # PPS + N -> PSPN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   1,   0,   0,  -1,    0], # PSPN -> PPS + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,   0,  -1,   0,   0,    1], # PPSN + N -> PSNPN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   0,   1,   0,   0,   -1], # PSNPN -> PPSN + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,   0,   0,   0,  -1,    1], # PSPN + N -> PSNPN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   0,   0,   0,   1,   -1], # PSNPN -> PSPN + N
            [-1,  0,  0,  0, -1,  0,   1,   0,   0,   0,   0,   0,    0], # PN + S -> PSN
            [ 1,  0,  0,  0,  1,  0,  -1,   0,   0,   0,   0,   0,    0], # PSN -> PN + S
            [-1,  0,  0,  0,  0,  0,   0,  -1,   0,   1,   0,   0,    0], # PPN + S -> PPSN
            [ 1,  0,  0,  0,  0,  0,   0,   1,   0,  -1,   0,   0,    0], # PPSN -> PPN + S
            [-1,  0,  0,  0,  0,  0,   0,  -1,   0,   0,   0,   1,    0], # PPN + S -> PSPN
            [ 1,  0,  0,  0,  0,  0,   0,   1,   0,   0,   0,  -1,    0], # PSPN -> PPN + S
            [-1,  0,  0,  0,  0,  0,   0,   0,   0,   0,  -1,   0,    1], # PNPN + S -> PSNPN
            [ 1,  0,  0,  0,  0,  0,   0,   0,   0,   0,   1,   0,   -1], # PSNPN -> PNPN + S
        ]
    )
    
    return propensities, rxnMatrix


# compare observed equilibrium constants with true values
def keq_deviation(EQUILIBRIUM, equiParms, tol):

    def zeroDivid(equi, rxn):
        '''To capture zero division error and irreversible reactions'''
        for i, ai in enumerate(rxn):
            if ai < 0 and equi[i]==0:
                return True
        return False

    # get parameters
    KPP, KPS, KPN, gamma, V = equiParms

    C0 = 0.6022

    # define the reaction matrix
    rxnMatrix = np.array(
        [
            # S,  N,  P, PP, PN, PS, PSN, PPN, PPS,PPSN,PNPN,PSPN,PSNPN
            [ 0,  0, -2,  1,  0,  0,   0,   0,   0,   0,   0,   0,    0], # P + P -> PP
            [ 0, -1, -1,  0,  1,  0,   0,   0,   0,   0,   0,   0,    0], # P + N -> PN
            [-1,  0, -1,  0,  0,  1,   0,   0,   0,   0,   0,   0,    0], # P + S -> PS
            [ 0,  0, -1,  0,  0, -1,   0,   0,   1,   0,   0,   0,    0], # P + PS -> PPS
            [ 0,  0, -1,  0, -1,  0,   0,   1,   0,   0,   0,   0,    0], # P + PN -> PPN
            [ 0,  0, -1,  0,  0,  0,  -1,   0,   0,   1,   0,   0,    0], # P + PSN -> PPSN
            [ 0, -1,  0, -1,  0,  0,   0,   1,   0,   0,   0,   0,    0], # PP + N -> PPN
            [-1,  0,  0, -1,  0,  0,   0,   0,   1,   0,   0,   0,    0], # PP + S -> PPS
            [ 0,  0,  0,  0, -2,  0,   0,   0,   0,   0,   1,   0,    0], # PN + PN -> PNPN
            [ 0,  0,  0,  0, -1,  0,  -1,   0,   0,   0,   0,   0,    1], # PSN + PN -> PSNPN
            [ 0,  0,  0,  0, -1, -1,   0,   0,   0,   0,   0,   1,    0], # PN + PS -> PSPN
            [ 0, -1,  0,  0,  0, -1,   1,   0,   0,   0,   0,   0,    0], # PS + N -> PSN
            [ 0, -1,  0,  0,  0,  0,   0,  -1,   0,   0,   1,   0,    0], # PPN + N -> PNPN
            [ 0, -1,  0,  0,  0,  0,   0,   0,  -1,   1,   0,   0,    0], # PPS + N -> PPSN
            [ 0, -1,  0,  0,  0,  0,   0,   0,  -1,   0,   0,   1,    0], # PPS + N -> PSPN
            [ 0, -1,  0,  0,  0,  0,   0,   0,   0,  -1,   0,   0,    1], # PPSN + N -> PSNPN
            [ 0, -1,  0,  0,  0,  0,   0,   0,   0,   0,   0,  -1,    1], # PSPN + N -> PSNPN
            [-1,  0,  0,  0, -1,  0,   1,   0,   0,   0,   0,   0,    0], # PN + S -> PSN
            [-1,  0,  0,  0,  0,  0,   0,  -1,   0,   1,   0,   0,    0], # PPN + S -> PPSN
            [-1,  0,  0,  0,  0,  0,   0,  -1,   0,   0,   0,   1,    0], # PPN + S -> PSPN
            [-1,  0,  0,  0,  0,  0,   0,   0,   0,   0,  -1,   0,    1], # PNPN + S -> PSNPN
        ]
    )
    # calculate true equilibrium constants
    KeqList = np.array(
        [
            KPP, 
            KPN, 
            KPS, 
            2*KPP,
            2*KPP,
            2*KPP,
            2*KPN, 
            2*KPS,
            gamma*KPP,
            2*gamma*KPP,
            2*gamma*KPP,
            gamma*KPN,
            gamma*KPN/2,
            gamma*KPN,
            gamma*KPN,
            gamma*KPN,
            gamma*KPN,
            gamma*KPS,
            gamma*KPS,
            gamma*KPS,
            2*gamma*KPS,
        ]
    )
    # observed equilibrium constants
    for i, rxn in enumerate(rxnMatrix):
        if KeqList[i] is np.nan or KeqList[i] == 0 or KeqList[i] == np.inf:
            pass
        elif zeroDivid(EQUILIBRIUM, rxn):
            pass
        elif abs(np.prod((EQUILIBRIUM/V)**rxn) / KeqList[i] - 1) > tol:
            return True
    return False

# determine whether fluctuation is too large
def flut_toolarge(solution, tol):
    for i, soli in enumerate(solution):
        if soli[-1] == 0:
            if np.std(soli[-10:]) > tol:
                return True
        elif np.std(soli[-10:])/soli[-1] > tol:
            return True
    return False

# ----------------------------------------
# ----------------------------------------
# The main function

labelstring = 'ID, S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PNPSN'
labels = labelstring.split(', ')

def rxnNetwork(parm, full_output=False):

    ## the initial concentrations
    CP0:float = parm['CP0'] # protein
    CN0:float = parm['CN0'] # nonspecific sites
    CS0:float = parm['CS0'] # specific sites
    ## equilibrium constants
    KPN:float = parm['KPN'] # P-N
    KPS:float = parm['KPS'] # P-S
    KPP:float = parm['KPP'] # P-P
    ## dimension factor
    gamma:float = parm['gamma']
    # calculate input for Gillespie 
    ## default system parameters
    NP0_sys = 1e6
    V = NP0_sys / CP0

    # ------------- set up solver --------------
    # S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PSNPN
    if KPP == np.inf:
        ini_count = np.array(
            [
                round(V*CS0), round(V*CN0), 0, NP0_sys/2, 0, 0,
                0, 0, 0, 
                0, 0, 0, 0,
            ]
        )
    else:
        ini_count = np.array(
            [
                round(V*CS0), round(V*CN0), NP0_sys, 0, 0, 0,
                0, 0, 0, 
                0, 0, 0, 0,
            ]
        )
    # S, N, P
    totalCopy = [round(V*CS0), round(V*CN0), NP0_sys]
    # define how to calculate the total number of monomers
    # S, N, P, PP, PN, PS, 
    # PSN, PPN, PPS, 
    # PPSN, PNPN, PSPN, PSNPN
    totalSum = np.array([
        [
            1, 0, 0, 0, 0, 1, 
            1, 0, 1, 
            1, 0, 1, 1,
        ], # S
        [
            0, 1, 0, 0, 1, 0, 
            1, 1, 0, 
            1, 2, 1, 2,
        ], # N
        [
            0, 0, 1, 2, 1, 1, 
            1, 2, 2, 
            2, 2, 2, 2,
        ] # P
    ])

    ## reaction parameters
    kaN, kaS = 10, 10
    kbN = kaN/KPN
    kbS = kaS/KPS
    if KPP == 0:
        kaP = 0
        kbP = 0
    else:
        kaP = 10
        kbP = kaP/KPP
    # kaP, kbP, kaS, kbS, kaN, kbN, gamma, V
    kineticParms = [kaP, kbP, kaS, kbS, kaN, kbN, gamma, V]
    # KPP, KPS, KPN, gamma, V
    equiParms = [KPP, KPS, KPN, gamma, V]
    
    # define a function which equals zero at equilibrium
    flow_func = lambda t, conc: ODE_function(conc, 0, kineticParms, reversible_dimer)
    # define a functino for determining if observed Keq's have deviations
    keq_dev = lambda count, tol: keq_deviation(count, equiParms, tol)
    # --------------------------------------------------
    # --------------------------------------------------

    # run simulation
    ntrys = 0
    # Time interval for integration
    t_span = (0, 1e9)
    y0 = ini_count
    n_limit = 10
    while ntrys < n_limit:
        dynamics = solve_ivp(
            flow_func, t_span, y0, 
            t_eval=np.logspace(0, np.log10(t_span[1]), 1000), 
            method='LSODA'
        )
        solution = dynamics.y
        # determine whether the system is in equilibrium
        # time fluctuation v.s. last value
        # compare observed Keq's v.s. true values
        if keq_dev(solution[:,-1], 1e-3) or flut_toolarge(solution, 1e-3):
            t_span = [0, t_span[-1]*10]
            ntrys += 1
        else:
            # output
            if any((np.sum(solution[:,-1].flatten()*totalSum, axis=1) - totalCopy)/totalCopy > 1e-3):
                print('# Mass conservation fails!',flush=True)
            if full_output:
                return parm['ID'], solution
            else:
                return [parm['ID']]+[value for value in solution[:,-1]]

    print('Failed. ID:',parm['ID'],flush=True)

