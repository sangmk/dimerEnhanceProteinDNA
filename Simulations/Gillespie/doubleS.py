import numpy as np
import sys
from datetime import datetime
from Gillespie import main_process
from pathlib import Path

# define the reactions for monomer forming dimers and binding to DNA
# input parameters:
#    S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PSNPN, PSPS, PSPSN, PSNPSN
#    kaP, kbP, kaS, kbS, kaN, kbN, gamma, V
def reversible_dimer(current_counts, parms):
    # get parameters and currernt state
    kaP, kbP, kaS, kbS, kaN, kbN, gamma, V = parms
    S, N, P, PP, PN, PS2, PS2N, PPN, PPS2, PPS2N, PNPN, PS2PN, PS2NPN, PSPS, PSPSN, PSNPSN = np.array(current_counts)
    # here we use S for S2, unless PSPS, PSPSN, and PSNPSN are themselves.
    # Note that the concentration of S2 should be set to S/2
    
    C0 = 0.6022
    
    # 3D reaction (1) ---------------------
    # P + P <-> PP
    a_P_P_PP = kaP*P*P/V    
    if P == 1: a_P_P_PP = 0 # no dimerization if there is only one P left
    a_PP_P_P = kbP*PP
    # 3D and 1D transitions (7) ---------------------
    # P + N <-> PN
    a_P_N_PN = kaN*P*N/V
    a_PN_P_N = kbN*PN
    # P + S <-> PS
    a_P_S_PS2 = kaS*P*S/V   
    if S == 1: a_P_S_PS2 = 0
    a_PS2_P_S = kbS*PS2
    # P + PS2 <-> PPS2
    a_P_PS2_PPS2 = 2*kaP*P*PS2/V
    a_PPS2_P_PS2 = kbP*PPS2
    # P + PN <-> PPN
    a_P_PN_PPN = 2*kaP*P*PN/V
    a_PPN_P_PN = kbP*PPN    
    # P + PSN <-> PPSN
    a_P_PS2N_PPS2N = 2*kaP*P*PS2N/V
    a_PPS2N_P_PS2N = kbP*PPS2N
    # PP + N <-> PPN
    a_PP_N_PPN = 2*kaN*PP*N/V
    a_PPN_PP_N = kbN*PPN
    # PP + S <-> PPS
    a_PP_S_PPS2 = 2*kaS*PP*S/V   
    if S == 1: a_PP_S_PPS2 = 0
    a_PPS2_PP_S = kbS*PPS2
    # 1D dimerizations (3) ----------------------
    # PN + PN <-> PNPN (gamma)
    a_PN_PN_PNPN = gamma*kaP*PN*PN/V    
    if PN == 1: a_PN_PN_PNPN = 0
    a_PNPN_PN_PN = kbP*PNPN
    # PSN + PN <-> PSNPN (gamma)
    a_PS2N_PN_PS2NPN = 2*gamma*kaP*PS2N*PN/V
    a_PS2NPN_PS2N_PN = kbP*PS2NPN
    # PN + PS <-> PSPN (gamma)
    a_PN_PS2_PS2PN = 2*gamma*kaP*PN*PS2/V
    a_PS2PN_PN_PS2 = kbP*PS2PN
    # 1D nonspecific bindings (5) ----------------------
    # PS + N <-> PSN (gamma)
    a_PS2_N_PS2N = gamma*kaN*PS2*N/V
    a_PS2N_PS2_N = kbN*PS2N
    # PPN + N <-> PNPN (gamma)
    a_PPN_N_PNPN = gamma*kaN*PPN*N/V
    a_PNPN_PPN_N = 2*kbN*PNPN
    # PPS + N <-> PPSN (gamma)
    a_PPS2_N_PPS2N = gamma*kaN*PPS2*N/V
    a_PPS2N_PPS2_N = kbN*PPS2N
    # PPS + N <-> PSPN (gamma)
    a_PPS2_N_PS2PN = gamma*kaN*PPS2*N/V
    a_PS2PN_PPS2_N = kbN*PS2PN
    # PPSN + N <-> PSNPN (gamma)
    a_PPS2N_N_PS2NPN = gamma*kaN*PPS2N*N/V
    a_PS2NPN_PPS2N_N = kbN*PS2NPN
    # PSPN + N <-> PSNPN (gamma)
    a_PS2PN_N_PS2NPN = gamma*kaN*PS2PN*N/V
    a_PS2NPN_PS2PN_N = kbN*PS2NPN
    # 1D target bindings (4) ----------------------
    # PN + S <-> PSN (gamma)
    a_PN_S_PS2N = gamma*kaS*PN*S/V   
    if S == 1: a_PN_S_PS2N = 0
    a_PS2N_PN_S = kbS*PS2N
    # PPN + S <-> PPSN (gamma)
    a_PPN_S_PPS2N = gamma*kaS*PPN*S/V  
    if S == 1: a_PPN_S_PPS2N = 0
    a_PPS2N_PPN_S = kbS*PPS2N
    # PPN + S <-> PSPN (gamma)
    a_PPN_S_PS2PN = gamma*kaS*PPN*S/V  
    if S == 1: a_PPN_S_PS2PN = 0
    a_PS2PN_PPN_S = kbS*PS2PN
    # PNPN + S <-> PSNPN (gamma)
    a_PNPN_S_PS2NPN = 2*gamma*kaS*PNPN*S/V  
    if S == 1: a_PNPN_S_PS2NPN = 0
    a_PS2NPN_PNPN_S = kbS*PS2NPN
    # 1D association of the second target (4) ----------------------
    # PPS2 <-> PSPS (C0)
    a_PPS2_PSPS = C0*kaS*PPS2
    a_PSPS_PPS2 = 2*kbS*PSPS
    # PS2PN <-> PSPSN (C0)
    a_PS2PN_PSPSN = C0*kaS*PS2PN
    a_PSPSN_PS2PN = kbS*PSPSN
    # PPS2N <-> PSPSN (C0)
    a_PPS2N_PSPSN = C0*kaS*PPS2N
    a_PSPSN_PPS2N = kbS*PSPSN
    # PS2NPN <-> PSNPSN (C0)
    a_PS2NPN_PSNPSN = C0*kaS*PS2NPN
    a_PSNPSN_PNPS2N = 2*kbS*PSNPSN
    
    
    propensities = np.array(
        [
            a_P_P_PP, a_PP_P_P, # P + P <-> PP
            a_P_N_PN, a_PN_P_N, # P + N <-> PN
            a_P_S_PS2, a_PS2_P_S, # P + S <-> PS
            a_P_PS2_PPS2, a_PPS2_P_PS2, # P + PS <-> PPS
            a_P_PN_PPN, a_PPN_P_PN, # P + PN <-> PPN
            a_P_PS2N_PPS2N, a_PPS2N_P_PS2N, # P + PSN <-> PPSN
            a_PP_N_PPN, a_PPN_PP_N, # PP + N <-> PPN
            a_PP_S_PPS2, a_PPS2_PP_S, # PP + S <-> PPS
            a_PN_PN_PNPN, a_PNPN_PN_PN, # PN + PN <-> PNPN
            a_PS2N_PN_PS2NPN, a_PS2NPN_PS2N_PN, # PSN + PN <-> PSNPN
            a_PN_PS2_PS2PN, a_PS2PN_PN_PS2, # PN + PS <-> PSPN
            a_PS2_N_PS2N, a_PS2N_PS2_N, # PS + N <-> PSN
            a_PPN_N_PNPN, a_PNPN_PPN_N, # PPN + N <-> PNPN
            a_PPS2_N_PPS2N, a_PPS2N_PPS2_N, # PPS + N <-> PPSN
            a_PPS2_N_PS2PN, a_PS2PN_PPS2_N, # PPS + N <-> PSPN
            a_PPS2N_N_PS2NPN, a_PS2NPN_PPS2N_N, # PPSN + N <-> PSNPN
            a_PS2PN_N_PS2NPN, a_PS2NPN_PS2PN_N, # PSPN + N <-> PSNPN
            a_PN_S_PS2N, a_PS2N_PN_S, # PN + S <-> PSN
            a_PPN_S_PPS2N, a_PPS2N_PPN_S, # PPN + S <-> PPSN
            a_PPN_S_PS2PN, a_PS2PN_PPN_S, # PPN + S <-> PSPN
            a_PNPN_S_PS2NPN, a_PS2NPN_PNPN_S, # PNPN + S <-> PSNPN
            a_PPS2_PSPS, a_PSPS_PPS2, # PPS <-> PSPS
            a_PS2PN_PSPSN, a_PSPSN_PS2PN, # PSPN <-> PSPSN
            a_PPS2N_PSPSN, a_PSPSN_PPS2N, # PPSN <-> PSPSN
            a_PS2NPN_PSNPSN, a_PSNPSN_PNPS2N, # PSNPN <-> PSNPSN
        ]
    )
    
    rxnMatrix = np.array(
        [
            # S,  N,  P, PP, PN, PS, PSN, PPN, PPS,PPSN,PNPN,PSPN,PSNPN,PSPS,PSPSN,PSNPSN
            [ 0,  0, -2,  1,  0,  0,   0,   0,   0,   0,   0,   0,    0,   0,    0,     0], # P + P -> PP
            [ 0,  0,  2, -1,  0,  0,   0,   0,   0,   0,   0,   0,    0,   0,    0,     0], # PP -> P + P
            [ 0, -1, -1,  0,  1,  0,   0,   0,   0,   0,   0,   0,    0,   0,    0,     0], # P + N -> PN
            [ 0,  1,  1,  0, -1,  0,   0,   0,   0,   0,   0,   0,    0,   0,    0,     0], # PN -> P + N
            [-2,  0, -1,  0,  0,  1,   0,   0,   0,   0,   0,   0,    0,   0,    0,     0], # P + S -> PS
            [ 2,  0,  1,  0,  0, -1,   0,   0,   0,   0,   0,   0,    0,   0,    0,     0], # PS -> P + S
            [ 0,  0, -1,  0,  0, -1,   0,   0,   1,   0,   0,   0,    0,   0,    0,     0], # P + PS -> PPS
            [ 0,  0,  1,  0,  0,  1,   0,   0,  -1,   0,   0,   0,    0,   0,    0,     0], # PPS -> P + PS
            [ 0,  0, -1,  0, -1,  0,   0,   1,   0,   0,   0,   0,    0,   0,    0,     0], # P + PN -> PPN
            [ 0,  0,  1,  0,  1,  0,   0,  -1,   0,   0,   0,   0,    0,   0,    0,     0], # PPN -> P + PN
            [ 0,  0, -1,  0,  0,  0,  -1,   0,   0,   1,   0,   0,    0,   0,    0,     0], # P + PSN -> PPSN
            [ 0,  0,  1,  0,  0,  0,   1,   0,   0,  -1,   0,   0,    0,   0,    0,     0], # PPSN -> P + PSN
            [ 0, -1,  0, -1,  0,  0,   0,   1,   0,   0,   0,   0,    0,   0,    0,     0], # PP + N -> PPN
            [ 0,  1,  0,  1,  0,  0,   0,  -1,   0,   0,   0,   0,    0,   0,    0,     0], # PPN -> PP + N
            [-2,  0,  0, -1,  0,  0,   0,   0,   1,   0,   0,   0,    0,   0,    0,     0], # PP + S -> PPS
            [ 2,  0,  0,  1,  0,  0,   0,   0,  -1,   0,   0,   0,    0,   0,    0,     0], # PPS -> PP + S
            [ 0,  0,  0,  0, -2,  0,   0,   0,   0,   0,   1,   0,    0,   0,    0,     0], # PN + PN -> PNPN
            [ 0,  0,  0,  0,  2,  0,   0,   0,   0,   0,  -1,   0,    0,   0,    0,     0], # PNPN -> PN + PN
            [ 0,  0,  0,  0, -1,  0,  -1,   0,   0,   0,   0,   0,    1,   0,    0,     0], # PSN + PN -> PSNPN
            [ 0,  0,  0,  0,  1,  0,   1,   0,   0,   0,   0,   0,   -1,   0,    0,     0], # PSNPN -> PSN + PN
            [ 0,  0,  0,  0, -1, -1,   0,   0,   0,   0,   0,   1,    0,   0,    0,     0], # PN + PS -> PSPN
            [ 0,  0,  0,  0,  1,  1,   0,   0,   0,   0,   0,  -1,    0,   0,    0,     0], # PSPN -> PN + PS
            [ 0, -1,  0,  0,  0, -1,   1,   0,   0,   0,   0,   0,    0,   0,    0,     0], # PS + N -> PSN
            [ 0,  1,  0,  0,  0,  1,  -1,   0,   0,   0,   0,   0,    0,   0,    0,     0], # PSN -> PS + N
            [ 0, -1,  0,  0,  0,  0,   0,  -1,   0,   0,   1,   0,    0,   0,    0,     0], # PPN + N -> PNPN
            [ 0,  1,  0,  0,  0,  0,   0,   1,   0,   0,  -1,   0,    0,   0,    0,     0], # PNPN -> PPN + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,  -1,   1,   0,   0,    0,   0,    0,     0], # PPS + N -> PPSN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   1,  -1,   0,   0,    0,   0,    0,     0], # PPSN -> PPS + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,  -1,   0,   0,   1,    0,   0,    0,     0], # PPS + N -> PSPN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   1,   0,   0,  -1,    0,   0,    0,     0], # PSPN -> PPS + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,   0,  -1,   0,   0,    1,   0,    0,     0], # PPSN + N -> PSNPN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   0,   1,   0,   0,   -1,   0,    0,     0], # PSNPN -> PPSN + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,   0,   0,   0,  -1,    1,   0,    0,     0], # PSPN + N -> PSNPN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   0,   0,   0,   1,   -1,   0,    0,     0], # PSNPN -> PSPN + N
            [-2,  0,  0,  0, -1,  0,   1,   0,   0,   0,   0,   0,    0,   0,    0,     0], # PN + S -> PSN
            [ 2,  0,  0,  0,  1,  0,  -1,   0,   0,   0,   0,   0,    0,   0,    0,     0], # PSN -> PN + S
            [-2,  0,  0,  0,  0,  0,   0,  -1,   0,   1,   0,   0,    0,   0,    0,     0], # PPN + S -> PPSN
            [ 2,  0,  0,  0,  0,  0,   0,   1,   0,  -1,   0,   0,    0,   0,    0,     0], # PPSN -> PPN + S
            [-2,  0,  0,  0,  0,  0,   0,  -1,   0,   0,   0,   1,    0,   0,    0,     0], # PPN + S -> PSPN
            [ 2,  0,  0,  0,  0,  0,   0,   1,   0,   0,   0,  -1,    0,   0,    0,     0], # PSPN -> PPN + S
            [-2,  0,  0,  0,  0,  0,   0,   0,   0,   0,  -1,   0,    1,   0,    0,     0], # PNPN + S -> PSNPN
            [ 2,  0,  0,  0,  0,  0,   0,   0,   0,   0,   1,   0,   -1,   0,    0,     0], # PSNPN -> PNPN + S
            [ 0,  0,  0,  0,  0,  0,   0,   0,  -1,   0,   0,   0,    0,   1,    0,     0], # PPS -> PSPS
            [ 0,  0,  0,  0,  0,  0,   0,   0,   1,   0,   0,   0,    0,  -1,    0,     0], # PSPS -> PPS
            [ 0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,  -1,    0,   0,    1,     0], # PSPN -> PSPSN
            [ 0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   1,    0,   0,   -1,     0], # PSPSN -> PSPN
            [ 0,  0,  0,  0,  0,  0,   0,   0,   0,  -1,   0,   0,    0,   0,    1,     0], # PPSN -> PSPSN
            [ 0,  0,  0,  0,  0,  0,   0,   0,   0,   1,   0,   0,    0,   0,   -1,     0], # PSPSN -> PPSN
            [ 0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   -1,   0,    0,     1], # PSNPN -> PSNPSN
            [ 0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0,   0,    1,   0,    0,    -1], # PSNPSN -> PSNPN
        ]
    )
    
    return propensities, rxnMatrix


def main_Gillespie(parms):

    equilibrium, parameters, repeati, rMaxT, rMinRatio, NP0_sys, pdir, maxT, tStart, tEnd, = parms

    # label all molecules in the system
    labelstring = 'S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PNPSN, PSPS, PSPSN, PSNPSN'
    labels = labelstring.split(', ')
    # define how to calculate the total number of monomers
    sumMat = np.array([
        [
            1, 0, 0, 0, 0, 1, 
            1, 0, 1, 
            1, 0, 1, 1,
            2, 2, 2
        ], # S
        [
            0, 1, 0, 0, 1, 0, 
            1, 1, 0, 
            1, 2, 1, 2,
            0, 1, 2
        ], # N
        [
            0, 0, 1, 2, 1, 1, 
            1, 2, 2, 
            2, 2, 2, 2,
            2, 2, 2
        ] # P
    ])
    equi_counts = np.array(equilibrium[labels].tolist())
    totalCopy_equi = np.sum(equi_counts * sumMat, axis=1)
    totalCopy = totalCopy_equi*NP0_sys/totalCopy_equi[-1]
    # --------------------------------------------------

    # run simulation
    kbPN = parameters['kbPN']
    tStepSize = rMinRatio/kbPN
    orig_stdout = sys.stdout

    print('ID: %d - repeat %d, Start Time:'%(parameters['ID'], repeati), datetime.now(),flush=True)
    Path(pdir+'OUTPUTS_doubleS/').mkdir(parents=True, exist_ok=True)
    fout = open(pdir+'OUTPUTS_doubleS/out_%d_r%d'%(parameters['ID'], repeati), 'w')
    sys.stdout = fout
    print('# Time:', datetime.now(),flush=True)
    print('# Random state', flush=True)
    # Set the random seed based on the current time.
    import time
    rndState = int(time.time())+repeati
    np.random.seed(rndState)
    print(rndState)

    main_process(
        reversible_dimer, 'S', NP0_sys, labels, 
        parameters, sumMat, totalCopy, equi_counts, tStepSize, 
        getSurvivalProb=True, getResT=True, maxT=maxT, tStart=tStart, tEnd=tEnd,
    )
        
    print('# Time:',datetime.now(),flush=True)
    fout.close()
    sys.stdout = orig_stdout
    print('ID: %d - repeat %d, End Time:'%(parameters['ID'], repeati), datetime.now(),flush=True)


