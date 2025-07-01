import numpy as np
import sys
from datetime import datetime
from Gillespie import main_process
from pathlib import Path

# input parameters:
#    S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PSNPN
#    kaP, kbP, kaS, kbS, kaN, kbN, gamma, V
def reversible_dimer(current_counts, parms):
    # get parameters and currernt state
    kaP, kbP, kaS, kbS, kaN, kbN, gamma, V = parms
    S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PSNPN = np.array(current_counts)
    
    C0 = 0.6022
    
    ## 3D reaction
    # P + P <-> PP
    a_P_P_PP = kaP*P*P/V
    if P == 1: a_P_P_PP = 0 # no dimerization if there is only one P left
    a_PP_P_P = kbP*PP

    ## 3D <-> 1D reactions
    # P + N <-> PN
    a_P_N_PN = kaN*P*N/V
    a_PN_P_N = kbN*PN
    # P + S <-> PS
    a_P_S_PS = kaS*P*S/V
    a_PS_P_S = kbS*PS
    # PP + N <-> PPN
    a_PP_N_PPN = 2*kaN*PP*N/V
    a_PPN_PP_N = kbN*PPN
    # P + PN <-> PPN
    a_P_PN_PPN = 2*kaP*P*PN/V
    a_PPN_P_PN = kbP*PPN
    # PP + S <-> PPS
    a_PP_S_PPS = 2*kaS*PP*S/V
    a_PPS_PP_S = kbS*PPS
    # P + PS <-> PPS
    a_P_PS_PPS = 2*kaP*P*PS/V
    a_PPS_P_PS = kbP*PPS
    # P + PSN <-> PPSN
    a_P_PSN_PPSN = 2*kaP*P*PSN/V
    a_PPSN_P_PSN = kbP*PPSN

    ## 1D bimolecular reactions
    # PN + S <-> PSN (gamma)
    a_PN_S_PSN = gamma*kaS*PN*S/V
    a_PSN_PN_S = kbS*PSN
    # PPN + S <-> PPSN (gamma)
    a_PPN_S_PPSN = gamma*kaS*PPN*S/V
    a_PPSN_PPN_S = kbS*PPSN
    # PSN + P <-> PPSN
    a_PSN_P_PPSN = 2*kaP*PSN*P/V
    a_PPSN_PSN_P = kbP*PPSN
    # PN + PN <-> PNPN (gamma)
    a_PN_PN_PNPN = gamma*kaP*PN*PN/V   
    if PN == 1: a_PN_PN_PNPN = 0 # no dimerization if there is only one PN left
    a_PNPN_PN_PN = kbP*PNPN
    # PPN + S <-> PSPN (gamma)
    a_PPN_S_PSPN = gamma*kaS*PPN*S/V
    a_PSPN_PPN_S = kbS*PSPN
    # PN + PS <-> PSPN (gamma)
    a_PN_PS_PSPN = 2*gamma*kaP*PN*PS/V
    a_PSPN_PN_PS = kbP*PSPN
    # PNPN + S <-> PSNPN (gamma)
    a_PNPN_S_PSNPN = 2*gamma*kaS*PNPN*S/V
    a_PSNPN_PNPN_S = kbS*PSNPN
    # PSN + PN <-> PSNPN (gamma)
    a_PSN_PN_PSNPN = 2*gamma*kaP*PSN*PN/V
    a_PSNPN_PSN_PN = kbP*PSNPN

    ## 1D nonepsicific reactions
    # PPS + N <-> PPSN (gamma)
    a_PPS_N_PPSN = gamma*kaN*PPS*N/V
    a_PPSN_PPS_N = kbN*PPSN
    # PPN + N <-> PNPN (gamma)
    a_PPN_N_PNPN = gamma*kaN*PPN*N/V
    a_PNPN_PPN_N = 2*kbN*PNPN
    # PPSN + N <-> PSNPN (gamma)
    a_PPSN_N_PSNPN = gamma*kaN*PPSN*N/V
    a_PSNPN_PPSN_N = kbN*PSNPN
    # PSPN + N <-> PSNPN (gamma)
    a_PSPN_N_PSNPN = gamma*kaN*PSPN*N/V
    a_PSNPN_PSPN_N = kbN*PSNPN
    
    propensities = np.array(
        [
            a_P_P_PP, a_PP_P_P, 
            a_P_N_PN, a_PN_P_N, a_P_S_PS, a_PS_P_S, 
            a_PN_S_PSN, a_PSN_PN_S, 
            a_PP_N_PPN, a_PPN_PP_N, a_P_PN_PPN, a_PPN_P_PN,
            a_PP_S_PPS, a_PPS_PP_S, a_P_PS_PPS, a_PPS_P_PS,
            a_P_PSN_PPSN, a_PPSN_P_PSN,
            a_PPN_S_PPSN, a_PPSN_PPN_S, a_PSN_P_PPSN, a_PPSN_PSN_P,
            a_PN_PN_PNPN, a_PNPN_PN_PN, 
            a_PPN_N_PNPN, a_PNPN_PPN_N,
            a_PPN_S_PSPN, a_PSPN_PPN_S, a_PN_PS_PSPN, a_PSPN_PN_PS,
            a_PNPN_S_PSNPN, a_PSNPN_PNPN_S, a_PSN_PN_PSNPN, a_PSNPN_PSN_PN,
            a_PPS_N_PPSN, a_PPSN_PPS_N,
            a_PPSN_N_PSNPN, a_PSNPN_PPSN_N,
            a_PSPN_N_PSNPN, a_PSNPN_PSPN_N,
        ]
    )
    
    rxnMatrix = np.array(
        [
            # S,  N,  P, PP, PN, PS, PSN, PPN, PPS,PPSN,PNPN,PSPN,PSNPN,
            [ 0,  0, -2,  1,  0,  0,   0,   0,   0,   0,   0,   0,    0,], # P + P -> PP
            [ 0,  0,  2, -1,  0,  0,   0,   0,   0,   0,   0,   0,    0,], # PP -> P + P
            [ 0, -1, -1,  0,  1,  0,   0,   0,   0,   0,   0,   0,    0,], # P + N -> PN
            [ 0,  1,  1,  0, -1,  0,   0,   0,   0,   0,   0,   0,    0,], # PN -> P + N
            [-1,  0, -1,  0,  0,  1,   0,   0,   0,   0,   0,   0,    0,], # P + S -> PS
            [ 1,  0,  1,  0,  0, -1,   0,   0,   0,   0,   0,   0,    0,], # PS -> P + S
            [-1,  0,  0,  0, -1,  0,   1,   0,   0,   0,   0,   0,    0,], # PN + S -> PSN
            [ 1,  0,  0,  0,  1,  0,  -1,   0,   0,   0,   0,   0,    0,], # PSN -> PN + S
            [ 0, -1,  0, -1,  0,  0,   0,   1,   0,   0,   0,   0,    0,], # PP + N -> PPN
            [ 0,  1,  0,  1,  0,  0,   0,  -1,   0,   0,   0,   0,    0,], # PPN -> PP + N
            [ 0,  0, -1,  0, -1,  0,   0,   1,   0,   0,   0,   0,    0,], # P + PN -> PPN
            [ 0,  0,  1,  0,  1,  0,   0,  -1,   0,   0,   0,   0,    0,], # PPN -> P + PN
            [-1,  0,  0, -1,  0,  0,   0,   0,   1,   0,   0,   0,    0,], # PP + S -> PPS
            [ 1,  0,  0,  1,  0,  0,   0,   0,  -1,   0,   0,   0,    0,], # PPS -> PP + S
            [ 0,  0, -1,  0,  0, -1,   0,   0,   1,   0,   0,   0,    0,], # PS + P -> PPS
            [ 0,  0,  1,  0,  0,  1,   0,   0,  -1,   0,   0,   0,    0,], # PPS -> PS + P
            [ 0,  0, -1,  0,  0,  0,  -1,   0,   0,   1,   0,   0,    0,], # P + PSN -> PPSN
            [ 0,  0,  1,  0,  0,  0,   1,   0,   0,  -1,   0,   0,    0,], # PPSN -> P + PSN
            [-1,  0,  0,  0,  0,  0,   0,  -1,   0,   1,   0,   0,    0,], # PPN + S -> PPSN
            [ 1,  0,  0,  0,  0,  0,   0,   1,   0,  -1,   0,   0,    0,], # PPSN -> PPN + S
            [ 0,  0, -1,  0,  0,  0,  -1,   0,   0,   1,   0,   0,    0,], # PSN + P -> PPSN
            [ 0,  0,  1,  0,  0,  0,   1,   0,   0,  -1,   0,   0,    0,], # PPSN -> PSN + P
            [ 0,  0,  0,  0, -2,  0,   0,   0,   0,   0,   1,   0,    0,], # PN + PN -> PNPN
            [ 0,  0,  0,  0,  2,  0,   0,   0,   0,   0,  -1,   0,    0,], # PNPN -> PN + PN
            [ 0, -1,  0,  0,  0,  0,   0,  -1,   0,   0,   1,   0,    0,], # PPN + N -> PNPN
            [ 0,  1,  0,  0,  0,  0,   0,   1,   0,   0,  -1,   0,    0,], # PNPN -> PPN + N
            [-1,  0,  0,  0,  0,  0,   0,  -1,   0,   0,   0,   1,    0,], # PPN + S -> PSPN
            [ 1,  0,  0,  0,  0,  0,   0,   1,   0,   0,   0,  -1,    0,], # PSPN -> PPN + S
            [ 0,  0,  0,  0, -1, -1,   0,   0,   0,   0,   0,   1,    0,], # PN + PS -> PSPN
            [ 0,  0,  0,  0,  1,  1,   0,   0,   0,   0,   0,  -1,    0,], # PSPN -> PN + PS
            [-1,  0,  0,  0,  0,  0,   0,   0,   0,   0,  -1,   0,    1,], # PNPN + S -> PSNPN
            [ 1,  0,  0,  0,  0,  0,   0,   0,   0,   0,   1,   0,   -1,], # PSNPN -> PNPN + S
            [ 0,  0,  0,  0, -1,  0,  -1,   0,   0,   0,   0,   0,    1,], # PSN + PN -> PSNPN
            [ 0,  0,  0,  0,  1,  0,   1,   0,   0,   0,   0,   0,   -1,], # PSNPN -> PSN + PN
            [ 0, -1,  0,  0,  0,  0,   0,   0,  -1,   1,   0,   0,    0,], # PPS + N -> PPSN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   1,  -1,   0,   0,    0,], # PPSN -> PPS + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,   0,  -1,   0,   0,    1,], # PPSN + N -> PSNPN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   0,   1,   0,   0,   -1,], # PSNPN -> PPSN + N
            [ 0, -1,  0,  0,  0,  0,   0,   0,   0,   0,   0,  -1,    1,], # PSPN + N -> PSNPN
            [ 0,  1,  0,  0,  0,  0,   0,   0,   0,   0,   0,   1,   -1,], # PSNPN -> PSPN + N
        ]
    )
    
    return propensities, rxnMatrix
    

def main_Gillespie(parms):

    equilibrium, parameters, repeati, rMaxT, rMinRatio, NP0_sys, pdir, maxT, tStart, tEnd, = parms

    # label all molecules in the system
    labelstring = 'S, N, P, PP, PN, PS, PSN, PPN, PPS, PPSN, PNPN, PSPN, PNPSN'
    labels = labelstring.split(', ')
    # define how to calculate the total number of monomers
    sumMat = np.array([
        [
            1, 0, 0, 0, 0, 1, 
            1, 0, 1, 
            1, 0, 1, 1
        ], # S
        [
            0, 1, 0, 0, 1, 0, 
            1, 1, 0, 
            1, 2, 1, 2
        ], # N
        [
            0, 0, 1, 2, 1, 1, 
            1, 2, 2, 
            2, 2, 2, 2
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
    Path(pdir+'OUTPUTS_singleS/').mkdir(parents=True, exist_ok=True)
    fout = open(pdir+'OUTPUTS_singleS/out_%d_r%d'%(parameters['ID'], repeati), 'w')
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


