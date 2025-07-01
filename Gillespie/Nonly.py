import numpy as np
import sys
from datetime import datetime
from Gillespie import main_process
from pathlib import Path

# input parameters:
#    N, P, PP, PN, PPN, PNPN
#    kaP, kbP, kaN, kbN, gamma, V
def reversible_dimer_Nonly(current_counts, parms):
    # get parameters and currernt state
    kaP, kbP, kaN, kbN, gamma, V = parms
    N, P, PP, PN, PPN, PNPN = np.array(current_counts)
    
    C0 = 0.6022

    # def get3DmicroOnRate(kon, D=0.6, sigma=2):
    #     '''calculate the 3D micro rate
    #     Input:
    #         kon: association rate constant
    #         D: diffusion coefficient
    #         sigma: bond length
    #     '''
    #     if kon == 0:
    #         return 0
    #     else:
    #         return (1/kon - 1/(4*np.pi*D*sigma))**(-1)
    
    # def get1DbimolecularRates(ka, kb, NP, D=0.6, L=V/1000, sigma=2):
    #     '''calculate the 1D bimolecular rates
    #     Input:
    #         ka, kb: association and dissociation rate constants
    #         N: number of N molecules
    #         D = D1+D2: total diffusion coefficient
    #         L: length of the space
    #         sigma: bond length
    #     '''
    #     if ka == 0:
    #         return 0, 0
    #     else:
    #         # on rate
    #         k_on = ((L/max(NP,1)-sigma)/3/D + 1/ka)**-1
    #         # off rate
    #         k_off = kb*k_on/ka
    #         return k_on, k_off
    
    ## 3D reaction
    # P + P <-> PP
    a_P_P_PP = kaP*P*P/V
    if P == 1: a_P_P_PP = 0 # no dimerization if there is only one P left
    a_PP_P_P = kbP*PP

    ## 3D <-> 1D reactions
    # P + N <-> PN
    a_P_N_PN = kaN*P*N/V
    a_PN_P_N = kbN*PN
    # PP + N <-> PPN
    a_PP_N_PPN = 2*kaN*PP*N/V
    a_PPN_PP_N = kbN*PPN
    # P + PN <-> PPN
    a_P_PN_PPN = 2*kaP*PN*P/V
    a_PPN_P_PN = kbP*PPN

    ## 1D bimolecular reactions
    # get 1D bimolecular rates
    # kaP1D, kbP1D = get1DbimolecularRates(gamma*get3DmicroOnRate(kaP, D=1.5*2*1e6, sigma=2), kbP, PN, D=1.2*1e6, sigma=2)
    # kaN1D, kbN1D = get1DbimolecularRates(gamma*get3DmicroOnRate(kaN, D=1.5/2*1e6, sigma=1), kbN, max(PPN, N), D=(1/1.5+1/0.6+1)**(-1)*1e6, sigma=1)
    kaP1D, kbP1D = gamma*kaP, kbP
    kaN1D, kbN1D = gamma*kaN, kbN
    # PN + PN <-> PNPN (gamma)
    a_PN_PN_PNPN = kaP1D*PN*PN/V   
    if PN == 1: a_PN_PN_PNPN = 0 # no dimerization if there is only one PN left
    a_PNPN_PN_PN = kbP1D*PNPN
    # PPN + N <-> PNPN (gamma)
    a_PPN_N_PNPN = kaN1D*PPN*N/V
    a_PNPN_PPN_N = 2*kbN1D*PNPN
    
    
    propensities = np.array(
        [
            a_P_P_PP, a_PP_P_P, 
            a_P_N_PN, a_PN_P_N, 
            a_PP_N_PPN, a_PPN_PP_N, 
            a_P_PN_PPN, a_PPN_P_PN,
            a_PN_PN_PNPN, a_PNPN_PN_PN, 
            a_PPN_N_PNPN, a_PNPN_PPN_N,
        ]
    )
    
    rxnMatrix = np.array(
        [
            # N,  P, PP, PN,PPN,PNPN,
            [ 0, -2,  1,  0,  0,   0,], # P + P -> PP
            [ 0,  2, -1,  0,  0,   0,], # PP -> P + P
            [-1, -1,  0,  1,  0,   0,], # P + N -> PN
            [ 1,  1,  0, -1,  0,   0,], # PN -> P + N
            [-1,  0, -1,  0,  1,   0,], # PP + N -> PPN
            [ 1,  0,  1,  0, -1,   0,], # PPN -> PP + N
            [ 0, -1,  0, -1,  1,   0,], # P + PN -> PPN
            [ 0,  1,  0,  1, -1,   0,], # PPN -> P + PN
            [ 0,  0,  0, -2,  0,   1,], # PN + PN -> PNPN
            [ 0,  0,  0,  2,  0,  -1,], # PNPN -> PN + PN
            [-1,  0,  0,  0, -1,   1,], # PPN + N -> PNPN
            [ 1,  0,  0,  0,  1,  -1,], # PNPN -> PPN + N
        ]
    )
    
    return propensities, rxnMatrix

def main_Gillespie(parms):

    equilibrium, parameters, repeati, rMaxT, rMinRatio, NP0_sys, pdir, maxT, tStart, tEnd,  = parms

    # label all molecules in the system
    labelstring = 'N, P, PP, PN, PPN, PNPN'
    labels = labelstring.split(', ')

    # define how to calculate the total number of monomers
    sumMat = np.array([
        [
            1, 0, 0, 1, 1, 2 
        ], # N
        [
            0, 1, 2, 1, 2, 2, 
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
    Path(pdir+'OUTPUTS_Nonly/').mkdir(parents=True, exist_ok=True)
    fout = open(pdir+'OUTPUTS_Nonly/out_%d_r%d'%(parameters['ID'], repeati), 'w')
    sys.stdout = fout
    print('# Time:', datetime.now(),flush=True)
    print('# Random state', flush=True)
    # Set the random seed based on the current time.
    import time
    rndState = int(time.time())+repeati
    np.random.seed(rndState)
    print(rndState)

    main_process(
        reversible_dimer_Nonly, 'N', NP0_sys, labels, 
        parameters, sumMat, totalCopy, equi_counts, tStepSize, 
        getSurvivalProb=True, getResT=True, maxT=maxT, tStart=tStart, tEnd=tEnd,
    )
        
    print('# Time:',datetime.now(),flush=True)
    fout.close()
    sys.stdout = orig_stdout
    print('ID: %d - repeat %d, End Time:'%(parameters['ID'], repeati), datetime.now(),flush=True)
