import numpy as np
def __get_parms(parm):
    C0 = 0.6022
    # the initial concentrations
    CP0:float = parm['CP0'] # protein
    CN0:float = parm['CN0'] # nonspecific sites
    CS0:float = parm['CS0'] # specific sites
    # equilibrium constants
    KPN:float = parm['KPN'] # P-N
    KPS:float = parm['KPS'] # P-S
    KPP:float = parm['KPP'] # P-P
    # 1d enhancement
    gamma = parm['gamma']
    # dimensionless affinities
    chiPS = KPS*CS0
    chiPN = KPN*CN0
    alpha = KPP*CP0
    return C0, CP0, CN0, CS0, KPN, KPS, KPP, gamma, chiPS, chiPN, alpha

def __ana_PtotFrac(enh, alpha):
    if alpha == 0:
        return 1
    else:
        return (-1 + np.sqrt(1 + 8*enh*alpha)) / (4*enh*alpha)

def __ana_enh_ns(parm):
    C0, CP0, CN0, CS0, KPN, KPS, KPP, gamma, chiPS, chiPN, alpha = __get_parms(parm)
    return (1 + 2*chiPN + gamma*chiPN**2)/(1 + chiPN)**2

def __ana_enh_II(parm, numClusterS):
    '''
    Assume St = S0
    '''
    C0, CP0, CN0, CS0, KPN, KPS, KPP, gamma, chiPS, chiPN, alpha = __get_parms(parm)
    if numClusterS == 2:
        nu = C0*KPS*chiPS*(gamma**2*chiPN**2 + 2*gamma*chiPN + 1) \
            + gamma*chiPN*(chiPN + 4*chiPS + 2*gamma*chiPS*chiPN) + 2*chiPS + 2*chiPN + 1
    elif numClusterS == 1:
        nu = gamma*chiPN*(chiPN + 4*chiPS + 2*gamma*chiPS*chiPN) + 2*chiPS + 2*chiPN + 1
    de = (1 + chiPN + chiPS + gamma*chiPS*chiPN)**2
    return nu/de

def __ana_enh_III(parm, numClusterS):
    C0, CP0, CN0, CS0, KPN, KPS, KPP, gamma, chiPS, chiPN, alpha = __get_parms(parm)
    enh_ns = __ana_enh_ns(parm)
    CPtot = __ana_PtotFrac(enh_ns, (CP0-CS0)*KPP)*(CP0-CS0)
    if numClusterS == 2:
        return enh_ns + np.exp(np.log(occ*CS0 / 2) - np.log(KPP) - 2*np.log(CPtot))
    elif numClusterS == 1:
        return enh_ns * CPtot**2 / (CPtot+CS0)**2
    
def ana_occupancy(parm, numClusterS):
    '''
    calculate the occupancy of targets
    '''
    C0, CP0, CN0, CS0, KPN, KPS, KPP, gamma, chiPS, chiPN, alpha = __get_parms(parm)
    CPeq = __ana_PtotFrac(__ana_enh_II(parm, numClusterS), alpha)*CP0 / (1 + chiPN + chiPS + gamma*chiPS*chiPN)
    EnhChiPN = gamma*chiPN
    if numClusterS == 0:
        raise ValueError('There is no targets, thus no target occupancy!')
    elif numClusterS == 2:
        dimerPart = (KPP*CPeq**2) * KPS * ((2 + 4*EnhChiPN + 2*EnhChiPN**2) + 2*C0*KPS*(1+2*EnhChiPN+EnhChiPN**2))
    elif numClusterS == 1:
        dimerPart = (KPP*CPeq**2) * KPS * (2 + 4*EnhChiPN + 2*EnhChiPN**2)
    monomerPart = KPS*CPeq * (EnhChiPN + 1)
    if dimerPart + monomerPart < 1:
        return dimerPart + monomerPart
    elif dimerPart + monomerPart < 0:
        return 0
    else:
        return 1

def ana_enh(parm, numClusterS):
    '''
    calculate the dimerization enhancement / effective dimerization strength
    numClusterS: 0 for model A, 1 for model B, 2 for model C
    '''
    if numClusterS == 0:
        return __ana_enh_ns(parm)
    else:
        occ = ana_occupancy(parm, numClusterS)
        if occ < 1:
            return __ana_enh_II(parm, numClusterS)
        else:
            return __ana_enh_III(parm, numClusterS)

def ana_equi_CP(parm, numClusterS):
    '''
    calculate the equilibrium concentration of protein monomers
    '''
    C0, CP0, CN0, CS0, KPN, KPS, KPP, gamma, chiPS, chiPN, alpha = __get_parms(parm)
    if numClusterS == 0:
        CPeq = __ana_PtotFrac(__ana_enh_ns(parm), alpha) * CP0 / (1 + chiPN)
    else:
        occ = ana_occupancy(parm, numClusterS)
        if occ < 1:
            de = (1 + chiPN + chiPS + gamma*chiPS*chiPN)
            CPeq = __ana_PtotFrac(__ana_enh_II(parm, numClusterS), alpha) * CP0 / de
        else:
            alpha = KPP * (CP0 - CS0)
            CPeq = __ana_PtotFrac(__ana_enh_ns(parm), alpha) * (CP0 - CS0) / (1 + chiPN)
            
    return CPeq
        
def ana_bound_ratio(parm, numClusterS):
    '''
    calculate the ratio of proteins bound to DNA
    '''
    C0, CP0, CN0, CS0, KPN, KPS, KPP, gamma, chiPS, chiPN, alpha = __get_parms(parm)
    CPeq = ana_equi_CP(parm, numClusterS)
    pBounRatio = 1 - CPeq/CP0 - 2*KPP*CPeq**2/CP0
    if 0 <= pBounRatio <= 1:
        return pBounRatio
    else:
        raise ValueError('nan')

def ana_resTime_balance(parm, numClusterS):
    '''
    calculate the residence time at equilibrium
    Assume St = S0
    '''
    # by default, all ka=1e4
    C0, CP0, CN0, CS0, KPN, KPS, KPP, gamma, chiPS, chiPN, alpha = __get_parms(parm)
    konS, koffN, konN, konP = parm[['kaPS', 'kbPN', 'kaPN', 'kaPP']]
    CPeq = ana_equi_CP(parm, numClusterS)
    if numClusterS == 0:
        nu = 1 + 4*KPP*CPeq + 2*gamma*chiPN*KPP*CPeq
        de = 1 + 4*KPP*CPeq + 2*(konP/koffN * CPeq)
        resT = 1/koffN * nu / de
    else:
        pBound = CP0 - CPeq - 2*KPP*CPeq**2
        SNBinding = (konN*CN0 + konS*CS0) * (CPeq + 4*KPP*CPeq**2)
        PBinding = 2*konP * CPeq * (chiPN + chiPS + chiPS*gamma*chiPN) * CPeq
        resT = pBound / (SNBinding + PBinding)
    return resT