import numpy as np
# calculate residence times
def calc_resT_modelC(parm, equi):
    # get parameters
    NPsys = np.sum(equi[['P', 'PN', 'PS', 'PSN']]) \
          + 2*np.sum(equi[['PP', 'PPN', 'PPS', 'PSPS', 'PSPN', 'PPSN', 
                           'PNPN', 'PSPSN', 'PNPSN', 'PSNPSN']])
    v = NPsys / parm['CP0']
    kaPS = parm['kaPS']
    kaPN = parm['kaPN']
    kaPP = parm['kaPP']
    # get equilibrium 
    Peq = equi['P']
    PPeq = equi['PP']
    pbound = np.sum(equi[['PN', 'PS', 'PSN']]) \
           + 2*np.sum(equi[['PPN', 'PPS', 'PSPS', 'PSPN', 'PPSN', 
                            'PNPN', 'PSPSN', 'PNPSN', 'PSNPSN']])
    # calculate the DNA on rate
    r_on = (kaPS*equi['S'] + kaPN*equi['N'])/v * (Peq + 2*2*PPeq) \
         + (equi['PN'] + equi['PS'] + equi['PSN'])/v * 2*kaPP * Peq
    return pbound/r_on
    
def calc_resT_modelB(parm, equi):
    # get parameters
    NPsys = np.sum(equi[['P', 'PN', 'PS', 'PSN']]) \
          + 2*np.sum(equi[['PP', 'PPN', 'PPS', 'PSPN', 'PPSN', 'PNPN', 'PNPSN']])
    v = NPsys / parm['CP0']
    kaPS = parm['kaPS']
    kaPN = parm['kaPN']
    kaPP = parm['kaPP']
    # get equilibrium 
    Peq = equi['P']
    PPeq = equi['PP']
    pbound = np.sum(equi[['PN', 'PS', 'PSN']]) \
           + 2*np.sum(equi[['PPN', 'PPS', 'PSPN', 'PPSN', 'PNPN', 'PNPSN']])
    # calculate the DNA on rate
    r_on = (kaPS*equi['S'] + kaPN*equi['N'])/v * (Peq + 2*2*PPeq) \
         + (equi['PN'] + equi['PS'] + equi['PSN'])/v * 2*kaPP * Peq
    return pbound/r_on

def calc_resT_modelA(parm, equi):
    # get parameters
    NPsys = np.sum(equi[['P', 'PN']]) + 2*np.sum(equi[['PP', 'PPN', 'PNPN']])
    v = NPsys / parm['CP0']
    kaPN = parm['kaPN']
    kaPP = parm['kaPP']
    # get equilibrium 
    Peq = equi['P']
    PPeq = equi['PP']
    pbound = np.sum(equi[['PN']]) + 2*np.sum(equi[['PPN', 'PNPN']])
    # calculate the DNA on rate
    r_on = kaPN*equi['N']/v * (Peq + 2*2*PPeq) + equi['PN']/v * 2*kaPP * Peq
    return pbound/r_on

# calculate residence time on S
def calc_resTonS_modelC(parm, equi):
    # get parameters
    NPsys = np.sum(equi[['P', 'PN', 'PS', 'PSN']]) \
          + 2*np.sum(equi[['PP', 'PPN', 'PPS', 'PSPS', 'PSPN', 'PPSN', 
                           'PNPN', 'PSPSN', 'PNPSN', 'PSNPSN']])
    v = NPsys / parm['CP0']
    kaPS = parm['kaPS']
    gamma = parm['gamma']
    C0 = 0.6022
    # get equilibrium 
    Peq = equi['P']
    PPeq = equi['PP']
    sOcc = np.sum(equi[['PS', 'PSN', 'PPS', 'PSPN', 'PPSN', 'PNPSN']]) \
           + 2*np.sum(equi[['PSPS', 'PSPSN', 'PSNPSN']])
    # calculate the specific site on rate
    r_on = kaPS*equi['S']/v * (Peq + 4*PPeq) \
            + gamma*kaPS*equi['S']/v * (equi['PN'] + 4*equi['PPN'] + 4*equi['PNPN'])
    return sOcc/r_on

def calc_resTonS_modelB(parm, equi):
    # get parameters
    NPsys = np.sum(equi[['P', 'PN', 'PS', 'PSN']]) \
          + 2*np.sum(equi[['PP', 'PPN', 'PPS', 'PSPN', 'PPSN', 'PNPN', 'PNPSN']])
    v = NPsys / parm['CP0']
    kaPS = parm['kaPS']
    gamma = parm['gamma']
    # get equilibrium 
    Peq = equi['P']
    PPeq = equi['PP']
    sOcc = np.sum(equi[['PS', 'PSN', 'PPS', 'PSPN', 'PPSN', 'PNPSN']])
    # calculate the specific site on rate
    r_on = kaPS*equi['S']/v * (Peq + 2*PPeq) \
            + gamma*kaPS*equi['S']/v * (equi['PN'] + 2*equi['PPN'] + 2*equi['PNPN'])
    return sOcc/r_on

# calculate S site occupancies
def calc_occS_modelC(parm, equi):
    # get equilibrium
    Socc = np.sum(equi[['PS','PSN','PPS','PPSN','PSPN', 'PNPSN']])\
            + 2*np.sum(equi[['PSPS','PSPSN','PSNPSN']])
    return Socc/(Socc+equi['S'])

def calc_occS_modelB(parm, equi):
    # get equilibrium
    Socc = np.sum(equi[['PS','PSN','PPS','PPSN','PSPN', 'PNPSN']])
    return Socc/(Socc+equi['S'])
    
# calculate protein bound ratio
def calc_BoundRatio_modelC(parm, equi):
    # get equilibrium
    Pbound = np.sum(equi[['PN', 'PS', 'PSN']]) \
           + 2*np.sum(equi[['PPN', 'PPS', 'PSPS', 'PSPN', 'PPSN', 
                            'PNPN', 'PSPSN', 'PNPSN', 'PSNPSN']])
    return Pbound/(Pbound+equi['P']+2*equi['PP'])

def calc_BoundRatio_modelB(parm, equi):
    # get equilibrium
    Pbound = np.sum(equi[['PN', 'PS', 'PSN']]) \
           + 2*np.sum(equi[['PPN', 'PPS', 'PSPN', 'PPSN', 'PNPN', 'PNPSN']])
    return Pbound/(Pbound+equi['P']+2*equi['PP'])
    
def calc_BoundRatio_modelA(parm, equi):
    # get equilibrium
    Pbound = np.sum(equi[['PN']]) + 2*np.sum(equi[['PPN', 'PNPN']])
    return Pbound/(Pbound+equi['P']+2*equi['PP'])
    
# calculate dimerization enhancement
def calc_enhance_modelC(parm, equi):
    # get parameters
    v = 1e6/parm['CP0']
    KPP = parm['KPP']
    # get equilibrium
    dimers = np.sum(equi[['PP', 'PPN', 'PPS', 'PSPS', 'PSPN', 'PPSN', 
                'PNPN', 'PSPSN', 'PNPSN', 'PSNPSN']])
    monomers = np.sum(equi[['P', 'PN', 'PS', 'PSN']])
    return v*dimers/monomers**2/KPP
    
def calc_enhance_modelB(parm, equi):
    # get parameters
    v = 1e6/parm['CP0']
    KPP = parm['KPP']
    # get equilibrium
    dimers = np.sum(equi[['PP', 'PPN', 'PPS', 'PSPN', 'PPSN', 'PNPN', 'PNPSN']])
    monomers = np.sum(equi[['P', 'PN', 'PS', 'PSN']])
    return v*dimers/monomers**2/KPP

def calc_enhance_modelA(parm, equi):
    # get parameters
    v = 1e6/parm['CP0']
    KPP = parm['KPP']
    # get equilibrium
    dimers = np.sum(equi[['PP', 'PPN', 'PNPN']])
    monomers = np.sum(equi[['P', 'PN']])
    return v*dimers/monomers**2/KPP

def processDATA(parm_file, equi_file_Nonly, equi_file_singleS, equi_file_doubleS):
    resT_Nonly = []
    resT_singleS = []
    resT_doubleS = []
    resT_onS_singleS = []
    resT_onS_doubleS = []
    rate_search_singleS = []
    rate_search_doubleS = []
    sOcc_singleS = []
    sOcc_doubleS = []
    enh_Nonly = []
    enh_singleS = []
    enh_doubleS = []
    pBound_Nonly = []
    pBound_singleS = []
    pBound_doubleS = []
    ratioSNS_singleS = []
    ratioSNS_doubleS = []
    for iloc in progressbar(range(parm_file.shape[0])):
        parmi = parm_file.iloc[iloc]
        equi_Nonlyi = equi_file_Nonly.iloc[iloc]
        equi_singleSi = equi_file_singleS.iloc[iloc]
        equi_doubleSi = equi_file_doubleS.iloc[iloc]
        # residence times
        resT_Nonly.append(calc_resT_modelA(parmi, equi_Nonlyi))
        resT_singleS.append(calc_resT_modelB(parmi, equi_singleSi))
        resT_doubleS.append(calc_resT_modelC(parmi, equi_doubleSi))
        # target binding dynamics
        resT_onS_singleS.append(calc_resTonS_modelB(parmi, equi_singleSi))
        resT_onS_doubleS.append(calc_resTonS_modelC(parmi, equi_doubleSi))
        # S site occupancies
        sOcc_singleS.append(calc_occS_modelB(parmi, equi_singleSi))
        sOcc_doubleS.append(calc_occS_modelC(parmi, equi_doubleSi))
        # enhancement
        enh_Nonly.append(calc_enhance_modelA(parmi, equi_Nonlyi))
        enh_singleS.append(calc_enhance_modelB(parmi, equi_singleSi))
        enh_doubleS.append(calc_enhance_modelC(parmi, equi_doubleSi))
        # protein bound ratio
        pBound_Nonly.append(calc_BoundRatio_modelA(parmi, equi_Nonlyi))
        pBound_singleS.append(calc_BoundRatio_modelB(parmi, equi_singleSi))
        pBound_doubleS.append(calc_BoundRatio_modelC(parmi, equi_doubleSi))
    
    resT_Nonly = np.array(resT_Nonly)
    resT_singleS = np.array(resT_singleS)
    resT_doubleS = np.array(resT_doubleS)
    resT_onS_singleS = np.array(resT_onS_singleS)
    resT_onS_doubleS = np.array(resT_onS_doubleS)
    sOcc_singleS = np.array(sOcc_singleS)
    sOcc_doubleS = np.array(sOcc_doubleS)
    enh_Nonly = np.array(enh_Nonly)
    enh_singleS = np.array(enh_singleS)
    enh_doubleS = np.array(enh_doubleS)
    pBound_Nonly = np.array(pBound_Nonly)
    pBound_singleS = np.array(pBound_singleS)
    pBound_doubleS = np.array(pBound_doubleS)
    
    return (resT_Nonly, resT_singleS, resT_doubleS, 
            resT_onS_singleS, resT_onS_doubleS, 
            sOcc_singleS, sOcc_doubleS, 
            enh_Nonly, enh_singleS, enh_doubleS, 
            pBound_Nonly, pBound_singleS, pBound_doubleS)

class EQUIDATA():
    def __init__(self, parm_file, equi_file_Nonly, equi_file_singleS, equi_file_doubleS):
        results = processDATA(parm_file, equi_file_Nonly, equi_file_singleS, equi_file_doubleS)
        (resT_Nonly, resT_singleS, resT_doubleS, 
        resT_onS_singleS, resT_onS_doubleS, 
        sOcc_singleS, sOcc_doubleS, 
        enh_Nonly, enh_singleS, enh_doubleS, 
        pBound_Nonly, pBound_singleS, pBound_doubleS) = results
        # save parameters
        self.parm_file = parm_file
        # save results
        self.resT_Nonly = resT_Nonly
        self.resT_singleS = resT_singleS
        self.resT_doubleS = resT_doubleS
        self.resT_onS_singleS = resT_onS_singleS
        self.resT_onS_doubleS = resT_onS_doubleS
        self.sOcc_singleS = sOcc_singleS
        self.sOcc_doubleS = sOcc_doubleS
        self.enh_Nonly = enh_Nonly
        self.enh_singleS = enh_singleS
        self.enh_doubleS = enh_doubleS
        self.pBound_Nonly = pBound_Nonly
        self.pBound_singleS = pBound_singleS
        self.pBound_doubleS = pBound_doubleS