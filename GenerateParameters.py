import numpy as np
import pandas as pd

# Generate parameters
# input units: M
# output units: nm & s
# conversion: 0.6022 nm^-3 = 1 M

def __staticParms(
        numTargets:int = 0, kbPP_fixed = np.nan, 
        kaPN_given = np.nan, kaPS_given = np.nan,
        lengthScale=1
        ):
    '''Generate static parameters for the model.
    Input:
        numTargets: int, whether the system has targets
        kbPP_fixed: float, the fixed value of kbPP. Set to np.nan if it is not fixed.
        lengthScale: float, multiple the length by this factor.
    Output:
        kaPP, kbPP, kaPN, kbPN, kaPS, kbPS, V0, VtoL, CN0, CS0, C0
    '''
    
    if kaPN_given is np.nan:
        kaPN = lambda K: np.round(200,10)
    else:
        kaPN = lambda K: np.round(kaPN_given,10)
    kbPN = lambda K: np.round(kaPN(K)/K,10)
    
    if kaPS_given is np.nan:
        kaPS = lambda K: np.round(1000,10)
    else:
        kaPS = lambda K: np.round(kaPS_given,10)
    kbPS = lambda K: np.round(kaPS(K)/K,10)
    
    def kbPP(K):
        if K==0:
            return 10
        elif K == np.inf:
            return 0
        elif kbPP_fixed is np.nan:
            return 10 / (K*C0/1e1)**(4/9)
        else:
            return kbPP_fixed
    def kaPP(K):
        if K==0:
            return 0
        elif K == np.inf:
            return 1e7
        else:
            return np.round(K*kbPP(K), 10)
    
    scale = lengthScale
    L = 21 * scale # nm
    VtoL = 1e3 # nm^2
    V0 = VtoL * L # nm^3

    if isinstance(numTargets, int):
        CN0 = (12*scale-numTargets)/V0 # nm^-3
        CS0 = numTargets/V0
        C0 = 0.6022 # 0.6022 nm^-3 = 1 M 
    else:
        raise ValueError('numTargets should be an integer!')
    
    return kaPP, kbPP, kaPN, kbPN, kaPS, kbPS, V0, VtoL, CN0, CS0, C0

def GenParameters(
    filename='', ifwrite=False, numTargets:int = 0,
    KPS=lambda KPN: 1e3*KPN, kaPS=np.nan,
    CP0=[np.nan], NP0=[np.nan], KPP=[np.nan], kbPP_fixed = np.nan,
    KPN=[np.nan], kaPN=np.nan, 
    area3Dto1D=[np.nan], gamma=[np.nan], lengthScale=1,
):
    '''Generate parameters for the model.
    Input:
        filename: str, the name of the output CSV file.
        ifwrite: bool, whether to write the data to a CSV file.
        numTargets: int, whether the system has targets.
        KPS: function, the function to calculate KPS according to KPN.
        CP0: list, the protein concentration in M. CP0 and NP0 cannot be both np.nan.
        NP0: list, the protein number. If CP0 is provided, NP0 will be calculated.
        KPN: list, the equilibrium constant for P-N binding.
        KPP: list, the equilibrium constant for P-P binding.
        kbPP_fixed: float, the fixed value of kbPP. Set to np.nan if it is not fixed.
        area3Dto1D: list, to calculate gamma = V/L / area3Dto1D.
        lengthScale: float, multiple the length by this factor.
    Output:
        pd.DataFrame, the parameters generated.
    '''
    headers = [
        "ID", "CN0", "CS0", "CP0", 
        "KPS", "KPN", "KPP", "gamma", 
        "kaPS", "kaPN", "kaPP", 
        "kbPS", "kbPN", "kbPP"
    ]

    # generate data
    data = []
    number = 0
    kaPP, kbPP, kaPN, kbPN, kaPS, kbPS, V0, VtoL, CN0, CS0, C0 = \
    __staticParms(numTargets, kbPP_fixed, kaPN, kaPS, lengthScale)

    # These parameters are fixed
    if np.isnan(CP0).any() & np.isnan(NP0).any():
        raise ValueError('No protein provided! Set CP0 or NP0!')
    elif not np.isnan(CP0).any():
        CP0_list = np.array(CP0)*C0 # convert M to nm^-3
    else:
        CP0_list = np.array(NP0) / V0
        
    if np.isnan(KPN).any():
        raise ValueError('No KPN provided!')
    else:
        KPN_list = np.array(KPN)/C0
        
    if np.isnan(KPP).any():
        raise ValueError('No KPP provided!')
    else:
        KPP_list = np.array(KPP)/C0
        
    if np.isnan(area3Dto1D).any():
        if np.isnan(gamma).any():
            raise ValueError('No gamma nor area3Dto1D provided!')
        else:
            area3Dto1D_list = VtoL / np.array(gamma)
    else:
        area3Dto1D_list = np.array(area3Dto1D)
    
    for _CP0 in CP0_list:
        # These parameters are varied
        for _KPN in KPN_list:
            _KPS = KPS(_KPN*C0)/C0
            for _area3Dto1D in area3Dto1D_list:
                for _KPP in KPP_list:
                    # _gamma = V0 / (V0/VtoL - excludeL1D*(_CP0*V0))
                    _gamma = VtoL / _area3Dto1D
                    entry = {
                        'ID':number,
                        "CN0":CN0,
                        "CS0":CS0,
                        "CP0":_CP0,
                        "KPN":np.round(_KPN, 10),
                        "kaPN":kaPN(_KPN),
                        "kbPN":kbPN(_KPN),
                        "KPS":np.round(_KPS, 10),
                        "kaPS":kaPS(_KPS),
                        "kbPS":kbPS(_KPS),
                        "KPP":np.round(_KPP, 10),
                        "kaPP":kaPP(_KPP),
                        "kbPP":kbPP(_KPP),
                        "gamma":_gamma
                    }
                    data.append(entry)
                    number += 1

    if ifwrite:
        # Specify the output CSV file name
        output_file = filename
        # # Write the data to the CSV file
        import csv
        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        print("CSV file generated successfully!")
    
    return pd.DataFrame(data, columns=headers)


def getParmRange(
        parameters, V0=2.1e4
    ):
    ''' Get the range of parameters.
    Input:
        parameters: pd.DataFrame, the parameters generated.
        V0: float, the volume of the system.
    Output:
        print out the range of parameters, and the number of unique values.
    '''
    C0 = 0.6022 # 0.6022 nm^-3 = 1 M
    print('Range of parameters (min, max, number of dp) (in Molar):')
    print('N0:', parameters['CN0'].min()*V0, parameters['CN0'].max()*V0, len(parameters['CN0'].nunique()))
    print('S0:', parameters['CS0'].min()*V0, parameters['CS0'].max()*V0, len(parameters['CS0'].nunique()))
    print('P0:', parameters['CP0'].min()*V0, parameters['CP0'].max()*V0, len(parameters['CP0'].nunique()))
    print(
        'KPS:', 
        '1e%.0f'%np.log10(parameters['KPS'].min()/C0), 
        '1e%.0f'%np.log10(parameters['KPS'].max()/C0), 
        len(parameters['KPS'].nunique())
    )
    print(
        'KPN:', 
        '1e%.0f'%np.log10(parameters['KPN'].min()/C0), 
        '1e%.0f'%np.log10(parameters['KPN'].max()/C0), 
        len(parameters['KPN'].nunique())
    )
    print(
        'kbPP:', 
        '1e%.0f'%np.log10(parameters['KPP'].min()/C0), 
        '1e%.0f'%np.log10(parameters['KPP'].max()/C0), 
        len(parameters['KPP'].nunique())
    )
    if parameters['KPP'].nunique() == np.inf:
        print('KPP: inf, inf, 1')
    else:
        print(
            'KPP:', 
            '1e%.0f'%np.log10(parameters['KPP'].min()/C0), 
            '1e%.0f'%np.log10(parameters['KPP'].max()/C0), 
            len(parameters['KPP'].nunique())
        )
    