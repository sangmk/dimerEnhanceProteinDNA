import numpy as np
from scipy.special import erfcx

'''
Define functions used for theoretical solutions
'''

def irr_distinct(kon:float, A0:float, B0:float, t):
    '''
    Theoretical solution for irreversible reaction A + B -> C

    Input:
    kon is the macroscopic reaction rate. Refer to Table 1 in (Fu et al. 2019)
    A0 is the initial concentration of molecule A
    B0 is the initial concentration of molecule B
    t is the time range

    Output:
    At, the concentration of molecule A
    Bt, the concentration of molecule B
    Ct, the concentration of product C
    '''
    if abs(A0 - B0) < 10**-10:
        At = A0 * np.exp( (A0-B0) * kon * t ) * (1 - A0/B0) / (1 - A0/B0 * np.exp( (A0-B0) * kon * t ))
        Bt = B0 + At - A0
        Ct = A0 - At
    else:
        At = A0 / ( kon * t * A0 + 1 )
        Bt = A0 / ( kon * t * A0 + 1 )
        Ct = A0 - At

    return At, Bt, Ct

def Torney(N:int, L:float, D:float, currt):
    '''
    Calculate the survival probability from Torney's theory

    Input:
    N, initial number of molecules
    L, length of the fiber (nm)
    D, diffusion constant (nm^2/us)
    currt, time in s
    '''
    t = currt*10**6
    A0 = N/L
    z = A0**2*D*t
    return erfcx((8*z)**0.5)*N

def equi_self(A0:int, vol:float, keq:float) -> float:
    
    a = 2*keq
    b = 1
    c = -A0/vol
    delta = b**2 - 4*a*c
    cA = (-b + np.sqrt(delta))/2/a
    
    return cA*vol

def equi_self_dense(A0, vol, exc_vol, keq):
    a = 2*keq - exc_vol
    b = exc_vol*A0 + vol
    c = -A0*vol
    return (-b + np.sqrt(b**2 - 4*a*c)) / 2/a


# def equi_self_exp(A0, vol, keq, dist_max=20):

#     dxxx = 0.001
#     xxx = np.arange(dxxx, dist_max, dxxx)
#     ave_dist = vol/A0
#     pdf_xxx = np.exp(-xxx/ave_dist)/ave_dist
#     equi_xxx = equi_self(1, xxx, keq)
#     self_equi = sum(equi_xxx*pdf_xxx)*dxxx*A0
    
#     return self_equi

def equi(A0:int, B0:int, vol:float, keq:float, target:str='A') -> float:

    '''
    Equilirbium solution for reversible reaction A + B <-> C

    Input:
    A0 is the initial concentration of molecule A
    B0 is the initial concentration of molecule B
    keq is the equilibrium constant
    target:
        'A' for A
        'B' for B

    Output:
    The concentration of molecule {target}
    '''
    
    b = keq*B0/vol - keq*A0/vol + 1
    delta = b**2 + 4*keq*A0/vol
    cA = (-b + np.sqrt(delta))/2/keq
    
    if target == 'A':
        return cA*vol
    elif target == 'B':
        return B0 - (A0 - cA*vol)