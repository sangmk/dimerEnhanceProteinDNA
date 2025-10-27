from .nerdssData import nerdssData
from . import theory
import matplotlib.pyplot as plt
import numpy as np


def Torney(model: nerdssData, targetMol: str, targetCmplx: str,
           legend_sim='', legend_theroy='', ls_sim='--', ls_theory='-'):
    '''
    Parameters
    ----------
    model: nerdssData
        Data read from file
    targetMol: str
        The molecule you are interested in, is used to find its number. 
        This can be found in nerdssData.parms['molecules'].
    targetCmplx: str
        The target complex is used to fetch time sequence data. 
        It should be in the form A(a), and can be found in nerdssData.reactions

    Outputs
    -------
    l_theory: matplotlib.lines.Line2D
        line handel for theory curve
    l_sim: matplotlib.lines.Line2D
        line handel for simulation curve
    '''
    model.readCopy()
    theory_Torney = theory.Torney(N=model.parms.molecules[targetMol],
                                  L=model.parms.waterbox[0],
                                  D=model.molecules[targetMol].D[0],
                                  currt=model.time)
    if legend_theroy == '':
        l_theory, = plt.plot(model.time, theory_Torney, ls=ls_theory)
    else:
        l_theory, = plt.plot(model.time, theory_Torney,
                             ls=ls_theory, label=legend_theroy)
    if legend_sim == '':
        l_sim, = plt.plot(model.time, model.MOLS[targetCmplx], ls=ls_sim)
    else:
        l_sim, = plt.plot(
            model.time, model.MOLS[targetCmplx], ls=ls_sim, label=legend_sim)

    return l_theory, l_sim


def rev_3D(model: nerdssData, MoleculeA: str, MoleculeB: str,
           targetRxn: str, targetCmplx: str, targetmol: str,
           legend_sim='', legend_theroy='', ls_sim='--', ls_theory='-'):
    '''
    Compare simulation with theory for reaction A+B<->C
    Parameters
    ----------
    model: nerdssData
        Data read from file
    MoleculeA: str
        The molecules you are interested in, are used to find its number. 
        This can be found in nerdssData.parms['molecules'].
    MoleculeB: str
        Same as MoleculeA
        Now only support heterogeneous reaction
    targetRxn: str
        The reaction that you are interested in, can be found in nerdssData.reactions
        This is in the form of A(a)+B(b) <-> A(a)B(b)
    targetCmplx: str
        The target complex is used to fetch time sequence data. 
        It should be in the form A(a), and can be found in nerdssData.reactions
    targetmol: str
        The molecule you are interested in.
        'A' for MoleculeA
        'B' for MoleculeB

    Outputs
    -------
    l_theory: matplotlib.lines.Line2D
        line handel for theory curve
    l_sim: matplotlib.lines.Line2D
        line handel for simulation curve
    '''
    model.readCopy()
    reaction = model.reactions[targetRxn]
    equi_stat = theory.equi(A0=model.parms.molecules[MoleculeA],
                            B0=model.parms.molecules[MoleculeB],
                            vol=model.parms.WBvol,
                            keq=reaction.get_keq(),
                            target=targetmol) * np.ones_like(model.time)
    if legend_theroy == '':
        l_theory, = plt.plot(model.time, equi_stat, ls=ls_theory)
    else:
        l_theory, = plt.plot(model.time, equi_stat,
                             ls=ls_theory, label=legend_theroy)
    if legend_sim == '':
        l_sim, = plt.plot(model.time, model.MOLS[targetCmplx], ls=ls_sim)
    else:
        l_sim, = plt.plot(
            model.time, model.MOLS[targetCmplx], ls=ls_sim, label=legend_sim)

    return l_theory, l_sim


def rev_1D(model: nerdssData, MoleculeA: str, MoleculeB: str,
           targetRxn: str, targetCmplx: str, targetmol: str,
           legend_sim='', legend_theroy='', ls_sim='--', ls_theory='-'):
    '''
    Compare simulation with theory for reaction A+B<->C
    Parameters
    ----------
    model: nerdssData
        Data read from file
    MoleculeA: str
        The molecules you are interested in, are used to find its number. 
        This can be found in nerdssData.parms['molecules'].
    MoleculeB: str
        Same as MoleculeA
        Now only support heterogeneous reaction
    targetRxn: str
        The reaction that you are interested in, can be found in nerdssData.reactions
        This is in the form of A(a)+B(b) <-> A(a)B(b)
    targetCmplx: str
        The target complex is used to fetch time sequence data. 
        It should be in the form A(a), and can be found in nerdssData.reactions
    targetmol: str
        The molecule you are interested in.
        'A' for MoleculeA
        'B' for MoleculeB

    Outputs
    -------
    l_theory: matplotlib.lines.Line2D
        line handel for theory curve
    l_sim: matplotlib.lines.Line2D
        line handel for simulation curve
    '''
    model.readCopy()
    reaction = model.reactions[targetRxn]
    equi_stat = theory.equi(A0=model.parms.molecules[MoleculeA],
                            B0=model.parms.molecules[MoleculeB],
                            vol=model.parms.waterbox[0],
                            keq=reaction.get_keq(),
                            target=targetmol) * np.ones_like(model.time)
    if legend_theroy == '':
        l_theory, = plt.plot(model.time, equi_stat, ls=ls_theory)
    else:
        l_theory, = plt.plot(model.time, equi_stat,
                             ls=ls_theory, label=legend_theroy)
    if legend_sim == '':
        l_sim, = plt.plot(model.time, model.MOLS[targetCmplx], ls=ls_sim)
    else:
        l_sim, = plt.plot(
            model.time, model.MOLS[targetCmplx], ls=ls_sim, label=legend_sim)

    return l_theory, l_sim


def rev_1D_self(model: nerdssData, MoleculeA: str,
                targetRxn: str, targetCmplx: str,
                legend_sim='', legend_theroy='', ls_sim='--', ls_theory='-'):
    '''
    Compare simulation with theory for reaction A+B<->C
    Parameters
    ----------
    model: nerdssData
        Data read from file
    MoleculeA: str
        The molecules you are interested in, are used to find its number. 
        This can be found in nerdssData.parms['molecules'].
    targetRxn: str
        The reaction that you are interested in, can be found in nerdssData.reactions
        This is in the form of A(a)+A(b) <-> A(a)A(a)
    targetCmplx: str
        The target complex is used to fetch time sequence data. 
        It should be in the form A(a), and can be found in nerdssData.reactions

    Outputs
    -------
    l_theory: matplotlib.lines.Line2D
        line handel for theory curve
    l_sim: matplotlib.lines.Line2D
        line handel for simulation curve
    '''
    model.readCopy()
    reaction = model.reactions[targetRxn]
    equi_stat = theory.equi_self_exp(A0=model.parms.molecules[MoleculeA],
                                     vol=model.parms.waterbox[0],
                                     keq=reaction.get_keq()) \
        * np.ones_like(model.time)
    if legend_theroy == '':
        l_theory, = plt.plot(model.time, equi_stat, ls=ls_theory)
    else:
        l_theory, = plt.plot(model.time, equi_stat,
                             ls=ls_theory, label=legend_theroy)
    if legend_sim == '':
        l_sim, = plt.plot(model.time, model.MOLS[targetCmplx], ls=ls_sim)
    else:
        l_sim, = plt.plot(
            model.time, model.MOLS[targetCmplx], ls=ls_sim, label=legend_sim)

    return l_theory, l_sim
