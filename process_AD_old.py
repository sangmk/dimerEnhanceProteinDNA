import numpy as np

# Reusable helper to apply one parsed reaction line and update protein/complex state
# It returns currt (seconds) from rxnInfo.
def _apply_rxn_update(rxntype, rxnInfo, proteinList, complexList, __add_protein_to_list, __add_newComplex, debug=False):
    if rxntype == 'rxnPRO':
        pro1, pro1ID, pro2, pro2ID, rxn, currt = rxnInfo
        __add_protein_to_list(proteinList, complexList, pro1, pro1ID)
        __add_protein_to_list(proteinList, complexList, pro2, pro2ID)
        if rxn == 'BOND':
            # move pro2 into pro1's complex
            del complexList[proteinList[pro2ID].multimerID]
            complexList[proteinList[pro1ID].multimerID].addComponent(proteinList[pro2ID], currt)
        elif rxn == 'BREAK':
            if proteinList[pro1ID].multimerID != -1:
                protein = complexList[proteinList[pro1ID].multimerID].removeLastComponent(currt)
                protein.updateBoundState(False, currt)
                __add_newComplex(complexList, protein)
        return currt
    elif rxntype == 'rxnSUB':
        pro, proID, sub, subID, rxn, currt = rxnInfo
        __add_protein_to_list(proteinList, complexList, pro, proID)
        if rxn == 'BOND':
            proteinList[proID].addSubstrate(sub)
            complexList[proteinList[proID].multimerID].updateBoundState(currt)
        elif rxn == 'BREAK':
            if sub in proteinList[proID].substrates:
                proteinList[proID].removeSubstrate(sub)
                complexList[proteinList[proID].multimerID].updateBoundState(currt)
        return currt
    else:
        # IGNORE or unexpected
        return rxnInfo[-1]

def __read_A_D_line(line, dt, proteins, substrates):
    '''read and process a line from assoc_dissoc_time.dat

    Input:
        line: a line from assoc_dissoc_time.dat
        dt: time step (s)
        proteins: list of protein names
        substrates: list of substrate names
    Output:
        rxntype: 'rxnPRO' or 'rxnSUB'
        rxnInfo: 
            tuple of (pro, proID, sub, subID, rxn, currt) if rxntype is 'rxnSUB'
            tuple of (pro1, pro1ID, pro2, pro2ID, rxn, currt) if rxntype is 'rxnPRO'
    '''
    linelist = line.split(',')
    currt = float(linelist[0].split(':')[1])*dt
    mol1, mol1id, mol2, mol2id = linelist[2], linelist[3], linelist[5], linelist[6]
    reaction = linelist[1]
    if (mol1 in proteins) and (mol2 in substrates):
        return 'rxnSUB', (mol1, mol1id, mol2, mol2id, reaction, currt)
    elif (mol1 in substrates) and (mol2 in proteins):
        return 'rxnSUB', (mol2, mol2id, mol1, mol1id, reaction, currt)
    elif (mol1 in proteins) and (mol2 in proteins):
        return 'rxnPRO', (mol1, mol1id, mol2, mol2id, reaction, currt)
    elif (mol1 in substrates) and (mol2 in substrates):
        return 'IGNORE', (mol1, mol1id, mol2, mol2id, reaction, currt)
    else:
        return KeyError('Molecules: %s %s do not fit types:'%(mol1, mol2), proteins, substrates)

class PROTEIN():
    '''Info of a protein'''
    def __init__(self, name:str, proID: int):
        self.name = name
        self.id = proID
        self.substrates = [] # the substrates this protein binds to
        self.multimerID = -1
        self.boundstate = False
        self.boundStartTime = -1 # set to -1 after dissociation
        self.boundEndTime = -1 # set to -1 after association
        self.resTimeList = [] # the time this protein has been in bound state
        self.searchTimeList = [] # the time this protein has been in search state

    def boundToSubstrates(self):
        return bool(self.substrates)

    def addSubstrate(self, substrate: str):
        '''Add a substrate to the protein. Update the bound state of the protein.\n 
        Notice that the start and end time of binding are not updated here'''
        self.substrates.append(substrate)
        self.boundstate = self.boundToSubstrates() # if there is any substrate, then it is in bound state

    def removeSubstrate(self, substrate: str):
        '''Remove a substrate from the protein. Update the bound state of the protein.\n
        Notice that the start and end time of binding are not updated here'''
        self.substrates.remove(substrate)
        self.boundstate = self.boundToSubstrates() # if there is any substrate, then it is in bound state

    def updateMultimer(self, multimerID: int):
        self.multimerID = multimerID

    def updateBoundState(self, complexBound: bool, currt: float):
        self.boundstate = complexBound or self.boundToSubstrates()
        self.updateBoundTime(currt)

    def updateBoundTime(self, currt: float):
        '''update the start and end time of binding. Also calculate the residence time and search time'''
        # decide whether the current time is the start or end time of binding
        if self.boundStartTime < 0:
            # the complex was in search state
            if self.boundstate == True:
                self.boundStartTime = currt
                if self.boundEndTime > 0:
                    self.searchTimeList.append(self.boundStartTime - self.boundEndTime)
                    self.boundEndTime = -1
                else:
                    # do not count first arrival
                    pass
            else:
                # remains in search state
                pass
        else:
            # the complex was in bound state
            if self.boundstate == False:
                self.boundEndTime = currt
                if self.boundStartTime > 0:
                    self.resTimeList.append(self.boundEndTime - self.boundStartTime)
                    self.boundStartTime = -1
                else:
                    # in case this protein is in bound state at the beginning
                    pass
            else:
                # remains in bound state
                pass
        
    def __str__(self):
        return 'Protein %s: %s'%(self.id, self.name)
    
    def __repr__(self):
        return self.__str__()
            
class DIMER():
    '''Info of a multimer'''
    def __init__(self, cmplxID: int):
        self.id = cmplxID
        self.components:list[PROTEIN] = [] # the proteins this complex has
        self.boundstate = False
    
    def addComponent(self, protein: PROTEIN, currt: float):
        '''Add a protein to the complex. Update this complex and its components'''
        self.components.append(protein)
        # update the multimerID of the new protein
        protein.updateMultimer(self.id)
        # update the bound state of all proteins in the complex
        self.updateBoundState(currt)
    
    def removeLastComponent(self, currt: float):
        '''Remove a protein from the complex. Update this complex and its components'''
        protein = self.components.pop()
        # if no protein in the complex is in bound state, then the complex is in search state
        self.updateBoundState(currt)
        # return the protein for further processing
        return protein
    
    def isEmpty(self):
        return not self.components
    
    def updateBoundState(self, currt: float):
        self.boundstate = any([pro.boundToSubstrates() for pro in self.components])
        for pro in self.components:
            pro.updateBoundState(self.boundstate, currt)

    def __str__(self):
        return 'Complex %d: %s'%(self.id, ['%s:%s'%(pro.name, pro.id) for pro in self.components])

    def __repr__(self):
        return self.__str__()

    
def readResT_from_NERDSS(
        file_assoc_dissoc_times:str, dt:float, proteins=['P'], substrates=['S','N'],
        debug:bool=False, startT:float=0
):
    '''Read residence time and search time from assoc_dissoc_time.dat. Return numpy arrays of residence time and search time
    
    Input:
        file_assoc_dissoc_times: path to assoc_dissoc_time.dat
        dt: time step (s)
        proteins: list of protein names
        substrates: list of substrate names
        debug: print out the information of each protein
    Output:
        resTimesAll: numpy array of residence time
        searchTimesAll: numpy array of search time
    '''

    def __maxComplexID(complexList):
        if not complexList:
            # if the complex list is empty
            return 0
        else:
            return max(complexList.keys())
    
    def __add_newComplex(complexList, protein):
        # create a new multimer
        newMultimerID = __maxComplexID(complexList) + 1
        newMultimer = DIMER(newMultimerID)
        newMultimer.addComponent(protein, currt=-1)
        complexList[newMultimerID] = newMultimer
    
    def __add_protein_to_list(proteinList, complexList, pro, proID):
        if proID not in proteinList:
            # create a new protein
            proteinList[proID] = PROTEIN(pro, proID)
            # add this protein to complex list, the id is the next integer
            __add_newComplex(complexList, proteinList[proID])

    with open(file_assoc_dissoc_times, 'r') as f:
        complexList:dict[int,DIMER] = {} # complex id: DIMER
        proteinList:dict[int,PROTEIN] = {} # protein id: PROTEIN
        for line in f:
            if debug: print(line.strip())
            rxntype, rxnInfo = __read_A_D_line(line, dt, proteins, substrates)
            if debug: print(rxntype, rxnInfo)
            if rxnInfo[-1] < startT:
                # skip the reaction if it happens before the start time
                continue
            currt = _apply_rxn_update(rxntype, rxnInfo, proteinList, complexList, __add_protein_to_list, __add_newComplex, debug)
            if debug and rxntype == 'rxnPRO':
                pro1, pro1ID, pro2, pro2ID, _, _ = rxnInfo
                print(pro1ID, 'startT: %.3f'%proteinList[pro1ID].boundStartTime, 'endT: %.3f'%proteinList[pro1ID].boundEndTime)
                print(pro2ID, 'startT: %.3f'%proteinList[pro2ID].boundStartTime, 'endT: %.3f'%proteinList[pro2ID].boundEndTime)
            elif debug and rxntype == 'rxnSUB':
                pro, proID, *_ = rxnInfo
                for memberPro in complexList[proteinList[proID].multimerID].components:
                    print('startT: %.3f'%memberPro.boundStartTime, 'endT: %.3f'%memberPro.boundEndTime)
        
        if debug: print()
    # finished reading the file
    # get the residence time and search time of each protein
    resTimesAll = []
    searchTimesAll = []
    for proID in proteinList:
        if debug: print(proID, proteinList[proID].resTimeList)
        resTimesAll.extend(proteinList[proID].resTimeList)
        searchTimesAll.extend(proteinList[proID].searchTimeList)
    # return numpy arrays of residence time and search time
    return np.array(resTimesAll), np.array(searchTimesAll)

# New: inner-state transitions using the same helper
import csv
from collections import defaultdict

STATE_LABELS = {0: 'PP', 1: 'PPN', 2: 'PNPN'}

def readInnerStateTransitions(
    file_assoc_dissoc_times: str,
    dt: float,
    proteins=['P'],
    substrates=['S','N'],
    startT: float = 0.0,
    exclude_pp_to_ppn: bool = True,
    debug: bool = False,
):
    def __maxComplexID(complexList):
        return max(complexList.keys()) if complexList else 0
    def __add_newComplex(complexList, protein):
        newMultimerID = __maxComplexID(complexList) + 1
        newMultimer = DIMER(newMultimerID)
        newMultimer.addComponent(protein, currt=-1)
        complexList[newMultimerID] = newMultimer
    def __add_protein_to_list(proteinList, complexList, pro, proID):
        if proID not in proteinList:
            proteinList[proID] = PROTEIN(pro, proID)
            __add_newComplex(complexList, proteinList[proID])

    complexList: dict[int, DIMER] = {}
    proteinList: dict[int, PROTEIN] = {}
    last_state = {}
    last_time = {}
    transitions = []

    with open(file_assoc_dissoc_times, 'r') as f:
        for line in f:
            rxntype, rxnInfo = __read_A_D_line(line, dt, proteins, substrates)
            currt = rxnInfo[-1]
            if currt < startT:
                continue
            currt = _apply_rxn_update(rxntype, rxnInfo, proteinList, complexList, __add_protein_to_list, __add_newComplex, debug)
            # evaluate inner state of dimers only
            for cid, cmplx in list(complexList.items()):
                if len(cmplx.components) != 2:
                    last_state.pop(cid, None)
                    last_time.pop(cid, None)
                    continue
                nbound = sum(1 for pro in cmplx.components if pro.boundToSubstrates())
                if nbound < 0: nbound = 0
                if nbound > 2: nbound = 2
                state = STATE_LABELS[nbound]
                if cid not in last_state:
                    last_state[cid] = state
                    last_time[cid] = currt
                elif state != last_state[cid]:
                    fr, to = last_state[cid], state
                    t0, t1 = last_time[cid], currt
                    if not (exclude_pp_to_ppn and fr == 'PP' and to == 'PPN'):
                        transitions.append((fr, to, t0, t1, t1 - t0))
                    last_state[cid] = state
                    last_time[cid] = currt
    return transitions