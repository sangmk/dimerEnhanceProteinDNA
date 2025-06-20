'''
Define a class used to read NERDSS data and get statistics
Created by Mankun Sang on Apr. 17 2020

nerdssData(N_rep, path)

defaults:
    self.dataCoor = 'RESULT_final_coor'
    self.dataCopy = 'copy_numbers_time.dat'
    self.dataHist = 'histogram_complexes_time.dat'
'''

# Import modules
from typing import Dict
import numpy as np
import networkx as nx
import io
# import os

# store molecule information
class _rxnMolecule():
    '''
    molecule containing three attributes
    id: the number same in NERDSS
    type: molecular name
    cluster: the cluster this molecule belongs to
    '''
    def __init__(self, mol:str):
        self.id = int(mol.split(':')[0])
        self.type = mol.split(':')[1].strip()
        self.cluster = -1
        
    def bind(self, cluster_id):
        self.cluster = cluster_id
        
    def unbind(self):
        self.cluster = -1
        
    def istype(self, types:list)->bool:
        for _type in types:
            if self.type == _type:
                return True
        return False

    def __eq__(self, other) -> bool:
        if not isinstance(other, _rxnMolecule):
            return False
        return f'{self.type}:{self.id}' == f'{other.type}:{other.id}'
    
    def __hash__(self) -> int:
        return id(self)

    def __str__(self):
        info = f'{self.type}:{self.id}'
        return info

    def __repr__(self):
        return str(self)

# store cluster information
class _rxnCluster():

    '''
    A class to store the information of clusters. A cluster contains all the moleucles binding together.
    The connection of molecules are determined by the help of networkx using Graph.
    Proteins and substrates are equal in the graph.
    self.events: dict{time: info}
        The times when the cluster changes, molecule types and numbers
    self.members: list
        all members at the end
    self.sustrates:
        all substrates ever exist
    self.proteins:
        all proteins ever exist
    '''
    
    def __init__(self, members:list, substrates:list, time:float, bind=False):
        self.substrateNames = substrates
        self.events={}
        self.members = []
        self.proteins = []
        self.substrates = []
        self.G = nx.Graph()
        
        # store the particles
        self.add_member(members)
        # update graph
        self.G.add_nodes_from(members)
        if bind:
            if len(members) == 2:
                self.G.add_edges_from([members])
            else:
                raise ValueError(f'Too many ({len(members)}) members for initiation, should be less than 2.')
        # initiate events
        self.events.update({time:self.report()})
    
    def update_graph(self, graph):
        self.G = graph
        
    def add_member(self, new_members:list):
        for new in new_members:
            if new in self.members:
                pass
            else:
                self.members.append(new)
                # classify the particles
                if new.istype(self.substrateNames):
                    if new not in self.substrates: self.substrates.append(new)
                else:
                    if new not in self.proteins: self.proteins.append(new)
                
    def clear(self):
        self.members = []
        self.G = nx.Graph()
        
    def bind(self, newmembers:list, time:float):
        # add the new particle(s) to complex
        self.add_member(newmembers)
        # update the graph
        self.G.add_edges_from([newmembers])
        self.events.update({time:self.report()})
    
    def unbind(self, oldmembers:list, time:float):
        # update the graph
        self.G.remove_edges_from([oldmembers])
        # find existing clusters
        connected = list(sorted(nx.connected_components(self.G), key=len, reverse=True))
        # remove dissociated particles
        if len(connected) == 1: # all particles are still connected
            self.events.update({time:self.report()})
            return None
        else: # the old cluster is splited to two
            if len(connected[0]) == 1:
                list(connected[0])[0].unbind()
            else:
                pass
            self.G.remove_nodes_from(connected[1])
            for item in connected[1]:
                self.remove(item)
                item.unbind()
        # return result clusters
        if len(connected[0]) == 1: # all particles are free
            self.clear()
            self.events.update({time:self.report()})
            return None
        elif len(connected[1]) == 1: # one particle is free
            self.events.update({time:self.report()})
            return None
        else: # two clusters
            newG = self.G.subgraph(connected[1])
            newCluster = _rxnCluster(list(connected[1]), self.substrateNames, time)
            newCluster.update_graph(newG)
            self.events.update({time:self.report()})
            return newG
        
    def remove(self, item):
        self.members.remove(item)

    def add(self, other, binding:list, time:float):
        newC = _rxnCluster(self.members, self.substrateNames, time)
        newC.events.update(self.events)
        newC.add_member(other.members)
        newC.update_graph(nx.union(self.G, other.G))
        newC.G.add_edges_from([binding])
        return newC
    
    def report(self): # return the number of each type molecules
        mem_types = []
        mem_counts = {}
        for pro in self.members:
            if pro.type in mem_types:
                mem_counts[pro.type] += 1
            else:
                mem_types.append(pro.type)
                mem_counts[pro.type] = 1
        info = ''
        for mem in mem_types:
            info += mem + '(' + str(mem_counts[mem]) + ')'
        return info
    
    def __str__(self):
        info = 'Cluster: {' + f'PROTEINS: {self.proteins}, ' + f'SUBSTRATES: {self.substrates}' + '}'
        return info

    def __repr__(self):
        return str(self)
    
    def calc_residenceT(self, startT:float):
        if list(self.events.values())[-1] == '':
            times = np.sort(list(self.events.keys()))
            startTime = np.max([startT, times[0]])
            self.residenceT = times[-1] - startTime
        else:
            self.residenceT = 0

## read molecule from smt_ractions_time.dat
def _readmol(f, molecules_ids, molecules) -> _rxnMolecule:
    '''
    f:
        file handle
    molecules_ids: list of int
        a list of the indeces of molecules
    molecules: list of _rxnMolecule
        a list of all molecules, matching with molecules_ids
    '''
    mol_info = f.readline().strip()
    mol_id = int(mol_info.split(':')[0])
    if mol_id not in molecules_ids:
        molecules_ids.update({mol_id:len(molecules)})
        molecules.append(_rxnMolecule(mol_info))
    return molecules[molecules_ids[mol_id]]

## process molecule binding
def _bind_molecules(mol1, mol2, cluster_traj, substrates, t_curr):
    '''
    This function is used to bind molecules and form clusters
    mol1, mol2: _rxnMolecule
        the molecules that forming bond
    cluster_traj:
        list of all clusters, used to find the corresponding clusters or create new cluster
    substrates: str
        molecule types recognized as substrates. 
        The meaning is to classify different molecules for further calculations
    t_curr: float
        current time
    '''
    if mol1.cluster == -1 and mol2.cluster == -1:
        cluster_id = len(cluster_traj)
        cluster_traj.append(_rxnCluster([mol1, mol2], substrates, t_curr, True))
        mol1.cluster = cluster_id
        mol2.cluster = cluster_id
    elif mol1.cluster != -1 and mol2.cluster == -1:
        cluster_traj[mol1.cluster].bind([mol1, mol2], t_curr)
        mol2.cluster = mol1.cluster
    elif mol1.cluster == -1 and mol2.cluster != -1:
        cluster_traj[mol2.cluster].bind([mol1, mol2], t_curr)
        mol1.cluster = mol2.cluster
    elif mol1.cluster == mol2.cluster:
        cluster_traj[mol1.cluster].bind([mol1, mol2], t_curr)
    else:
        clus1 = cluster_traj[mol1.cluster]
        clus2 = cluster_traj[mol2.cluster]
        cluster_id = len(cluster_traj)
        cluster_traj.append(clus1.add(clus2, [mol1, mol2], t_curr))
        for mol in cluster_traj[cluster_id].members:
            mol.cluster = cluster_id
            
## process molecule dissociation
def _unbind_molecules(mol1, mol2, cluster_traj, t_curr):
    '''
    This function is used to unbind molecules and update clusters
    mol1, mol2: _rxnMolecule
        the molecules that forming bond
    cluster_traj:
        list of all clusters, used to find the corresponding clusters or create new cluster
    t_curr: float
        current time
    '''
    if mol1.cluster == -1 or mol2.cluster == -1:
        pass
    elif mol1.cluster == mol2.cluster:
        newClus = cluster_traj[mol1.cluster].unbind([mol1, mol2], t_curr)
        if newClus:
            cluster_id = len(cluster_traj)
            cluster_traj.append(newClus)
            for mol in newClus.members:
                mol.cluster = cluster_id
    else:
        pass


# return the reaction name in the format of NERDSS reactions
def Reaction_Name(mol1, mol2, iface1, iface2, ifreversible=True) -> str:
    if ifreversible:
        return f"{mol1}({iface1}) + {mol2}({iface2}) <-> {mol1}({iface1}!1).{mol2}({iface2}!1)"
    else:
        return f"{mol1}({iface1}) + {mol2}({iface2}) -> {mol1}({iface1}!1).{mol2}({iface2}!1)"


class _Reaction_nerdss():

    def __init__(self):
        self.onRate3Dka = 0
        self.onRate3DMacro = 0
        self.offRatekb = 0
        self.offRateMacro = 0
        self.norm1 = [0, 0, 1]
        self.norm2 = [0, 0, 1]
        self.sigma = 1
        self.assocAngles = '[nan, nan, nan, nan, nan]'
        self.rate = 0
        self.area3Dto1D = np.nan
        self.length3dto2d = np.nan
        self.loopcoopfactor = 1
        self.bindRadSameCom = 1.1

        self.isnewrxn = True
    
    def update(self):
        if self.area3Dto1D is np.nan:
            self.area3Dto1D = 4*np.pi*self.sigma**2
        if self.length3dto2d is np.nan:
            self.length3dto2d = 2*self.sigma

    def get_keq(self) -> float:
        if self.onRate3Dka != 0 and self.offRatekb != 0:
            return self.onRate3Dka/self.offRatekb/1e-6
        elif self.onRate3DMacro != 0 and self.offRateMacro != 0:
            return self.onRate3DMacro/self.offRateMacro
    
    def get_keq_1D(self) -> float:
        return self.get_keq() / self.area3Dto1D

    def get_keq_2D(self) -> float:
        return self.get_keq() / self.length3dto2d

    def set_assocAngles(self, assocAngles: list):
        self.assocAngles = '['
        for i, angle in enumerate(assocAngles):
            if angle is np.nan:
                item = 'nan'
            elif abs(angle-np.pi) < 0.0001:
                item = 'M_PI'
            else:
                item = str(angle)
            if i != len(assocAngles) - 1:
                self.assocAngles += (item + ', ')
            else:
                self.assocAngles += (item + ']')

def Write_CrdInp(crdFileName: str, fixCoords: Dict[str, list] = {}, WaterBox: list = [0,0,0]):
    with open(crdFileName, 'w') as f:
        for key in fixCoords:
            # there is really something need to be fixed
            if len(fixCoords[key]) > 0:
                # make sure the coordinates are not fake
                if len(fixCoords[key][0]) > 0:
                    f.write("NEWTYPE\n")
                    f.write(key+'\n')
                    f.write(f"{len(fixCoords[key])}\n")
                    for siteCrd in fixCoords[key]:
                        if isinstance(siteCrd, (float, int)):
                            if abs(siteCrd) > WaterBox[0]:
                                raise ValueError(f'Coordinate outside of waterbox: {key}:{siteCrd}')
                            else:
                                f.write(f"{siteCrd:.2f}\t0\t0\n")
                        elif isinstance(siteCrd, (list, np.ndarray)):
                            if len(siteCrd) == 3:
                                if abs(siteCrd[0]) > WaterBox[0]:
                                    raise ValueError(f'Coordinate outside of waterbox: {key}:{siteCrd}')
                                else:
                                    x = siteCrd[0]
                                if abs(siteCrd[1]) > WaterBox[1]:
                                    raise ValueError(f'Coordinate outside of waterbox: {key}:{siteCrd}')
                                else:
                                    y = siteCrd[1]
                                if abs(siteCrd[2]) > WaterBox[2]:
                                    raise ValueError(f'Coordinate outside of waterbox: {key}:{siteCrd}')
                                else:
                                    z = siteCrd[2]
                                f.write(f"{x:.2f}\t{y:.2f}\t{z:.2f}\n")
                            else:
                                raise ValueError(f'Wrong coordinate format: {key}:{siteCrd}')
                        else:
                            raise ValueError(f'Wrong coordinate format: {key}:{siteCrd}. Should be numbers or list/array.')

def Write_Inp(parmFileName: str, comment: str = '',
              nItr: int = 1e6, timeStep: float = 1000, timeWrite: int = 5e1,
              pdbWrite: int = 1e6, trajWrite: int = 5e1,
              restartWrite: int = 1e6, checkPoint: int = 1e6,
              overlapSepLimit: float = 6.0, WaterBox: list = [347, 39, 39],
              molList: Dict[str, int] = {'A': 20, 'F': 199, 'G': 5},
              reactions: Dict[str, _Reaction_nerdss] = {'A(dbd) + F(bs) <-> A(dbd!1).F(bs!1)': _Reaction_nerdss()}, 
              crdFileName: str = '', fixCoords: Dict[str, list] = {}):

    f = open(parmFileName, 'w')

    f.write(f'# Input file {comment}\n\n')
    f.write('start parameters\n')
    f.write(f'    nItr = {nItr:.0f} #iterations\n')
    f.write(f'    timeStep = {timeStep}\n')
    f.write(f'    timeWrite = {timeWrite:.0f}\n')
    f.write(f'    pdbWrite = {pdbWrite:.0f}\n')
    f.write(f'    trajWrite = {trajWrite:.0f}\n')
    f.write(f'    restartWrite = {restartWrite:.0f}\n')
    f.write(f'    checkPoint = {checkPoint:.0f}\n')
    f.write(f'    overlapSepLimit = {overlapSepLimit}\n')
    f.write('end parameters\n\n')

    f.write('start boundaries\n')
    f.write(f'    WaterBox = {str(WaterBox)}\n')
    f.write('end boundaries\n\n')

    f.write('start molecules\n')
    for molname in molList:
        f.write(f'    {molname} : {molList[molname]:.0f}\n')
    f.write('end molecules\n\n')

    f.write('start reactions\n')
    for rxnName in reactions:
        rxn = reactions[rxnName]
        f.write(f'    {rxnName}\n')
        writeZeros = True
        if rxn.onRate3Dka != 0:
            f.write(f'    onRate3Dka = {rxn.onRate3Dka}\n')
            writeZeros = False
        if rxn.offRatekb != 0:
            f.write(f'    offRatekb = {rxn.offRatekb}\n')
            writeZeros = False
        if rxn.onRate3DMacro != 0:
            f.write(f'    onRate3DMacro = {rxn.onRate3DMacro}\n')
            writeZeros = False
        if rxn.offRateMacro != 0:
            f.write(f'    offRateMacro = {rxn.offRateMacro}\n')
            writeZeros = False
        if writeZeros:
            f.write(f'    onRate3Dka = 0\n')
            f.write(f'    offRatekb = 0\n')
        if rxn.area3Dto1D is not np.nan:
            f.write(f'    area3Dto1D = {rxn.area3Dto1D}\n')
        f.write(f'    norm1 = {str(rxn.norm1)}\n')
        f.write(f'    norm2 = {str(rxn.norm2)}\n')
        f.write(f'    sigma = {rxn.sigma}\n')
        f.write(f'    assocAngles = {rxn.assocAngles}\n')
        f.write(f'    loopcoopfactor = {rxn.loopcoopfactor}\n')
        f.write(f'    bindRadSameCom = {rxn.bindRadSameCom}\n')
        f.write('\n')
    f.write('end reactions\n')

    f.close()

    if crdFileName == '':
        crdFileName = 'crd' + parmFileName
    Write_CrdInp(crdFileName, fixCoords, WaterBox)

class _Parm_nerdss():
    def __init__(self) -> None:
        self.nItr = 0
        self.timeStep = 0
        self.molecules = {}
        self.waterbox = [] # [x, y, z]
        self.WBvol = 0
        self.timeWrite = 0
        self.scaleMaxDisplace = 100

class _molecule_nerdss():
    def __init__(self) -> None:
        self.Dr = 0
        self.Dr_mean = 0
        self.D = 0
        self.D_mean = 0
        self.name = 0
        self.ispromoter = ''

class nerdssData():
    
    def __init__(self, N_rep, path, subpath='', parmName='parms.inp', ifprint=True):

        '''
        N_rep = 0 for single trace in current folder
              = int > 0 for multiple traces in sub folders
              = list for multiple traces in sub folders with given names
        '''
        
        if ifprint: print(f'>>>>>> parsing {path}')

        self.dataCoor = 'final_coords.xyz'
        self.dataCopy = 'copy_numbers_time.dat'
        self.dataHist = 'histogram_complexes_time.dat'
        self.dataSmt = 'smt_reactions_time.dat'
        if isinstance(N_rep, int):
            if N_rep == 0:
                self.N_rep = 1
                self.paths = [f"{path}"]
            else:
                self.N_rep = N_rep
                self.paths = [f"{path}/{i}" for i in range(N_rep)]
        elif isinstance(N_rep, (list, tuple)):
            self.N_rep = len(N_rep)
            self.paths = [f"{path}/{i}" for i in N_rep]
        
        if subpath == '':
            pass
        else:
            self.paths = [p + "/" + subpath for p in self.paths]
        
        self.MOLNAMES = []
        self.time = np.array([])
        self.parms = _Parm_nerdss()
        self.reactions = {}
        self.bindings = {}
        self.occupancies = {}
        self.substrates = {}
        self.__readParamaters(f"{self.paths[0]}/{parmName}", ifprint=ifprint)
        self.molecules = self.__readMolecules(self.paths[0])
        
        if ifprint: print(f'<<<<<< finished parsing')
        if ifprint: print('------------------------------------------------------')

    # read bindings from smt_reactions_time.dat
    def readBindings(self, substrates:list, dataSmt='', t_step=np.nan, excludedBonds=[])->None:
        '''
        substrares: 
            the molecules that sits on membrane or chromatin, list of str.
        excludedBonds: 
            if there are some bonds not expected to be calculated, [['mol1', 'mol2'], *]. Order of molecules does not matter.
        '''
        # change smt file name if needed
        if dataSmt != '': self.dataSmt = dataSmt

        # pre-process excluded bonds (sort each bond)
        excludedBonds_sorted = []
        for bondi in excludedBonds:
            excludedBonds_sorted.append('+'.join(np.sort(bondi)))
        
        Clusters = {}
        trajnum = 0
        if len(self.paths) == 1:
            traj_names = ["traj1"]
        else:
            traj_names = [f"traj{n+1}" for n in range(len(self.paths))]
        for path in self.paths:
            # start reading one of the outputs
            molecules = [] # list of all molecules
            molecules_ids = {} # id of each molecule
            cluster_traj = [] # all clusters
            with open(f"{path}/{self.dataSmt}", 'r') as f:
                times = []
                use_t = True
                t_curr = 0
                for line in f:
                    # read the current time and determine whether to use
                    if line[0:4] == 'Time':
                        t_curr = float(line.split(':')[1])
                        if t_step is np.nan:
                            use_t = True
                            times.append(t_curr)
                        else:
                            if len(times) == 0:
                                times.append(t_curr)
                                use_t = True
                            elif round(t_curr - times[-1], 6) == round(t_step, 6):
                                times.append(t_curr)
                                use_t = True
                            else:
                                use_t = False
                    # parse binding events
                    elif line[0:4] == 'Bind' and use_t:
                        # read in molecules binding
                        mol1 = _readmol(f, molecules_ids, molecules)
                        mol2 = _readmol(f, molecules_ids, molecules)
                        # process clusters
                        if "+".join(np.sort([mol1.type, mol2.type])) not in excludedBonds_sorted:
                            _bind_molecules(mol1, mol2, cluster_traj, substrates, t_curr)

                    elif line[0:6] == 'Unbind' and use_t:
                        mol1 = _readmol(f, molecules_ids, molecules)
                        mol2 = _readmol(f, molecules_ids, molecules)
                        # process clusters
                        if "+".join(np.sort([mol1.type, mol2.type])) not in excludedBonds_sorted:
                            _unbind_molecules(mol1, mol2, cluster_traj, t_curr)

                    elif line[0:6] == 'Rebind' and use_t:
                        mol1 = _readmol(f, molecules_ids, molecules)
                        mol2 = _readmol(f, molecules_ids, molecules)
                        mol3 = _readmol(f, molecules_ids, molecules)
                        # process clusters
                        if "+".join(np.sort([mol1.type, mol2.type])) not in excludedBonds_sorted:
                            _unbind_molecules(mol1, mol2, cluster_traj, t_curr)
                        if "+".join(np.sort([mol1.type, mol3.type])) not in excludedBonds_sorted:
                            _bind_molecules(mol1, mol3, cluster_traj, substrates, t_curr)

            Clusters.update({traj_names[trajnum]:cluster_traj})
            trajnum += 1

        self.bindings = Clusters
        self.time = np.array(times)

    def readOccupancies(self, substrates:list, dataSmt='', t_step=np.nan, excludedBonds=[])->None:
        '''
        substrares: 
            the molecules that sits on membrane or chromatin, list of str. Also the molecules occupancies are calculated
        excludedBonds: 
            if there are some bonds not expected to be calculated, [['mol1', 'mol2'], *]. Order of molecules does not matter.
        '''
        # change smt file name if needed
        if dataSmt != '': self.dataSmt = dataSmt

        # pre-process excluded bonds (sort each bond)
        excludedBonds_sorted = []
        for bondi in excludedBonds:
            excludedBonds_sorted.append('+'.join(np.sort(bondi)))
        
        Occupancies = {}
        trajnum = 0
        if len(self.paths) == 1:
            traj_names = ["traj1"]
        else:
            traj_names = [f"traj{n+1}" for n in range(len(self.paths))]
        for path in self.paths:
            # start reading one of the outputs
            molecules = [] # list of all molecules
            molecules_ids = {} # id of each molecule
            occup_traj = [] # only clusters involving 
            with open(f"{path}/{self.dataSmt}", 'r') as f:
                times = []
                use_t = True
                t_curr = 0
                for line in f:
                    # read the current time and determine whether to use
                    if line[0:4] == 'Time':
                        t_curr = float(line.split(':')[1])
                        if t_step is np.nan:
                            use_t = True
                            times.append(t_curr)
                        else:
                            if len(times) == 0:
                                times.append(t_curr)
                                use_t = True
                            elif round(t_curr - times[-1], 6) == round(t_step, 6):
                                times.append(t_curr)
                                use_t = True
                            else:
                                use_t = False
                    # parse binding events
                    elif line[0:4] == 'Bind' and use_t:
                        # read in molecules binding
                        mol1 = _readmol(f, molecules_ids, molecules)
                        mol2 = _readmol(f, molecules_ids, molecules)
                        # process clusters
                        if "+".join(np.sort([mol1.type, mol2.type])) not in excludedBonds_sorted:
                            if (mol1.type in substrates) or (mol2.type in substrates):
                                _bind_molecules(mol1, mol2, occup_traj, substrates, t_curr)

                    elif line[0:6] == 'Unbind' and use_t:
                        mol1 = _readmol(f, molecules_ids, molecules)
                        mol2 = _readmol(f, molecules_ids, molecules)
                        # process clusters
                        if "+".join(np.sort([mol1.type, mol2.type])) not in excludedBonds_sorted:
                            if (mol1.type in substrates) or (mol2.type in substrates):
                                _unbind_molecules(mol1, mol2, occup_traj, t_curr)

                    elif line[0:6] == 'Rebind' and use_t:
                        mol1 = _readmol(f, molecules_ids, molecules)
                        mol2 = _readmol(f, molecules_ids, molecules)
                        mol3 = _readmol(f, molecules_ids, molecules)
                        # process clusters
                        if "+".join(np.sort([mol1.type, mol2.type])) not in excludedBonds_sorted:
                            if (mol1.type in substrates) or (mol2.type in substrates):
                                _unbind_molecules(mol1, mol2, occup_traj, t_curr)
                        if "+".join(np.sort([mol1.type, mol3.type])) not in excludedBonds_sorted:
                            if (mol1.type in substrates) or (mol3.type in substrates):
                                _bind_molecules(mol1, mol3, occup_traj, substrates, t_curr)

            Occupancies.update({traj_names[trajnum]:occup_traj})
            trajnum += 1

        self.occupancies = Occupancies
        self.time = np.array(times)
    
    def processSubstrates(self)->None:
        '''
        extract the durations and binding times, using self.occupancies
        '''
        
        def calcBindindTimes(endT:float, startT:float, cluster):
            '''
            Given a cluster, calculate the duration it existed
            return:
                occupancyLen:list
                    the durations of binding
                startTimes:list
                    the time bindings started
            '''
            foundBond = False
            occupancyLen = []
            startTimes = []
            for timepoint in sorted(cluster.events):
                # loop the event times
                if cluster.events[timepoint] != '':
                    if foundBond:
                        pass # This site is being occupied by different proteins
                    else:
                        # This site changes from vacant to occupied
                        startTcluster = timepoint
                        foundBond = True
                else:
                    if not foundBond:
                        # the cluster has been formed before the observation began
                        occupancyLen.append(timepoint-startT)
                        startTimes.append(startT)
                    else:
                        occupancyLen.append(timepoint-startTcluster)
                        startTimes.append(startTcluster)
                        foundBond = False
            if foundBond:
                # There is a not dissociated bond till the end of simulation
                occupancyLen.append(endT-startTcluster)
                startTimes.append(startTcluster)
            return occupancyLen, startTimes

        startT = self.time[0]
        endT = self.time[-1]
        for traj in self.occupancies:
            clustersDict = {}
            for cluster in self.occupancies[traj]:
                substrate = cluster.substrates[0]
                if substrate not in clustersDict:
                    # found a new substrate, calcluate its residence time in the current cluster
                    duration, start = calcBindindTimes(endT, startT, cluster)
                    clustersDict.update({substrate:{'duration':duration, 'startT':start}})
                else:
                    duration, start = calcBindindTimes(endT, startT, cluster)
                    clustersDict[substrate]['duration'] = clustersDict[substrate]['duration'] + duration
                    clustersDict[substrate]['startT'] = clustersDict[substrate]['startT'] + start
        self.substrates = clustersDict

    def calcOccupancy(self) -> Dict[_rxnMolecule, float]:
        '''
        occupancies of each substrate
        '''
        totT = (self.time[-1] - self.time[0])*self.N_rep
        occupancies = {}
        for substrate in self.substrates:
            occupancies.update({substrate:sum(self.substrates[substrate]['duration'])/totT})
        return occupancies

    def readCoor(self,  dataCoor=''):
        '''
        Read the coordinates of molecules from output.
        default: dataCoor = final_coords.xyz
        '''
        if dataCoor != '':
            self.dataCoor = dataCoor
        molnames = []
        molCoords = {}
        for i, path in enumerate(self.paths):
            with open(f"{path}/{self.dataCoor}", 'r') as f:
                number = int(f.readline())
                info = f.readline()
                for line in f:
                    linelist = line.split()
                    name = linelist[0]
                    coord = float(linelist[1])
                    if name in molnames:
                        molCoords[name].append(coord)
                    else:
                        molCoords[name] = [coord]
                        molnames.append(name)
        self.MOLNAMES = molnames
        self.MOLCOORD = {}
        for i in self.MOLNAMES:
            self.MOLCOORD[i] = np.array(molCoords[i])
        print(self.MOLCOORD.keys())
        
    def readHist(self, dataHist='', ifprint=True):
        '''
        Read the histgrams from the output. 
        default: dataHist = histogram_complexes_time.dat
        '''
        if dataHist != '':
            self.dataHist = dataHist
        for i, path in enumerate(self.paths):
            with open(f"{path}/{self.dataHist}", 'r') as f:
                time = []
                mols = {}
                molnames = []
                zeros = [] # used to generate a new type of temporal sequence
                for line in f:
                    if "Time" in line:
                        for cmplx in molnames:
                            if len(mols[cmplx]) != len(time):
                                mols[cmplx].append(0)
                        # find current time and enlongate zero sequence
                        time.append(float(line.strip().split(':')[-1]))
                        zeros.append(0)
                    else:
                        linelist = line.split('\t')
                        num = float(linelist[0])
                        cmplx = linelist[1].strip()
                        if cmplx in molnames:
                            mols[cmplx].append(num)
                        else:
                            if ifprint: print('Found '+cmplx+f' at output {len(zeros)} (time {time[-1]} s).')
                            molnames.append(cmplx)
                            mols[cmplx] = zeros[:]
                            mols[cmplx][-1] = num
            for cmplx in molnames:
                if len(mols[cmplx]) == len(time)-1:
                    mols[cmplx].append(0)
                elif len(mols[cmplx]) != len(time):
                    raise ValueError(f'Sequence length of ({cmplx}: len = {len(mols[cmplx])}) does not match length of time: {len(time)}')
            if i == 0:
                # initial data storage
                self.time = np.array(time)
                self.MOLNAMES = molnames
                self.MOLS = {}
                for cmplx in mols.keys():
                    self.MOLS[cmplx] = [np.array(mols[cmplx])]
            else:
                # correct MOLNAMES
                for cmplx in molnames:
                    if cmplx not in self.MOLNAMES:
                        self.MOLNAMES.append(cmplx)
                        self.MOLS[cmplx] = [np.zeros(len(mols[cmplx]))]
                # correct time and timporal sequences
                size = min(len(time), len(self.time))
                if size != len(self.time):
                    self.time = np.array(time)
                for cmplx in self.MOLNAMES:
                    if cmplx in molnames:
                        self.MOLS[cmplx] = [item[:size] for item in self.MOLS[cmplx]] + [np.array(mols[cmplx][:size])]
                    else:
                        self.MOLS[cmplx] = [item[:size] for item in self.MOLS[cmplx]]
                    # print(i, cmplx, np.size(self.MOLS[cmplx]), size)
            if ifprint: print(f'Finish read replication {i+1}.')
        self.MOLS_err = {}
        for cmplx in self.MOLNAMES:
            self.MOLS_err[cmplx] = np.std(self.MOLS[cmplx], axis=0)
            self.MOLS[cmplx] = np.mean(self.MOLS[cmplx], axis=0)
            
    def __readParamaters(self, path2parms, ifprint=True) -> None:
        # read the parameters and molecule informations
        with open(path2parms, "r") as f:
            for line in f:
                if line.strip() == '':
                    pass
                elif line.strip()[0] == '#':
                    pass
                elif line.strip().lower() == 'start parameters':
                    next_line = f.readline().strip()
                    while next_line.lower() != 'end parameters':
                        if next_line.strip() != '':
                            linelist = next_line.split('=')
                            if next_line.strip()[0] == '#':
                                pass
                            elif linelist[0].strip() == 'nItr':
                                num = linelist[1].split('#')[0].strip()
                                self.parms.nItr = int(float(num))
                            elif linelist[0].strip() == 'timeStep':
                                num = linelist[1].split('#')[0].strip()
                                self.parms.timeStep = float(num)
                            elif linelist[0].strip() == 'timeWrite':
                                num = linelist[1].split('#')[0].strip()
                                self.parms.timeWrite = float(num)
                            elif linelist[0].strip() == 'scaleMaxDisplace':
                                num = linelist[1].split('#')[0].strip()
                                self.parms.scaleMaxDisplace = float(num)
                        next_line = f.readline().strip()

                elif line.strip().lower() == 'start molecules':
                    molecules = {}
                    next_line = f.readline().strip()
                    while next_line.lower() != 'end molecules':
                        if next_line.strip() != '':
                            if next_line.strip()[0] == '#':
                                pass
                            else:
                                linelist = next_line.split(':')
                                num = linelist[1].split('#')[0].strip()
                                molecules[linelist[0].strip()] = int(float(num))
                        next_line = f.readline().strip()
                    self.parms.molecules = molecules
                    if ifprint: print(self.parms.molecules)

                elif line.strip().lower() == 'start reactions':
                    self.reactions = self.__readReactions(f)
                    if ifprint: 
                        print('Reactions:')
                        for reaction in self.reactions.keys():
                            print(reaction)

                elif line.strip().lower()[:8] == 'waterbox':
                    linelist = line.strip().split('=')
                    num = linelist[1].split('#')[0].strip()
                    waterbox = eval(num)
                    self.parms.waterbox = waterbox
                    self.parms.WBvol = waterbox[0] * waterbox[1] * waterbox[2]
                    if ifprint: print(line.strip())
    
    def __readMolecules(self, path2mols) -> Dict[str, _molecule_nerdss]:
        # read the parameters and molecule informations
        found_molecules = {}
        for mol in self.parms.molecules.keys():
            newmol = _molecule_nerdss()
            molfile = path2mols + '/' + mol + '.mol'
            with open(molfile, "r") as f:
                for line in f:
                    linelist_space = line.split()
                    if line.strip() == '':
                        pass
                    elif line.strip()[0] == '#':
                        pass
                    else:
                        if linelist_space[0].strip() == 'Dr':
                            linelist = line.strip().split('=')
                            num = linelist[1].split('#')[0].strip()
                            Dlist = eval(num)
                            newmol.Dr = Dlist
                            newmol.Dr_mean = np.mean(Dlist)
                        
                        elif linelist_space[0].strip() == 'D':
                            linelist = line.strip().split('=')
                            num = linelist[1].split('#')[0].strip()
                            Dlist = eval(num)
                            newmol.D = Dlist
                            newmol.D_mean = np.mean(Dlist)
                        
                        elif line.strip().lower()[:4] == 'name':
                            linelist = line.strip().split('=')
                            num = linelist[1].split('#')[0].strip()
                            newmol.name = num
                        
                        elif line.strip().lower()[:10] == 'ispromoter':
                            linelist = line.strip().split('=')
                            num = linelist[1].split('#')[0].strip()
                            newmol.ispromoter = num
            found_molecules[mol] = newmol
        return found_molecules
    
    def __readReactions(self, file: io.TextIOWrapper) -> Dict[str, _Reaction_nerdss]:
        reactions = {}
        newrxn = _Reaction_nerdss()
        for line in file:
            if line.strip() == '':
                pass
            elif line.strip()[0] == '#':
                pass
            else:
                if "->" in line or "<->" in line:
                    if newrxn.isnewrxn:
                        rxnname = line.strip()
                    else:
                        reactions[rxnname] = newrxn
                        newrxn = _Reaction_nerdss()
                        rxnname = line.strip()
                elif line.strip().lower() == 'end reactions':
                    newrxn.update()
                    reactions[rxnname] = newrxn
                    return reactions
                else:
                    newrxn.isnewrxn = False
                    linelist = line.split('=')
                    num = linelist[1].split('#')[0].strip()
                    keyword = linelist[0].strip().lower()
                    if keyword == 'onrate3dka':
                        newrxn.onRate3Dka = float(num)
                    elif keyword == 'onrate3dmacro':
                        newrxn.onRate3DMacro = float(num)
                    elif keyword == 'offratekb':
                        newrxn.offRatekb = float(num)
                    elif keyword == 'offratemacro':
                        newrxn.offRateMacro = float(num)
                    elif keyword == 'area3dto1d': # case 5
                        newrxn.area3Dto1D = float(num)
                    elif keyword == 'length3dto2d':
                        newrxn.length3dto2d = float(num)
                    elif keyword == 'loopcoopfactor':
                        newrxn.loopcoopfactor = float(num)
                    elif keyword == 'rate':
                        newrxn.rate = float(num)
                    elif keyword == 'sigma': # case 9
                        newrxn.sigma = float(num)


