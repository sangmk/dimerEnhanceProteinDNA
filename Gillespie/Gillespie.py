import numpy as np
from datetime import datetime

# define a class containing the system information
class Gillespie_sys():
    def __init__(self, labels, t0, NP0_sys=1e3, iftrack=True, get_resT_only=False):
        # define constant parameters from input
        ## the total number of proteins in system
        self.NP0 = int(NP0_sys)
        ## labels
        self.labels = labels
        self.iftrack = iftrack
        self.noSave = False
        self.get_resT_only = get_resT_only
        if self.get_resT_only: self.noSave = True
        
        if self.iftrack:
            # initialize proteins as all free ('F'=free, 'S'/'N'=bound) (id:1/0)
            self.protein = {i:[''] for i in range(self.NP0)}
            # initialize bound state as all free (0=free, 1=bound)
            self.protein_bound = {i:[0] for i in range(self.NP0)}
            # initialize target bound as no bond (0=free, 1=bound)
            self.protein_target = {i:[0] for i in range(self.NP0)}
            # initialize the number of dissociations
            self.protein_diss = {i:0 for i in range(self.NP0)}
            # Initialize complex list with empty lists (label:ids)
            self.complex = {i:[] for i in labels}
            # all proteins are in free monomer state
            self.complex['P'] = [[i] for i in self.protein]
        
        # Initialize some properties
        self.time = t0
        self.parms = []
        self.sumMatrix = None
        self.totalCopy = None
        self.curr_counts = np.zeros(len(labels), dtype=np.int64)
           
    def mass_conservation(self, sumMatrix=None, totalCopy=None):
        '''
        Input:
          sumMatrix: N*M ndarray
          totalCopy: 1*N ndarray or 'init'
        Output:
          if totalCopy == 'init': no return
          else: bool (whether mass conservation holds)
        '''
        # whether sumMatrix is given
        if sumMatrix is not None:
            self.sumMatrix = sumMatrix
        elif self.sumMatrix is None:
            raise ValueError('Please define the sumMatrix!')
        else:
            pass

        initialization = False 
        if isinstance(totalCopy, str):
            if totalCopy == 'init':
                initialization = True

        if initialization:
            self.totalCopy = np.sum(self.curr_counts * self.sumMatrix, axis=1)
        elif totalCopy is not None:
            if isinstance(totalCopy, (np.ndarray, list, tuple)):
                self.totalCopy = totalCopy
            else:
                raise ValueError('Unknown input of totalCopy')
        elif self.totalCopy is None:
                raise ValueError('Please initilize the totalCopy!')
        else:
            pass

        # perform test
        if initialization:
            pass
        else:
            if any((np.sum(self.curr_counts*self.sumMatrix, axis=1) - self.totalCopy)/self.totalCopy > 1e-3):
                return False
            else:
                return True
    
    def set_parameters(self, parm, version='new', model='S'):
        if model not in ['S', 'N']:
            raise ValueError('Model should be S or N')
        C0 = 0.6022
        # the initial concentrations
        CP0:float = parm['CP0'] # protein
        CN0:float = parm['CN0'] # nonspecific sites
        if model=='S': CS0:float = parm['CS0'] # specific sites
        # equilibrium constants
        KPN:float = parm['KPN'] # P-N
        if model=='S': KPS:float = parm['KPS'] # P-S
        KPP:float = parm['KPP'] # P-P
        # dimension factor
        gamma:float = parm['gamma']
        # 1d enhancement
        # gamma = 1/CN0 / area3Dto1D
        V = self.NP0 / parm['CP0']
        # kinetics rates
        if version == 'new':
            kaPN:float = parm['kaPN'] # P + N
            if model=='S': kaPS:float = parm['kaPS'] # P + S
            kaPP:float = parm['kaPP'] # P + P
            kbPN:float = parm['kbPN'] # PN ->
            if model=='S': kbPS:float = parm['kbPS'] # PS ->
            kbPP:float = parm['kbPP'] # PP ->
            if model=='S': 
                # kaP, kbP, kaS, kbS, kaN, kbN, gamma, V
                self.parms = [kaPP, kbPP, kaPS, kbPS, kaPN, kbPN, gamma, V]
            else:
                # kaP, kbP, kaN, kbN, gamma, V
                self.parms = [kaPP, kbPP, kaPN, kbPN, gamma, V]
        elif version == 'old':
            kbPN = 1
            if model=='S': kbPS = 0.1
            kbPP = 0.1
            kaPN:float = KPN*kbPN
            if model=='S': kaPS:float = KPS*kbPS
            kaPP:float = KPP*kbPP
            if model=='S': 
                # kaP, kbP, kaS, kbS, kaN, kbN, gamma, V
                self.parms = [kaPP, kbPP, kaPS, kbPS, kaPN, kbPN, gamma, V]
            else:
                # kaP, kbP, kaN, kbN, gamma, V
                self.parms = [kaPP, kbPP, kaPN, kbPN, gamma, V]
        if model=='S': 
            self.curr_counts[0] = CS0*V
            self.curr_counts[1] = CN0*V
            self.curr_counts[2] = self.NP0
        else:
            self.curr_counts[1] = CN0*V
            self.curr_counts[2] = self.NP0
        
    def __convert_bindingLabel(self, label):
        # sort the tmpLabel
        a = list(label)
        a.sort()
        a.reverse()
        return ''.join(a)
    
    def update(self, reaction, tau:float, save:bool):
        if len(self.curr_counts) != len(reaction):
            raise ValueError(f'Reaction has different dimension ({len(reaction)}) than counts ({len(self.curr_counts)})')
        ## reactant_label: 
        ##  {'S':-1}, {'N':-1}, {'P--':[id]}
        ## Either 'S' or 'N' dissociate, both unbound cannot happen
        ## product_label:
        ## {'anything'}, can repeat
        if self.iftrack:
            reactant_label = {}
            product_label = []
            for react_i in range(len(reaction)):
                # which complex is reacting
                cmplx_name = self.labels[react_i]
                if reaction[react_i] == -1:
                    if cmplx_name in ['S', 'N']:
                        reactant_label.update({cmplx_name:-1})
                    else:
                        # randomly select an index
                        rnd_int = np.random.choice(len(self.complex[cmplx_name]))
                        # remove that protein from reacted complex and collect this id
                        pop_id = self.complex[cmplx_name].pop(rnd_int)
                        # add id to list
                        reactant_label.update({cmplx_name:pop_id})
                elif reaction[react_i] == -2:
                    # randomly select an index
                    if cmplx_name in ['S', 'N']:
                        reactant_label.update({cmplx_name:-1})
                    else:
                        pop_id = [] # [id, id]
                        for i in range(2):
                            rnd_int = np.random.choice(len(self.complex[cmplx_name]))
                            # remove that protein from reacted complex and collect this id
                            pop_id = pop_id + self.complex[cmplx_name].pop(rnd_int)
        #                     cmplx_id.append(pop_id[-1])
                        # add id to list
                        reactant_label.update({cmplx_name:pop_id})
                elif reaction[react_i] == 1:
                    product_label.append(cmplx_name)
                elif reaction[react_i] == 2:
                    product_label = product_label + [cmplx_name, cmplx_name]
                else:
                    pass
        
        ## update current counts and save change if needed
        ## only save the new state when needed otherwise just change the current
        self.curr_counts = self.curr_counts + reaction
        self.time = self.time + tau
        
        if self.iftrack:
            if save and not self.noSave:
                for pro_id in self.protein:
                    self.protein[pro_id].append(self.protein[pro_id][-1])
                    self.protein_bound[pro_id].append(self.protein_bound[pro_id][-1])
            else:
                pass

            ## Process state changes
            if len(product_label) == 1:
                # association happens (only one product)
                cmplx_id = [] # to get [id, id, ..]
                changebound = ''
                for rlabel in reactant_label:
                    if rlabel in ['S', 'N']:
                        changebound = rlabel
                    else:
                        cmplx_id = cmplx_id + reactant_label[rlabel]
                # update the product
                self.complex[product_label[0]].append(cmplx_id)
                # update proteins
                # add the new binding site to a protein not bound in this type
                if bool(changebound):
                    np.random.shuffle(cmplx_id)
                    for pro_id in cmplx_id:
                        if changebound not in self.protein[pro_id][-1]:
                            # temporary label
                            tmpLabel = self.__convert_bindingLabel(self.protein[pro_id][-1] + changebound)
                            if tmpLabel in product_label[0].split('P'):
                                self.protein[pro_id][-1] = tmpLabel
                                break
                        else:
                            pass
                else:
                    pass
                # update bound status
                # if at least one protein is bound, this complex is bound
                if ('S' in product_label[0]) or ('N' in product_label[0]):
                    for pro_id in cmplx_id:
                        self.protein_bound[pro_id][-1] = 1
                elif ('S' in product_label[0]):
                    for pro_id in cmplx_id:
                        self.protein_target[pro_id][-1] = 1
                # update residence time
                if self.get_resT_only: 
                    for pro_id in cmplx_id:
                        if self.protein_bound[pro_id][-1] == 1:
                            if self.proetin_startT[pro_id] < 0:
                                self.proetin_startT[pro_id] = self.time
                        else:
                            pass
                        if self.protein_target[pro_id][-1] == 1:
                            if self.protein_target_startT[pro_id] < 0:
                                self.protein_target_startT[pro_id] = self.time
                            if self.protein_target_endT[pro_id] > 0:
                                self.targetSearchTimes.append(self.time-self.protein_target_endT[pro_id])
                                self.protein_target_endT[pro_id] = -1
                        else:
                            pass
            else:
                # dissociation happens
                changebound = ''
                target_cmplx = [] # ['P--', ..]
                for rlabel in product_label:
                    if rlabel in ['S', 'N']:
                        changebound = rlabel
                    else:
                        target_cmplx.append(rlabel)
                # collect the id of proteins first
                r_cmplx_id = [] # to get [id, id, ..]
                for rlabel in reactant_label:
                    r_cmplx_id = r_cmplx_id + reactant_label[rlabel]
                # dissociation from chromatin
                # update product
                if len(target_cmplx) == 1:
                    self.complex[target_cmplx[0]].append(r_cmplx_id)
                else: # it has to be 2
                    self.complex[target_cmplx[0]].append([r_cmplx_id[0]])
                    self.complex[target_cmplx[1]].append([r_cmplx_id[1]])
                # update proteins
                np.random.shuffle(r_cmplx_id)
                if bool(changebound): # len(target_cmplx) has to be 1
                    for pro_id in r_cmplx_id:
                        if (changebound in self.protein[pro_id][-1]):
                            tmpLabel = self.protein[pro_id][-1].replace(changebound, '')
                            if tmpLabel in target_cmplx[0].split('P'):
                                self.protein[pro_id][-1] = tmpLabel
                                break
                # update bound status
                for tcmplxi in target_cmplx:
                    if ('S' not in tcmplxi) and ('N' not in tcmplxi):
                        for pro_id in self.complex[tcmplxi][-1]:
                            self.protein_bound[pro_id][-1] = 0
                    else:
                        for pro_id in self.complex[tcmplxi][-1]:
                            self.protein_bound[pro_id][-1] = 1
                # update residence time
                if self.get_resT_only: 
                    for pro_id in r_cmplx_id:
                        if self.protein_bound[pro_id][-1] == 0:
                            if self.proetin_startT[pro_id] >= 0:
                                self.resTimes.append(self.time-self.proetin_startT[pro_id])
                                self.proetin_startT[pro_id] = -1
                                self.protein_diss[pro_id] += 1 # update the number of dissociations
                            else:
                                pass
                        else:
                            pass
                        if self.protein_target[pro_id][-1] == 0:
                            if self.protein_target_startT[pro_id] >= 0:
                                self.targetResTimes.append(self.time-self.protein_target_startT[pro_id])
                                self.protein_target_startT[pro_id] = -1
                            if self.protein_target_endT[pro_id] < 0:
                                self.protein_target_endT[pro_id] = self.time

        
        return self.curr_counts

    def set_counts(self, counts):
        counts_array = np.array(counts)
        # scale the number by protein number
        scaler = self.NP0 / np.sum(counts_array * self.sumMatrix, axis=1)[-1]
        count_int = np.round(counts_array*scaler)
        total_delta = self.totalCopy - np.sum(count_int * self.sumMatrix, axis=1)
        for i, value in enumerate(total_delta):
            count_int[i] = max(count_int[i] + value, 0)
        # if the total protein overflows, change the list
        NP_now = np.sum(count_int * self.sumMatrix, axis=1)[-1]
        if self.iftrack and NP_now > self.NP0:
            self.NP0 = int(np.round(NP_now))
            # re-initialize proteins as all free ('F'=free, 'S'/'N'=bound) (id:1/0)
            self.protein = {i:[''] for i in range(self.NP0)}
            # re-initialize bound state as all free (0=free, 1=bound)
            self.protein_bound = {i:[0] for i in range(self.NP0)}
            self.protein_target = {i:[0] for i in range(self.NP0)}
            # re-initialize the number of dissociations
            self.protein_diss = {i:0 for i in range(self.NP0)}
            # all proteins are in free monomer state
            self.complex['P'] = [[i] for i in self.protein]
        for ci, count_i in enumerate(count_int):
            self.curr_counts[ci] = count_i
        print('# \_ Initialize tracker', flush=True)
        if self.iftrack:
            # set complexes, bound state and binding labels
            for i, labeli in enumerate(self.labels):
                N_p = labeli.count('P')
                bindingLabel = labeli.split('P')[1:]
                print('#   \_ %s ... '%labeli, end='', flush=True)
                if N_p == 0 or labeli=='P':
                    pass
                    print('Skipped', flush=True)
                else:
                    # number of copies of this complex
                    print('Number of copies: %d ... '%self.curr_counts[i], end='', flush=True)
                    for i_count in range(self.curr_counts[i]):
                        selected_p = []
                        for sii in range(N_p):
                            selected_p.append(self.complex['P'].pop()[0])
                        # add to target complex
                        self.complex[labeli].append(selected_p)
                        # for proteins
                        for i_P in range(N_p):
                            pro_id = selected_p[i_P]
                            # just update the binding label
                            self.protein[pro_id][-1] = bindingLabel[i_P]
                            # update binding state
                            if ('S' in bindingLabel) or ('N' in bindingLabel) or ('SN' in bindingLabel):
                                self.protein_bound[pro_id][-1] = 1
                                if ('S' in bindingLabel) or ('SN' in bindingLabel):
                                    self.protein_target[pro_id][-1] = 1
                    print('Finished', flush=True)
        print('# \_ Initialize residence time calculator', flush=True)
        if self.get_resT_only:
            if not self.iftrack:
                raise ValueError('Cannot get residence time without tracking!')  
            else:
                self.resTimes = []
                self.targetSearchTimes = []
                self.targetResTimes = []
                self.proetin_startT = {i:-1 for i in range(self.NP0)}
                self.protein_target_startT = {i:-1 for i in range(self.NP0)}
                self.protein_target_endT = {i:-1 for i in range(self.NP0)}
                for pro_id in range(self.NP0):
                    if self.protein_bound[pro_id][-1] == 1:
                        self.proetin_startT[pro_id] = self.time
                        if self.protein_target[pro_id][-1] == 1:
                            self.protein_target_startT[pro_id] = self.time
                    else:
                        pass

    def flush_protein(self, filename, tList, start=0):
        with open(filename, 'a') as f:
            for i in range(start, len(self.protein_bound[0])):
                f.write(str(tList[i])+',')
                for j in range(self.NP0):
                    f.write(str(self.protein_bound[j][i])+',')
                f.write('\n')
        for i in range(self.NP0):
            self.protein[i] = self.protein[i][-1:]
            self.protein_bound[i] = self.protein_bound[i][-1:]


# Define the Gillespie algorithm function
def gillespie_algorithm(GSYS:Gillespie_sys, reactions, save:bool):

    # Calculate the propensity functions and reaction matrix
    propList, rxnMatrix = reactions(GSYS.curr_counts, GSYS.parms)
    
    if any(propList < -1e-12):
        raise ValueError('Propensity<0!  '+str(propList))
    
    # Calculate the total propensity function
    propTot = np.sum(propList)

    # Calculate the time of the next reaction
    tau = np.random.exponential(1 / propTot)

    # Choose which reaction occurs next
    rv = np.random.uniform(0, propTot)
    rxnID = len(propList) - np.sum(rv < np.cumsum(propList))
    new_counts = GSYS.update(rxnMatrix[rxnID], tau, save)

    return new_counts, tau

# perform ODE solver based on the reactions used for Gillespie input
def ODE_function(current_counts, t, parms, reactions):
    propList, rxnMatrix = reactions(current_counts, parms)
    return np.matmul(rxnMatrix.T, propList)

# define the exit condition
def Exit_time(tLimit='auto'):
    def Exit_condition(GSYS, t):
        if isinstance(tLimit, (int, float)):
            if t > tLimit:
                return True
            return False
        elif isinstance(tLimit, str):
            if tLimit.lower() == 'auto':
                N_tot_diss = 0
                for proId in GSYS.protein_diss:
                    if GSYS.protein_diss[proId] < 10:
                        return False
                    else:
                        pass
                    N_tot_diss += GSYS.protein_diss[proId]
                if N_tot_diss > 1000*GSYS.NP0:
                    return True
                else:
                    return False
    return Exit_condition

def iterate_reactions(
        GSYS:Gillespie_sys, reactions, 
        t:float, tMax, time_points:list,
        saveSize, saveFileName, save, 
        trackParticle, particle_points:list,
        tForOneBar, N_tbar, N_tStep, tStepSize, dissMultiply,
    ):
    ''' Perform one iteration of the Gillespie algorithm, save data, and print progress bar.
    '''
    # perform the Gillespie algorithm
    current_counts, tau = gillespie_algorithm(GSYS, reactions, save)
    if save:
        time_points.append(t)
        if trackParticle: particle_points.append(current_counts)
        N_tStep += 1
    t += tau
    time_points[-1] = t
    if trackParticle: particle_points[-1] = current_counts
    # update the save flag, save according to time step
    if t > N_tStep*tStepSize:
        save = True
    else:
        save = False
    if tMax != 0:
        # print the time bar
        if t > N_tbar*tForOneBar:
            print('*', end='', flush=True)
            N_tbar += 1
    else:
        # check if the number of dissociations is enough
        N_tot_diss = np.sum([GSYS.protein_diss[proId] for proId in GSYS.protein_diss])
        if dissMultiply*GSYS.NP0*100 < N_tot_diss:
            dissMultiply += 1
            print('# ', datetime.now(), end='\t', flush=True)
            print('%.0f dissociations happened at %.3f s.'%(N_tot_diss,t), flush=True)
    # save the data to cache to reduce RAM usage
    if saveFileName!='':
        if (len(GSYS.protein_bound)*len(GSYS.protein_bound[0])) > saveSize:
            if len(time_points) == len(GSYS.protein_bound[0]):
                GSYS.flush_protein(saveFileName, time_points, 0)
                time_points = time_points[-1:]
            else:
                GSYS.flush_protein(saveFileName, time_points, 1)
                time_points = time_points[-1:]
    return t, N_tbar, N_tStep, save, dissMultiply

def run_Gillespie(
    t, GSYS:Gillespie_sys, reactions, ExitCondition, tStepSize=1, tMax=0, 
    saveSize=5e6, saveFileName='', trackParticle=False
):
    ''' Run the Gillespie algorithm until the exit condition is met. 
    Save data, print progress bar, and return the time points and particle points.
    '''
    # initialize the time points and particle points
    time_points = [t]
    particle_points = []
    if trackParticle:
        particle_points = [GSYS.curr_counts]
    # set the time bar
    dissMultiply = 0
    if tMax != 0:
        print('Gillespie simulation:', flush=True)
        print('-'*9+'10'+'-'*8+'20'+'-'*8+'30'+'-'*8+'40'+'-'*8+'50'+'-'*8+'60'+'-'*8+'70'+'-'*8+'80'+'-'*8+'90'+'-'*8+'100%', flush=True)
        tForOneBar = tMax/100
        N_tbar = 1
    else:
        dissMultiply = 1 # to monitor the number of dissociations
        tForOneBar = np.inf
        N_tbar = 1
        print('# %.0d proteins in total.'%GSYS.NP0, flush=True)
    N_tStep = 0
    # set the save flag
    save = True
    # run the simulation until exit condition is met
    while not ExitCondition(GSYS, t):
        t, N_tbar, N_tStep, save, dissMultiply = iterate_reactions(
            GSYS, reactions, 
            t, tMax, time_points,
            saveSize, saveFileName, save, 
            trackParticle, particle_points,
            tForOneBar, N_tbar, N_tStep, tStepSize, dissMultiply,
        )
    # print the last bars
    if tMax != 0:
        if N_tbar < 100:
            for i  in range(100-N_tbar):
                print('*', end='', flush=True)
    # save the last data
    if saveFileName!='':
        if len(time_points) == len(GSYS.protein_bound[0]):
            GSYS.flush_protein(saveFileName, time_points, 0)
            time_points = time_points[-1:]
        else:
            GSYS.flush_protein(saveFileName, time_points, 1)
            time_points = time_points[-1:]
    print()
    # return the time points and particle points (if needed)
    return np.array(time_points), np.array(particle_points).T



# calculate residence time
def calc_resT(binding_series, tList, tStart=0, tEnd=np.inf):
    residenceT = []
    bound = False
    firstBound = True
    # calculate residence time, ignore the first and the last binding event
    startT = 0
    for ti, t in enumerate(tList[1:]):
        if tStart <= t <= tEnd:
            if binding_series[ti] == 1:
                if firstBound:
                    pass
                else:
                    if not bound:
                        bound = True
                        startT = t
            elif (binding_series[ti] == 0):
                if firstBound:
                    firstBound = False
                elif bound:
                    bound = False
                    if startT >= 0:
                        residenceT.append(t-startT)
                    else:
                        pass
    return residenceT

def calc_resT_by_time(bindings:list, boundstatus:list, startT:list):
    # bindings: [t, '0'/'1', ...], each element is for a protein
    # boundstatus: [False/True, ...], each element is for a protein
    # startT: [float, ...], the binding time for each protein
    t = float(bindings[0])
    residenceT = []
    for i, b in enumerate(bindings[1:]):
        if b == '1' and (not boundstatus[i]):
            boundstatus[i] = True
            startT[i] = t
        elif b == '0' and boundstatus[i]:
            boundstatus[i] = False
            if startT[i] >= 0:
                residenceT.append(t-startT[i])
            else:
                pass
    return residenceT


# obtain survival probability
def get_survival_prob(resident_times, bins=100):

    # get survival probabilities
    h = np.histogram(resident_times, bins=bins, density=True)
    return h[1], np.append([1], 1 - np.cumsum(h[0]*np.diff(h[1])))

def readDataFromCache(saveFileName):
    print('# Time:', datetime.now(),flush=True)
    print('# Reading binding data...', flush=True)
    print('-'*9+'10'+'-'*8+'20'+'-'*8+'30'+'-'*8+'40'+'-'*8+'50'+'-'*8+'60'+'-'*8+'70'+'-'*8+'80'+'-'*8+'90'+'-'*8+'100%', flush=True)
    # first, count the total number of records
    with open(saveFileName, "r") as f:
        totlineNum = sum([1 for _ in f])
    tForOneBar = (totlineNum-1)/100
    N_tbar = 1
    # read the binding data
    # obtain residence times
    import csv
    with open(saveFileName, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        startT = [0 for h in headers]
        row_t0 = next(reader)
        boundstatus = [b=='1' for b in row_t0]
        residenceT = []
        i = 1
        for row in reader:
            if row != []:
                tmpResT = calc_resT_by_time(row, boundstatus, startT)
                residenceT = residenceT + tmpResT
                i += 1
                if i > N_tbar*tForOneBar:
                    print('*', end='', flush=True)
                    N_tbar += 1
            else:
                pass

    import os
    if os.path.exists(saveFileName): os.remove(saveFileName)
    print('*')
    # output
    print('Number of dissociation events:', len(residenceT))
    print('# Mean Residence Time \n%.4e'%np.mean(residenceT), flush=True)

    return residenceT

def main_process(
        model, modeltype, NP0_sys, labels,
        parameters, sumMat, totalCopy, equi_counts, tStepSize,  
        saveSize=-1, maxT=0, saveFileName='',
        getSurvivalProb=False, getResT=True, tStart=0, tEnd=np.inf,
    ):
    # to prevent the crash
    try:
        # Initiation
        # prepare the data log file
        if saveSize > 0:
            if saveFileName == '':
                raise ValueError('Saving... File name not provided!')
            with open(saveFileName, 'w') as f:
                f.write('0,')
                for i in range(NP0_sys):
                    f.write(str(i)+',')
                f.write('\n')
        else:
            saveFileName = ''
        GSYS = Gillespie_sys(labels, 0, NP0_sys, True, getResT)
        if getResT: print('# Tracking residence time only', flush=True)
        # kaP, kbP, kaS, kbS, kaN, kbN, gamma, V
        print('# Setting parameters...', flush=True)
        GSYS.set_parameters(parameters, version='new', model=modeltype)
        print('# Enforce mass conservation...', flush=True)
        GSYS.mass_conservation(sumMat, totalCopy)
        print('# Setting initial counts...', flush=True)
        GSYS.set_counts(equi_counts)
        print('# Initiation finished!', flush=True)
        print('# current counts:\n# ', end='', flush=True)
        for i, labeli in enumerate(labels):
            print('%s: %.0f'%(labeli, GSYS.curr_counts[i]), end=', ')
        print('', flush=True)
        if isinstance(maxT, str):
            tMax = 0
        else:
            tMax = maxT
        # run simulation
        tList, _ = run_Gillespie(
            t=0, GSYS=GSYS, reactions=model, 
            ExitCondition=Exit_time(maxT), 
            tStepSize=tStepSize,
            tMax=tMax,
            saveSize=saveSize,
            saveFileName=saveFileName
        )
        success = True
    except Exception as error:
        print('Crashed!\t', error, flush=True)
        success = False
    
    if success == True:
        if getResT:
            # residence times were calculated during the simulation
            residenceT = GSYS.resTimes
            print('Number of dissociation events:', len(residenceT))
            print('# Mean Residence Time \n%.4e'%np.mean(residenceT), flush=True)
        else:
            if saveSize > 0 and saveFileName!='':
                # binding data was saved to cache
                residenceT = readDataFromCache(saveFileName)
            else:
                residenceT = []
                for mol_bound in GSYS.protein_bound:
                    residenceT = residenceT + calc_resT(GSYS.protein_bound[mol_bound], tList, tStart, tEnd)
                # output
                print('Number of dissociation events:', len(residenceT))
                print('# Mean Residence Time \n%.4e'%np.mean(residenceT), flush=True)
        
        # calculate survival probability
        if getSurvivalProb == True:
            minResT = min(residenceT)
            maxResT = max(residenceT)
            if minResT > maxResT/1e6:
                t_survP = np.arange(0, maxResT, minResT)
            else:
                fastEvents = np.arange(0, 1e6*minResT, 10*minResT)
                slowEvents = np.arange(1e6*minResT, maxResT, maxResT/1e5)
                t_survP = np.concatenate((fastEvents, slowEvents))
            print('# Calculating survival probability...', end='', flush=True)
            survP = get_survival_prob(residenceT, bins=t_survP)[1]
            print('done', flush=True)
            print("# START SURVIVAL PROB",flush=True)
            for i, resTi in enumerate(t_survP):
                print(resTi, survP[i], flush=True)
            print("# END SURVIVAL PROB", flush=True)