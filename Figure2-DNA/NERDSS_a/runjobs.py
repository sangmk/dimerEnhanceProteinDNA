# file handeling
import shutil
# import helper modules
import sys
from pathlib import Path
# import ionerdss
sys.path.append("../../../ionerdss/") # path to ionerdss repo
import ionerdss as ion

def runjobs(N_rep, current_dir, subdir='', coordinate=False, simFirstIndex=1):
    workdir = current_dir + subdir
    # prepare an input directory and move input files inside
    directory_input = workdir + '/nerdss_input/'
    path = Path(directory_input)
    if path.exists():
        shutil.rmtree(directory_input)  # Remove existing input files
    path.mkdir(parents=True, exist_ok=True)  # Recreate empty
    shutil.copy(current_dir+'/P.mol', directory_input)
    shutil.copy(current_dir+'/N.mol', directory_input)
    if coordinate: shutil.copy(current_dir+'/fixCoordinates.pdb', directory_input)
    shutil.copy(current_dir+'/parms.inp', directory_input)
    # use ionerdss to start simulations
    model = ion.Simulation(workdir)
    sim_indices = [i+simFirstIndex for i in range(N_rep)]
    print(f'sim_indeces from {sim_indices[0]} to {sim_indices[-1]}')
    model.run_new_simulations(
        sim_indices=sim_indices, coordinate=coordinate, verbose=False, parallel=True,
        nerdss_dir='/home/local/WIN/msang2/mankun/nerdss_development/'
    )

# # parent folder
# pfolder = '/association/kt/'
# # sub-folders
# N_rep = 30
# workdir = str(Path.cwd()) + f'/{pfolder}/L200_ka01_D01/'
# runjobs(N_rep, workdir, simFirstIndex=1)
# workdir = str(Path.cwd()) + f'/{pfolder}/L500_ka01_D01/'
# runjobs(N_rep, workdir, simFirstIndex=1)
# workdir = str(Path.cwd()) + f'/{pfolder}/L1000_ka01_D01/'
# runjobs(N_rep, workdir, simFirstIndex=1)

# parent folder
pfolder = './'
# sub-folders
N_rep = 3
workdir = str(Path.cwd()) + f'/{pfolder}/kpp1E6koffP01/'
runjobs(N_rep, workdir, simFirstIndex=1)
workdir = str(Path.cwd()) + f'/{pfolder}/kpp1E6koffP1/'
runjobs(N_rep, workdir, simFirstIndex=1)
workdir = str(Path.cwd()) + f'/{pfolder}/kpp1E6koffP10/'
runjobs(N_rep, workdir, simFirstIndex=1)
workdir = str(Path.cwd()) + f'/{pfolder}/monomer/'
runjobs(N_rep, workdir, simFirstIndex=1)
workdir = str(Path.cwd()) + f'/{pfolder}/dimer/'
runjobs(N_rep, workdir, simFirstIndex=1)