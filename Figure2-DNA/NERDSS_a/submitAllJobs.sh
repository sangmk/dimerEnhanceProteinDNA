#bash

current_dir=`pwd`
pfolder='1DNA'

# cd $pfolder/dimer
# sbatch submitjob.sh
# cd $current_dir

# cd $pfolder/monomer
# sbatch submitjob.sh
# cd $current_dir

cd $pfolder/kpp1E6koffP01
sbatch submitjob.sh
cd $current_dir

cd $pfolder/kpp1E6koffP1
sbatch submitjob.sh
cd $current_dir

cd $pfolder/kpp1E6koffP10
sbatch submitjob.sh
cd $current_dir