#bash

current_dir=`pwd`
pfolder='./'

cd $pfolder/dimers
sbatch runjob.sh
cd $current_dir

cd $pfolder/fourInRow
sbatch runjob.sh
cd $current_dir

cd $pfolder/monomers
sbatch runjob.sh
cd $current_dir