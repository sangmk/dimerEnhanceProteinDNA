#bash

current_dir=`pwd`
pfolder='./'

cd $pfolder/kpp0
sbatch runjobs.sh
cd $current_dir

cd $pfolder/kppE2
sbatch runjobs.sh
cd $current_dir

cd $pfolder/kppE3
sbatch runjobs.sh
cd $current_dir

cd $pfolder/kppE4
sbatch runjobs.sh
cd $current_dir