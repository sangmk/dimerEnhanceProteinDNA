#bash

current_dir=`pwd`
pfolder='./'

for subd in kpp0kpnE1.0  kppE1.0kpnE1.0  kppE2.0kpnE1.0  kppE3.0kpnE1.0  kppE4.0kpnE1.0  kppE5.0kpnE1.0  kppE6.0kpnE1.0
do 
    cd $pfolder/$subd
    sbatch runjob.sh
    cd $current_dir
done