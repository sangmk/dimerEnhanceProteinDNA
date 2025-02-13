#!/bin/bash
#SBATCH --job-name=EquiSpatial
#SBATCH --time=72:0:0
#SBATCH --partition=parallel
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=48
#### load and unload modules you may need

module load gsl 

pfolders="kppE2.0kpnE1.0"
restartfile="restart.dat"
# parmfile="parms.inp"
# coordfile="coordinate_fix.inp"
current_dir=`pwd`

for dir in $pfolders
do 
    for i in {0..47}
    do
        cd $dir/$i
        rm -f nerdss
        cp /home/msang2/mankun/nerdss_development/bin/nerdss ./nerdss
        sed -i -e 's/pdbWrite = .*/pdbWrite = 10000000/' $restartfile
        ./nerdss -r $restartfile > OUTPUT &
        cd $current_dir
    done
done
wait
# done
