#!/bin/bash
#SBATCH --job-name=kbp10
#SBATCH --time=72:0:0
#SBATCH --partition=parallel
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=48
#### load and unload modules you may need

module load gsl 


current_dir=`pwd`


for i in {0..47}
do
    rm -rf nerdss_output/$i
    mkdir -p nerdss_output/$i
    cp nerdss_input/parms.inp nerdss_output/$i
    cp nerdss_input/N.mol nerdss_output/$i
    cp nerdss_input/P.mol nerdss_output/$i
    cd ./nerdss_output/$i
    cp /home/msang2/mankun/nerdss_development/bin/nerdss ./nerdss
    ./nerdss -f parms.inp > OUTPUT &
    cd $current_dir
done
wait
# done
