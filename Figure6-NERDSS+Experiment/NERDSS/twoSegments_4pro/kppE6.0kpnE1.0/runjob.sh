#!/bin/bash
#SBATCH --job-name=kppE6_4pro
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
    cp ./fixCoordinates.pdb nerdss_output/$i
    cp ./parms.inp nerdss_output/$i
    cp ./N.mol nerdss_output/$i
    cp ./P.mol nerdss_output/$i
    cp ./S.mol nerdss_output/$i
    cp ./nuc.mol nerdss_output/$i
    cd ./nerdss_output/$i
    cp /home/msang2/mankun/nerdss_development/bin/nerdss ./nerdss
    ./nerdss -f parms.inp -c fixCoordinates.pdb > OUTPUT &
    cd $current_dir
done
wait
# done