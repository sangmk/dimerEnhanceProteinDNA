max_load=80

cwd=$(pwd)
# $1 for the path to the directory containing inputs
if [ "$1" == "" ]; then
    echo "Please provide the path to the directory containing the inputs"
    exit 1
fi
for j in `ls -d "$1"/*`
# for j in $1/kppE1.0kpnE1.0 $1/kppE2.0kpnE1.0
do
    echo $j
    cd $j
    currNum=$(ls -d */ | wc -l)
    for i in {0..47}
    do
        ## create a new directory to restart, the folder name is currNum + i
        # newfolder=`echo "$currNum+$i" | bc`
        # rm -f -r $newfolder
        # mkdir -p $newfolder
        # cp $i/*.mol $newfolder
        # cp $i/parms.inp $newfolder
        # cp $i/RESTARTS/restart10000000.dat $newfolder/restart.dat
        # cd $newfolder
        # ln -s ~/mankun/nerdss_development/bin/nerdss ./nerdss
        cd $i
        cp RESTARTS/restart1000.dat ./restart.dat
        sed -i -e 's/numItr = .*/numItr = 100001/' \
           -e 's/trajWrite = .*/trajWrite = 100000/' \
           -e 's/checkPoint = .*/checkPoint = 100000/' restart.dat
        while true; do
            current_load=$(bash ~/mankun/Notebooks/cpuLoad 1)
            if (( $(echo "$current_load < $max_load" | bc -l) )); then
                nohup ./nerdss -r restart.dat > OUTPUT &
                break
            else
                sleep 1
            fi
        done
        cd ..
    done
    cd $cwd
done
