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
    # for i in `ls -d "$j"/*`
    for i in {0..47}
    do
        cp ./molFiles/*.mol $j/$i/
        cd $j/$i
        rm -rf DATA
        rm -rf PDB
        rm -rf RESTARTS
        rm -f ./nerdss
        ln -s ~/mankun/nerdss_development/bin/nerdss ./nerdss
        while true; do
            current_load=$(bash ~/mankun/Notebooks/cpuLoad 1)
            if (( $(echo "$current_load < $max_load" | bc -l) )); then
                nohup ./nerdss -f parms.inp -c fixCoordinates.inp > OUTPUT &
                break
            else
                sleep 1
            fi
        done
        cd $cwd
    done
done
