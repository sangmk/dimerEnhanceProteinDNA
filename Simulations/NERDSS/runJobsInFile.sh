max_load=80
cwd=$(pwd)
for dir in `cat filesToRun`
do
    cd $dir
    while true; do
        current_load=$(bash ~/mankun/Notebooks/cpuLoad 1)
        if (( $(echo "$current_load < $max_load" | bc -l) )); then
            # nohup ./nerdss -f parms.inp -c fixCoordinates.inp > OUTPUT &
            nohup ./nerdss -r restart.dat > OUTPUT &
            break
        else
            sleep 1
        fi
    done
cd $cwd
done