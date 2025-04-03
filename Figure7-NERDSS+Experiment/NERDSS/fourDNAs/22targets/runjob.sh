for i in {0..14}
do
    rm -rf ./$i
    mkdir -p $i # Create a directory for each job
    cp *.mol ./$i
    cp parms.inp ./$i
    cp fixCoordinates.inp ./$i
    # cp restart.dat ./$i
    cd $i
    # rm -f nerdss
    cp ~/mankun/nerdss_development/bin/nerdss ./nerdss
    # rm -rf PDB
    # rm -rf RESTARTS
    # rm -rf DATA
    # nohup ./nerdss -r restart.dat > OUTPUT &
    nohup ./nerdss -f parms.inp -c fixCoordinates.inp > OUTPUT &
    # bash ../runNERDSSandMonitor.sh $PWD &
    cd ..
done