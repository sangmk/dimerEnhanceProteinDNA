for i in {0..29}
do
    mkdir -p $i # Create a directory for each job
    cp *.mol ./$i
    cp parms.inp ./$i
    cp restart.dat ./$i
    cd $i
    rm -f nerdss
    ln -s ~/mankun/nerdss_development/bin/nerdss ./
    rm -rf PDB
    rm -rf RESTARTS
    rm -rf DATA
    nohup ./nerdss -r restart.dat > OUTPUT &
    # bash ../runNERDSSandMonitor.sh $PWD &
    cd ..
done