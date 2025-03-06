for i in {0..29}
do
    mkdir $i
    cp *.mol ./$i
    cp parms.inp ./$i
    cp restart.dat ./$i
    cd $i
    rm -f nerdss
    ln -s ~/mankun/nerdss_development/bin/nerdss ./
    nohup ./nerdss -r restart.dat > OUTPUT &
    cd ..
done