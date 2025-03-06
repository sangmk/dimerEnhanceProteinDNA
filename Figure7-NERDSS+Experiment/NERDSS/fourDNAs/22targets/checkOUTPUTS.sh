for i in {0..29}
do
    cd $i
    # if Exiting... is in the last line of the OUTPUT file, then restart the job
    if [ `tail -n1 OUTPUT | grep "Exiting..." | wc -l` -eq 1 ]
    then
        echo "Restarting job $i"
        rm -r DATA
        rm -r RESTARTS
        rm -r PDB
        nohup ./nerdss -r restart.dat > OUTPUT &
    fi
    cd ..
done