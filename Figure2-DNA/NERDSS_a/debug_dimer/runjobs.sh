current_dir=`pwd`


for i in {0..30}
do
    # prepare directory
    rm -rf nerdss_output/$i
    mkdir -p nerdss_output/$i
    # prepare NERDSS
    cp nerdss_input/parms.inp nerdss_output/$i
    cp nerdss_input/N.mol nerdss_output/$i
    cp nerdss_input/P.mol nerdss_output/$i
    cd ./nerdss_output/$i
    cp ~/mankun/nerdss_development/bin/nerdss ./nerdss
    # run NERDSS
    nohup ./nerdss -f parms.inp > OUTPUT &
    # go back to current directroy
    cd $current_dir
done

# done
