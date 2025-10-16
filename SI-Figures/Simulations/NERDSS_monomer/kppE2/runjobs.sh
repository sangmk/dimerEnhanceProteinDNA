# pfolder='nerdss_output'
# # parmfile="parms.inp"
# # coordfile="coordinate_fix.inp"
# current_dir=`pwd`

# for i in {1..30}
# do
#     rm -rf $pfolder/$i
#     mkdir -p $pfolder/$i
#     cp ./N.mol $pfolder/$i
#     cp ./P.mol $pfolder/$i
#     cp ./parms.inp $pfolder/$i
#     cd $pfolder/$i
#     cp /home/local/WIN/msang2/mankun/nerdss_development/bin/nerdss ./nerdss
#     nohup ./nerdss -f parms.inp > OUTPUT &
#     cd $current_dir
# done

pfolder='REPEAT'
# parmfile="parms.inp"
# coordfile="coordinate_fix.inp"
current_dir=`pwd`

for i in {1..30}
do
    cd $pfolder/$i
    rm -rf DATA
    rm -rf PDB
    rm -rf RESTARTS
    rm ./nerdss
    cp /home/local/WIN/msang2/mankun/nerdss_development/bin/nerdss ./nerdss
    nohup ./nerdss -r restart.dat > OUTPUT &
    cd $current_dir
done