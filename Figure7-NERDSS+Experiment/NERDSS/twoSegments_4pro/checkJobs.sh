max_load=80

cwd=$(pwd)

for j in `ls -d ./kpp*/`
# for j in $1/kppE1.0kpnE1.0 $1/kppE2.0kpnE1.0
do
    numBONDtot=0
    numBREAKtot=0
    echo $j
    # for i in `ls -d "$j"/*`
    for i in "$j"/{0..47}
    do
        numBOND=$(grep -c "BOND" $i/DATA/assoc_dissoc_time.dat)
        numBREAK=$(grep -c "BREAK" $i/DATA/assoc_dissoc_time.dat)
        info="$i: "
        info="$info $(grep "Time (s)" $i/DATA/histogram_complexes_time.dat | tail -n1)"
        # info="$info numBOND: $numBOND    numBREAK: $numBREAK"
        info="$info `head -n1 $i/DATA/assoc_dissoc_time.dat`"
        echo $info
        # head -n1 $i/DATA/assoc_dissoc_time.dat
        # numBONDtot=$(($numBONDtot+$numBOND))
        # numBREAKtot=$(($numBREAKtot+$numBREAK))
    done
    # echo Totally: 
    # echo "numBOND: $numBONDtot" 
    # echo "numBREAK: $numBREAKtot"
done
