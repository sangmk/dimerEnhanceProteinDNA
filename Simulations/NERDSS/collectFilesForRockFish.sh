max_load=80

cwd=$(pwd)
# $1 for the path to the directory containing inputs
if [ "$1" == "" ]; then
    echo "Please provide the path to the directory containing the inputs"
    exit 1
fi
for j in `ls -d "$1"/*`
do
    echo $j
    cd $j
    currNum=$(ls -d */ | wc -l)
    for i in {0..47}
    do
        mkdir -p $cwd/filesToRockFish/$j/$i
        cp $i/restart.dat $cwd/filesToRockFish/$j/$i
        cp $i/*.inp $cwd/filesToRockFish/$j/$i
        cp $i/*.mol $cwd/filesToRockFish/$j/$i
        cd $cwd/filesToRockFish/$j/$i
        sed -i -e 's/numItr = .*/numItr = 2000000001/' \
           -e 's/trajWrite = .*/trajWrite = 1000000000/' \
           -e 's/checkPoint = .*/checkPoint = 1000000000/' \
           -e 's/pdbWrite = .*/pdbWrite = 10000000/' restart.dat
        cd $cwd/$j
    done
    cd $cwd
done
