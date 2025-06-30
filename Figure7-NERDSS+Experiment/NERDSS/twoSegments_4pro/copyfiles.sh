folder=kppE6.0kpnE1.0
# folder=kpp0kpnE1.0
mkdir -p $folder
path=msang2@login.rockfish.jhu.edu:/home/msang2/mankun/twoSegments_4pro/
# scp $path/$folder/0/restart.dat $path/$folder/0/*.inp $path/$folder/0/*.mol $folder
scp $path/$folder/0/restart.dat $path/$folder/0/*.inp $path/$folder/0/*.mol $folder