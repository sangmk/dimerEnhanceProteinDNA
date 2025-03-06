jobsToKill=""
numJobs=`ps -u msang2 | grep nerdss | wc -l`
numJobsToKill=0
for pid in `ps -u msang2 | grep nerdss | awk '{print $1}'`
do 
    echo $pid
    # exam the histogram_complexes_time.dat file for the pid
    grep "Time (s)" /proc/$pid/cwd/DATA/histogram_complexes_time.dat | tail -n1 
    ETA=`tail -n100 /proc/$pid/cwd/OUTPUT | grep "Estimated end time" | tail -n1 `
    echo $ETA
    ETAday=`echo $ETA | awk '{print $4}' | awk -F- '{print $3}'`
    ETAhour=`echo $ETA | awk '{print $5}' | awk -F: '{print $1}'`
    # pass check if ETAday is empty
    if [ -z $ETAday ]
    then
        continue
    fi
    # add the pid to jobsToKill if ETA is earlier than now-1hour
    if [ $ETAday -lt $(echo `date +%d` | bc) ] 
    then
        jobsToKill="$jobsToKill $pid"
        numJobsToKill=$(echo "$numJobsToKill + 1" | bc)
    elif [ $ETAday -eq $(echo `date +%d` | bc) ] && [ $ETAhour -lt $(echo "`date +%H` - 1" | bc) ]
    then
        jobsToKill="$jobsToKill $pid"
        numJobsToKill=$(echo "$numJobsToKill + 1" | bc)
    fi
done
echo Jobs to kill \($numJobsToKill/$numJobs\): $jobsToKill
