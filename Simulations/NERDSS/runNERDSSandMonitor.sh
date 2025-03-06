#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <job_directory>"
    exit 1
fi

JOB_DIR=$1
cd "$JOB_DIR" || { echo "Failed to enter directory $JOB_DIR"; exit 1; }


nohup ./nerdss -r restart.dat > OUTPUT &
pid=$!
echo "Started job with PID $pid"

while true; do
    sleep 3600  # Adjust sleep time as needed
    
    if [ "$(tail -n1 OUTPUT | grep "Exiting..." | wc -l)" -eq 1 ]; then
        echo "Detected termination message, restarting job..." >> OUTPUT
        # Find the latest restart file
        RESTART_FILE=$(ls -v RESTARTS/restart*.dat 2>/dev/null | tail -n1)
        if [ -n "$RESTART_FILE" ]; then
            cp "$RESTART_FILE" ./restart.dat
        fi
        echo "Job exited unexpectedly, restarting job..." > OUTPUT
        kill $pid 2>/dev/null
        wait $pid 2>/dev/null
        nohup ./nerdss -r restart.dat > OUTPUT &
        pid=$!
    fi
    
    ETA=$(tail -n100 OUTPUT | grep "Estimated end time" | tail -n1)
    if [[ -n "$ETA" ]]; then
        ETAday=$(echo "$ETA" | awk '{print $NF}' | cut -d'-' -f3)
        ETAhour=$(echo "$ETA" | awk '{print $(NF-1)}' | cut -d':' -f1)
        
        CUR_DAY=$(date +%d | bc)
        CUR_HOUR=$(date +%H | bc)
        
        if [ "$ETAday" -lt "$CUR_DAY" ] || { [ "$ETAday" -eq "$CUR_DAY" ] && [ "$ETAhour" -lt "$((CUR_HOUR - 1))" ]; }; then
            echo "ETA condition met, restarting job..." >> OUTPUT
            # Find the latest restart file
            RESTART_FILE=$(ls -v RESTARTS/restart*.dat 2>/dev/null | tail -n1)
            if [ -n "$RESTART_FILE" ]; then
                cp "$RESTART_FILE" ./restart.dat
            fi
            echo "Job stucked, restarting job..." > OUTPUT
            kill $pid 2>/dev/null
            wait $pid 2>/dev/null
            nohup ./nerdss -r restart.dat > OUTPUT &
            pid=$!
        fi
    fi

    if ! kill -0 $pid 2>/dev/null; then
        echo "Process $pid has ended. Exiting." >> OUTPUT
        exit 0
    fi

done

