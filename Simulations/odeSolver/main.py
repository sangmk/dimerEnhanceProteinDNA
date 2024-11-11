import pandas as pd
import multiprocessing
from datetime import datetime

def main(parm_df, model, labels, cores=0, ifprint=True):
    # Read parameters from CSV file
    if ifprint: print(datetime.now(),flush=True)
    parameters = []
    for i in range(parm_df.shape[0]):
        parameters.append(parm_df.iloc[i])
    # Run tasks in parallel
    num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores
    if cores >= 1:
        cores_used = min(len(parameters), cores)
    else:
        cores_used = min(len(parameters), num_processes-2)
    if cores_used == 1:
        results = [model(p) for p in parameters]
    else:
        # Run tasks in parallel
        if ifprint: print('Number of cores used: ',f'{cores_used}/{num_processes}')
        pool = multiprocessing.Pool(processes=cores_used)
        results = pool.map(model, parameters)
        pool.close()
        pool.join()
    if ifprint: print('Finished parallel processing.')
    return pd.DataFrame(results, columns=labels)

if __name__ == '__main__':
    main()