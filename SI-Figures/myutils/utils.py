import matplotlib.pyplot as plt
import numpy as np
from progressbar import progressbar
from scipy import stats
from scipy.special import erfc, erfcx
from scipy.optimize import root_scalar


def preact_1D(x0, sigma, D, dt, k):
    '''Calculate 1-D reaction probability from Smoluchovski's theory '''
    z = x0 - sigma
    P = erfc(z/np.sqrt(4*D*dt)) - \
        np.exp(-z**2/(4*D*dt))* \
        erfcx(z/np.sqrt(4*D*dt) + k*np.sqrt(dt/D))
    return P

def preact_3D(x0, sigma, D, dt, k):
    '''Calculate 3-D reaction probability from Smoluchovski's theory '''
    z = x0 - sigma
    kD = 4*np.pi*sigma*D
    alpha = np.sqrt(D)/sigma * (1 + k/kD)
    P = sigma/x0*k/(k + kD) * \
        (erfc(z/np.sqrt(4*D*dt)) - np.exp(-z**2/(4*D*dt)) * erfcx(z/np.sqrt(4*D*dt) + alpha*np.sqrt(dt)))
    return P

def beforeRun(N, L, D, sigma, k=np.nan, dimension='1D'):
    '''Return valid parameters for simulation'''
    if dimension == '1D':
        timeStep = 1/32/D*(L/2/N)**2
        print(f'Diffusion requires dt < {timeStep}')
        if k is not np.nan:
            if preact_1D(sigma, sigma, D, timeStep, k) < 0.5:
                print(f'The maximam association rate is {preact_1D(sigma, sigma, D, timeStep, k)} when dt = {timeStep}')
            else:
                timeStep_onRate_05 = root_scalar(lambda x: preact_1D(sigma, sigma, D, x, k)-0.5, bracket=[1e-16, timeStep]).root
                print(f'At least choose a dt < {timeStep_onRate_05} for association rate < 0.5.')
        # Rmax = sigma + 4.0 * np.sqrt(2.0 * D * dt)

    elif dimension == '3D':
        timeStep = 1/54/D*((3*L/4/np.pi/N + sigma**3)**(1/3)-sigma)**2
        # Rmax = sigma + 3.0 * np.sqrt(6.0 * D * dt)
        print(f'Diffusion requires dt < {timeStep}')
        if k is not np.nan:
            if preact_3D(sigma, sigma, D, timeStep, k) < 0.5:
                print(f'The maximam association rate is {preact_3D(sigma, sigma, D, timeStep, k)} when dt = {timeStep}')
            else:
                timeStep_onRate_05 = root_scalar(lambda x: preact_3D(sigma, sigma, D, x, k)-0.5, bracket=[1e-16, timeStep]).root
                print(f'At least choose a dt < {timeStep_onRate_05} for association rate < 0.5.')

def gdrive_link_convert(slink:str='') -> str:
    '''
    Google Drive sharing link is in the format of:
    "https://drive.google.com/file/d/<id>/view?usp=sharing",
    but we need a link directly to the picture itself. The target format is:
    "https://drive.google.com/uc?id=<id>"
    '''
    if slink == '':
        slink = input('Paste the link from google drive shareing here: ')
    else:
        pass
    return "https://drive.google.com/uc?id=" + slink.split('/')[5].strip()


def _draw_result(time, copynum, cutoff, end, label, color):

    equi_mean = np.mean(copynum[cutoff:end])
    equi_std = np.std(copynum[cutoff:end])

    label = label + f': {equi_mean:.2f}' + '$\pm$' + f'{equi_std:.2f}'

    plt.fill_between(time[cutoff:end], equi_mean - equi_std, equi_mean + equi_std,
                     color=color, alpha=0.7, label=label)
    plt.vlines(time[end-1], ymin=equi_mean-3*equi_std,
               ymax=equi_mean+3*equi_std, color=color)
    plt.vlines(time[cutoff], ymin=equi_mean-3*equi_std,
               ymax=equi_mean+3*equi_std, color=color)


def _plot_steady_state(time, copynum, opt_wsize):

    equi_mean = np.mean(copynum[-opt_wsize:])
    equi_std = np.std(copynum[-opt_wsize:])

    # plot data
    plt.plot(time, copynum)

    # plot the turning point
    plt.axvline(time[-opt_wsize], color='black')
    plt.axhline(equi_mean, color='black')

    lastN_third = int(opt_wsize/3)
    t_full = time[-opt_wsize]
    t_2of3 = time[-2*lastN_third]
    t_1of3 = time[-lastN_third]
    # 3/3 region
    _draw_result(time, copynum, -opt_wsize, -2*lastN_third,
                label=f'From {t_full:.2e}s to {t_2of3:.2e}s',
                color='darkorange')
    # 2/3 region
    _draw_result(time, copynum, -2*lastN_third, -lastN_third,
                label=f'From {t_2of3:.2e}s to {t_1of3:.2e}s',
                color='darkgoldenrod')
    # 1/3 region
    _draw_result(time, copynum, -lastN_third, len(time),
                label=f'From {t_1of3:.2e}s to {time[-1]:.2e}s',
                color='orangered')

    plt.legend(loc=[1.01, 0.2])

    plt.xlabel('Time / s', fontsize=17)
    plt.ylabel('Copynumber #', fontsize=17)

    plt.title(f'Average molecule number: {equi_mean:.2f}' + ' $\pm$ ' + f'{equi_std:.2f}' +
              f'\nThe turning time point is {time[-opt_wsize]:.2e} s')
    plt.show()


def _get_rel_slope(time, copynumber, start, end):
    fit = stats.linregress(time[start:end], copynumber[start:end])
    mean = np.mean(copynumber[start:end])
    return fit.slope / np.mean(copynumber[start:end]), mean


def find_equi_linear(time, copynumber, ifdraw=True):

    nan = np.nan

    N_data = len(time)
    N_upper = 1e4
    N_why_so_many_dp = 1e5
    start = max(5, int(len(time)/100))
    if N_data < 10:
        print(
            f'ERROR! Not enough data. Given {N_data} data, expect more than 10!')
        return nan, nan, nan
    elif N_data <= N_upper + 2*start:
        window_size = np.arange(start, int(len(time)-start), 1)
    elif N_data <= N_why_so_many_dp + 2*start:
        window_step = int((N_data - 2*start) / N_upper)
        window_size = np.arange(start, int(len(time)-start), window_step)
        print(
            f'To save time, only try {len(window_size)} window sizes. (No data points dropped)')
    else:
        data_step = int(N_data / N_why_so_many_dp)
        copynumber = copynumber[::data_step]
        time = time[::data_step]
        N_data = len(copynumber)
        window_step = int((N_data - 2*start) / N_upper)
        window_size = np.arange(start, int(len(time)-start), window_step)
        print(
            f'To save time, only try {len(window_size)} window sizes for {N_data} data points')

    slope_rel_equi = np.zeros_like(window_size, dtype=np.double)
    equi_mean = np.zeros_like(window_size, dtype=np.double)

    for i in progressbar(range(len(window_size))):
        # linear fitting and find percentage change, mean value
        slope_rel_equi[i], equi_mean[i] = _get_rel_slope(
            time, copynumber, -window_size[i], N_data)

    linear_region = np.where(abs(slope_rel_equi) < 0.01)
    num_candidates = np.size(linear_region)

    if num_candidates == 0:
        print(
            f'ERROR! No steady state found. The smallest relative slope is {np.min(slope_rel_equi)}')
        _plot_steady_state(time, copynumber, 1)
        return nan, nan, nan
    else:
        linear_winow_filtered = []
        linear_id_filtered = []
        for idw, linear_window in progressbar(enumerate(window_size[linear_region])):

            window_id = linear_region[0][idw]
            window_mean = equi_mean[window_id]
            window_std = np.std(copynumber[-linear_window:])

            # Separate the whole section into two parts
            one_half = int(linear_window/2)
            # for the first section
            one_half_slope, one_half_mean = _get_rel_slope(
                time, copynumber, -one_half, N_data)
            # for the last section
            two_half_slope, two_half_mean = _get_rel_slope(
                time, copynumber, -linear_window, -one_half)

            # if both two parts are flat
            if window_mean < 1:
                both_flat = True
            else:
                both_flat = abs(one_half_slope) < 0.1 and abs(
                    two_half_slope) < 0.1
            # if both two parts have mean value close to the whole mean
            both_close = abs(one_half_mean - window_mean) < window_std and abs(
                two_half_mean - window_mean) < window_std

            if both_flat or both_close:
                linear_winow_filtered.append(linear_window)
                linear_id_filtered.append(linear_region[0][idw])

        # find the "best" window size
        if len(linear_winow_filtered) == 0:
            print(f'WARNING! The steady state might be fake.')
            print('\t1)Please try running more trajectories,')
            print('\t2)or running longer.')
            print('\t3)or collecting more date points.')
            best_window_id = linear_region[0][np.argmax(
                window_size[linear_region])]
        else:
            print('Steady state found')
            best_window_id = linear_id_filtered[np.argmax(
                linear_winow_filtered)]

        bewt_window = window_size[best_window_id]
        window_mean = equi_mean[best_window_id]
        window_std = np.std(copynumber[-bewt_window:])
        window_time = time[-bewt_window]

        one_half = int(bewt_window/2)
        one_half_time = time[-one_half]
        # for the first section
        one_half_slope, one_half_mean = _get_rel_slope(time, copynumber, -one_half, N_data)
        # for the last section
        two_half_slope, two_half_mean = _get_rel_slope(time, copynumber, -bewt_window, -one_half)

        s_two_half = f'From {window_time:.2e}s to {one_half_time:.2e}s, '
        s_one_half = f'From {one_half_time:.2e}s to {time[-1]:.2e}s, '

        if window_mean < 1:
            print(f'This state has avearage = {window_mean:.2f} -> 0. '
                  'Please BE CAREFUL that this is special!')

            if abs(one_half_mean - window_mean) > window_std or abs(two_half_mean - window_mean) > window_std:
                print('This might be a meta-stable state:')
                print('\t' + s_two_half +
                      f'the average is {two_half_mean:.2f}%')
                print('\t' + s_one_half +
                      f'the average is {one_half_mean:.2f}%')

            if ifdraw: _plot_steady_state(time, copynumber, bewt_window)
            return bewt_window, slope_rel_equi[best_window_id], window_mean, window_std

        elif len(linear_winow_filtered) == 0:

            # print out warning messages

            if abs(one_half_slope) > 0.1 or abs(two_half_slope) > 0.1:
                print(s_two_half +
                      f'the relative slope is {two_half_slope*100:.2f}%')
                print(s_one_half +
                      f'the relative slope is {one_half_slope*100:.2f}%')
                print(
                    '(relative slope = slope (by linear fitting) / average molecule number)')

            if abs(one_half_mean - window_mean) > window_std or abs(two_half_mean - window_mean) > window_std:
                print(s_two_half + f'the average is {two_half_mean:.2f}%')
                print(s_one_half + f'the average is {one_half_mean:.2f}%')

            if ifdraw: _plot_steady_state(time, copynumber, bewt_window)
            return bewt_window, slope_rel_equi[best_window_id], window_mean, window_std

        else:

            # print out warning messages
            if abs(one_half_slope) > 0.1 or abs(two_half_slope) > 0.1:
                print('This state may have periodic fluctuation:')
                print('\t' + s_two_half +
                      f'the relative slope is {two_half_slope*100:.2f}%')
                print('\t' + s_one_half +
                      f'the relative slope is {one_half_slope*100:.2f}%')
                print(
                    '\t(relative slope = slope (by linear fitting) / average molecule number)')

            if abs(one_half_mean - window_mean) > window_std or abs(two_half_mean - window_mean) > window_std:
                print('This might be a meta-stable state:')
                print('\t' + s_two_half +
                      f'the average is {two_half_mean:.2f}%')
                print('\t' + s_one_half +
                      f'the average is {one_half_mean:.2f}%')

            if ifdraw: _plot_steady_state(time, copynumber, bewt_window)
            return bewt_window, slope_rel_equi[best_window_id], window_mean, window_std
