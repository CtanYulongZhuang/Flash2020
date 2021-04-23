# LS for 60%
import h5py
import numpy as np
import matplotlib.pyplot as plt

def r_mean_weighted(delay,diameter,error,bin_cen,win_hsize):
    #bin_cen = np.arange(-3, 80, 1.0)
    bin_dia = np.zeros_like(bin_cen)
    bin_dia_error = np.zeros_like(bin_cen)
    for i, b in enumerate(bin_cen):
        curr_bin_sel = np.where((delay < b + win_hsize) & (delay > b - win_hsize))[0]
        print(i,b,len(curr_bin_sel))
        diain = diameter[curr_bin_sel]
        errin = error[curr_bin_sel]
        bin_dia[i] = np.sum(diain/errin**2)/np.sum(1/errin**2)
        bin_dia_error[i] = np.sqrt(1/(np.sum(errin**(-2))))
        if (len(curr_bin_sel) < 10):
            bin_dia[i] = -999
            bin_dia_error[i] = -999
        #print(i, len(selds), bin_dia_error[i], selds.std())
        #sys.stderr.write('\r%d'%(i+1))
    return [bin_cen, bin_dia, bin_dia_error]

def read_blacklist(blacklist_name):
    file0 = open(blacklist_name,"r")
    Lines = file0.readlines()
    file0.close() #to change file access modes
    blacklist = np.array([int(i.split()[0]) for i in Lines])
    return blacklist


data1 = h5py.File('diameter_mask1.h5','r')
diameter01 = data1['diameter'][:]
error00 = data1['error'][:]
data1.close()
error01 = np.sqrt(error00**2 + 1.6090532403672295**2)

info1 = h5py.File('info_107_152.h5','r')
delay01 = info1['delay_ps'][:]
pp_attenuation = info1['pp_attenuation'][:]   #[0.175, 0.4  , 0.6  , 0.78 , 0.88 ]
gmd = info1['gmd_uJ'][:]
info1.close()
fnum = h5py.File('Run_num.h5', 'r')
run_num = fnum['Run_num'][:]
fnum.close()

pp012 = np.where(pp_attenuation == 0.88)[0]
pp022 = np.where(pp_attenuation == 0.78)[0]
pp040 = np.where(pp_attenuation == 0.6)[0]
pp060 = np.where(pp_attenuation == 0.4)[0]
pp082 = np.where(pp_attenuation == 0.175)[0]

com012 = np.unique(run_num[pp012], return_counts=True)
com022 = np.unique(run_num[pp022], return_counts=True)
com040 = np.unique(run_num[pp040], return_counts=True)
com060 = np.unique(run_num[pp060], return_counts=True)
com082 = np.unique(run_num[pp082], return_counts=True)

#window size 60
blacklist_name = 'blacklist_r0_60_68914.dat'
bl060_sb = read_blacklist(blacklist_name)
bl060 = run_num*0.0 + 1
bl060[pp060] = bl060_sb
bin_cen = np.arange(-3, 80, 0.1)
nrun0 = pp060
select_runs = [7,8,9,10]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com060[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0) & (delay01[nrun0] > -100))]
#nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (nKH[nrun0] == 0) & (delay01[nrun0] > -100))]

#full range standard deviation check

delay_range = delay01[nrun]
diameter_range = diameter01[nrun]
error_range = error01[nrun]
error0_range = error00[nrun]

bin_cen = np.arange(0, 80, 0.1)
sta_weighted_0 = r_mean_weighted(delay_range, diameter_range, error0_range, bin_cen,1.0)
plt.scatter(sta_weighted_0[0],sta_weighted_0[2],s=1,color='Black', label='sizing error, bin_size = 2.0')
sta_weighted_0 = r_mean_weighted(delay_range, diameter_range, error_range, bin_cen,1.0)
plt.scatter(sta_weighted_0[0],sta_weighted_0[2],s=1,color='blue', label='effective error, bin_size = 2.0')
plt.legend()
plt.xlabel('delay [ps]')
plt.ylabel('standard deviation [ps]')



h5 = h5py.File('PL60_selected.h5', 'w')
h5['delay'] = delay01[nrun]
h5['diameter'] = diameter01[nrun]
h5['error_eff'] = error01[nrun]
h5['error_sizing'] = error00[nrun]
h5.close()





#==================range_01==================.

d_range_01 = np.where((delay01[nrun] < 23) & (delay01[nrun] > 12))[0]

delay_range = delay01[nrun][d_range_01[0::1]]
diameter_range = diameter01[nrun][d_range_01[0::1]]
error_range = error01[nrun][d_range_01[0::1]]
error0_range = error00[nrun][d_range_01[0::1]]

from scipy.optimize import curve_fit
import scipy
#Gaussian profile with centre in the 0
def simple_model(x, a0, a1):
    return a0 + a1*x

Sopf_0,Scov_0 = curve_fit(simple_model,delay_range,diameter_range, sigma=error_range, maxfev=5000)

continum_0 = simple_model(delay_range, *Sopf_0)
dia_mod_0 = diameter_range-continum_0

from astropy import timeseries as ts
freq = np.arange(0.01,0.5,0.001)
probabilities = [0.1, 0.05, 0.01]
term_0 = ts.LombScargle(delay_range,dia_mod_0,error0_range,center_data=False, fit_mean=True, normalization='standard')
intens_0 = term_0.power(freq)
Fap_0 = term_0.false_alarm_probability(intens_0)
Pts_0 = term_0.false_alarm_level(probabilities)

Jxx = np.zeros((len(freq),len(t_fit)))
t_fit = np.linspace(12, 23)
for i in range(len(freq)):
    fff = freq[i]
    y_fit = term_0.model(t_fit, fff)
    Jxx[i] = y_fit-np.mean(y_fit)
    print(np.min(y_fit))
    #plt.plot(y_fit)

AVJ = np.average(Jxx, axis=0, weights = np.sqrt(intens_0))


freq_sig = freq[Fap_0 < 0.05]

Jxx_sig = np.zeros((len(freq_sig),len(t_fit)))
for i in range(len(freq_sig)):
    fff = freq_sig[i]
    y_fit = term_0.model(t_fit, fff)
    Jxx_sig[i] = y_fit-np.mean(y_fit)
    print(np.min(y_fit))
    #plt.plot(y_fit)

AVJ_sig = np.average(Jxx_sig, axis=0, weights = np.sqrt(intens_0[Fap_0 < 0.05]))

bin_cen = np.arange(12, 23, 0.1)
sta_weighted_0 = r_mean_weighted(delay_range, dia_mod_0, error_range, bin_cen,binsize0)
plt.scatter(sta_weighted_0[0],sta_weighted_0[1],s=1,color='Red', label='running mean, bin_size = 1.0')
plt.scatter(delay_range,dia_mod_0,s=1,color='black',alpha=0.1)
plt.plot(t_fit,AVJ, color='blue', label='model for all low frequency')
plt.plot(t_fit,AVJ_sig, color='purple', label='model for significant frequency')
plt.ylim(-2,2)
plt.legend()
plt.ylabel('diameter - linear_model')
plt.xlabel('delay')





plt.figure(figsize=(14,5))
plt.subplot(131)
binsize0 = 1
bin_cen = np.arange(12, 23, 0.1)
sta_weighted_0 = r_mean_weighted(delay_range, diameter_range, error_range, bin_cen,binsize0)
continum_1 = sta_weighted_0[1]
plt.scatter(delay_range, diameter_range ,s=1, alpha=0.1, color='Black')
plt.scatter(sta_weighted_0[0],sta_weighted_0[1],s=1,color='Red', label='running mean, bin_size = 1.0')
plt.scatter(delay_range, simple_model(delay_range, *Sopf_0),s=1,color='Black', label='linear fit' )
plt.xlabel('delay [ps]')
plt.ylabel('diameter [nm]')
plt.ylim(25,28)
plt.legend()


plt.subplot(132)
plt.plot(freq, intens_0, color='blue', label='sizing error')
plt.hlines(Pts_0[1],0,0.5, color='grey', label='5% significance')
#plt.legend()
plt.text(0.1, Pts_0[1], '5% significance')
plt.xlabel('frequency [1/ps]')
plt.ylabel('intens')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(133)
plt.plot(freq, Fap_0, color='blue', label='sizing error')
plt.hlines(0.05,0,0.5, color='grey', label='5% significance')
plt.text(0.1,0.1, '5% significance')
plt.ylim(0,1.1)
plt.xlabel('frequency [1/ps]')
plt.ylabel('probability to H0')





#==================range_02==================.

d_range_02 = np.where((delay01[nrun] > 30) & (delay01[nrun] < 60))[0]

delay_range = delay01[nrun][d_range_02[0::1]]
diameter_range = diameter01[nrun][d_range_02[0::1]]
error_range = error01[nrun][d_range_02[0::1]]
error0_range = error00[nrun][d_range_02[0::1]]

from scipy.optimize import curve_fit
import scipy
#Gaussian profile with centre in the 0
def simple_model(x, a0, a1):
    return a0 + a1*x

Sopf_0,Scov_0 = curve_fit(simple_model,delay_range,diameter_range, sigma=error_range, maxfev=5000)

continum_0 = simple_model(delay_range, *Sopf_0)
dia_mod_0 = diameter_range-continum_0

from astropy import timeseries as ts
freq = np.arange(0.01,0.5,0.001)
probabilities = [0.1, 0.05, 0.01]
term_0 = ts.LombScargle(delay_range,dia_mod_0,error0_range,center_data=False, fit_mean=True, normalization='standard')
intens_0 = term_0.power(freq)
Fap_0 = term_0.false_alarm_probability(intens_0)
Pts_0 = term_0.false_alarm_level(probabilities)

plt.figure(figsize=(14,5))
plt.subplot(131)
binsize0 = 1
bin_cen = np.arange(30, 60, 0.1)
sta_weighted_0 = r_mean_weighted(delay_range, diameter_range, error_range, bin_cen,binsize0)
continum_1 = sta_weighted_0[1]
plt.scatter(delay_range, diameter_range ,s=1, alpha=0.1, color='Black')
plt.scatter(sta_weighted_0[0],sta_weighted_0[1],s=1,color='Red', label='running mean, bin_size = 1.0')
plt.scatter(delay_range, simple_model(delay_range, *Sopf_0),s=1,color='Black', label='linear fit' )
plt.xlabel('delay [ps]')
plt.ylabel('diameter [nm]')
plt.ylim(27,28)
plt.legend()


plt.subplot(132)
plt.plot(freq, intens_0, color='blue', label='sizing error')
plt.hlines(Pts_0[1],0,0.5, color='grey', label='5% significance')
#plt.legend()
plt.text(0.1, Pts_0[1], '5% significance')
plt.xlabel('frequency [1/ps]')
plt.ylabel('intens')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(133)
plt.plot(freq, Fap_0, color='blue', label='sizing error')
plt.hlines(0.05,0,0.5, color='grey', label='5% significance')
plt.text(0.1,0.1, '5% significance')
plt.ylim(0,1.1)
plt.xlabel('frequency [1/ps]')
plt.ylabel('probability to H0')




#==================range_03==================.

d_range_03 = np.where((delay01[nrun] > 12) & (delay01[nrun] < 72))

delay_range = delay01[nrun][d_range_03]
diameter_range = diameter01[nrun][d_range_03]
error_range = error01[nrun][d_range_03]
error0_range = error00[nrun][d_range_03]

from scipy.optimize import curve_fit
import scipy
#Gaussian profile with centre in the 0
def plateau_model(x, a0, a1, a2, a3):
    return a0 - a1*np.exp(-(x-a2)/a3)


#initial_guess = [26.23794313,  0.08837063,  0.20073022,  0.67, -5.41961505]
#Sopf_1,Scov_1 = curve_fit(plateau_model,delay_range,diameter_range, sigma=error0_range, p0=initial_guess, maxfev=5000)
Sopf_1,Scov_1 = curve_fit(plateau_model,delay_range,diameter_range, sigma=error0_range, maxfev=5000)

#plt.scatter(delay_range, plateau_model(delay_range, *Sopf_1),s=1, label='exponent' )


continum_1 = plateau_model(delay_range, *Sopf_1)
dia_mod_1 = diameter_range-continum_1


from astropy import timeseries as ts
freq = np.arange(0.01,0.5,0.001)
probabilities = [0.1, 0.05, 0.01]
term_0 = ts.LombScargle(delay_range,dia_mod_1,error0_range,center_data=False, fit_mean=True, normalization='standard')
intens_0 = term_0.power(freq)
Fap_0 = term_0.false_alarm_probability(intens_0)
Pts_0 = term_0.false_alarm_level(probabilities)



plt.subplot(131)
binsize0 = 1
bin_cen = np.arange(12, 72, 0.1)
sta_weighted_0 = r_mean_weighted(delay_range, diameter_range, error_range, bin_cen,binsize0)
continum_1 = sta_weighted_0[1]
plt.scatter(delay_range, diameter_range ,s=1, alpha=0.1, color='blue')
plt.scatter(sta_weighted_0[0],sta_weighted_0[1],s=1,color='red', label='running mean, bin_size = 1.0')
plt.scatter(delay_range, plateau_model(delay_range, *Sopf_1),s=1, label='exponent' )
plt.xlabel('delay')
plt.ylabel('diameter - continuum')
plt.ylim(20,35)
plt.legend()


plt.subplot(132)
plt.plot(freq, intens_0, color='red', label='sizing error')
plt.hlines(Pts_0[0],0,0.5, color='darkred', label='10% significance')
plt.legend()
plt.xlabel('frequency [1/ps]')
plt.ylabel('intens')

plt.subplot(133)
plt.plot(freq, Fap_0, color='red', label='sizing error')
plt.ylim(0,1.1)
plt.xlabel('frequency [1/ps]')
plt.ylabel('pro')




# range02


cgcolor = ['red', 'orange', 'green', 'blue']

plt.figure(figsize=(14,5))
for i in range(3):
    run_n = np.where(run_num[nrun] == com060[0][select_runs[i]] )
    nrun_run = nrun[run_n]

    d_range_02 = np.where((delay01[nrun_run] > 30) & (delay01[nrun_run] < 60))[0]

    delay_range = delay01[nrun_run][d_range_02[0::1]]
    diameter_range = diameter01[nrun_run][d_range_02[0::1]]
    error_range = error01[nrun_run][d_range_02[0::1]]
    error0_range = error00[nrun_run][d_range_02[0::1]]


    plt.subplot(121)
    bin_cen = np.arange(30, 60, 0.1)
    sta_weighted_0 = r_mean_weighted(delay_range, diameter_range, error_range, bin_cen,binsize0)
    plt.plot(sta_weighted_0[0],sta_weighted_0[1],color=cgcolor[i], label='running mean, bin_size = 1.0')
    plt.ylim(27,29)
    plt.xlabel('delay [ps]')
    plt.ylabel('diameter')

    plt.subplot(122)
    Sopf_0,Scov_0 = curve_fit(simple_model,delay_range,diameter_range, sigma=error_range, maxfev=5000)
    continum_0 = simple_model(delay_range, *Sopf_0)
    dia_mod_0 = diameter_range-continum_0
    freq = np.arange(0.01,0.5,0.001)
    probabilities = [0.1, 0.05, 0.01]
    term_0 = ts.LombScargle(delay_range,dia_mod_0,error0_range,center_data=False, fit_mean=True, normalization='standard')
    intens_0 = term_0.power(freq)
    Fap_0 = term_0.false_alarm_probability(intens_0)
    Pts_0 = term_0.false_alarm_level(probabilities)

    plt.plot(freq, intens_0+0.01*i, color=cgcolor[i], label='run'+str(com060[0][select_runs[i]] ))
    plt.legend()
    plt.xlabel('frequency [1/ps]')



# range01


cgcolor = ['red', 'orange', 'green', 'blue']

plt.figure(figsize=(14,5))
for i in range(3):
    run_n = np.where(run_num[nrun] == com060[0][select_runs[i]] )
    nrun_run = nrun[run_n]

    d_range_02 = np.where((delay01[nrun_run] > 12) & (delay01[nrun_run] < 23))[0]

    delay_range = delay01[nrun_run][d_range_02[0::1]]
    diameter_range = diameter01[nrun_run][d_range_02[0::1]]
    error_range = error01[nrun_run][d_range_02[0::1]]
    error0_range = error00[nrun_run][d_range_02[0::1]]


    plt.subplot(121)
    bin_cen = np.arange(12, 23, 0.1)
    sta_weighted_0 = r_mean_weighted(delay_range, diameter_range, error_range, bin_cen,binsize0)
    plt.plot(sta_weighted_0[0],sta_weighted_0[1],color=cgcolor[i], label='running mean, bin_size = 1.0')
    plt.ylim(25,28)
    plt.xlabel('delay [ps]')
    plt.ylabel('diameter')

    plt.subplot(122)
    Sopf_0,Scov_0 = curve_fit(simple_model,delay_range,diameter_range, sigma=error_range, maxfev=5000)
    continum_0 = simple_model(delay_range, *Sopf_0)
    dia_mod_0 = diameter_range-continum_0
    freq = np.arange(0.01,0.5,0.001)
    probabilities = [0.1, 0.05, 0.01]
    term_0 = ts.LombScargle(delay_range,dia_mod_0,error0_range,center_data=False, fit_mean=True, normalization='standard')
    intens_0 = term_0.power(freq)
    Fap_0 = term_0.false_alarm_probability(intens_0)
    Pts_0 = term_0.false_alarm_level(probabilities)

    plt.plot(freq, intens_0+0.01*i, color=cgcolor[i], label='run'+str(com060[0][select_runs[i]] ))
    plt.legend()
    plt.xlabel('frequency [1/ps]')

















































#
