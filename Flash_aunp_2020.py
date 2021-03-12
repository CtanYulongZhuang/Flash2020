import h5py
import numpy as np
import matplotlib.pyplot as plt
def read_blacklist(blacklist_name):
    file0 = open(blacklist_name,"r")
    Lines = file0.readlines()
    file0.close() #to change file access modes
    blacklist = np.array([int(i.split()[0]) for i in Lines])
    return blacklist

def r_mean(delay,diameter,bin_cen,win_hsize):
    #bin_cen = np.arange(-3, 80, 1.0)
    bin_dia = np.zeros_like(bin_cen)
    bin_dia_error = np.zeros_like(bin_cen)
    for i, b in enumerate(bin_cen):
        curr_bin_sel = (delay < b + win_hsize) & (delay > b - win_hsize)
        selds = diameter[curr_bin_sel]
        bin_dia_error[i] = selds.std() / np.sqrt(len(selds) - 1)
        bin_dia[i] = selds.mean()
        #print(i, len(selds), bin_dia_error[i], selds.std())
        #sys.stderr.write('\r%d'%(i+1))
    return [bin_cen, bin_dia, bin_dia_error]

def r_mean_weighted(delay,diameter,error,bin_cen,win_hsize):
    #bin_cen = np.arange(-3, 80, 1.0)
    bin_dia = np.zeros_like(bin_cen)
    bin_dia_error = np.zeros_like(bin_cen)
    for i, b in enumerate(bin_cen):
        curr_bin_sel = (delay < b + win_hsize) & (delay > b - win_hsize)
        diain = diameter[curr_bin_sel]
        errin = error[curr_bin_sel]
        bin_dia[i] = np.sum(diain/errin**2)/np.sum(1/errin**2)
        bin_dia_error[i] = np.sqrt(1/(np.sum(errin**(-2))))
        if (len(curr_bin_sel) < 100):
            bin_dia[i] = 0
            bin_dia_error[i] = 0
        #print(i, len(selds), bin_dia_error[i], selds.std())
        #sys.stderr.write('\r%d'%(i+1))
    return [bin_cen, bin_dia, bin_dia_error]

data1 = h5py.File('diameter_mask1.h5','r')
diameter01 = data1['diameter'][:]
error01 = data1['error'][:]
data1.close()
error01_all = np.sqrt(error01**2 + 1.6090532403672295**2)

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


plt.step(com012[0], com012[1], color = "Cyan")
plt.step(com022[0], com022[1], color = "Green")
plt.step(com040[0], com040[1])
plt.step(com060[0], com060[1], color = "Red")
plt.step(com082[0], com082[1])


#================Plot_ppe 12%=================


blacklist_name = 'blacklist_r0_12_34000.dat'
bl012_sb = read_blacklist(blacklist_name)
bl012 = run_num*0.0 + 1
bl012[pp012] = bl012_sb

plt.figure(figsize = (12,7))
bin_cen = np.arange(-3, 22, 0.1)
plt.subplot(3,5,1)
nrun0 = pp012
select_runs = [1,3,4,5,10,11,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com012[0][select_runs]):
        nKH[i] = 0.0


nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(delay01[nrun],diameter01[nrun],alpha=0.1,s=1, color= 'grey')
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Selected', color='black')
#plt.errorbar(sta_weighted[0],sta_weighted[1],yerr=sta_weighted[2],alpha=0.8, color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.text(-2,26.1, 'N = '+str(int(len(nrun))))
plt.legend()
plt.ylim(26,28)
plt.xlim(-3,22)
plt.ylabel('weighted mean')
plt.legend(loc='upper center')

bin_cen = np.arange(-3, 22, 0.1)
for i in range(len(com012[0])):
    plt.subplot(3,5,i+1+1)
    nrun = np.where((run_num == com012[0][i]) & (diameter01 < 40) & (diameter01 > 10) & (bl012 == 0))[0]
    ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
    sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
    plt.scatter(delay01[nrun],diameter01[nrun],alpha=0.3,s=1, color= 'grey')
    plt.scatter(sta_weighted[0],sta_weighted[1],s=1,label='run '+str(int(com012[0][i])))
    #plt.errorbar(sta_weighted[0],sta_weighted[1],yerr=sta_weighted[2],alpha=0.5)
    plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
    plt.legend(loc='upper center')
    plt.ylim(26,28)
    plt.xlim(-3,22)
    #plt.text(-2,26.3, 'ave_Xe = '+str(ave_xe))
    plt.text(-2,26.1, 'N = '+str(int(len(nrun))))
    if ((i+1+1)%5 == 1):
        plt.ylabel('weighted mean')
    if ((i+1+1) > 10):
        plt.xlabel('delay [ps] ')

#================Plot_ppe 22%=================

blacklist_name = 'blacklist_r0_22_39427.dat'
bl022_sb = read_blacklist(blacklist_name)
bl022 = run_num*0.0 + 1
bl022[pp022] = bl022_sb

plt.figure(figsize = (12,7))
bin_cen = np.arange(-10, 100, 0.1)
plt.subplot(3,5,1)
nrun0 = pp022
select_runs = [0,1,3,6,7,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com022[0][select_runs]):
        nKH[i] = 0.0


nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl022[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(delay01[nrun],diameter01[nrun],alpha=0.1,s=1, color= 'grey')
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Selected', color='black')
#plt.errorbar(sta_weighted[0],sta_weighted[1],yerr=sta_weighted[2],alpha=0.8, color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.text(-2,25.1, 'N = '+str(int(len(nrun))))
plt.legend()
plt.ylim(25,30)
plt.xlim(-10,100)
plt.ylabel('weighted mean')
plt.legend(loc='upper center')

bin_cen = np.arange(-10, 100, 0.1)
for i in range(len(com022[0])):
    plt.subplot(3,5,i+1+1)
    nrun = np.where((run_num == com022[0][i]) & (diameter01 < 40) & (diameter01 > 10) & (bl022 == 0))[0]
    ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
    sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
    plt.scatter(delay01[nrun],diameter01[nrun],alpha=0.3,s=1, color= 'grey')
    plt.scatter(sta_weighted[0],sta_weighted[1],s=1,label='run '+str(int(com022[0][i])))
    #plt.errorbar(sta_weighted[0],sta_weighted[1],yerr=sta_weighted[2],alpha=0.5)
    plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
    plt.legend(loc='upper center')
    plt.ylim(25,30)
    plt.xlim(-10,100)
    #plt.text(-2,26.3, 'ave_Xe = '+str(ave_xe))
    plt.text(-2,25.1, 'N = '+str(int(len(nrun))))
    if ((i+1+1)%5 == 1):
        plt.ylabel('weighted mean')
    if ((i+1+1) > 10):
        plt.xlabel('delay [ps] ')



#================Plot_ppe 60%=================

blacklist_name = 'blacklist_r0_60_43736.dat'
bl060_sb = read_blacklist(blacklist_name)
bl060 = run_num*0.0 + 1
bl060[pp060] = bl060_sb

plt.figure(figsize = (12,7))
bin_cen = np.arange(-10, 100, 0.1)
plt.subplot(3,4,1)
nrun0 = pp060
select_runs = [7,8,9,10]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com060[0][select_runs]):
        nKH[i] = 0.0


nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(delay01[nrun],diameter01[nrun],alpha=0.1,s=1, color= 'grey')
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Selected', color='black')
#plt.errorbar(sta_weighted[0],sta_weighted[1],yerr=sta_weighted[2],alpha=0.8, color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.text(-2,25.1, 'N = '+str(int(len(nrun))))
plt.legend()
plt.ylim(25,33)
plt.xlim(-10,100)
plt.ylabel('weighted mean')
plt.legend(loc='upper center')

bin_cen = np.arange(-10, 100, 0.1)
for i in range(len(com060[0])):
    plt.subplot(3,4,i+1+1)
    nrun = np.where((run_num == com060[0][i]) & (diameter01 < 40) & (diameter01 > 10) & (bl060 == 0))[0]
    ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
    sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
    plt.scatter(delay01[nrun],diameter01[nrun],alpha=0.3,s=1, color= 'grey')
    plt.scatter(sta_weighted[0],sta_weighted[1],s=1,label='run '+str(int(com060[0][i])))
    #plt.errorbar(sta_weighted[0],sta_weighted[1],yerr=sta_weighted[2],alpha=0.5)
    plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
    plt.legend(loc='upper center')
    plt.ylim(25,33)
    plt.xlim(-10,100)
    #plt.text(-2,26.3, 'ave_Xe = '+str(ave_xe))
    plt.text(-2,25.1, 'N = '+str(int(len(nrun))))
    if ((i+1+1)%4 == 1):
        plt.ylabel('weighted mean')
    if ((i+1+1) > 8):
        plt.xlabel('delay [ps] ')




#+++++++++++++++++error

blacklist_name = 'blacklist_r0_12_34000.dat'
bl012 = read_blacklist(blacklist_name)

nrun0 = pp012
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012 == 0))]

bin_cen = np.arange(1.7, 3.75, 0.02)
sta = r_mean(np.log10(intens_tot1[nrun]), diameter01[nrun], bin_cen,0.2)
sigma_sample = np.sqrt(sta[2]**2 - err_model**2)
final_error = np.sqrt(np.mean(sigma_sample)**2 + err_model**2)


plt.scatter(sta[0],sigma_sample,s=1,color='Red', label=r'$\sigma_{sample}$')
plt.scatter(sta[0],sta[2],s=1,color='purple', label=r'$\sigma_{12\%}$')
plt.scatter(lgint_model,err_model,s=1,color='blue', label=r'$\sigma_{sizing}$')
plt.scatter(lgint_model,final_error,s=2,color='black', label=r'$\sigma_{final}$')
plt.xlabel('lg(intens)')
plt.legend()
plt.ylim(0,3)

r'$\alpha > \beta$'








#=====3 subplot
plt.subplot(131)
blacklist_name = 'blacklist_r0_12_34000.dat'
bl012_sb = read_blacklist(blacklist_name)
bl012 = run_num*0.0 + 1
bl012[pp012] = bl012_sb
bin_cen = np.arange(-3, 22, 0.1)
nrun0 = pp012
select_runs = [1,3,4,5,10,11,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com012[0][select_runs]):
        nKH[i] = 0.0


nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='All 12%', color='red')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='red')
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Selected', color='green')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='green')
plt.legend()
plt.ylim(26.75,27.25)
plt.xlim(-3,22)
plt.ylabel('diameter [nm]')
plt.xlabel('delay [ps]')
plt.legend(loc='upper center')


plt.subplot(132)
blacklist_name = 'blacklist_r0_22_39427.dat'
bl022_sb = read_blacklist(blacklist_name)
bl022 = run_num*0.0 + 1
bl022[pp022] = bl022_sb
bin_cen = np.arange(-10, 100, 0.1)
nrun0 = pp022
select_runs = [0,1,3,6,7,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com022[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl022[nrun0] == 0) )]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='All 22%', color='red')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='red')
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl022[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Selected', color='green')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='green')
plt.legend()
plt.ylim(26,28)
plt.xlim(-10,100)
plt.ylabel('diameter [nm]')
plt.xlabel('delay [ps]')
plt.legend(loc='upper center')



plt.subplot(133)
blacklist_name = 'blacklist_r0_60_68914.dat'
bl060_sb = read_blacklist(blacklist_name)
bl060 = run_num*0.0 + 1
bl060[pp060] = bl060_sb
bin_cen = np.arange(0, 80, 0.1)
nrun0 = pp060
select_runs = [7,8,9,10]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com060[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
#plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='all 60%', color='red')
#plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='red')
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Selected (subgroup 1)', color='green')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='green')
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 1))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Rejected (subgroup 2)', color='gray')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='gray')
plt.legend()
plt.ylim(26,32)
plt.xlim(0,80)
plt.ylabel('diameter [nm]')
plt.xlabel('delay [ps]')
plt.legend(loc='upper right')











#=====3 summary
plt.subplot(131)
blacklist_name = 'blacklist_r0_12_34000.dat'
bl012_sb = read_blacklist(blacklist_name)
bl012 = run_num*0.0 + 1
bl012[pp012] = bl012_sb
bin_cen = np.arange(-3, 22, 0.1)
nrun0 = pp012
select_runs = [1,3,4,5,10,11,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com012[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='selected samples', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(26.75,27.25)
plt.xlim(-3,22)
plt.ylabel('diameter [nm]')
plt.xlabel('delay [ps]')
plt.legend(loc='upper center')
sta_weighted_12 = sta_weighted

plt.subplot(132)
blacklist_name = 'blacklist_r0_22_39427.dat'
bl022_sb = read_blacklist(blacklist_name)
bl022 = run_num*0.0 + 1
bl022[pp022] = bl022_sb
bin_cen = np.arange(-10, 100, 0.1)
nrun0 = pp022
select_runs = [0,1,3,6,7,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com022[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl022[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='selected samples', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(26,28)
plt.xlim(-10,100)
plt.xlabel('delay [ps]')
plt.legend(loc='upper center')
sta_weighted_22 = sta_weighted


plt.subplot(133)
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

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='selected samples', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(25.5,28)
plt.xlim(-10,80)
plt.xlabel('delay [ps]')
plt.legend(loc='upper center')
sta_weighted_60 = sta_weighted


h5 = h5py.File('Summary.h5', 'w')
h5['sta_weighted_12'] = sta_weighted_12
h5['sta_weighted_22'] = sta_weighted_22
h5['sta_weighted_60'] = sta_weighted_60
h5.close()


#window size 12
blacklist_name = 'blacklist_r0_12_34000.dat'
bl012_sb = read_blacklist(blacklist_name)
bl012 = run_num*0.0 + 1
bl012[pp012] = bl012_sb
bin_cen = np.arange(-3, 22, 0.1)
nrun0 = pp012
select_runs = [1,3,4,5,10,11,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com012[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)

plt.subplot(221)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,0.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(26.8,27.2)
plt.xlim(-3,22)
plt.ylabel('weighted mean binsize=0.5')

plt.subplot(222)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(26.8,27.2)
plt.xlim(-3,22)
plt.ylabel('weighted mean binsize=1.0')

plt.subplot(223)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(26.8,27.2)
plt.xlim(-3,22)
plt.ylabel('weighted mean binsize=1.5')
plt.xlabel('delay [ps] ')

plt.subplot(224)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(26.8,27.2)
plt.xlim(-3,22)
plt.xlabel('delay [ps] ')
plt.ylabel('weighted mean binsize=2.0')













#window size 22
blacklist_name = 'blacklist_r0_22_39427.dat'
bl022_sb = read_blacklist(blacklist_name)
bl022 = run_num*0.0 + 1
bl022[pp022] = bl022_sb
bin_cen = np.arange(-10, 100, 0.1)
nrun0 = pp022
select_runs = [0,1,3,6,7,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com022[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl022[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)

plt.subplot(221)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,0.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(26,27.5)
plt.xlim(-10,100)
plt.ylabel('weighted mean binsize=0.5')

plt.subplot(222)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(26,27.5)
plt.xlim(-10,100)
plt.ylabel('weighted mean binsize=1.0')

plt.subplot(223)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(26,27.5)
plt.xlim(-10,100)
plt.ylabel('weighted mean binsize=1.5')
plt.xlabel('delay [ps] ')

plt.subplot(224)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(26,27.5)
plt.xlim(-10,100)
plt.ylabel('weighted mean binsize=2.0')
plt.xlabel('delay [ps] ')





#window size 60
blacklist_name = 'blacklist_r0_60_68914.dat'
bl060_sb = read_blacklist(blacklist_name)
bl060 = run_num*0.0 + 1
bl060[pp060] = bl060_sb
bin_cen = np.arange(-10, 100, 0.1)
nrun0 = pp060
select_runs = [7,8,9,10]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com060[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)

plt.subplot(221)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,0.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(27,29)
plt.xlim(-10,100)
plt.ylabel('weighted mean binsize=0.5')

plt.subplot(222)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(27,29)
plt.xlim(-10,100)
plt.ylabel('weighted mean binsize=1.0')

plt.subplot(223)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(27,29)
plt.xlim(-10,100)
plt.ylabel('weighted mean binsize=1.5')
plt.xlabel('delay [ps] ')

plt.subplot(224)
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,  color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
ave_xe = np.mean(gmd[nrun][gmd[nrun] > 0])//1
plt.legend()
plt.ylim(27,29)
plt.xlim(-10,100)
plt.ylabel('weighted mean binsize=2.0')
plt.xlabel('delay [ps] ')


















#Average rejected ones
#60
blacklist_name = 'blacklist_r0_60_68914.dat'
bl060_sb = read_blacklist(blacklist_name)
bl060 = run_num*0.0 + 1
bl060[pp060] = bl060_sb
bin_cen = np.arange(-10, 100, 0.1)
nrun0 = pp060
select_runs = [7,8,9,10]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com060[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='all 60%', color='red')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='red')
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Selected', color='green')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='green')
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 1))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='Rejected', color='gray')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='gray')
plt.legend()
plt.ylim(26,32)
plt.xlim(-10,100)
plt.ylabel('size')
plt.xlabel('delay')
plt.legend(loc='upper right')


nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 1))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='subgroup 2', color='Black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='Black')
plt.legend()
plt.ylim(26,31)
plt.xlim(-5,80)
plt.ylabel('diameter [nm]')
plt.xlabel('delay [ps]')
plt.legend(loc='upper right')


nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='subgroup 1', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(25.5,28)
plt.xlim(-5,80)
plt.ylabel('diameter [nm]')
plt.xlabel('delay [ps]')
plt.legend(loc='upper right')





#=====3 summary binsizes

blacklist_name = 'blacklist_r0_12_34000.dat'
bl012_sb = read_blacklist(blacklist_name)
bl012 = run_num*0.0 + 1
bl012[pp012] = bl012_sb
bin_cen = np.arange(-3, 22, 0.1)
nrun0 = pp012
select_runs = [1,3,4,5,10,11,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com012[0][select_runs]):
        nKH[i] = 0.0

plt.subplot(331)
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,0.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 1 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(26.75,27.25)
plt.xlim(-3,22)
plt.ylabel('diameter [nm]')
plt.legend(loc='upper center')

plt.subplot(334)
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 2 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(26.75,27.25)
plt.xlim(-3,22)
plt.ylabel('diameter [nm]')
plt.legend(loc='upper center')


plt.subplot(337)
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl012[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 4 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(26.75,27.25)
plt.xlim(-3,22)
plt.ylabel('diameter [nm]')
plt.xlabel('delay [ps]')
plt.legend(loc='upper center')


plt.subplot(332)
blacklist_name = 'blacklist_r0_22_39427.dat'
bl022_sb = read_blacklist(blacklist_name)
bl022 = run_num*0.0 + 1
bl022[pp022] = bl022_sb
bin_cen = np.arange(-10, 100, 0.1)
nrun0 = pp022
select_runs = [0,1,3,6,7,12]
nKH = run_num*0.0 + 1
for i in range(len(run_num)):
    if (run_num[i] in com022[0][select_runs]):
        nKH[i] = 0.0

nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl022[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,0.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 1 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(26,28)
plt.xlim(-10,100)
plt.legend(loc='upper center')

plt.subplot(335)
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl022[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 2 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(26,28)
plt.xlim(-10,100)
plt.legend(loc='upper center')

plt.subplot(338)
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl022[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 4 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(26,28)
plt.xlim(-10,100)
plt.xlabel('delay [ps]')
plt.legend(loc='upper center')



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

plt.subplot(333)
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,0.5)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 1 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(25.5,28.5)
plt.xlim(-5,80)
plt.legend(loc='upper center')

plt.subplot(336)
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,1.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 2 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(25.5,28.5)
plt.xlim(-5,80)
plt.legend(loc='upper center')

plt.subplot(339)
nrun = nrun0[np.where((diameter01[nrun0] < 40) & (diameter01[nrun0] > 10) & (bl060[nrun0] == 0) & (nKH[nrun0] == 0))]
sta_weighted = r_mean_weighted(delay01[nrun], diameter01[nrun], error01_all[nrun], bin_cen,2.0)
plt.scatter(sta_weighted[0],sta_weighted[1],s=2,label='binsize = 4 [ps]', color='black')
plt.fill_between(sta_weighted[0],sta_weighted[1]-sta_weighted[2],sta_weighted[1]+sta_weighted[2],alpha=0.4, color='black')
plt.legend()
plt.ylim(25.5,28.5)
plt.xlim(-5,80)
plt.xlabel('delay [ps]')
plt.legend(loc='upper center')
