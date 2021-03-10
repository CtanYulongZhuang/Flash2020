import sys
sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src/')
import detector
import reademc
import numpy as np
import ctypes
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import multiprocessing as mp
import matplotlib.pyplot as plt
#/asap3/flash/gpfs/bl1/2020/data/11007613/shared/aux/PL


import h5py
with h5py.File('../aux/det_intrad4_corr.h5', "r") as data_tem:
    keys = data_tem.keys()

data_tem = h5py.File('../aux/YZ_det_intrad4_corr.h5', "r")
qx = data_tem['qx'][:]
qy = data_tem['qy'][:]
qz = data_tem['qz'][:]
mask = data_tem['mask'][:]
data_tem.close()

def ellipse_r(a,b,t,tp):
    t_all = t + tp
    return  a*b/np.sqrt(a**2*np.sin(t_all)**2 + b**2*np.cos(t_all)**2)


a =3
b= 5
t = np.arange(1000)*0.02
tp = np.arange(60)/60*np.pi
r = ellipse_r(a,b,t,tp[15])
plt.scatter(r*np.sin(t), r*np.cos(t),s=1)

for i in range(10):
    r = ellipse_r(a,b,t,tp[i*3])
    plt.scatter(r*np.sin(t), r*np.cos(t),s=1)


#poisson
det = detector.Detector('../aux/det_intrad4.h5', mask_flag=True)
with open('../recon_0002/photons.txt', 'r') as f:
     emc_flist = [i.strip() for i in f.readlines()]

emc = reademc.EMCReader(emc_flist,det)
n_frames = emc.num_frames

clist = np.array([f['num_data'] for f in emc.flist])
running_num = np.zeros(np.max(clist))
running_num[0:clist[0]] = np.int( emc_flist[0][75:78] )
for i in range(len(clist)-1):
    running_num[clist[i]:clist[i+1]] = np.int( emc_flist[i+1][75:78] )





mask_emc0 = emc.get_frame(1).mask.ravel()
mask_emc0 = np.array([ not i for i in mask_emc0]).reshape(512, 256)
mask_center = np.load('../aux/mask_center_cmc_0.npy')                           #mask file

intrad = np.load('../aux/intrad_004.npy')
intrad = intrad[256:768,256:512]

#Absolute coordinate
x0,y0 = np.indices((512,256)); x0-=247; y0-=(255+67)
#plt.scatter(x0,y0,c = intrad)
floatang = np.arccos(x0/intrad)
divang = np.round(floatang, 1)
divrad = intrad//2*2

rad_pool = np.unique(divrad)
ang_pool = np.unique(divang)
for i in rad_pool:
    for j in ang_pool:
        ns = np.where((divrad == i) & (divang == j))[0]
        intens =




flotrad = intrad[mask_center]

s_s = np.load('../aux/int2std_model_r30_mask0_weighted.npz')
std_models = s_s['std_model']
int_models = s_s['intens_model']
std_smooth = savgol_filter(std_models, 51, 3)
fis = interp1d(np.log10(int_models), std_smooth,'cubic')


radcount = np.zeros(intrad.max() + 1)
np.add.at(radcount, intrad, mask_center)
radmask = (radcount > 10)
radcount[radcount == 0] = 1

det_dist = 100 # mm
qvals = 2. * np.sin(0.5 * np.arctan(np.arange(int(intrad.max())+1) * 0.075 / det_dist)) / 4.5
diameters = np.linspace(10, 50, 1200)
svals = np.outer(diameters, qvals) * np.pi
svals[svals==0] = 1e-6
spheremodels = (np.sin(svals) - svals * np.cos(svals))**2 / svals**6 * diameters[:,np.newaxis]**6


###############################################
intens_tot = mp.Array(ctypes.c_double, n_frames)
cc = mp.Array(ctypes.c_double, n_frames)
error = mp.Array(ctypes.c_double, n_frames)
dia = mp.Array(ctypes.c_double, n_frames)

def mp_worker(rank, indices, dia, intens_tot, cc, error):
    irange = indices[rank::nproc]

    for i in irange:
        f_i = emc.get_frame(i)
        data_i = f_i.data[mask_center]
        radavg = np.zeros_like(radcount)
        tdata = data_i
        np.add.at(radavg, flotrad, tdata)
        radavg /= radcount

        corrs = np.corrcoef(((radcount*radavg)[radmask]), (radcount*spheremodels)[:,radmask])[0, 1:]
        cc[i] = corrs.max()
        dia[i] = diameters[corrs.argmax()]
        intens_tot[i] = np.sum(tdata)
        if (intens_tot[i] >=10):
            error[i] = fis(np.log10(intens_tot[i]))
        if (intens_tot[i] < 10):
            error[i] = 100

        if rank == 0:
            print("CC (%d):"%i, ' intens = ', intens_tot[i], ' dia = ', dia[i], ' err = ', error[i])



nproc = 16
ind = np.arange(n_frames)
jobs = [mp.Process(target=mp_worker, args=(rank, ind, dia, intens_tot, cc, error)) for rank in range(nproc)]
[i.start() for i in jobs]
[i.join() for i in jobs]
dia = np.frombuffer(dia.get_obj())
intens_tot = np.frombuffer(intens_tot.get_obj())
cc = np.frombuffer(cc.get_obj())
error = np.frombuffer(error.get_obj())
###############################################
with h5py.File('diameter_weighted_error_mask0.h5', 'w') as f:
    f['Run_num'] = running_num
    f['diameter'] = dia
    f['intens_tot'] = intens_tot
    f['error'] = error
    f['cc'] = cc


np.savez('diameter_weighted_error_mask0.npz', dia=dia, intens=intens_tot, error=error, cc=cc)
