import sys
sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src/')
import detector
import reademc
import numpy as np
import ctypes
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import multiprocessing as mp

Path = '/asap3/flash/gpfs/bl1/2020/data/11007613/shared/'

#poisson
det = detector.Detector(Path + 'aux/det_intrad4.h5', mask_flag=True)
with open(Path + 'recon_0002/photons.txt', 'r') as f:
     emc_flist = [i.strip() for i in f.readlines()]

emc = reademc.EMCReader(emc_flist,det)
n_frames = emc.num_frames

clist = np.array([f['num_data'] for f in emc.flist])
running_num = np.zeros(np.max(clist))
running_num[0:clist[0]] = np.int( emc_flist[0][75:78] )
for i in range(len(clist)-1):
    running_num[clist[i]:clist[i+1]] = np.int( emc_flist[i+1][75:78] )


f = h5py.File(Path + 'aux/Geomotry_corr_yulong.h5','r')
intrad0 = f['intrad0'][:]
intrad1 = f['intrad1'][:]
xcorr0 = f['x0'][:]
ycorr0 = f['y0'][:]
xcorr1 = f['x1'][:]
ycorr1 = f['y1'][:]
mask_center = f['mask_center'][:]
f.close()

intrad = intrad0
flotrad = intrad[mask_center]



s_s = np.load(Path + 'aux/int2std_model_r30_mask0_weighted.npz')
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
with h5py.File(Path + 'aux/diameter_corr_mask0.h5', 'w') as f:
    f['Run_num'] = running_num
    f['diameter'] = dia
    f['intens_tot'] = intens_tot
    f['error'] = error
    f['cc'] = cc
