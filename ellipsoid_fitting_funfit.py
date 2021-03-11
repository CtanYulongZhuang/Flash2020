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
from scipy.optimize import curve_fit
#/asap3/flash/gpfs/bl1/2020/data/11007613/shared/aux/


import h5py
with h5py.File('../aux/det_intrad4_corr.h5', "r") as data_tem:
    keys = data_tem.keys()

data_tem = h5py.File('../aux/YZ_det_intrad4_corr.h5', "r")
qx = data_tem['qx'][:]
qy = data_tem['qy'][:]
qz = data_tem['qz'][:]
mask = data_tem['mask'][:]
data_tem.close()


def intens(X, a, b, tp, scl):
    rx,tx = X
    t_all = tx + tp
    radii = a*b/np.sqrt(a**2*np.sin(t_all)**2 + b**2*np.cos(t_all)**2)
    det_dist = 100
    qval = 2. * np.sin(0.5 * np.arctan(rx * 0.075 / det_dist)) / 4.5
    diameter = 2*radii
    #svals = np.outer(diameter, qval) * np.pi
    svals = diameter * qval * np.pi
    #svals[svals==0] = 1e-6
    sval = svals
    intensity = scl*(np.sin(sval) - sval * np.cos(sval))**2 / sval**6 * diameter**6
    return intensity


#[247,255] Closest point to the center rmin = 67
intrad0 = np.load('../aux/intrad_004.npy')
intrad = intrad0[256:768,256:512]
x0,y0 = np.indices((512,256)); x0-=247; y0-=(255+67)
plt.scatter(x0,y0,c = intrad)
floatang = np.arccos(x0/intrad)
flat_floatang = floatang.ravel()
flat_intrad = intrad.ravel()

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

flotrad = intrad[mask_center]
flotang = floatang[mask_center]
x1 = x0[mask_center]
y1 = y0[mask_center]

s_s = np.load('../aux/int2std_model_r30_mask0_weighted.npz')
std_models = s_s['std_model']
int_models = s_s['intens_model']
std_smooth = savgol_filter(std_models, 51, 3)
fis = interp1d(np.log10(int_models), std_smooth,'cubic')



X = [flotrad, flotang]
###############################################
intens_tot = mp.Array(ctypes.c_double, n_frames)
cc = mp.Array(ctypes.c_double, n_frames)
angle = mp.Array(ctypes.c_double, n_frames)
dia_a = mp.Array(ctypes.c_double, n_frames)
dia_b = mp.Array(ctypes.c_double, n_frames)
scale = mp.Array(ctypes.c_double, n_frames)
sum_i = mp.Array(ctypes.c_double, n_frames)

def mp_worker(rank, indices, dia_a, dia_b, angle, scale, sum_i):
    irange = indices[rank::nproc]

    for i in irange:
        f_i = emc.get_frame(i)
        data_i = f_i.data[mask_center]
        p0 = 17., 15., 0, 0.1
        #Pa, Pb = curve_fit(intens, X, data_i, p0, maxfev=100000)
        Pa = [0,0,0,0]
        try:
            Pa, Pb = curve_fit(intens, X, data_i, p0, maxfev=10000)
        except:
            pass
        #perr = np.sqrt(np.diag(Pb)).


        angle[i] = Pa[2]
        dia_a[i] = max(Pa[0],Pa[1])
        dia_b[i] = min(Pa[0],Pa[1])
        scale[i] = Pa[3]
        sum_i[i] = np.sum(data_i)

        if rank == 0:
            print("CC (%d):"%i, ' sum_i = ', np.sum(data_i),  ' dia_a = ', dia_a[i], ' dia_b = ', dia_b[i], ' angle = ', angle[i])


nproc = 16
ind = np.arange(n_frames)
jobs = [mp.Process(target=mp_worker, args=(rank, ind, dia_a, dia_b, angle, scale, sum_i)) for rank in range(nproc)]
[i.start() for i in jobs]
[i.join() for i in jobs]
dia_a = np.frombuffer(dia_a.get_obj())
dia_b = np.frombuffer(dia_b.get_obj())
angle = np.frombuffer(angle.get_obj())
scale = np.frombuffer(scale.get_obj())
sum_i = np.frombuffer(sum_i.get_obj())
###############################################
with h5py.File('diameter_ab.h5', 'w') as f:
    f['dia_a'] = dia_a
    f['dia_b'] = dia_b
    f['angle'] = angle
    f['scale'] = scale
    f['sum_i'] = sum_i







X = [flat_intrad, flat_floatang]
intensity_map = intens(X, dia_a[0], dia_b[0], angle[0])
plt.imshow(np.log10(intensity_map.reshape(512,256)))









        f_i = emc.get_frame(i)
        data_i = f_i.data[mask_center]
        p0 = 17., 15., 0, 0.1
        Pa, Pb = curve_fit(intens, X, data_i, p0)
        plt.subplot(121)
        plt.scatter(x1,y1,c=data_i,s=1)
        mc = intens(X, Pa[0], Pa[1], Pa[2], Pa[3])
        plt.subplot(122)
        plt.scatter(x1,y1,c=mc,s=1)
