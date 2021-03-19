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
Path = '/asap3/flash/gpfs/bl1/2020/data/11007613/shared/'


import h5py
with h5py.File(Path + 'aux/det_intrad4_corr.h5', "r") as data_tem:
    keys = data_tem.keys()

data_tem = h5py.File(Path + 'aux/YZ_det_intrad4_corr.h5', "r")
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
    svals = diameter * qval * np.pi
    sval = svals
    intensity = scl*(np.sin(sval) - sval * np.cos(sval))**2 / sval**6 * diameter**6
    return intensity


f = h5py.File(Path + 'aux/Geomotry_corr_yulong.h5','r')
intrad0 = f['intrad0'][:]
intrad1 = f['intrad1'][:]
xcorr0 = f['x0'][:]
ycorr0 = f['y0'][:]
xcorr1 = f['x1'][:]
ycorr1 = f['y1'][:]
mask_center = f['mask_center'][:]
f.close()


#[247,255] Closest point to the center rmin = 67
floatang0 = np.arccos(xcorr0/intrad0)


#poisson
det = detector.Detector(Path + 'aux/det_intrad4.h5', mask_flag=True)
with open(Path + 'recon_0002/photons.txt', 'r') as f:
     emc_flist = [i.strip() for i in f.readlines()]

emc = reademc.EMCReader(emc_flist,det)
n_frames = emc.num_frames

#Absolute coordinate
flotrad = intrad0[mask_center]
flotang = floatang0[mask_center]
x1 = xcorr0[mask_center]
y1 = ycorr0[mask_center]

X = [flotrad, flotang]
pie = np.pi



###############################################
dia_a = mp.Array(ctypes.c_double, n_frames)
dia_b = mp.Array(ctypes.c_double, n_frames)
angle = mp.Array(ctypes.c_double, n_frames)
scale = mp.Array(ctypes.c_double, n_frames)
angle_err = mp.Array(ctypes.c_double, n_frames)
dia_a_err = mp.Array(ctypes.c_double, n_frames)
dia_b_err = mp.Array(ctypes.c_double, n_frames)
sum_i = mp.Array(ctypes.c_double, n_frames)

def mp_worker(rank, indices, dia_a, dia_b, angle, scale, sum_i, dia_a_err, dia_b_err, angle_err):
    irange = indices[rank::nproc]

    for i in irange:
        f_i = emc.get_frame(i)
        data_i = f_i.data[mask_center]
        p0 = 15., 15., 0, 0.1
        #p_bounds = ((5.0,5.0,-4,0.0), (60.0,60.0,4,1.0))
        try:
            Pa, Pb = curve_fit(intens, X, data_i, p0)#, bounds=p_bounds)
        except:
            pass
        #Pa, Pb = curve_fit(intens, X, data_i, p0, bounds=p_bounds, maxfev=10000)
        if (Pa[0] >= Pa[1]):
            dia_a[i] = Pa[0]
            dia_b[i] = Pa[1]
            dia_a_err = Pb[0][0]
            dia_b_err = Pb[1][1]
        if (Pa[0] < Pa[1]):
            dia_a[i] = Pa[1]
            dia_b[i] = Pa[0]
            dia_a_err = Pb[1][1]
            dia_b_err = Pb[0][0]

        angle[i] = Pa[2]
        angle[i] = Pb[2][2]
        scale[i] = Pa[3]
        sum_i[i] = np.sum(data_i)

        if rank == 0:
            print("CC (%d):"%i, ' sum_i = ', np.sum(data_i),  ' dia_a = ', dia_a[i], ' dia_b = ', dia_b[i], ' angle = ', angle[i])




nproc = 16
ind = np.arange(n_frames)
jobs = [mp.Process(target=mp_worker, args=(rank, ind, dia_a, dia_b, angle, scale, sum_i, dia_a_err, dia_b_err, angle_err)) for rank in range(nproc)]
[i.start() for i in jobs]
[i.join() for i in jobs]
dia_a = np.frombuffer(dia_a.get_obj())
dia_b = np.frombuffer(dia_b.get_obj())
angle = np.frombuffer(angle.get_obj())
scale = np.frombuffer(scale.get_obj())
sum_i = np.frombuffer(sum_i.get_obj())
angle_err = np.frombuffer(angle_err.get_obj())
dia_a_err = np.frombuffer(dia_a_err.get_obj())
dia_b_err = np.frombuffer(dia_b_err.get_obj())


with h5py.File(Path+'aux/diameter_ab_cv_centre_corr.h5', 'w') as f:
    f['dia_a'] = dia_a
    f['dia_b'] = dia_b
    f['angle'] = angle
    f['scale'] = scale
    f['sum_i'] = sum_i
    f['angle_err'] = angle_err
    f['dia_a_err'] = dia_a_err
    f['dia_b_err'] = dia_b_err
