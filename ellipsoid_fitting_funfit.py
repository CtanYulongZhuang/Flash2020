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
#/asap3/flash/gpfs/bl1/2020/data/11007613/shared/ana


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
    svals = diameter * qval * np.pi
    sval = svals
    intensity = scl*(np.sin(sval) - sval * np.cos(sval))**2 / sval**6 * diameter**6
    return intensity


f = h5py.File('Geomotry_corr_yulong.h5','r')
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
det = detector.Detector('../aux/det_intrad4.h5', mask_flag=True)
with open('../recon_0002/photons.txt', 'r') as f:
     emc_flist = [i.strip() for i in f.readlines()]

emc = reademc.EMCReader(emc_flist,det)
n_frames = emc.num_frames

#clist = np.array([f['num_data'] for f in emc.flist])
#running_num = np.zeros(np.max(clist))
#running_num[0:clist[0]] = np.int( emc_flist[0][75:78] )
#for i in range(len(clist)-1):
#    running_num[clist[i]:clist[i+1]] = np.int( emc_flist[i+1][75:78] )


mask_emc0 = emc.get_frame(1).mask.ravel()
mask_emc0 = np.array([ not i for i in mask_emc0]).reshape(512, 256)
mask_center = np.load('../aux/mask_center_cmc_0.npy')                           #mask file


#Absolute coordinate
flotrad = intrad0[mask_center]
flotang = floatang0[mask_center]
x1 = xcorr0[mask_center]
y1 = ycorr0[mask_center]

X = [flotrad, flotang]
pie = np.pi
###############################################
angle = mp.Array(ctypes.c_double, n_frames)
dia_a = mp.Array(ctypes.c_double, n_frames)
dia_b = mp.Array(ctypes.c_double, n_frames)
scale = mp.Array(ctypes.c_double, n_frames)
sum_i = mp.Array(ctypes.c_double, n_frames)
pocv_diag = mp.Array(ctypes.c_double, n_frames*4)
def mp_worker(rank, indices, dia_a, dia_b, angle, scale, sum_i, pocv_diag):
    irange = indices[rank::nproc]

    for i in irange:
        f_i = emc.get_frame(i)
        data_i = f_i.data[mask_center]
        p0 = 15., 15., 0, 0.1
        p_bounds = ((5.0,5.0,-4,0.0), (60.0,60.0,4,1.0))
        Pa = [0,0,0,0]
        Pb = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        try:
            Pa, Pb = curve_fit(intens, X, data_i, p0, bounds=p_bounds)
        except:
            pass
        #Pa, Pb = curve_fit(intens, X, data_i, p0, bounds=p_bounds, maxfev=10000)
        angle[i] = Pa[2]
        dia_a[i] = max(Pa[0],Pa[1])
        dia_b[i] = min(Pa[0],Pa[1])
        scale[i] = Pa[3]
        sum_i[i] = np.sum(data_i)
        pocv_diag[i*4] = Pb[0][0]
        pocv_diag[i*4+1] = Pb[1][1]
        pocv_diag[i*4+2] = Pb[2][2]
        pocv_diag[i*4+3] = Pb[3][3]

        if rank == 0:
            print("CC (%d):"%i, ' sum_i = ', np.sum(data_i),  ' dia_a = ', dia_a[i], ' dia_b = ', dia_b[i], ' angle = ', angle[i])


nproc = 16
ind = np.arange(n_frames)
jobs = [mp.Process(target=mp_worker, args=(rank, ind, dia_a, dia_b, angle, scale, sum_i, pocv_diag)) for rank in range(nproc)]
[i.start() for i in jobs]
[i.join() for i in jobs]
dia_a = np.frombuffer(dia_a.get_obj())
dia_b = np.frombuffer(dia_b.get_obj())
angle = np.frombuffer(angle.get_obj())
scale = np.frombuffer(scale.get_obj())
sum_i = np.frombuffer(sum_i.get_obj())
pocv_diag = np.frombuffer(pocv_diag.get_obj())

pocvx_diag = pocv_diag.reshape(n_frames,4)
###############################################
with h5py.File('diameter_ab_cv_centre_corr.h5', 'w') as f:
    f['dia_a'] = dia_a
    f['dia_b'] = dia_b
    f['angle'] = angle
    f['scale'] = scale
    f['sum_i'] = sum_i
    f['pocvx_diag'] = pocvx_diag
