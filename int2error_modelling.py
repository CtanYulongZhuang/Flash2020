import numpy as np
from numpy import random
import multiprocessing as mp
import ctypes
#tt = [1,2,3,4,5,6,7]
#x = random.poisson(lam=tt, size=1000)

# Quick config parameters
run_num = 107
dark_run_num = 44
hitScoreThreshold = 480
aduThreshold = 160 # Gain 1
sizecc_threshold = 0.5
strong_hit_threshold = 1600 # For sizing
det_dist = 100 # mm
gap_top_pix = 144
gap_bottom_pix = 86

# Pre-calculations


det_dist = 100 # mm
intrad = np.load('../aux/intrad_004.npy')
intrad = intrad[256:768,256:512]

mask_center = np.load('../aux/mask_center_cmc_1.npy')
radcount = np.zeros(intrad.max() + 1)
np.add.at(radcount, intrad, mask_center)
radmask = (radcount > 10)
radcount[radcount == 0] = 1

#create 2D model frame for dia = 30
qvals = 2. * np.sin(0.5 * np.arctan(intrad * 0.075 / det_dist)) / 4.5
diameters_30 = 30
svals = np.outer(diameters_30, qvals) * np.pi
svals[svals==0] = 1e-6
spheremodel_30 = (np.sin(svals) - svals * np.cos(svals))**2 / svals**6 * diameters_30**6
model_30_mask = spheremodel_30.reshape(512,256)[mask_center]
model_30_mask0 = model_30_mask/np.sum(model_30_mask)
flotrad = intrad[mask_center]

#intens_model = np.arange(0, 20000, 100)
intens_model = 10**(np.arange(200)*0.03+1)
n_models = len(intens_model)

#fitting_model
det_dist = 100 # mm
qvals = 2. * np.sin(0.5 * np.arctan(np.arange(int(intrad.max())+1) * 0.075 / det_dist)) / 4.5
diameters = np.linspace(10, 50, 2000)
svals = np.outer(diameters, qvals) * np.pi
svals[svals==0] = 1e-6
spheremodels = (np.sin(svals) - svals * np.cos(svals))**2 / svals**6 * diameters[:,np.newaxis]**6
#dvals = np.arange(int(intrad.max())+1)

dia_all = np.zeros((n_models,1000))
cc_all = np.zeros((n_models,1000))
std_model = np.zeros(n_models)
for i in range(n_models):
    model_30_maski = model_30_mask0*intens_model[i]
    fake_datai = np.array([random.poisson(lam=i, size=1000) for i in model_30_maski]).T
    #fake_datai = random.poisson(lam=model_30_maski, size=(1000,len(model_30_maski)))

###############################################
    diamax = mp.Array(ctypes.c_double, 1000)
    ccmax = mp.Array(ctypes.c_double, 1000)

    def mp_worker(rank, indices, diamax, ccmax):
        jrange = indices[rank::nproc]

        for j in jrange:
            radavg = np.zeros_like(radcount)
            tdata = fake_datai[j]
            np.add.at(radavg, flotrad, tdata)
            radavg /= radcount

            corrs = np.corrcoef((radavg*radcount)[radmask], (radcount*spheremodels)[:,radmask])[0, 1:] #====weighted
            #corrs = np.corrcoef(radavg[radmask], spheremodels[:,radmask])[0, 1:] #====unweighted
            ccmax[j] = corrs.max()
            diamax[j] = diameters[corrs.argmax()]

            #if rank == 0:
                #print("CC (%d):"%j, ' CC = ', ccmax[j], ' dia = ', diamax[j])

    nproc = 16
    indices = np.arange(1000)
    jobs = [mp.Process(target=mp_worker, args=(rank, indices, diamax, ccmax)) for rank in range(nproc)]
    [j.start() for j in jobs]
    [j.join() for j in jobs]
    diamax = np.frombuffer(diamax.get_obj())
    ccmax = np.frombuffer(ccmax.get_obj())
###############################################
    std_model[i] = np.sqrt(np.sum((diamax-30)**2)/999)
    dia_all[i] = diamax
    cc_all[i] = ccmax
    print('i = ',i, ' std = ', std_model[i], ' dia = ', np.mean(dia_all[i]))


np.savez('int2std_model_r30_mask1_weighted.npz', cc_all=cc_all, dia_all=dia_all, std_model=std_model, intens_model=intens_model)

