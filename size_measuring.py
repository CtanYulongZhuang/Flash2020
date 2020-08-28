#measure size of each frames, by fitting radial profiles with model, and estimating error with the intensity and our int2error model
import sys
sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src/')
import detector
import reademc
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

#poisson
det = detector.Detector('../aux/det_intrad4.h5', mask_flag=True)
with open('../recon_0002/photons.txt', 'r') as f:
     emc_flist = [i.strip() for i in f.readlines()]

emc = reademc.EMCReader(emc_flist,det)
n_frames = emc.num_frames
mask_emc0 = emc.get_frame(1).mask.ravel()
mask_emc0 = np.array([ not i for i in mask_emc0]).reshape(512, 256)
mask_emc = np.load('../aux/mask_center_cmc_1.npy')

mask_h = np.load('../aux/mask_002_hitfinding6.npy')
mask_c = np.load('../aux/mask_cmc6.npy')
#mask_center = mask_h & mask_c
mask_center = mask_emc
intrad = np.load('../aux/intrad_004.npy')
intrad = intrad[256:768,256:512]
flotrad = intrad[mask_center]
s_s = np.load('../aux/int2std_model_r30_mask1.npz')
std_models = s_s['std_all']
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


dia = np.zeros(n_frames)
cc = np.zeros(n_frames)
std = np.zeros(n_frames)
intens_tot = np.zeros(n_frames)
error = np.zeros(n_frames)
for i in range(n_frames):
    f_i = emc.get_frame(i)
    data_i = f_i.data[mask_emc]
    radavg = np.zeros_like(radcount)
    tdata = data_i
    np.add.at(radavg, flotrad, tdata)
    radavg /= radcount

    corrs = np.corrcoef(radavg[radmask], spheremodels[:,radmask])[0, 1:]
    cc[i] = corrs.max()
    dia[i] = diameters[corrs.argmax()]
    intens_tot[i] = np.sum(tdata)
    if (intens_tot[i] >=10):
        error[i] = fis(np.log10(intens_tot[i]))
    if (intens_tot[i] < 10):
        error[i] = 100



    print('i = ',i, ' intens = ', intens_tot[i], ' dia = ', dia[i], ' err = ', error[i])



np.savez('diameter_error_mask1.npz', dia=dia, intens=intens_tot, std=std, error=error, cc=cc)
