from glob import glob
import SimpleITK as stik
import numpy as np
import json
import os
from tqdm import tqdm

'''
load_scans: loads scans for an individual patient into a numpy array
@param: full pathname to the folder with the patient's scans
@returns: numpy array of the patients scans
'''
def load_scans(path):
    flair = glob(path + '/*Flair*/*.mha')
    t1 = glob(path + '/*T1.*/*_n4.mha')
    t1c = glob(path + '/*T1c.*/*_n4.mha')
    t2 = glob(path + '/*T2.*/*.mha')
    gt = glob(path + '/*OT*/*.mha')
    paths = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    scans = [stik.GetArrayFromImage(stik.ReadImage(paths[mod]))
            for mod in range(len(paths))]
    scans = np.array(scans)
    return scans

''' norm_scans: normalizes all of a patients scans by calling norm_slice()
@parm: scans all of the scans (including all modalities) for a single patient
@returns: normalized np array
'''
def norm_scans(scans):
    normed_scans = np.zeros((155, 240, 240, 5)).astype(np.float32)
    normed_scans[:,:,:,4] = scans[4,:,:,:]
    for mod_idx in range(4):
        for slice_idx in range(155):
            normed_slice = norm_slice(scans[mod_idx,slice_idx,:,:])
            normed_scans[slice_idx,:,:,mod_idx] = normed_slice
    return normed_scans


''' norm_slice: normalizes a 2d slice of a patient's scans
@param: slice an np array representing 2d scan of just a single modality
@returns: scan with top and bottom 1% pixel intensity removed with each
modality normalized to zero mean and unit variance '''
def norm_slice(slice):
    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    img_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(img_nonzero) == 0:
        return slice
    else:
        normed = (slice - np.mean(img_nonzero)) / np.std(img_nonzero)
        return normed






