import cv2
import numpy as np
import h5py
import random
import os
import pandas as pd

from pyefd import elliptic_fourier_descriptors, plot_efd
from utils import multi_slice_viewer, load_data_lung2, plot_efd_edit
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes

from colorama import init
from colorama import Fore, Back, Style

init()

def resize_imag(ct, mask, img_resolution, plot_label=False):
    dim = ct.shape
    newdimX = int(round(img_resolution[0]*dim[0]))
    newdimY = int(round(img_resolution[1]*dim[1]))

    ct_new = np.empty((newdimX,newdimY,dim[2]))
    mask_new = np.empty((newdimX, newdimY, dim[2]))

    for i in list(range(dim[2])):
        ct_new[:,:,i] = cv2.resize(ct[:,:,i], dsize=(newdimX, newdimY), interpolation=cv2.INTER_CUBIC)
        mask_new[:, :, i] = cv2.resize(mask[:, :, i], dsize=(newdimX, newdimY), interpolation=cv2.INTER_NEAREST)

    if plot_label:
        multi_slice_viewer(ct)
        multi_slice_viewer(mask)
        multi_slice_viewer(ct_new)
        multi_slice_viewer(mask_new)

    return ct_new, mask_new

def margin_3D_histogram(ct_sele, mask_sele, lung_sele, img_resolution, itr_dist = 3, plot_label = False):
    ######################################
    # # resize the ct and mask to 1 mm
    # ct_sele, mask_sele = resize_imag(ct, mask, img_resolution, plot_label=True)

    # dilate and erode the mask
    assert ct_sele.shape == mask_sele.shape
    assert ct_sele.shape == lung_sele.shape

    mask_sele_inner = np.empty(ct_sele.shape)
    mask_sele_outer = np.empty(ct_sele.shape)
    mask_sele_core = np.empty(ct_sele.shape)

    iterate_num = np.round(itr_dist/img_resolution[0])

    # loop through the nonzero slices
    sum_vec = mask_sele.sum(0).sum(0)
    sele_idx = np.nonzero(sum_vec)
    for k in sele_idx[0]:
        kernel = np.ones((3, 3), np.uint8)
        mask_sele_2D = mask_sele[:,:,k] * 1
        mask_sele_2D = mask_sele_2D.astype(np.uint8)
        mask_dist = distance_transform_edt(mask_sele_2D)
        max_dist = np.amax(mask_dist)
        mask_erode = mask_dist > (max_dist/2)
        mask_erode = mask_erode * 1
        mask_sele_core[:, :, k] = mask_erode
        mask_sele_inner[:, :, k] = mask_sele_2D - mask_erode
        mask_dist2 = distance_transform_edt(1-mask_sele_2D)
        mask_dilate = mask_dist2 <= iterate_num
        mask_dilate = mask_dilate * 1
        mask_sele_outer[:, :, k] = mask_dilate

    # refine the analysis to the area within lung parenchyma
    lung_sele = np.logical_or(mask_sele, lung_sele) * 1
    mask_sele_outer = np.logical_and(lung_sele, mask_sele_outer) * 1
    mask_sele_core = mask_sele_core * 1
    mask_sele_inner = mask_sele_inner * 1

    if plot_label:
        multi_slice_viewer(mask_sele_inner)
        multi_slice_viewer(mask_sele_outer)

    #######################################
    # histogram comparison between inner and outer rings
    pixel_outer = ct_sele[mask_sele_outer == 1]
    pixel_inner = ct_sele[mask_sele_inner == 1]
    pixel_core = ct_sele[mask_sele_core == 1]

    frq1, edges1 = np.histogram(pixel_outer, bins=100, range=(-1000, 500))
    frq2, edges2 = np.histogram(pixel_inner, bins=100, range=(-1000, 500))
    frq3, edges3 = np.histogram(pixel_core, bins=100, range=(-1000, 500))
    # print(frq1)
    # print(frq2)

    # normalize to pdf
    frq1 = frq1 / (np.sum(frq1) + 1e-10)
    frq2 = frq2 / (np.sum(frq2) + 1e-10)
    frq3 = frq3 / (np.sum(frq3) + 1e-10)

    if plot_label:
        fig1, ax1 = plt.subplots()
        ax1.bar(edges1[:-1], frq1, width=np.diff(edges1), ec="k", align="edge")
        plt.show()
        fig2, ax2 = plt.subplots()
        ax2.bar(edges2[:-1], frq2, width=np.diff(edges2), ec="k", align="edge")
        plt.show()

    f_hist_compare1 = [
        cv2.compareHist(frq2.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_CORREL),
        cv2.compareHist(frq2.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_CHISQR),
        cv2.compareHist(frq2.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(frq2.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_BHATTACHARYYA)]

    f_hist_compare2 = [
        cv2.compareHist(frq3.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_CORREL),
        cv2.compareHist(frq3.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_CHISQR),
        cv2.compareHist(frq3.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(frq3.ravel().astype('float32'), frq1.ravel().astype('float32'), cv2.HISTCMP_BHATTACHARYYA)]

    f_hist_compare3 = [
        cv2.compareHist(frq3.ravel().astype('float32'), frq2.ravel().astype('float32'), cv2.HISTCMP_CORREL),
        cv2.compareHist(frq3.ravel().astype('float32'), frq2.ravel().astype('float32'), cv2.HISTCMP_CHISQR),
        cv2.compareHist(frq3.ravel().astype('float32'), frq2.ravel().astype('float32'), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(frq3.ravel().astype('float32'), frq2.ravel().astype('float32'), cv2.HISTCMP_BHATTACHARYYA)]

    return f_hist_compare1 + f_hist_compare2 + f_hist_compare3

if __name__ == '__main__':
    #Folders for the input
    cwd = os.getcwd()
    print(cwd)
    dataDir = os.path.join(cwd, 'Data');
    featureDir = os.path.join(cwd, 'Results');
    image_dir = [os.path.join(dataDir,'TCGA','data')]
    lung_dir = [os.path.join(dataDir,'TCGA','lung')]
    csv_file = [os.path.join(dataDir,'TCGA','TCGA_LUSC_data.csv')]

    h5_file = ['TCGA_LUSC']

    f_margin_label = 1;

    f_all_pt = []
    pt_list = []
    for i in list(range(len(h5_file))):
        df = pd.read_csv(csv_file[i])
        for index, row in df.iterrows():
            print(index)
            print(row)
            # load the data from MAT
            if row['run'] == 0:
                continue

            image_id = row['matFile']
            pt_name = row['matFile'].split('.')[0]
            path = os.path.join(image_dir[i], row['matFile'])
            path2 = os.path.join(lung_dir[i], pt_name + '-lung.mat')
            ct, mask, lung, img_resolution = load_data_lung2(path, path2)
            print(Fore.BLUE + image_id)
            print(Style.RESET_ALL)

            # load image resolution
            assert len(img_resolution) == 3
            print('image resolution: ', img_resolution)
            assert np.prod(img_resolution) < 10
            assert np.prod(img_resolution) > 0
            print('image resolution product: ', np.prod(img_resolution))

            #############################################################
            ###### feature type 2: MARGIN BASED FEATURES
            if f_margin_label:
                f_3D_histogram1 = margin_3D_histogram(ct, mask, lung, img_resolution, itr_dist=5, plot_label=False)
                print('Histogram: ', f_3D_histogram1)

                f_3D_histogram2 = margin_3D_histogram(ct, mask, lung, img_resolution, itr_dist=10, plot_label=False)
                print('Histogram: ', f_3D_histogram2)

                f_3D_all = np.concatenate([f_3D_histogram1, f_3D_histogram2])
                print('Aggregated 3D features: ', f_3D_all)

            # save the patient name
            pt_list.append(h5_file[i] + '_' + image_id)
            f_all_pt.append(f_3D_all)

        feature_file = os.path.join(featureDir, h5_file[i] + '_Regional_Variation_Feature.csv')
        with open(feature_file, 'w') as fout:
            out_list = [pt_list[i] + ',' + str(list(f_all_pt[i]))[1:-1] for i in range(len(pt_list))]
            fout.write('\n'.join(out_list))