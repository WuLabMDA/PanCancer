import h5py
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.io import loadmat


try:
    _range = xrange
except NameError:
    _range = range

# overlay the image and its ground truth mask
def overlay(image, mask):
    """Overlap Original Image with Mask
    """
    if len(image.shape) == 3:
        image = image[:, :, 0]
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    if np.amax(image) > 100:
        image = image / 255

    masked = np.ma.masked_where(mask == 0, mask)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, 'gray', interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(image, 'gray', interpolation='nearest')
    plt.imshow(masked, 'jet', interpolation='nearest', alpha=0.5)
    plt.show()

# overlay the image and mask probability
def overlay_prob(image, mask, cutoff=0.5):
    """Overlap Original Image with Mask
    """
    if len(image.shape) == 3:
        image = image[: ,: ,0]
    if len(mask.shape) == 3:
        mask = mask[: ,: ,0]
    if np.amax(image) > 100:
        image = image /255

    mask = mask>=cutoff
    mask = mask.astype(int)
    masked = np.ma.masked_where(mask == 0, mask)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, 'gray', interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(image, 'gray', interpolation='nearest')
    plt.imshow(masked, 'jet', interpolation='nearest', alpha=0.5)
    plt.show()

# save all the mat files to h5 file
def saveMat2h5(image_dir,csv_file,save_dir,h5_file):
    df = pd.read_csv(csv_file)
    h5_path_ct = os.path.join(save_dir,h5_file+'_ct')
    h5_path_mask = os.path.join(save_dir, h5_file + '_mask')
    hf1 = h5py.File(h5_path_ct, 'w')
    hf2 = h5py.File(h5_path_mask, 'w')
    for index, row in df.iterrows():
        image_id = row['matFile']
        path = os.path.join(image_dir, row['matFile'])
        ct, mask = load_data(path)
        hf1.create_dataset(image_id,data=ct,compression="gzip", compression_opts=9)
        hf2.create_dataset(image_id,data=mask,compression="gzip", compression_opts=9)
    hf1.close()
    hf2.close()

# load the image and mask data
def load_data(file_path):
    # Load image
    print(file_path)
    assert os.path.isfile(file_path)
    image_3D = loadmat(file_path)['CT']
    mask_3D = loadmat(file_path)['tumor']

    # Convert dimention to 512
    assert image_3D.shape == mask_3D.shape
    assert image_3D.shape[0] == 512
    assert image_3D.shape[1] == 512

    return image_3D.astype(np.int16), mask_3D.astype(np.bool)

# load the image and mask data
def load_data_lung2(file_path,file_path2):
    # Load image
    print(file_path)
    assert os.path.isfile(file_path)
    image_3D = loadmat(file_path)['CT']
    # tumor_3D = loadmat(file_path)['tumor1']
    tumor_3D = loadmat(file_path)['tumor']
    img_resolution = loadmat(file_path)['img_resolution']

    print(file_path2)
    assert os.path.isfile(file_path2)
    lung_3D = loadmat(file_path2)['lung']

    # Convert dimention to 512
    assert image_3D.shape == tumor_3D.shape
    assert image_3D.shape == lung_3D.shape

    return image_3D.astype(np.int16), tumor_3D.astype(np.bool), lung_3D.astype(np.bool), np.squeeze(img_resolution)


# load the image and mask data
def load_data_lung(file_path):
    # Load image
    print(file_path)
    assert os.path.isfile(file_path)
    image_3D = loadmat(file_path)['CT']
    tumor_3D = loadmat(file_path)['tumor']
    lung_3D = loadmat(file_path)['lung']

    # Convert dimention to 512
    assert image_3D.shape == tumor_3D.shape
    assert image_3D.shape == lung_3D.shape

    return image_3D.astype(np.int16), tumor_3D.astype(np.bool), lung_3D.astype(np.bool)

# load the image and mask data
def load_data_breast(file_path):
    # Load image
    print(file_path)
    assert os.path.isfile(file_path)
    image_3D1 = loadmat(file_path)['dce1']
    image_3D2 = loadmat(file_path)['dce2']
    image_3D3 = loadmat(file_path)['dce3']
    tumor_3D = loadmat(file_path)['tumor1']
    bpe_3D = loadmat(file_path)['bpe']
    ser = loadmat(file_path)['ser']
    img_resolution = loadmat(file_path)['img_resolution']

    print(image_3D1.dtype)
    print(image_3D2.dtype)
    print(image_3D3.dtype)
    print(tumor_3D.dtype)
    print(bpe_3D.dtype)
    print(ser.dtype)
    print(img_resolution.dtype)
    print(image_3D1.shape)
    print(image_3D2.shape)
    print(image_3D3.shape)
    print(tumor_3D.shape)
    print(bpe_3D.shape)
    print(ser.shape)
    print(img_resolution.shape)

    # Convert dimention to 512
    assert image_3D1.shape == tumor_3D.shape
    assert image_3D2.shape == tumor_3D.shape
    assert image_3D3.shape == tumor_3D.shape
    assert bpe_3D.shape == tumor_3D.shape
    assert ser.shape == tumor_3D.shape

    return ser, tumor_3D.astype(np.bool), bpe_3D.astype(np.bool), np.squeeze(img_resolution)

# generate the input for training generator
def inputGenerator(ct_file,mask_file):
    with h5py.File(ct_file, 'r') as f:
        image_info = []
        for key in f.keys():
            ct = np.array(f.get(key))
            for slice_num in list(range(ct.shape[2])):
                info = {
                    "image_id": key + str(slice_num),
                    "h5_key": key,
                    "slice_index": slice_num,
                    "ct_path": ct_file,
                    "mask_path": mask_file
                }
                image_info.append(info)
    return image_info

# import hdf5 file
def load_hdf5(ct_file,mask_file,preprocess=True):
    levels = (-1000,500)
    # stack all the CT image
    with h5py.File(ct_file, 'r') as f:
        image = []
        for key in f.keys():
            print(key)
            ct = np.array(f.get(key))
            if preprocess:
                ct = (ct - levels[0]) / (levels[1] - levels[0])
                ct[ct > 1] = 1
                ct[ct < 0] = 0

                # Contrast stretching
                p2, p98 = np.percentile(ct, (2, 98))
                ct_rescale = exposure.rescale_intensity(ct, in_range=(p2, p98))
                # Equalization
                ct_eq = exposure.equalize_hist(ct)

                data = np.stack((ct,ct_rescale,ct_eq), axis=-1)
                image.append(data)

        ct_all = np.concatenate(image,axis=2)
        ct_all = np.transpose(ct_all,[2,0,1,3])
        # multi_slice_viewer(ct)
        input("Press Enter to continue...")

    # stack all the segmentation
    with h5py.File(mask_file, 'r') as f:
        mask = []
        for key in f.keys():
            print(key)
            seg = np.array(f.get(key))
            data = np.stack((seg,) * 1, axis=-1)
            mask.append(data)

        mask_all = np.concatenate(mask,axis=2)
        mask_all = np.transpose(mask_all, [2, 0, 1, 3])

    return ct_all, mask_all


##########################################################################
# plot the fourier decomposition results
def plot_efd_edit(coeffs, locus=(0., 0.), image=None, contour=None, mask=None, n=300, interval=10):
    """Plot a ``[2 x (N / 2)]`` grid of successive truncations of the series.

    .. note::

        Requires `matplotlib <http://matplotlib.org/>`_!

    :param numpy.ndarray coeffs: ``[N x 4]`` Fourier coefficient array.
    :param list, tuple or numpy.ndarray locus:
        The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
    :param int n: Number of points to use for plotting of Fourier series.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot: matplotlib was not installed.")
        return

    N = coeffs.shape[0]
    N_half = int(np.ceil(N / (2*interval)))
    n_rows = 2

    t = np.linspace(0, 1.0, n)
    xt = np.ones((n,)) * locus[0]
    yt = np.ones((n,)) * locus[1]

    for n in _range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )

        if n % interval == 0:
            if mask is not None:
                idx1, idx2 = get_bbox(mask,dilate=10)
            ax = plt.subplot2grid((n_rows, N_half), (n // (N_half*interval), n //interval % N_half))
            ax.set_title(str(n + 1))
            if contour is not None:
                ax.plot(contour[:, 0]-idx2[0], contour[:, 1]-idx1[0], "y--", linewidth=2)
            ax.plot(xt-idx2[0], yt-idx1[0], "r", linewidth=2)
            if image is not None:
                image_crop = image[idx1[0]:idx1[1], idx2[0]:idx2[1]]
                ax.imshow(image_crop, plt.cm.gray)

    plt.show()

# get the bounding box for the given mask
def get_bbox(mask, dilate=10):
    # labeled_array, num_features = scpimg.label(mask)
    index = np.where(mask!=0)
    dim_0 = (np.amin(index[0]), np.amax(index[0]) + 1)
    dim_1 = (np.amin(index[1]), np.amax(index[1]) + 1)
    dim_0 = (np.max([0,dim_0[0]-dilate]),np.min([mask.shape[1],dim_0[1]+dilate]))
    dim_1 = (np.max([0, dim_1[0] - dilate]), np.min([mask.shape[1], dim_1[1] + dilate]))

    return dim_0, dim_1
##########################################################################


##########################################################################
# visualization in 3D
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    ax.imshow(volume[:,:,ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    ax.images[0].set_array(volume[:,:,ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[:,:,ax.index])

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

############################################################################