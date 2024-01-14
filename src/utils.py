import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms

def load_image(path):
    """
    Load image file to image tensor.

    Parameters:
    - path: Path of image file

    Returns:
    - Loaded image tensor
    """

    fp = open(path, 'rb')
    image = Image.open(fp).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0   #(425, 425, 3)
    fp.close()
    image = image[None].transpose(0, 3, 1, 2)           # (1, 3, 425, 425)
    image = torch.from_numpy(image)
    return image

def save_image(samples, path):     
    """
    Save image tensor to image file.

    Parameters:
    - samples: Input image tensors.
    - path: Path to save the images

    Returns:
    - No return
    """

    samples = 255 * samples.clamp(0,1)    # (1, 3, 425, 425)
    samples = samples.detach().numpy()
    samples = samples.transpose(0, 2, 3, 1)       #(1, 425, 425, 3)
    image = samples[0]                            #(425, 425, 3)
    image = Image.fromarray(image.astype(np.uint8))
    image.save(path)

def encode_img(input_img, vae):
    """
    Encode the image tensors to the VAE latents

    Parameters:
    - input_img: image tensors.
    - vae: VAE for encoding latent.

    Returns:
    - Encoded latents from image tensors by vae.
    """

    # Single image -> single latent in a batch (so size 1, 4, 53, 53)
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img*2 - 1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def decode_img(latents, vae):
    """
    Decode the VAE latents back to image tensor.

    Parameters:
    - latents: VAE latent.
    - vae: VAE for decoding latent.

    Returns:
    - Decoded image from latents by vae.
    """

    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach()
    return image

def transform(image):
    """
    Transform the input image to the size (512, 512).

    Parameters:
    - image: Input image as a tensor.

    Returns:
    - Transformed image with size (512, 512).
    """
    return transforms.Resize([512, 512])(image)

def to_PIL(tensor):
    """
    Transform the input tensor to a PIL image.

    Parameters:
    - tensor: Input image as a tensor. 
    
    Returns:
    - Transformed image.
    """
    return torchvision.transforms.ToPILImage()(tensor)


def DoOneRoiMask(data_path, subject_num, data, LR, roi, maskedFmri):
    """
    Perform ROI Mask one time on maskedFmri

    Parameters:
    - data_path (string): path to folder for subj0x
    - subject_num (int): subject number for training ex: 1
    - data: target's fMRI data
    - LR (string "left" or "right"): specify the fmri data is left or right hemisphere
    - roi: region we want to select with ROI Mask
    - maskedFmri: Target fMRI to perform with

    Returns:
    - Masked fMRI
    """

    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = 'prf-visualrois'
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = 'floc-bodies'
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = 'floc-faces'
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = 'floc-places'
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = 'floc-words'
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = 'streams'

    # load mask file
    roi_space_dir = os.path.join(data_path, 'subj0{}'.format(subject_num), 'roi_masks',
                                 LR[0]+'h.'+roi_class+'_space.npy')
    roi_map_dir = os.path.join(data_path, 'subj0{}'.format(subject_num), 'roi_masks',
                               'mapping_'+roi_class+'.npy')
    roi_space = np.load(roi_space_dir)
    roi_map = np.load(roi_map_dir, allow_pickle=True).item()

    # Select the vertices corresponding to the ROI of interest
    roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]

    target_index = np.where(np.isin(roi_space, roi_mapping))[0]

    for i in target_index:
        maskedFmri[:, i] = data[:, i]

    return maskedFmri

def GetRoiMaskedFmri(data_path, subject_num, LR, regions):
    """
    Get fMRI with selected ROI masks

    Parameters:
    - data_path (string): path to folder for subj0x
    - subject_num (int): subject number for training ex: 1
    - LR (string "left" or "right"): specify the fmri data is left or right hemisphere
    - regoins: list of strings of interseted regions, split by bankspace -- ex: ["FFA-1", "OPA"]

    Returns:
    masked fmri data -- (n, 19004) or (n, 20054)
    """
    if LR == "left":
        data = np.load(os.path.join(data_path, 'subj0{}'.format(subject_num), 'training_split/training_fmri',
        LR[0]+'h_training_fmri.npy'))
        assert (data.shape[1] == 19004)
    elif LR == "right":
        data = np.load(os.path.join(data_path, 'subj0{}'.format(subject_num), 'training_split/training_fmri',
        LR[0]+'h_training_fmri.npy'))
        assert (data.shape[1] == 20544)

    maskedFmriData = np.zeros_like(data, dtype=float)

    for region in regions:
        DoOneRoiMask(data_path=data_path, subject_num=subject_num, data=data, LR=LR, roi=region, maskedFmri=maskedFmriData)

    return maskedFmriData

def GetRoiMaskedLR(ROI_num, data_path):
    """
    Get masked LR fMRI with selected ROI masks

    Parameters:
    - ROI_num: select which ROI mask set to choose (1, 2, 3)

    Returns:
    masked fmri data, concatenate both Left and Right
    """

    # determine the datapath and training subject
    subject_num = 1

    # get masked fmri data
    ROIs_test1 = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "FFA-1", "FFA-2", "PPA"]
    ROIs_test2 = ["EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC"]
    ROIs_test3 = ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    ROIs = ['', ROIs_test1, ROIs_test2, ROIs_test3]

    lh = GetRoiMaskedFmri(data_path=data_path, subject_num=subject_num, LR="left", regions=ROIs[ROI_num])
    rh = GetRoiMaskedFmri(data_path=data_path, subject_num=subject_num, LR="right", regions=ROIs[ROI_num])
    lrh = np.concatenate((lh, rh), axis=1)
    return lrh

def GetROI(image_idx, select_ROI):
    """
    Get specific image's fMRI, with selected ROI Mask

    Parameters:
    - image_idx: select which image's fMRI to get
    - select_ROI: select which ROI mask set to choose (1, 2, 3)

    Returns:
    masked fmri data, concatenate both Left and Right
    """
    select_ROI = 0
    # determine the datapath and training subject
    data_path = '../dataset/'
    subject_num = 1
    
    # get masked fmri data
    ROIs_test1 = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "FFA-1", "FFA-2", "PPA"]
    ROIs_test2 = ["EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC"]
    ROIs_test3 = ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    ROIs_tests = [ROIs_test1, ROIs_test2, ROIs_test3]
    
    lh = GetRoiMaskedFmri(data_path=data_path, subject_num=subject_num, LR="left", regions=ROIs_tests[select_ROI])
    rh = GetRoiMaskedFmri(data_path=data_path, subject_num=subject_num, LR="right", regions=ROIs_tests[select_ROI])
    lrh = np.concatenate((lh, rh), axis=1)
    return lrh[image_idx]