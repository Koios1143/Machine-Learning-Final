import os
import numpy as np

def DoOneRoiMask(data_path, subject_num, data, LR, roi, maskedFmri):

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
    data_path: string -- path to folder for subj0x
    subject_num: int -- subject number for training ex: 1
    LR: string "left" or "right" -- specify the fmri data is left or right hemisphere
    regoins: list of strings of interseted regions, split by bankspace -- ex: ["FFA-1", "OPA"]

    return: masked fmri data -- (n, 19004) or (n, 20054)
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

def GetROI(image_idx, select_ROI):
    select_ROI = 0
    # determine the datapath and training subject
    data_path = './dataset/'
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