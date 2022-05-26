import numpy as np
import scipy.optimize
#from numba import njit, prange
from pathlib import Path
from typing import Tuple
import cv2
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

models_base = Path(__file__).absolute().parent / "sem_landmarks_models"

device = "cpu" if not torch.cuda.is_available() else "cuda"
# device = "cpu"

SEG_MASK_CUTOFF = 0.5
TAKE_TOP_N_MASKS = 10
TAKE_TOP_SCORE = 0.45


def _conv_mask(mask):
    return ((mask.cpu().detach().numpy()[0] > SEG_MASK_CUTOFF)).astype(np.uint8)


def _conv_img(img):
    return (img.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

#def _conv_img_np(img):
#    return (img).astype(np.uint8)


# @njit(parallel=True, cache=True)
# def _score(mask1, mask2, displ):
#     score = 0
#     if displ < 0:
#         for i in prange(abs(displ), mask1.shape[1]):
#             for j in range(mask1.shape[0]):
#                 score += mask1[j, i] * mask2[j, i - abs(displ)]
#     if displ > 0:
#         for i in prange(abs(displ), mask2.shape[1]):
#             for j in range(mask2.shape[0]):
#                 score += mask1[j, i - abs(displ)] * mask2[j, i]
#     return score




def _matches_to_alignment(kp1, kp2, matches):
    if len(matches) == 0:
        return 0

    distances = []
    for m in matches:
        kp_img2 = kp2[m.trainIdx]
        kp_img1 = kp1[m.queryIdx]
        dist = kp_img2.pt[0] - kp_img1.pt[0]
        distances.append(dist)

    distances = np.array(distances)

    # get higher granularity in the histogram, by filtering out outliers
    # distances = distances[np.abs(distances - alignment) < HORIZONTAL_MAX]
    if len(distances) > 4:
        q_025, q_975 = np.quantile(distances, [.025, .975])
        distances = distances[(distances > q_025) & (distances < q_975)]
    hist, edges = np.histogram(distances, bins=100)
    argmax = np.argmax(hist)
    distances_filtered = distances[(distances >= edges[argmax]) & (distances <= edges[argmax + 1])]
    alignment = np.mean(distances_filtered)

    return alignment


def _good_matches(matches, good_matches):
    if not good_matches:
        return [m[0] for m in matches if len(m) > 0]
    return [m[0] for m in matches
            if len(m) == 1 or (len(m) == 2 and m[0].distance < .75 * m[1].distance)]


def match_sift(im1, im2, vertical_limit: int=None, match_mask: Tuple[np.ndarray, np.ndarray]=None):
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.0)

    kp1, des1 = sift.detectAndCompute(_conv_img(_process_img(im1)), match_mask[0].astype(np.uint8) if match_mask is not None else None)
    kp2, des2 = sift.detectAndCompute(_conv_img(_process_img(im2)), match_mask[1].astype(np.uint8) if match_mask is not None else None)

    # kp1, des1 = sift.detectAndCompute(_conv_img(im1), None)
    # kp2, des2 = sift.detectAndCompute(_conv_img(im2), None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, 2)
    matches = _good_matches(matches, True)
    # apply vertical limit post matching
    if vertical_limit is not None:
        matches = list(filter(lambda m: kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1] < vertical_limit, matches))
    # elif match_mask is not None:
    #     matches_filtered = []
    #     for m in matches:
    #         pos1 = np.array(kp1[m.queryIdx].pt).astype(int)
    #         pos2 = np.array(kp2[m.trainIdx].pt).astype(int)
    #         kp1_lies_in_mask = match_mask[0][pos1[1], pos1[0]]
    #         kp2_lies_in_mask = match_mask[1][pos2[1], pos2[0]]
    #         if kp1_lies_in_mask and kp2_lies_in_mask:
    #             matches_filtered.append(m)
    #     matches = matches_filtered

    return _matches_to_alignment(kp1, kp2, matches)



def _score(mask1, mask2, displ):
    iss = np.arange(abs(displ), mask1.shape[1])
    if displ <= 0:
        return np.sum(mask1[:, iss] * mask2[:, iss - abs(displ)])
    if displ > 0:
        return np.sum(mask1[:, iss - abs(displ)] * mask2[:, iss])


def _masks_added(sgm, im1, im2):
    mask1 = np.zeros(im1.shape[1:], dtype=bool)
    taken = 0
    for i, mask in enumerate(sgm[0]["masks"]):
        mask = _conv_mask(mask) > 0
        if np.mean(mask) > 1 / 3:
            continue
        taken += 1
        mask1 |= mask
        if sgm[0]["scores"][i] < TAKE_TOP_SCORE and taken > TAKE_TOP_N_MASKS:
            break
    if taken == 0:
        mask = _conv_mask(sgm[0]["masks"][0])
        mask1 |= mask

    taken = 0
    mask2 = np.zeros(im2.shape[1:], dtype=bool)
    for i, mask in enumerate(sgm[1]["masks"]):
        mask = _conv_mask(mask) > 0
        if np.mean(mask) > 1 / 3:
            continue
        taken += 1
        mask2 |= mask
        if sgm[1]["scores"][i] < TAKE_TOP_SCORE and taken > TAKE_TOP_N_MASKS:
            break
    if taken == 0:
        mask = _conv_mask(sgm[1]["masks"][0])
        mask1 |= mask

    return mask1, mask2


def match_segmentation_by_shift_color(sgm, im1, im2):
    mask1, mask2 = _masks_added(sgm, im1, im2)

    im1_gray = cv2.cvtColor(_conv_img(im1), cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(_conv_img(im2), cv2.COLOR_BGR2GRAY)

    shifts = np.arange(-mask1.shape[1], mask1.shape[1])
    # mask1_padded = np.zeros((mask1.shape[0], mask1.shape[1] * 3))
    # mask1_padded[:, mask1.shape[1]: mask1.shape[1] * 2] = mask1
    scores = [_score(mask1 * im1_gray, mask2 * im2_gray, i) for i in shifts]
    best_shift = np.argmax(scores)
    best_shift = shifts[best_shift]
    # best_shift = best_shift - mask1.shape[1]
    return best_shift


def match_segmentations_by_positions(sgm, horizontal_lim_before=100, outlier_rejection=True, vertical_limit=50):
    positions1 = []
    positions2 = []

    for i, mask in enumerate(sgm[0]["masks"]):
        mask = _conv_mask(mask)
        # pos = np.min(np.argwhere(mask > 0), axis=0)
        if np.mean(mask) > 1/3:
            continue
        pos = np.mean(np.argwhere(mask > 0), axis=0)
        positions1.append(pos)
        if sgm[0]["scores"][i] < TAKE_TOP_SCORE:
            break
    if len(positions1) == 0:
        mask = _conv_mask(sgm[0]["masks"][0])
        pos = np.mean(np.argwhere(mask > 0), axis=0)
        positions1.append(pos)

    for i, mask in enumerate(sgm[1]["masks"]):
        mask = _conv_mask(mask)
        # pos = np.min(np.argwhere(mask > 0), axis=0)
        if np.mean(mask) > 1/3:
            continue
        pos = np.mean(np.argwhere(mask > 0), axis=0)
        positions2.append(pos)
        if sgm[1]["scores"][i] < TAKE_TOP_SCORE:
            break
    if len(positions2) == 0:
        mask = _conv_mask(sgm[1]["masks"][0])
        pos = np.mean(np.argwhere(mask > 0), axis=0)
        positions2.append(pos)

    positions1 = np.array(positions1)
    positions2 = np.array(positions2)
    C = scipy.spatial.distance.cdist(positions1, positions2) # matrix of pairwise distances

    if horizontal_lim_before is not None:
        HC = scipy.spatial.distance.cdist(positions1[:, [1]], positions2[:, [1]])
        C[HC > horizontal_lim_before] = 1e6
    if vertical_limit is not None:
        HC = scipy.spatial.distance.cdist(positions1[:, [0]], positions2[:, [0]])
        C[HC > vertical_limit] = 1e6

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)  # solve min euklidean distance assignment

    if horizontal_lim_before is not None or vertical_limit is not None:
        costs = C[row_ind, col_ind]
        row_ind = row_ind[costs < 1e6]
        col_ind = col_ind[costs < 1e6]

    distances = positions2[col_ind, 1] - positions1[row_ind, 1]  # alignment diffs between matched detections
    if len(distances) == 0:
        distances = positions2[:, 1] - positions1[:, 1]  # alignment diffs between matched detections

    if outlier_rejection and len(distances) > 4:
        q_025, q_975 = np.quantile(distances, [.025, .975])
        distances = distances[(distances > q_025) & (distances < q_975)]

    return np.median(distances)


align_sc = lambda sgm, ima, imb: match_segmentation_by_shift_color(sgm, ima, imb)
align_p = lambda sgm, ima, imb: match_segmentations_by_positions(sgm, vertical_limit=30, outlier_rejection=True)


sem_landmarks_nn_models = {
    "autodidact32": ("Autodidact32", "autodidact32_6_05_12-05-22-17_17_51.st", align_sc),
    "autodidact64": ("Autodidact64", "autodidact64_6_05_12-05-22-17_44_53.st", align_sc),
    "temporal": ("Temporal", "temporal_6_05_12-05-22-16_53_20.st", align_p),
    "optflow": ("OptFlow", "optflow_6_05_12-05-22-16_23_45.st", align_p),
}

def get_model_instance_segmentation(num_classes, transfer_learning=False):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # COMENTED FOR TRANSFER LEARNING

    # get number of input features for the classifier

    # if transfer_learning:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def load_model(name, model_name):
    global device, models_base
    device = torch.device(device)
    model_file = models_base / model_name
    print(f"Loading {model_file}")
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()
    return model


def _process_img(img):
    img = cv2.resize(img, (640, 480))
    img = cv2.pyrDown(img)
    img = torchvision.transforms.functional.to_tensor(img)
    return img


def align_images(model, ima, imb, align_f):
    global device, models_base
    old_width = ima.shape[1]
    ima = _process_img(ima)[None]
    imb = _process_img(imb)[None]

    data = torch.cat((ima, imb), 0).to(device)
    sgms = model(data)
    sgms = list(zip(sgms[:len(ima)], sgms[len(imb):]))
    sgm = sgms[0]
    ima = ima[0]
    imb = imb[0]

    alignment = align_f(sgm, ima, imb)

    # alignment *= old_width / 320  # rescale back to original 

    return alignment

