import numpy as np
from base_classes import DisplacementEstimator
import torch as t
from torch.nn import functional as F
from torchvision import transforms
import rospy
from cv_bridge import CvBridge
import os
from bearnav2.msg import SensorsInput
from typing import List
from scipy import interpolate


PAD = 32
NEWTORK_DIVISION = 8.0
RESIZE_H = int(320 // NEWTORK_DIVISION)
RESIZE_W = int(512 // NEWTORK_DIVISION)


class CrossCorrelation(DisplacementEstimator):

    def __init__(self):
        super(CrossCorrelation, self).__init__()
        self.supported_message_type = SensorsInput
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        # init neural network
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(RESIZE_H)
        self.cv_parser = CvBridge()
        self.alignment_processing = False
        self.histograms = None
        self.distances_probs = None
        rospy.logwarn("Cross correlation displacement estimator sucessfully initialized!")

    def _displacement_message_callback(self, msg: SensorsInput) -> List[np.ndarray]:
        rospy.logwarn("started processing")
        self.alignment_processing = True
        self.process_msg(msg)
        return self.histograms

    def health_check(self) -> bool:
        return True

    def process_msg(self, msg):
        hist = self.forward(msg.map_images.data, msg.live_images.data)      # not sure about .data here
        rospy.logwarn(np.shape(np.linspace(0, int(RESIZE_W * NEWTORK_DIVISION), len(hist[0]))))
        rospy.logwarn(np.shape(hist[0]))
        f = interpolate.interp1d(np.linspace(0, int(RESIZE_W * NEWTORK_DIVISION), len(hist[0])), hist[0], kind="cubic")
        interp_hist = f(np.arange(0, int(RESIZE_W * NEWTORK_DIVISION)))
        rospy.logwarn(np.shape(interp_hist))
        zeros = np.zeros(np.size(interp_hist)//2)
        ret = np.concatenate([zeros, interp_hist, zeros])    # siam can do only -0.5 to 0.5 img so extend both sides by sth
        self.histograms = [ret]

    def forward(self, map_images, live_images):
        """
        map_images: list of Image messages (map images)
        live_images: list of Image messages (live feed) - right now is supported size 1
        """
        tensor1 = self.image_to_tensor(map_images)
        tensor2 = self.image_to_tensor(live_images)
        rospy.logwarn("Aligning using crosscorr " + str(tensor1.shape[0]) + " to " + str(tensor2.shape[0]) + " images")
        tensor2 = tensor2.repeat(tensor1.shape[0], 1, 1, 1)
        with t.no_grad():
            hists = self._match_corr(tensor1, tensor2, padding=PAD).cpu().numpy()
        return hists[0][0]

    def _match_corr(self, embed_ref, embed_srch, padding=None):
        if padding is None:
            padding = self.padding
        b, c, h, w = embed_srch.shape
        _, _, h_ref, w_ref = embed_ref.shape

        match_map = F.conv2d(F.pad(embed_srch.view(1, b * c, h, w), pad=(padding, padding, 1, 1), mode='circular'),
                             embed_ref, groups=b)

        match_map = t.max(match_map.permute(1, 0, 2, 3), dim=2)[0].unsqueeze(2)
        return match_map

    def image_to_tensor(self, imgs):
        image_list = [self.resize(self.to_tensor(np.array(self.cv_parser.imgmsg_to_cv2(img))).to(self.device))
                      for img in imgs]
        stacked_tensor = t.stack(image_list)
        return stacked_tensor
