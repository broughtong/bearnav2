import numpy as np
from base_classes import ProbabilityDistanceEstimator, DisplacementEstimator, AbsoluteDistanceEstimator, RepresentationsCreator
import torch as t
from backends.siamese.siam_model import get_parametrized_model, load_model
from torchvision import transforms
import rospy
import os
from bearnav2.msg import SensorsInput, ImageList, Features
from typing import List
from scipy import interpolate
import ros_numpy


PAD = 32
PEAK_MULT = 0.5
NEWTORK_DIVISION = 8.0
RESIZE_W = 512


class SiameseCNN(DisplacementEstimator, ProbabilityDistanceEstimator,
                 AbsoluteDistanceEstimator, RepresentationsCreator):

    def __init__(self):
        # TODO: all sensor inputs should be changed from ImageList to Features for major speedup
        super(SiameseCNN, self).__init__()
        self.supported_message_type = SensorsInput
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        # init neural network
        model = get_parametrized_model(False, 3, 256, 0, 3, self.device)
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(model, os.path.join(file_path, "./model_eunord.pt")).to(self.device)
        self.model.eval()
        self.to_tensor = transforms.ToTensor()
        self.alignment_processing = False
        self.histograms = None
        self.distances_probs = None
        rospy.logwarn("Siamese-NN displacement/distance estimator successfully initialized!")

    def _displacement_message_callback(self, msg: SensorsInput) -> List[np.ndarray]:
        self.alignment_processing = True
        self.process_msg(msg)
        return self.histograms

    def _prob_dist_message_callback(self, msg: SensorsInput) -> List[float]:
        if not self.alignment_processing:
            self.process_msg(msg)
        return self.distances_probs

    def _abs_dist_message_callback(self, msg: SensorsInput) -> float:
        if not len(msg.distances) > 0:
            rospy.logwarn("You cannot assign absolute distance to ")
            raise Exception("Absolute distant message callback for siamese network.")
        if not self.alignment_processing:
            self.process_msg(msg)
        return self.distances[np.argmax(self.distances_probs)]

    def _to_representation(self, msg: ImageList) -> Features:
        tensor_in = self.image_to_tensor(msg.data)
        reprs = self.model.get_repr(tensor_in)
        ret_features = []
        for repr in reprs:
            f = Features()
            f.shape = repr.shape
            f.values = t.flatten(repr).cpu().numpy()
            ret_features.append(f)
        return ret_features

    def _from_feature(self, msg):
        raise Exception()

    def health_check(self) -> bool:
        return True

    def process_msg(self, msg):
        hist = self.forward(msg.map_images.data, msg.live_images.data)      # not sure about .data here
        f = interpolate.interp1d(np.linspace(0, RESIZE_W, len(hist[0])), hist, kind="cubic")
        interp_hist = f(np.arange(0, RESIZE_W))
        self.distances_probs = np.max(interp_hist, axis=1)
        ret = []
        for hist in interp_hist:
            zeros = np.zeros(np.size(hist)//2)
            ret.append(np.concatenate([zeros, hist, zeros]))    # siam can do only -0.5 to 0.5 img so extend both sides by sth
        self.histograms = ret

    def forward(self, map_images, live_images):
        """
        map_images: list of Image messages (map images)
        live_images: list of Image messages (live feed) - right now is supported size 1
        """
        tensor1 = self.image_to_tensor(map_images)
        tensor2 = self.image_to_tensor(live_images)
        rospy.logwarn("aligning using NN " + str(tensor1.shape[0]) + " to " + str(tensor2.shape[0]) + " images")
        tensor2 = tensor2.repeat(tensor1.shape[0], 1, 1, 1)
        with t.no_grad():
            hists = self.model(tensor1, tensor2, padding=PAD).cpu().numpy()
        return hists

    def image_to_tensor(self, imgs):
        desired_height = int(imgs[0].height * RESIZE_W / imgs[0].width)
        image_list = [self.bgr_to_rgb(transforms.Resize(desired_height)(self.to_tensor(ros_numpy.numpify(img)).to(self.device)))
                      for img in imgs]
        stacked_tensor = t.stack(image_list)
        return stacked_tensor

    def bgr_to_rgb(self, tensor):
        # b = tensor[0, :, :].clone()
        # r = tensor[2, :, :].clone()
        # tensor[0, :, :] = r
        # tensor[2, :, :] = b
        return tensor
