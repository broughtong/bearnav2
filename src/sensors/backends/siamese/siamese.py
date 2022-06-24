import numpy as np
from base_classes import ProbabilityDistanceEstimator, DisplacementEstimator, AbsoluteDistanceEstimator
import torch as t
from backends.siamese.siam_model import get_parametrized_model, load_model
from torchvision import transforms
import rospy
from cv_bridge import CvBridge
import os
from bearnav2.msg import SensorsInput
from typing import List
from scipy import interpolate


PAD = 32
PEAK_MULT = 0.5
NEWTORK_DIVISION = 8.0
RESIZE_H = 320
RESIZE_W = 512


class SiameseCNN(DisplacementEstimator, ProbabilityDistanceEstimator, AbsoluteDistanceEstimator):

    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.supported_message_type = SensorsInput
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        # init neural network
        model = get_parametrized_model(False, 3, 256, 0, 3, self.device)
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(model, os.path.join(file_path, "./model_eunord.pt")).to(self.device)
        self.model.eval()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(RESIZE_H)
        self.cv_parser = CvBridge()
        self.alignment_processing = False
        self.histograms = None
        self.distances_probs = None
        rospy.logwarn("Siamese model sucessfully initialized!")

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

    def health_check(self) -> bool:
        return True

    def process_msg(self, msg):
        hist = self.forward(msg.map_images.data, msg.live_images.data)      # not sure about .data here
        f = interpolate.interp1d(np.linspace(0, RESIZE_W, len(hist[0])), hist, kind="cubic")
        interp_hist = f(np.arange(0, RESIZE_W))
        self.distances_probs = np.max(interp_hist, axis=1)
        self.histograms = list(interp_hist)

    def forward(self, map_images, live_images):
        """
        map_images: list of Image messages (map images)
        live_images: list of Image messages (live feed) - right now is supported size 1
        """
        tensor1 = self.image_to_tensor(map_images)
        tensor2 = self.image_to_tensor(live_images)
        rospy.logwarn("aligning " + str(tensor1.shape[0]) + " to " + str(tensor2.shape[0]) + " images")
        tensor2 = tensor2.repeat(tensor1.shape[0], 1, 1, 1)
        with t.no_grad():
            hists = self.model(tensor1, tensor2, padding=PAD).cpu().numpy()
        return hists

    def image_to_tensor(self, imgs):
        image_list = [self.resize(self.to_tensor(np.array(self.cv_parser.imgmsg_to_cv2(img))).to(self.device))
                      for img in imgs]
        stacked_tensor = t.stack(image_list)
        return stacked_tensor
