import torch as t
from backends.siam_model import Siamese, load_model, get_parametrized_model
from torchvision import transforms
import os
import numpy as np
from cv_bridge import CvBridge

PAD = 32
RESIZE_H = 320

class SiameseNetwork:

    def __init__(self):
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        # init neural network
        model = get_parametrized_model(False, 3, 256, 0, 3, self.device)
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(model, os.path.join(file_path,
                                                    "../siamese/backends/model_eunord.pt")).to(self.device)
        self.model.eval()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(RESIZE_H)
        self.cv_parser = CvBridge()

    def forward(self, map_images, live_images):
        """
        imgs1: list of Image messages (map images)
        imgs2: list of Image messages (live feed) - right now is supported size 1
        """
        tensor1 = self.image_to_tensor(map_images)
        tensor2 = self.image_to_tensor(live_images)
        tensor2 = tensor2.repeat(tensor1.shape[0], 1, 1, 1)
        with t.no_grad():
            hists = self.model(tensor1, tensor2, padding=PAD)
        return hists

    def image_to_tensor(self, imgs):
        image_list = [self.resize(self.to_tensor(np.array(self.cv_parser.imgmsg_to_cv2(img))).unsqueeze(0))
                      for img in imgs]
        stacked_tensor = t.stack(image_list).to(self.device)
        return stacked_tensor