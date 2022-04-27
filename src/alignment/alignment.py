#!/usr/bin/env python
import yaml
import histogram
import numpy as np

class Alignment:

    def __init__(self):

        self.method = "SIAM"
        self.traditionalMethods = ["SIFT", "SURF", "KAZE", "AKAZE", "BRISK", "ORB"]
        
        if self.method == "SIAM":
            from backends.siam_model import Siamese, load_model, get_parametrized_model
            import torch as t
            self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
            # init neural network
            model = get_parametrized_model(False, 3, 256, 0, 3, self.device)
            file_path = os.path.dirname(os.path.abspath(__file__))
            self.model = load_model(model, os.path.join(file_path, "./backend/model_eunord.pt")).to(self.device)
            self.model.eval()
            self.to_tensor = transforms.ToTensor()

    def process(self, imgA, imgB):

        peak, uncertainty = 0, 0
        hist = []

        if self.method in self.traditionalMethods: 
            from backends import traditional
            kpsA, desA = traditional.detect(imgA, self.method)
            kpsB, desB = traditional.detect(imgB, self.method)

            if kpsA is None or kpsB is None:
                return peak, 0, hist

            displacements = traditional.match(kpsA, desA, kpsB, desB)
            displacements = [int(x) for x in displacements]

            hist = histogram.slidingHist(displacements, 10)
            peak, n = histogram.getHistPeak(hist)

            h = {}
            for i in hist:
                h.update(i)

            yVals = []
            for x in range(min(h), max(h) + 1):
                yVals.append(h[x])
            hist = yVals

            print(peak, n)

            if n < 10:
                peak = 0

        elif self.method == "VGG":
            from backends import vgg

            print(imgA.shape, imgB.shape, "SHAPRES")

            if imgA.shape[-1] == 4:
                print("WARNING 4D image!")
                imgA = imgA[:,:,:3]
            
            peak, val, hist = vgg.align(imgA, imgB)
            print(peak, val)

        elif self.method == "SIAM":
            with t.no_grad():
                if self.map_tensor is not None:
                    rospy.loginfo('Image pair received ...')
                    curr_tensor = self.image_to_tensor(imgB)
                    map_tensor = self.image_to_tensor(imgA)
                    print("Passing tensors:", map_tensor.shape, curr_tensor.shape)
                    hist = self.model(map_tensor, curr_tensor).cpu().numpy()
                    peak = int(np.argmax(hist) - hist//2)
                    # TODO: interpolate the histogram!
                    print("Outputed histogram", hist.shape)
            return peak, 0, hist 
        rospy.logwarn("No image matching scheme selected! Not correcting heading!")
        return peak, 0, hist

    def image_to_tensor(self, msg):
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        image_tensor = self.to_tensor(im).unsqueeze(0).to(self.device)
        return image_tensor
