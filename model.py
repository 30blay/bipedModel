import skimage.segmentation as seg
from skimage import io
from skimage.transform import resize
import numpy as np
from PIL import Image


class BipedModel:
    def __init__(self):
        pass

    def extract_features(self, img_path):
        image = io.imread(img_path)
        mask = self.isolate_sock(image)
        features = self.histogram(image, mask)
        return features

    def isolate_sock(self, image):
        # first resize the image for quicker calculations
        aspect_ratio = image.shape[1] / image.shape[0]
        resized_height = 100
        resized_width = int(resized_height * aspect_ratio)
        image = resize(image, (resized_height, resized_width), anti_aliasing=False)

        # low min size factor because we don't want the algo to drop a segment
        mask = seg.slic(image, n_segments=2, min_size_factor=0.01, max_size_factor=2, compactness=4)
        image = image * np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # make a 1D vector with all the unmasked pixels
        #sock = image[np.nonzero(image.mean(axis=2))]
        t2 = time.process_time()
        return mask

    def get_similarity(self, feature1, feature2):
        distance = np.linalg.norm(feature1, feature2)
        similarity = 1-distance
        return similarity

    def histogram(self, image, mask):
        pil_image = Image.fromarray(image)
        count_histogram = pil_image.histogram()
        density_histogram = count_histogram/np.linalg.norm(count_histogram, ord=2, keepdims=True)
        return density_histogram
