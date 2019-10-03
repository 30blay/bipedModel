import skimage.segmentation as seg
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine as cosine_distance
import matplotlib.pyplot as plt


class BipedModel:
    def __init__(self):
        pass

    def extract_features(self, img_path):
        image = io.imread(img_path)
        image = self.resize(image)
        image, mask = self.isolate_sock(image)
        features = self.histogram(image, mask)
        return features

    def resize(self, image):
        # first resize the image for quicker calculations
        aspect_ratio = image.shape[1] / image.shape[0]
        resized_height = 100
        resized_width = int(resized_height * aspect_ratio)
        image = resize(image, (resized_height, resized_width), anti_aliasing=False, preserve_range=True)
        return image

    def isolate_sock(self, image):
        # low min size factor because we don't want the algo to drop a segment
        mask = seg.slic(image, n_segments=2, min_size_factor=0.01, max_size_factor=2, compactness=10)
        image = image * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        image = image.astype(int)
        #plt.imshow(image)
        #plt.show()
        return image, mask

    def get_similarity(self, feature1, feature2):
        similarity = 1-cosine_distance(feature1, feature2)
        return similarity

    def histogram(self, image, mask, method='pil_histogram'):
        # make a linear image with all the unmasked pixels
        sock = image[np.nonzero(image.mean(axis=2))]
        sock = sock[:, np.newaxis, :]

        if method == 'pil_histogram':
            sock = np.uint8(sock)
            pil_image = Image.fromarray(sock)
            count_histogram = pil_image.histogram()

        if method == 'hue':
            hsv = rgb2hsv(sock)
            pass

        density_histogram = count_histogram/np.linalg.norm(count_histogram, ord=2, keepdims=True)
        return density_histogram
