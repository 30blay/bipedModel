import skimage.segmentation as seg
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, rgb2gray
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine as cosine_distance


class BipedModel:
    def __init__(self):
        pass

    def extract_features(self, img_path, return_isolated=False):
        image = self.isolate_sock(img_path)
        features = self.histogram(image)

        if return_isolated:
            return features, image

        return features

    def resize(self, image):
        # first resize the image for quicker calculations
        aspect_ratio = image.shape[1] / image.shape[0]
        resized_height = 100
        resized_width = int(resized_height * aspect_ratio)
        image = resize(image, (resized_height, resized_width), anti_aliasing=False, preserve_range=True)
        return image

    def isolate_sock(self, img_path, method='morphological_chan_vese'):
        image = io.imread(img_path)
        image = self.resize(image)
        # low min size factor because we don't want the algo to drop a segment
        methods = {
            'slic': self.slic,
            'morphological_chan_vese': self.morphological_cv,
        }
        mask = methods[method](image)
        mask = mask != 0
        image = image * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        image = image.astype(int)
        return image

    def get_similarity(self, feature1, feature2, method='intersection'):
        if method == 'cosine':
            similarity = 1-cosine_distance(feature1, feature2)

        elif method == 'intersection':
            minima = np.minimum(feature1, feature2)
            maxima = np.maximum(feature1, feature2)
            similarity = np.true_divide(np.sum(minima), np.sum(maxima))
        return similarity

    def histogram(self, image, method='3D_rgb'):
        image = image.astype(int)
        # make a linear image with all the unmasked pixels
        sock = image[np.nonzero(image.mean(axis=2))]
        sock = sock[:, np.newaxis, :]
        sock = np.uint8(sock)

        if method == 'pil_histogram':
            pil_image = Image.fromarray(sock)
            histogram = pil_image.histogram()

        if method == '3D_rgb':
            sock = sock[:, 0, :]
            bins_per_dim = 17
            bin_limits = [pow(255, i/bins_per_dim)for i in range(bins_per_dim+1)]
            bin_limits[0] = 0
            bins = np.array([bin_limits, bin_limits, bin_limits])
            histogram, edges = np.histogramdd(sock, bins=bins)

        if method == 'hue':
            sock = rgb2hsv(sock)
            sock = sock[:, 0, :]
            weights = sock[:, 1] * sock[:, 2] # value times saturation
            histogram, edges = np.histogramdd(sock[:, 0], bins=[40])

        return histogram.flatten()

    def slic(self, image):
        mask = seg.slic(image, n_segments=2, min_size_factor=0.01, max_size_factor=2, compactness=10)
        return mask

    def morphological_cv(self, image):
        image = rgb2gray(image)
        mask = seg.morphological_chan_vese(image, iterations=1000, init_level_set='circle')
        return mask
