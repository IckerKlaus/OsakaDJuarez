import cv2
import numpy as np

EDGE_REGISTRY = {}

def register_edge(name):
    def decorator(cls):
        EDGE_REGISTRY[name] = cls()
        return cls
    return decorator


class EdgeStrategy:
    def compute(self, gray):
        raise NotImplementedError


# @register_edge("sobel")
# class SobelEdge(EdgeStrategy):

#     def compute(self, gray):
#         gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#         mag = cv2.magnitude(gx, gy)
#         return cv2.normalize(mag, None, 0, 255,
#                              cv2.NORM_MINMAX).astype(np.uint8)


# @register_edge("canny")
# class CannyEdge(EdgeStrategy):

#     def compute(self, gray):
#         sigma = np.std(gray)
#         lower = int(max(0, 0.66 * sigma))
#         upper = int(min(255, 1.33 * sigma))
#         return cv2.Canny(gray, lower, upper)


# @register_edge("scharr")
# class ScharrEdge(EdgeStrategy):

#     def compute(self, gray):
#         gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
#         gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
#         mag = cv2.magnitude(gx, gy)
#         return cv2.normalize(mag, None, 0, 255,
#                              cv2.NORM_MINMAX).astype(np.uint8)


@register_edge("laplacian")
class LaplacianEdge(EdgeStrategy):

    def compute(self, gray):

        # Step 1: Gaussian smoothing (IMPORTANT)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Step 2: Laplacian
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

        # Step 3: Absolute value
        lap = np.absolute(lap)

        # Step 4: Convert to uint8 safely
        lap = np.uint8(lap)

        return lap
    