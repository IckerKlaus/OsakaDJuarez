import cv2
import numpy as np


def apply_bilateral_filter(image: np.ndarray,
                           diameter: int = 9,
                           sigma_color: int = 75,
                           sigma_space: int = 75) -> np.ndarray:
    """
    Smooth textures while preserving edges.
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def apply_clahe_lab(image: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE on the L channel in LAB space to normalize lighting.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def convert_color_space(image: np.ndarray,
                        space: str = "LAB") -> np.ndarray:
    """
    Convert image to LAB or HSV for later processing phases.
    """
    if space.upper() == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif space.upper() == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Unsupported color space. Use 'LAB' or 'HSV'.")


def phase1_preprocess(image_path: str,
                      output_path: str = None,
                      color_space: str = "LAB") -> np.ndarray:
    """
    Full Phase I pipeline:
        1. Bilateral filter
        2. CLAHE contrast normalization
        3. Color space conversion

    Returns processed image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Step 1: Bilateral filtering
    filtered = apply_bilateral_filter(image)

    # Step 2: CLAHE (lighting normalization)
    enhanced = apply_clahe_lab(filtered)

    # Step 3: Convert color space (for later segmentation use)
    converted = convert_color_space(enhanced, color_space)

    if output_path:
        cv2.imwrite(output_path, converted)

    return converted
    