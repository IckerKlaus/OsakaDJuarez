import os
import cv2
from multiprocessing import Pool, cpu_count
from edge_strategies import EDGE_REGISTRY
from segmentation_strategies import SEGMENT_REGISTRY
from utils import colorize_labels, draw_bounding_boxes


def run_combination(args):
    image, edge_name, seg_name, output_root, disable_box_filtering = args

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edge = EDGE_REGISTRY[edge_name]
    segmenter = SEGMENT_REGISTRY[seg_name]

    gradient = edge.compute(gray)
    labels = segmenter.segment(image.copy(), gradient)

    folder = os.path.join(
        output_root,
        f"{edge_name}_{seg_name}"
    )
    os.makedirs(folder, exist_ok=True)

    # Save gradient
    cv2.imwrite(
        os.path.join(folder, "gradient.png"),
        gradient
    )

    # Color visualization
    colored = colorize_labels(labels)
    cv2.imwrite(
        os.path.join(folder, "labels_color.png"),
        colored
    )

    overlay = cv2.addWeighted(
        image, 0.6,
        colored, 0.4,
        0
    )
    cv2.imwrite(
        os.path.join(folder, "overlay.png"),
        overlay
    )

    boxed = draw_bounding_boxes(
        image,
        labels,
        disable_filtering=disable_box_filtering
    )

    cv2.imwrite(
        os.path.join(folder, "boxes.png"),
        boxed
    )

    print(f"Finished: {edge_name} + {seg_name}")


def run_parallel(image,
                 output_root="output",
                 disable_box_filtering=False):

    # ------------------------------------
    # Active combinations
    # ------------------------------------
    combinations = [
        ("canny", "watershed"),
        ("canny", "voronoi"),
    ]

    args_list = [
        (image, edge, seg, output_root, disable_box_filtering)
        for edge, seg in combinations
    ]

    workers = min(len(combinations), cpu_count())

    print(f"Running in parallel using {workers} workers")

    with Pool(processes=workers) as pool:
        pool.map(run_combination, args_list)
        