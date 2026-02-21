import argparse
import cv2
from runner import run_parallel
from edge_strategies import EDGE_REGISTRY
from segmentation_strategies import SEGMENT_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Input image path"
    )

    # Always running all combinations now
    parser.add_argument(
        "--no_box_filtering",
        action="store_true",
        help="Disable box merging and filtering"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError("Image not found.")

    # Always run all combinations
    run_parallel(
        image,
        disable_box_filtering=args.no_box_filtering
    )


if __name__ == "__main__":
    main()
    