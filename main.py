import argparse
import cv2
import shutil
import os
from runner import run_parallel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Input image path"
    )

    parser.add_argument(
        "--no_box_filtering",
        action="store_true",
        help="Disable box merging and filtering"
    )

    return parser.parse_args()


def reset_output():
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output", exist_ok=True)


def main():
    args = parse_args()

    reset_output()  # 🔥 reset everything every run

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError("Image not found.")

    run_parallel(
        image,
        disable_box_filtering=args.no_box_filtering
    )


if __name__ == "__main__":
    main()
    