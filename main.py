import argparse
import cv2
import shutil
import os
from runner import run_parallel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Product segmentation pipeline for retail shelf images."
    )

    # ── positional ────────────────────────────────────────────────────────────
    parser.add_argument(
        "image",
        help="Path to the input retail shelf image."
    )

    # ── filtering ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no_box_filtering",
        action="store_true",
        help="Disable all bounding-box merging and filtering."
    )
    parser.add_argument(
        "--area_percentile",
        type=float,
        default=20.0,
        help=(
            "Percentile threshold for area-based box removal (default: 20). "
            "Boxes whose area is below this percentile are discarded."
        )
    )
    parser.add_argument(
        "--iou_merge_threshold",
        type=float,
        default=0.3,
        help="IoU threshold for merging overlapping boxes (default: 0.3)."
    )
    parser.add_argument(
        "--duplicate_threshold",
        type=float,
        default=0.7,
        help=(
            "IoU threshold for removing duplicate crop boxes (default: 0.7). "
            "Boxes above this threshold are considered the same product."
        )
    )

    # ── evaluation ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Enable evaluation mode.  Requires --gt_path.  "
            "Computes Precision, Recall, F1, and Mean IoU against "
            "ground-truth bounding boxes."
        )
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help=(
            "Path to the ground-truth bounding-box file (required when "
            "--evaluate is set).  Supported formats: JSON (.json) or "
            "plain text (.txt, one box per line: x y w h)."
        )
    )
    parser.add_argument(
        "--eval_iou_threshold",
        type=float,
        default=0.5,
        help=(
            "IoU threshold used to count a predicted box as a true positive "
            "during evaluation (default: 0.5)."
        )
    )

    args = parser.parse_args()

    # Validate evaluation arguments
    if args.evaluate and not args.gt_path:
        parser.error("--evaluate requires --gt_path to be specified.")

    return args


def reset_output():
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output", exist_ok=True)


def main():
    args = parse_args()

    reset_output()  # reset everything every run

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Image not found or could not be read: {args.image}")

    run_parallel(
        image,
        disable_box_filtering=args.no_box_filtering,
        area_percentile=args.area_percentile,
        iou_merge_threshold=args.iou_merge_threshold,
        duplicate_threshold=args.duplicate_threshold,
        evaluate=args.evaluate,
        gt_path=args.gt_path,
        eval_iou_threshold=args.eval_iou_threshold,
    )


if __name__ == "__main__":
    main()
    