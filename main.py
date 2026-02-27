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
        "--max_box_ratio",
        type=float,
        default=0.9,
        help=(
            "Maximum fraction of image area a single box may cover "
            "(default: 0.9). Boxes larger than this are discarded as "
            "background. Applied before all other filters."
        )
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
        "--aspect_ratio_sigma",
        type=float,
        default=2.0,
        help=(
            "Number of standard deviations from the mean aspect ratio "
            "allowed before a box is discarded (default: 2.0). "
            "Applied after merging and before duplicate removal."
        )
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
    parser.add_argument(
        "--diagonal_gap_ratio",
        type=float,
        default=0.3,
        help=(
            "Relative diagonal-length gap that separates two size groups "
            "(default: 0.3). Boxes are sorted by diagonal; a new group starts "
            "whenever consecutive diagonals differ by more than this fraction. "
            "Groups with too few members are discarded."
        )
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=2,
        help=(
            "Minimum number of boxes a diagonal-size group must contain to be "
            "kept (default: 2). Groups smaller than this are considered fragment "
            "crops and are discarded."
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
        max_box_ratio=args.max_box_ratio,
        area_percentile=args.area_percentile,
        iou_merge_threshold=args.iou_merge_threshold,
        aspect_ratio_sigma=args.aspect_ratio_sigma,
        duplicate_threshold=args.duplicate_threshold,
        diagonal_gap_ratio=args.diagonal_gap_ratio,
        min_group_size=args.min_group_size,
        evaluate=args.evaluate,
        gt_path=args.gt_path,
        eval_iou_threshold=args.eval_iou_threshold,
    )


if __name__ == "__main__":
    main()
    