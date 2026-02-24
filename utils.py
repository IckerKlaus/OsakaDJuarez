import numpy as np
import cv2
import os
import json

# -------------------------------------------------
# Colorize labels
# -------------------------------------------------
def colorize_labels(labels):
    np.random.seed(42)
    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for label in np.unique(labels):
        if label == 0:
            continue
        color = np.random.randint(0, 255, size=3)
        colored[labels == label] = color

    return colored


# -------------------------------------------------
# Extract raw bounding boxes
# -------------------------------------------------
def extract_boxes(labels):

    boxes = []

    for label in np.unique(labels):
        if label == 0:
            continue

        mask = (labels == label).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h, w * h))

    return boxes


# -------------------------------------------------
# IoU between two boxes (x, y, w, h, area)
# -------------------------------------------------
def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    union = boxA[4] + boxB[4] - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


# -------------------------------------------------
# 1a) Percentile-based area filtering
#     Removes boxes whose area is below the given
#     percentile of the distribution of all box areas.
# -------------------------------------------------
def percentile_filter(boxes, percentile=20):
    """
    Remove boxes whose area falls below `percentile` of all box areas.

    Args:
        boxes:      list of (x, y, w, h, area)
        percentile: float in [0, 100]; boxes below this percentile are dropped

    Returns:
        Filtered list of boxes.
    """
    if not boxes:
        return []

    areas = np.array([b[4] for b in boxes], dtype=float)
    threshold = np.percentile(areas, percentile)

    return [b for b in boxes if b[4] >= threshold]


# -------------------------------------------------
# 1b) Remove fully contained boxes
#     A box is removed if it is completely enclosed
#     within any other box (boundary-inclusive).
# -------------------------------------------------
def remove_contained(boxes):
    """
    Remove any box that is fully contained within another box.

    Containment is defined as:
        boxA.x1 >= boxB.x1  AND  boxA.y1 >= boxB.y1
        AND boxA.x2 <= boxB.x2  AND  boxA.y2 <= boxB.y2
    where A != B.

    Returns:
        List with contained boxes removed.
    """
    result = []

    for i, boxA in enumerate(boxes):
        ax1, ay1 = boxA[0], boxA[1]
        ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]

        is_contained = False

        for j, boxB in enumerate(boxes):
            if i == j:
                continue

            bx1, by1 = boxB[0], boxB[1]
            bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

            if ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2:
                # boxA is fully inside boxB — drop it
                is_contained = True
                break

        if not is_contained:
            result.append(boxA)

    return result


# -------------------------------------------------
# 1c) IoU-based iterative box merging
#     Merges pairs of boxes that overlap above
#     `iou_threshold` based on their *original*
#     extents (not growing merged extents), which
#     prevents weak-overlap cascade merging.
# -------------------------------------------------
def merge_boxes(boxes, iou_threshold=0.3):
    """
    Iteratively merge boxes that have IoU > iou_threshold.

    To prevent cascade merging (where A+B → AB grows to absorb C even
    though original A and C had IoU=0), each merge pass evaluates IoU
    only on the *input* boxes at the start of that pass.  The merged
    bounding rectangle is the minimal axis-aligned rectangle covering
    both boxes.  Passes repeat until no further merges occur.

    Args:
        boxes:         list of (x, y, w, h, area)
        iou_threshold: minimum IoU to trigger a merge

    Returns:
        Merged list of boxes.
    """
    boxes = list(boxes)

    changed = True
    while changed:
        changed = False
        n = len(boxes)
        used = [False] * n
        new_boxes = []

        for i in range(n):
            if used[i]:
                continue

            # Find all boxes that directly overlap with boxes[i]
            # (evaluated against the ORIGINAL boxes[i], not a growing merge)
            group = [i]
            for j in range(i + 1, n):
                if used[j]:
                    continue
                if compute_iou(boxes[i], boxes[j]) > iou_threshold:
                    group.append(j)

            if len(group) > 1:
                # Merge all boxes in the group into one bounding rectangle
                x1 = min(boxes[k][0] for k in group)
                y1 = min(boxes[k][1] for k in group)
                x2 = max(boxes[k][0] + boxes[k][2] for k in group)
                y2 = max(boxes[k][1] + boxes[k][3] for k in group)
                w, h = x2 - x1, y2 - y1
                new_boxes.append((x1, y1, w, h, w * h))
                for k in group:
                    used[k] = True
                changed = True
            else:
                new_boxes.append(boxes[i])
                used[i] = True

        boxes = new_boxes

    return boxes


# -------------------------------------------------
# 2) Remove duplicate crops using IoU similarity
#    Boxes with IoU > duplicate_threshold are
#    considered the same product; keep the larger.
# -------------------------------------------------
def remove_duplicate_boxes(boxes, duplicate_threshold=0.7):
    """
    After all merging/cleaning, remove boxes that are near-duplicates of
    each other (IoU > duplicate_threshold).  When two duplicates are found
    the larger box (by area) is retained.

    This prevents saving multiple cropped images of the same product.

    Args:
        boxes:               list of (x, y, w, h, area)
        duplicate_threshold: IoU threshold above which two boxes are
                             considered duplicates (default 0.7)

    Returns:
        De-duplicated list of boxes.
    """
    if not boxes:
        return []

    # Sort largest-first so that when duplicates are found we keep the
    # largest by construction (first encountered wins).
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

    kept = []
    for i, boxA in enumerate(boxes):
        is_dup = False
        for boxB in kept:
            if compute_iou(boxA, boxB) > duplicate_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(boxA)

    return kept


# -------------------------------------------------
# Full post-processing pipeline
# -------------------------------------------------
def postprocess_boxes(
    boxes,
    area_percentile=20,
    iou_merge_threshold=0.3,
    duplicate_threshold=0.7,
):
    """
    Apply the full Phase IV post-processing chain in order:
        1a. Percentile-based area filtering
        1b. Remove fully contained boxes
        1c. Iterative IoU merging
         2. Remove near-duplicate boxes

    All thresholds are configurable.
    """
    boxes = percentile_filter(boxes, percentile=area_percentile)
    boxes = remove_contained(boxes)
    boxes = merge_boxes(boxes, iou_threshold=iou_merge_threshold)
    boxes = remove_duplicate_boxes(boxes, duplicate_threshold=duplicate_threshold)
    return boxes


# -------------------------------------------------
# Final Phase IV draw
# -------------------------------------------------
def draw_bounding_boxes(
    image,
    labels,
    disable_filtering=False,
    area_percentile=20,
    iou_merge_threshold=0.3,
    duplicate_threshold=0.7,
):
    output = image.copy()
    boxes = extract_boxes(labels)

    if not disable_filtering:
        boxes = postprocess_boxes(
            boxes,
            area_percentile=area_percentile,
            iou_merge_threshold=iou_merge_threshold,
            duplicate_threshold=duplicate_threshold,
        )

    for box in boxes:
        x, y, w, h, _ = box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

    return output


# -------------------------------------------------
# Jaccard similarity (pixel mask IoU)
# -------------------------------------------------
def compute_jaccard(img1, img2):

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, bin1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

    bin1 = bin1 > 0
    bin2 = bin2 > 0

    intersection = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()

    if union == 0:
        return 0.0

    return intersection / union


# -------------------------------------------------
# Load ground truth images (for Jaccard matching)
# -------------------------------------------------
def load_ground_truth(folder="results"):

    images = {}

    if not os.path.exists(folder):
        return images

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            images[file] = cv2.imread(os.path.join(folder, file))

    return images


# -------------------------------------------------
# Load ground truth bounding boxes for evaluation
#
# Supported formats
#   JSON  – list of {"x":…,"y":…,"w":…,"h":…} or [x,y,w,h]
#   TXT   – one box per line: x y w h
# -------------------------------------------------
def load_gt_boxes(path):
    """
    Load ground-truth bounding boxes from a file.

    Supported formats
    -----------------
    JSON  :  A JSON array of objects {"x", "y", "w", "h"}
             or plain arrays [x, y, w, h].
    TXT   :  One box per line with whitespace-separated x y w h values.

    Returns
    -------
    list of (x, y, w, h, area) tuples
    """
    ext = os.path.splitext(path)[1].lower()
    boxes = []

    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, dict):
                x, y, w, h = item["x"], item["y"], item["w"], item["h"]
            else:
                x, y, w, h = item[:4]
            boxes.append((int(x), int(y), int(w), int(h), int(w) * int(h)))

    elif ext == ".txt":
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    boxes.append((x, y, w, h, w * h))

    else:
        raise ValueError(f"Unsupported ground-truth format: {ext}  (use .json or .txt)")

    return boxes


# -------------------------------------------------
# Evaluation metrics
# -------------------------------------------------
def evaluate_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compare predicted boxes against ground-truth boxes and compute:
        Precision, Recall, F1-score, Mean IoU (over true positives).

    A predicted box is a True Positive (TP) if its best IoU with any
    unmatched GT box exceeds `iou_threshold`.

    Args:
        pred_boxes:    list of (x, y, w, h, area)
        gt_boxes:      list of (x, y, w, h, area)
        iou_threshold: IoU required to count as a match (default 0.5)

    Returns:
        dict with keys: precision, recall, f1, mean_iou, tp, fp, fn
    """
    tp = 0
    fp = 0
    matched_gt = set()
    iou_scores = []

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            iou_scores.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou  = float(np.mean(iou_scores)) if iou_scores else 0.0

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "mean_iou":  mean_iou,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }
