#!/usr/bin/env python3
"""
Official Competition Metric Evaluator for AIC2026 Multimodal Object Detection
=============================================================================
Strictly implements the official competition evaluation protocol defined in:
《【竞赛细则】面向城市场景的视觉多模态目标检测》 Section 六 (性能指标要求):

1. IoU Thresholds:
   T = {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95} (10 thresholds)
2. Greedy TP / FP Matching:
   - Max 100 detections per image enforced (sorted by confidence descending across all classes).
   - Per category and per IoU threshold t:
     Traverse predictions sorted by confidence descending.
     Greedily match to highest-IoU unmatched ground truth on the same image with IoU >= t.
     If IoU >= t and unmatched: TP, mark GT as matched.
     Else: FP.
3. 101-point Interpolated AP:
   - Recall sampling points: r in {0.00, 0.01, ..., 1.00} (101 points).
   - P_interp(r) = max_{r' >= r} P(r')
   - AP_c(t) = 1/101 * sum_{r} P_interp(r)
4. Macro-Average across all 12 classes:
   mAP(t) = 1/12 * sum_{c=0}^{11} AP_c(t)
5. Overall Competition Score:
   mAP@50-95 = 1/10 * sum_{t=0.50}^{0.95} mAP(t)

Supported Inputs:
  - Ground Truth: COCO JSON (`instances_val.json`) or YOLO/AIC TXT folder (`<stem>.txt`)
  - Predictions: TXT submission directory, submission .zip file, or Python detection dict/list
"""

import os
import sys
import glob
import json
import zipfile
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

# Official 12 Categories Definition
AIC_CATEGORIES = [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "boat"},
    {"id": 2, "name": "animal"},
    {"id": 3, "name": "seat"},
    {"id": 4, "name": "sign"},
    {"id": 5, "name": "bicycle"},
    {"id": 6, "name": "car"},
    {"id": 7, "name": "ball"},
    {"id": 8, "name": "light"},
    {"id": 9, "name": "garbage can"},
    {"id": 10, "name": "uav"},
    {"id": 11, "name": "tricycle"},
]
ID2NAME = {c["id"]: c["name"] for c in AIC_CATEGORIES}
NAME2ID = {c["name"]: c["id"] for c in AIC_CATEGORIES}
NUM_CLASSES = len(AIC_CATEGORIES)


def box_iou_numpy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box [x1, y1, x2, y2] and multiple boxes [N, 4].
    Works for both normalized [0, 1] and pixel coordinates.
    """
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


class OfficialCompetitionEvaluator:
    """
    Evaluator implementing the exact official AIC2026 evaluation logic.
    """

    def __init__(self, ground_truth_source: str):
        """
        Initialize with ground truth source.
        ground_truth_source: Path to instances_val.json OR path to directory of .txt label files.
        """
        self.iou_thresholds = np.linspace(0.50, 0.95, 10, dtype=np.float32)
        self.recall_thresholds = np.linspace(0.00, 1.00, 101, dtype=np.float32)
        self.gts_by_stem_cls = defaultdict(lambda: defaultdict(list))
        self.gt_counts_by_cls = defaultdict(int)
        self.image_meta = {}  # stem -> {"width": W, "height": H}
        self.total_gt = 0

        self._load_ground_truth(ground_truth_source)

    def _load_ground_truth(self, gt_path: str):
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth path not found: {gt_path}")

        if os.path.isfile(gt_path) and gt_path.endswith(".json"):
            # Load COCO JSON
            with open(gt_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            id2stem = {}
            for img in data["images"]:
                stem = os.path.splitext(img["file_name"])[0]
                id2stem[img["id"]] = stem
                self.image_meta[stem] = {
                    "width": float(img.get("width", 640)),
                    "height": float(img.get("height", 360)),
                }

            for ann in data["annotations"]:
                stem = id2stem.get(ann["image_id"])
                if stem is None:
                    continue
                cid = int(ann["category_id"])
                if cid < 0 or cid >= NUM_CLASSES:
                    continue

                W = self.image_meta[stem]["width"]
                H = self.image_meta[stem]["height"]
                # In COCO bbox is [x, y, w, h] in absolute pixels
                x, y, w, h = ann["bbox"]
                # Convert to normalized [x1, y1, x2, y2]
                box_norm = np.array([
                    np.clip(x / W, 0.0, 1.0),
                    np.clip(y / H, 0.0, 1.0),
                    np.clip((x + w) / W, 0.0, 1.0),
                    np.clip((y + h) / H, 0.0, 1.0),
                ], dtype=np.float32)

                self.gts_by_stem_cls[stem][cid].append(box_norm)
                self.gt_counts_by_cls[cid] += 1
                self.total_gt += 1

        elif os.path.isdir(gt_path):
            # Load TXT folder
            txt_files = glob.glob(os.path.join(gt_path, "*.txt"))
            for tf in txt_files:
                stem = os.path.splitext(os.path.basename(tf))[0]
                with open(tf, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cid = int(float(parts[0]))
                        if cid < 0 or cid >= NUM_CLASSES:
                            continue
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = np.clip(cx - w / 2.0, 0.0, 1.0)
                        y1 = np.clip(cy - h / 2.0, 0.0, 1.0)
                        x2 = np.clip(cx + w / 2.0, 0.0, 1.0)
                        y2 = np.clip(cy + h / 2.0, 0.0, 1.0)
                        box_norm = np.array([x1, y1, x2, y2], dtype=np.float32)
                        self.gts_by_stem_cls[stem][cid].append(box_norm)
                        self.gt_counts_by_cls[cid] += 1
                        self.total_gt += 1
        else:
            raise ValueError(f"Unsupported ground truth format: {gt_path}. Must be .json file or directory of .txt files.")

    def parse_predictions_from_txt_dir(
        self,
        pred_source: str,
        max_boxes_per_image: int = 100,
        score_thresh: float = 0.0,
    ) -> Dict[str, List[Tuple[float, int, np.ndarray]]]:
        """
        Parses predictions from a folder or zip file containing .txt files.
        Enforces official rule:
        - Max 100 boxes per image across all categories (sorted by confidence descending).
        - Format: <class_id> <cx> <cy> <w> <h> <confidence>
        - Discards invalid coordinates or classes.
        """
        preds_by_stem = defaultdict(list)

        def process_lines(stem: str, lines: List[str]):
            img_cands = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h, sc = map(float, parts[1:6])
                except ValueError:
                    continue

                if cid < 0 or cid >= NUM_CLASSES:
                    continue
                if sc < score_thresh or sc <= 0.0 or sc > 1.0 or np.isnan(sc):
                    continue
                if w <= 0.0 or h <= 0.0 or np.isnan(cx) or np.isnan(cy) or np.isnan(w) or np.isnan(h):
                    continue

                x1 = np.clip(cx - w / 2.0, 0.0, 1.0)
                y1 = np.clip(cy - h / 2.0, 0.0, 1.0)
                x2 = np.clip(cx + w / 2.0, 0.0, 1.0)
                y2 = np.clip(cy + h / 2.0, 0.0, 1.0)
                img_cands.append((float(sc), cid, np.array([x1, y1, x2, y2], dtype=np.float32)))

            # Sort by score descending and truncate to max_boxes_per_image
            img_cands.sort(key=lambda x: -x[0])
            preds_by_stem[stem] = img_cands[:max_boxes_per_image]

        if zipfile.is_zipfile(pred_source):
            with zipfile.ZipFile(pred_source, "r") as z:
                for fname in z.namelist():
                    if fname.endswith(".txt") and not fname.startswith("__MACOSX"):
                        stem = os.path.splitext(os.path.basename(fname))[0]
                        content = z.read(fname).decode("utf-8", errors="ignore")
                        lines = content.splitlines()
                        process_lines(stem, lines)
        elif os.path.isdir(pred_source):
            for txt_file in glob.glob(os.path.join(pred_source, "*.txt")):
                stem = os.path.splitext(os.path.basename(txt_file))[0]
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    process_lines(stem, lines)
        else:
            raise ValueError(f"Invalid prediction source: {pred_source}. Must be a directory or .zip file.")

        return preds_by_stem

    def parse_predictions_from_dict(
        self,
        pred_dict: Dict,
        max_boxes_per_image: int = 100,
        score_thresh: float = 0.0,
        id_to_stem_map: Optional[Dict] = None,
    ) -> Dict[str, List[Tuple[float, int, np.ndarray]]]:
        """
        Parses predictions from internal dictionary or tensors:
        pred_dict: {img_id_or_stem: {"boxes": ..., "scores": ..., "labels": ...}}
        """
        preds_by_stem = defaultdict(list)
        for key, det in pred_dict.items():
            stem = id_to_stem_map.get(key, str(key)) if id_to_stem_map else str(key)
            boxes = det["boxes"]
            scores = det["scores"]
            labels = det["labels"]

            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()

            W = self.image_meta.get(stem, {}).get("width", 640.0)
            H = self.image_meta.get(stem, {}).get("height", 360.0)

            img_cands = []
            for b, s, l in zip(boxes, scores, labels):
                cid = int(l)
                sc = float(s)
                if cid < 0 or cid >= NUM_CLASSES:
                    continue
                if sc < score_thresh or sc <= 0.0 or sc > 1.0 or np.isnan(sc):
                    continue

                # Box could be [x1, y1, x2, y2] absolute or normalized
                if np.max(b) > 1.5:
                    # Absolute pixel coordinates
                    x1 = np.clip(b[0] / W, 0.0, 1.0)
                    y1 = np.clip(b[1] / H, 0.0, 1.0)
                    x2 = np.clip(b[2] / W, 0.0, 1.0)
                    y2 = np.clip(b[3] / H, 0.0, 1.0)
                else:
                    # Already normalized [0, 1]
                    x1 = np.clip(b[0], 0.0, 1.0)
                    y1 = np.clip(b[1], 0.0, 1.0)
                    x2 = np.clip(b[2], 0.0, 1.0)
                    y2 = np.clip(b[3], 0.0, 1.0)

                img_cands.append((sc, cid, np.array([x1, y1, x2, y2], dtype=np.float32)))

            img_cands.sort(key=lambda x: -x[0])
            preds_by_stem[stem] = img_cands[:max_boxes_per_image]

        return preds_by_stem

    def evaluate(
        self,
        predictions: Union[str, Dict],
        max_boxes_per_image: int = 100,
        score_thresh: float = 0.0,
        id_to_stem_map: Optional[Dict] = None,
    ) -> Dict:
        """
        Runs the full official competition metric calculation.
        """
        if isinstance(predictions, str):
            preds_by_stem = self.parse_predictions_from_txt_dir(
                predictions,
                max_boxes_per_image=max_boxes_per_image,
                score_thresh=score_thresh,
            )
        elif isinstance(predictions, dict):
            preds_by_stem = self.parse_predictions_from_dict(
                predictions,
                max_boxes_per_image=max_boxes_per_image,
                score_thresh=score_thresh,
                id_to_stem_map=id_to_stem_map,
            )
        else:
            raise TypeError("predictions must be a path string or a dictionary.")

        # Re-organize predictions by category
        preds_by_cls = defaultdict(list)
        pred_counts_by_cls = defaultdict(int)
        total_predictions = 0

        for stem, cands in preds_by_stem.items():
            for sc, cid, box in cands:
                preds_by_cls[cid].append((sc, stem, box))
                pred_counts_by_cls[cid] += 1
                total_predictions += 1

        # Per-class AP computation across all 10 IoU thresholds
        ap_matrix = np.zeros((NUM_CLASSES, len(self.iou_thresholds)), dtype=np.float32)

        for cid in range(NUM_CLASSES):
            N_gt = self.gt_counts_by_cls[cid]
            cls_preds = sorted(preds_by_cls[cid], key=lambda x: -x[0])

            if N_gt == 0:
                # No GT targets for this class
                ap_matrix[cid, :] = 0.0
                continue

            if len(cls_preds) == 0:
                # No predictions made for this class
                ap_matrix[cid, :] = 0.0
                continue

            for t_idx, t in enumerate(self.iou_thresholds):
                matched_gt = defaultdict(set)
                tp = np.zeros(len(cls_preds), dtype=np.float32)
                fp = np.zeros(len(cls_preds), dtype=np.float32)

                for k, (score, stem, box) in enumerate(cls_preds):
                    gt_boxes = self.gts_by_stem_cls[stem][cid]
                    if len(gt_boxes) == 0:
                        fp[k] = 1.0
                        continue

                    ious = box_iou_numpy(box, np.array(gt_boxes, dtype=np.float32))

                    # Greedy match to best unmatched GT with IoU >= t
                    best_iou = -1.0
                    best_idx = -1
                    for g_idx, iou_val in enumerate(ious):
                        if g_idx not in matched_gt[stem] and iou_val >= t and iou_val > best_iou:
                            best_iou = iou_val
                            best_idx = g_idx

                    if best_idx >= 0:
                        tp[k] = 1.0
                        matched_gt[stem].add(best_idx)
                    else:
                        fp[k] = 1.0

                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                recall = tp_cum / max(N_gt, 1)
                precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

                # 101-point Interpolated AP
                p_interp = np.zeros(len(self.recall_thresholds), dtype=np.float32)
                for r_idx, r in enumerate(self.recall_thresholds):
                    mask = recall >= r
                    if np.any(mask):
                        p_interp[r_idx] = np.max(precision[mask])
                    else:
                        p_interp[r_idx] = 0.0

                ap_matrix[cid, t_idx] = np.mean(p_interp)

        # Macro averages
        # mAP(t) = 1/12 * sum_{c} AP_c(t)
        map_per_threshold = np.mean(ap_matrix, axis=0)  # [10]
        # mAP@50-95 = 1/10 * sum_{t} mAP(t)
        overall_map_50_95 = float(np.mean(map_per_threshold))
        overall_map_50 = float(map_per_threshold[0])  # t = 0.50
        overall_map_75 = float(map_per_threshold[5])  # t = 0.75

        per_class_results = {}
        for cid in range(NUM_CLASSES):
            cname = ID2NAME[cid]
            per_class_results[cname] = {
                "class_id": cid,
                "ap50_95": float(np.mean(ap_matrix[cid, :])),
                "ap50": float(ap_matrix[cid, 0]),
                "ap75": float(ap_matrix[cid, 5]),
                "gt_count": int(self.gt_counts_by_cls[cid]),
                "pred_count": int(pred_counts_by_cls[cid]),
                "pred_gt_ratio": float(pred_counts_by_cls[cid] / max(self.gt_counts_by_cls[cid], 1)),
            }

        return {
            "mAP_50_95": overall_map_50_95,
            "mAP_50": overall_map_50,
            "mAP_75": overall_map_75,
            "map_per_threshold": [float(v) for v in map_per_threshold],
            "total_predictions": total_predictions,
            "total_gt": self.total_gt,
            "num_evaluated_images": len(preds_by_stem),
            "per_class": per_class_results,
        }

    def print_report(self, results: Dict, title: str = "Official Competition Metric Report"):
        """
        Prints a formatted report matching the official tournament specification.
        """
        print("\n" + "=" * 92)
        print(f" {title.upper()} ".center(92, "="))
        print("=" * 92)
        print(f"{'Class':<14} | {'GT':<5} | {'Pred':<6} | {'Ratio':<6} | {'AP@50:95':<10} | {'AP@50':<8} | {'AP@75':<8}")
        print("-" * 92)

        for cid in range(NUM_CLASSES):
            cname = ID2NAME[cid]
            info = results["per_class"][cname]
            flag = ""
            if info["pred_gt_ratio"] > 4.0 and info["gt_count"] > 10:
                flag = " [!] OVER-DET"
            elif info["pred_gt_ratio"] < 0.3 and info["gt_count"] > 10:
                flag = " [?] UNDER-DET"

            print(
                f"{cname:<14s} | {info['gt_count']:<5d} | {info['pred_count']:<6d} | "
                f"{info['pred_gt_ratio']:<6.2f} | {info['ap50_95']:<10.4f} | "
                f"{info['ap50']:<8.4f} | {info['ap75']:<8.4f}{flag}"
            )

        print("-" * 92)
        print(
            f"{'OVERALL (12-CLASS MACRO)':<37} | {results['mAP_50_95']:<10.4f} | "
            f"{results['mAP_50']:<8.4f} | {results['mAP_75']:<8.4f}"
        )
        print("=" * 92)
        print(f" Evaluated Images: {results['num_evaluated_images']} | Total Predictions: {results['total_predictions']} (avg {results['total_predictions']/max(1, results['num_evaluated_images']):.1f} boxes/img)")
        print(f" Total Ground Truth Targets: {results['total_gt']}")
        print(f" Official Competition Score (mAP@50-95): {results['mAP_50_95'] * 100.0:.2f} %")
        print("=" * 92 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Official Competition Metric Evaluator")
    parser.add_argument("-p", "--pred", required=True, help="Path to predictions folder or .zip file")
    parser.add_argument(
        "-g", "--gt",
        default="work_dir/deimv2_dataset/valid/annotations/instances_val.json",
        help="Path to instances_val.json or valid/labels_txt folder",
    )
    parser.add_argument("--max-boxes", type=int, default=100, help="Max boxes per image (default: 100)")
    parser.add_argument("--score-thresh", type=float, default=0.0, help="Optional score filter before eval")
    args = parser.parse_args()

    evaluator = OfficialCompetitionEvaluator(args.gt)
    results = evaluator.evaluate(
        args.pred,
        max_boxes_per_image=args.max_boxes,
        score_thresh=args.score_thresh,
    )
    evaluator.print_report(results, title=f"Evaluation: {os.path.basename(args.pred)}")


if __name__ == "__main__":
    main()
