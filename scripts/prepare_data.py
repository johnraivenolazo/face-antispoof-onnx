from __future__ import annotations

import argparse
import os
import shutil
from multiprocessing import Pool, cpu_count
from typing import Tuple

import cv2
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset preparation (crop + resize)")
    p.add_argument("--orig_dir", required=True, help="Path to dataset root")
    p.add_argument("--crop_dir", required=True, help="Output folder for cropped images")
    p.add_argument("--size", type=int, default=224, help="Output image size (square)")
    p.add_argument(
        "--bbox_expansion_factor", type=float, default=1.5, help="Crop expansion factor"
    )
    p.add_argument(
        "--label_dir",
        default="metas/labels",
        help="Relative label folder under orig_dir (default: metas/labels)",
    )
    p.add_argument(
        "--spoof_types",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 7, 8, 9],
        help="Keep these label type codes (if applicable)",
    )
    return p.parse_args()


def _process_single_image(args_bundle: Tuple[str, str, str, int, float]) -> None:
    _, full_img_path, save_path, target_size, bbox_expansion_factor = args_bundle

    img = cv2.imread(full_img_path)
    if img is None:
        return

    original_height, original_width = img.shape[:2]

    bbox_file = full_img_path.replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
    if not os.path.exists(bbox_file):
        return

    try:
        with open(bbox_file, "r", encoding="utf-8") as f:
            line = f.readline().strip().split(" ")
            x_ref, y_ref, w_ref, h_ref = map(float, line[:4])

        x = int(x_ref * (original_width / 224))
        w = int(w_ref * (original_width / 224))
        y = int(y_ref * (original_height / 224))
        h = int(h_ref * (original_height / 224))
    except Exception:
        return

    center_x = x + w // 2
    center_y = y + h // 2
    side_len = int(max(w, h) * bbox_expansion_factor)

    x1 = center_x - side_len // 2
    y1 = center_y - side_len // 2
    x2 = x1 + side_len
    y2 = y1 + side_len

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - original_height)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - original_width)

    if pad_top or pad_bottom or pad_left or pad_right:
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return

    h_crop, w_crop = face_crop.shape[:2]
    crop_size = min(h_crop, w_crop)
    interp = cv2.INTER_LANCZOS4 if crop_size < target_size else cv2.INTER_AREA
    final_img = cv2.resize(face_crop, (target_size, target_size), interpolation=interp)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final_img)


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args()

    orig_dir = args.orig_dir
    if not orig_dir.endswith("/") and not orig_dir.endswith("\\"):
        orig_dir = orig_dir + os.sep

    label_dir = os.path.join(orig_dir, args.label_dir)

    train_json = os.path.join(label_dir, "train_label.json")
    test_json = os.path.join(label_dir, "test_label.json")

    if not os.path.exists(train_json) or not os.path.exists(test_json):
        raise FileNotFoundError(f"Label JSON files not found under: {label_dir}")

    train_df = pd.read_json(train_json, orient="index")
    test_df = pd.read_json(test_json, orient="index")

    if 40 in train_df.columns and args.spoof_types:
        train_df = train_df[train_df[40].isin(args.spoof_types)]
    if 40 in test_df.columns and args.spoof_types:
        test_df = test_df[test_df[40].isin(args.spoof_types)]

    all_files = pd.concat([train_df, test_df])

    tasks = []
    for index_path, _ in all_files.iterrows():
        full_img_path = os.path.join(orig_dir, str(index_path))
        rel_path = str(index_path)
        save_path = os.path.join(args.crop_dir, rel_path)
        tasks.append(
            (rel_path, full_img_path, save_path, args.size, args.bbox_expansion_factor)
        )

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(_process_single_image, tasks), total=len(tasks)))

    out_label_dir = os.path.join(args.crop_dir, "metas", "labels")
    os.makedirs(out_label_dir, exist_ok=True)
    shutil.copy(train_json, out_label_dir)
    shutil.copy(test_json, out_label_dir)

    print(f"done | wrote cropped dataset to: {args.crop_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
