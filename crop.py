import os
import json
import cv2
import argparse
import numpy as np
from tqdm import tqdm


def add_padding_to_coordinates(args, x1, x2, y1, y2):
    """Add padding to bounding box coordinates."""
    try:
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 0 or box_height < 0:
            raise ValueError("Negative width or height")

        if args.padding_percent == 0:
            w_pad = args.w_pad
            h_pad = args.h_pad
        else:
            w_pad = int(box_width * (args.padding_percent * 0.75) / 100)
            h_pad = int(box_height * (args.padding_percent * 0.75) / 100)

        x1 = max(0, x1 - w_pad)
        y1 = max(0, y1 - h_pad)
        x2 = x2 + w_pad
        y2 = y2 + h_pad

    except Exception as e:
        print(f"[SKIP] Padding error: {e}. Coordinates: {(x1, y1, x2, y2)}")
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return x1, y1, x2, y2


def resize_with_padding(image, target_w, target_h):
    """Resize image while preserving aspect ratio and add padding."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

    new_image = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    offset_y = (target_h - resized_image.shape[0]) // 2
    offset_x = (target_w - resized_image.shape[1]) // 2
    new_image[offset_y:offset_y + resized_image.shape[0],
              offset_x:offset_x + resized_image.shape[1]] = resized_image
    return new_image


def find_image_path(image_folder, image_filename):
    """Find image path, trying multiple extensions if needed."""
    img_path = os.path.join(image_folder, image_filename)
    if os.path.exists(img_path):
        return img_path

    base, _ = os.path.splitext(img_path)
    for ext in ['.JPG', '.jpg', '.jpeg', '.png']:
        alt_path = base + ext
        if os.path.exists(alt_path):
            return alt_path
    return None


def parse_points(points_obj, image_id, object_id):
    """Parse points information from JSON."""
    if isinstance(points_obj, dict):
        return points_obj
    elif isinstance(points_obj, str):
        try:
            points = json.loads(points_obj)
            if not isinstance(points, dict):
                raise ValueError("Points is not a dict")
            return points
        except Exception as e:
            print(f"[SKIP] Failed to parse points in {image_id}, object {object_id}: {e}")
            return None
    else:
        print(f"[SKIP] Invalid points type in {image_id}, object {object_id}: {type(points_obj)}")
        return None


def process_annotations(json_path, image_folder, output_root, args):
    """Process JSON annotations and save cropped images."""
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    os.makedirs(output_root, exist_ok=True)

    for image_id, items in tqdm(annotations.items(), desc="Processing images"):
        if not isinstance(image_id, str) or image_id.strip().lower() in ["null", "none", ""]:
            continue

        image_filename = f"{image_id}.JPG"
        image_path = find_image_path(image_folder, image_filename)
        if image_path is None:
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        for object_id, data in items.items():
            if not isinstance(data, dict) or "points" not in data:
                continue

            points = parse_points(data["points"], image_id, object_id)
            if points is None or "top_left" not in points or "bottom_right" not in points:
                continue
            if "sku_id" not in data or "split_id" not in data:
                continue

            try:
                sku_id = str(data["sku_id"])
                split_id = str(data["split_id"])
                x1, y1 = map(float, points["top_left"])
                x2, y2 = map(float, points["bottom_right"])
            except Exception:
                continue

            x1, y1, x2, y2 = add_padding_to_coordinates(args, x1, x2, y1, y2)
            x1 = int(max(0, min(x1, w - 1)))
            x2 = int(max(0, min(x2, w)))
            y1 = int(max(0, min(y1, h - 1)))
            y2 = int(max(0, min(y2, h)))

            if x2 <= x1 or y2 <= y1:
                continue

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            if args.post_padding_method == "resize":
                cropped = cv2.resize(cropped, (args.target_w, args.target_h))
            elif args.post_padding_method == "resize_with_padding":
                cropped = resize_with_padding(cropped, args.target_w, args.target_h)

            save_dir = os.path.join(output_root, split_id, sku_id)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{image_id}_{object_id}.JPG")
            cv2.imwrite(save_path, cropped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON annotations")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder with original images")
    parser.add_argument("--output_root", type=str, required=True, help="Folder to save cropped images")

    parser.add_argument("--padding_percent", type=int, default=10, help="Padding percent around bounding boxes")
    parser.add_argument("--w_pad", type=int, default=0, help="Fixed width padding (pixels)")
    parser.add_argument("--h_pad", type=int, default=0, help="Fixed height padding (pixels)")
    parser.add_argument("--post_padding_method", type=str, default="resize",
                        choices=["resize", "resize_with_padding", "none"], help="Method after cropping")
    parser.add_argument("--target_w", type=int, default=224, help="Target width after resizing")
    parser.add_argument("--target_h", type=int, default=224, help="Target height after resizing")

    args = parser.parse_args()
    process_annotations(args.json_path, args.image_folder, args.output_root, args)
