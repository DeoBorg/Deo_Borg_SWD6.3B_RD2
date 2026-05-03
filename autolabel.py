import os
import requests
from ultralytics import YOLO
from pillow_heif import register_heif_opener
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# CONFIG
API_KEY          = os.getenv("API_KEY")
PROJECT          = os.getenv("PROJECT")
WORKSPACE        = os.getenv("WORKSPACE")
IMAGES_FOLDER    = os.getenv("IMAGES_FOLDER")
CONVERTED_FOLDER = os.getenv("CONVERTED_FOLDER")


COCO_CLASSES = {0: "person", 67: "smartphone"}

def convert_heic_to_jpg(heic_path, output_folder):
    register_heif_opener()
    filename = os.path.splitext(os.path.basename(heic_path))[0] + ".jpg"
    output_path = os.path.join(output_folder, filename)
    if not os.path.exists(output_path):
        img = Image.open(heic_path)
        img = img.convert("RGB") 
        img.save(output_path, "JPEG", quality=95)
    return output_path, filename

def upload_image(jpg_path, jpg_name):
    """Upload image to Roboflow and return the image ID."""
    url = f"https://api.roboflow.com/dataset/{PROJECT}/upload"
    with open(jpg_path, "rb") as img_file:
        response = requests.post(
            url,
            params={
                "api_key": API_KEY,
                "name": jpg_name,
                "split": "train"
            },
            files={"file": img_file}
        )
    data = response.json()
    if response.status_code == 200 and "id" in data:
        return data["id"]
    else:
        print(f"  Image upload failed for {jpg_name}: {data}")
        return None

def upload_annotations(image_id, boxes, img_width, img_height):
    """Upload annotations in YOLO format as a text file."""
    lines = []
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id not in COCO_CLASSES:
            continue
        x_center, y_center, w, h = box.xywhn[0].tolist()
        # Map COCO class id to our class index (0=person, 1=smartphone)
        our_cls = 0 if cls_id == 0 else 1
        lines.append(f"{our_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    if not lines:
        print(f"  No relevant objects detected, skipping annotations.")
        return

    # Write YOLO format annotation to a temp text file
    annotation_text = "\n".join(lines)

    url = f"https://api.roboflow.com/dataset/{PROJECT}/annotate/{image_id}"
    response = requests.post(
        url,
        params={
            "api_key": API_KEY,
            "name": image_id + ".txt",
            "format": "yolo"
        },
        headers={"Content-Type": "text/plain"},
        data=annotation_text
    )

    if response.status_code == 200:
        print(f"  Uploaded {len(lines)} annotations successfully.")
    else:
        print(f"  Annotation upload failed: {response.text}")

def main():
    os.makedirs(CONVERTED_FOLDER, exist_ok=True)
    model = YOLO("yolov8n.pt")

    heic_files = [
        f for f in os.listdir(IMAGES_FOLDER)
        if f.lower().endswith(".heic")
    ]

    print(f"Found {len(heic_files)} HEIC images. Processing...\n")

    for i, heic_name in enumerate(heic_files, 1):
        heic_path = os.path.join(IMAGES_FOLDER, heic_name)
        print(f"[{i}/{len(heic_files)}] Processing: {heic_name}")

        # Convert HEIC to JPG
        jpg_path, jpg_name = convert_heic_to_jpg(heic_path, CONVERTED_FOLDER)
        print(f"  Converted to JPG.")

        # Upload image to Roboflow
        image_id = upload_image(jpg_path, jpg_name)
        if not image_id:
            continue
        print(f"  Uploaded image. ID: {image_id}")

        # Run YOLOv8 detection
        results = model(jpg_path, verbose=False)
        boxes = results[0].boxes
        img_height, img_width = results[0].orig_shape

        # Upload annotations
        upload_annotations(image_id, boxes, img_width, img_height)
        print()

    print("Done. Go to Roboflow to review your annotations.")

if __name__ == "__main__":
    main()