import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os
from utils.generate_random_name import generate_random_name


model_types = None
model_diseases = None

try:
  model_types = YOLO("../models/model_types/best.pt")
  print("model_types loaded successfully!")

except Exception as e:
    print(f"model_types is not loaded: {e}")

try:
  model_diseases = YOLO("../models/model_diseases/best.pt")
  print("model_diseases loaded successfully!")

except Exception as e:
    print(f"model_diseases is not loaded: {e}")

class MyError(Exception):
    
    def __init__(self, message="Preprocessing Error"):
        self.message = message
        super().__init__(self.message)

def predict_fish(base64_image_input, conf_threshold=0.5):
    
    """
    Args:
        base64_image_input (str): image yang telah diencode menjadi base64.
        conf_threshold (float): threshold berapa battas prediksi yang diambil, default 50% confidence.

    Returns:
        str: image yang telah memiliki bonding box, telah juga diencode menjadi base64.
    """

    try:
        if not base64_image_input:
          raise MyError("Base64 image input is empty.")
        
        # Decode image
        # Cek apabila encoded memiliki prefixes "data:image/jpeg;base64"

        if "base64," in base64_image_input:
            base64_image_input = base64_image_input.split("base64,")[1]

        img_bytes = base64.b64decode(base64_image_input)

        np_arr = np.frombuffer(img_bytes, np.uint8)

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (640, 640))  # Resize di sini

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
          raise MyError("imdecode is not valid.")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except Exception as e:
        raise MyError(f"Error decoding Base64 image: {e}")

    # Simpan image temporari untuk prediksi
    img_name = generate_random_name() + ".jpg"
    temp_img_path = f"assets/images/{img_name}"
    success = cv2.imwrite(temp_img_path, img) 

    if not success:
      raise MyError("Failed to write temporary image file.")

    # Lakukan prediksi jenis
    results_types = model_types.predict(source=temp_img_path, conf=conf_threshold, save=False, verbose=False, imgsz=640)

    detections = []

    for type in results_types:
        boxes = type.boxes
        if not boxes:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model_types.names[cls_id]

            if class_name in ["bawal", "cupang", "koi", "lele", "mujair"]:
                coords = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)

                detection_info = {
                    'coords': (x1, y1, x2, y2),
                    'conf': conf,
                    'class_name': class_name,
                }

                detections.append(detection_info)
                

    detections.sort(key=lambda x: x['conf'], reverse=True)
    top_1_type = detections[0]

    fish_condition = None
    fish_type = None

    if not top_1_type:
        print("No 'fish', or model_types confidence really low")
        return None
    
    else:
        x1, y1, x2, y2 = top_1_type['coords']
        conf = top_1_type['conf']
        class_name = top_1_type['class_name']

        fish_type = {"type": class_name, "confidence": float(conf)} 
    
        color = (255, 0, 0)

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 1)

        label = f'{class_name}: {conf:.2f}'
        cv2.putText(img_rgb, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        print(f"[{class_name}] Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf:.2f}")

        cropped_img = img[y1:y2, x1:x2]
        temp_img_path = f"assets/images/{img_name}"
        success = cv2.imwrite(temp_img_path, cropped_img)

        results_diseases = model_diseases(temp_img_path, save=False, verbose=False, imgsz=640)

        condition = results_diseases[0]

        if condition.probs is not None:
            probs = condition.probs.cpu().numpy().data # Konversi ke numpy array

            top_prob_idx = np.argmax(probs)
            confidence_score = float(probs[top_prob_idx]) # Konversi ke float biasa

            predicted_class_name = str(model_diseases.names[top_prob_idx]) # Konversi ke string biasa

            fish_condition = { "condition": predicted_class_name, "confidence": float(confidence_score)}  

    img_processed_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode('.jpg', img_processed_bgr)
    base64_output_image = base64.b64encode(buffer).decode('utf-8')

    if os.path.exists(temp_img_path):
      os.remove(temp_img_path)

    return { "fish-condition": fish_condition, "fish-type": fish_type, "img" : base64_output_image }