# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO("yolov8l.pt")  # Use fine-tuned model if trained

# # Define class labels
# fruit_labels = ["apple", "banana", "orange", "grapes", "pineapple", "watermelon"]

# # Function to process image and detect objects
# def detect_fruits(image):
#     # Convert PIL Image to OpenCV format (BGR) if needed
#     if isinstance(image, np.ndarray):
#         # If already NumPy array, ensure correct format
#         image_rgb = image.copy()
#         if len(image.shape) == 3 and image.shape[2] == 3:
#             image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
#         else:
#             image_bgr = image_rgb
#     else:
#         # Convert PIL Image to NumPy array
#         image_rgb = np.array(image)
#         image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
#     results = model(image_bgr)  # Run YOLOv8 on image
#     detections = []
    
#     for result in results:
#         boxes = result.boxes
#         for i in range(len(boxes)):
#             x1, y1, x2, y2 = map(int, boxes.xyxy[i])  # Get bounding box coordinates
#             cls_id = int(boxes.cls[i])
#             conf = float(boxes.conf[i])
            
#             # Check if the detected class is in our fruit list or is a reference object
#             if cls_id == 67:  # 67 is cell phone in COCO dataset
#                 label = "cell_phone"  # More explicit label
#                 detections.append((label, conf, (x1, y1, x2, y2)))
#             elif 0 <= cls_id < 80:  # COCO dataset has 80 classes
#                 coco_names = model.names
#                 coco_label = coco_names[cls_id].lower()
                
#                 # Map COCO fruit classes to our fruit_labels if they match
#                 if coco_label in fruit_labels:
#                     detections.append((coco_label, conf, (x1, y1, x2, y2)))
    
#     return detections, image_rgb  # Return the RGB image for display

# # Known object dimensions (in cm)
# REFERENCE_WIDTH_CM = 7.5  # Average cell phone width in cm (corrected)
# PIXELS_PER_CM = None  # To be calculated

# def estimate_calories(detections):
#     global PIXELS_PER_CM
    
#     reference_box = None
#     fruit_boxes = []
    
#     for label, confidence, (x1, y1, x2, y2) in detections:
#         if label.lower() == "cell_phone":  # Updated to match the new label
#             reference_box = (x2 - x1)  # Width of reference object
#         else:
#             fruit_boxes.append((label, confidence, x1, y1, x2, y2))
    
#     if reference_box:
#         PIXELS_PER_CM = reference_box / REFERENCE_WIDTH_CM  # Compute scale
    
#     # Estimating fruit size and calories (per 100g)
#     fruit_calories = {
#         "apple": 52, 
#         "banana": 89, 
#         "orange": 47, 
#         "grapes": 69, 
#         "pineapple": 50, 
#         "watermelon": 30
#     }
    
#     results = []
    
#     for label, confidence, x1, y1, x2, y2 in fruit_boxes:
#         if PIXELS_PER_CM:
#             width_cm = (x2 - x1) / PIXELS_PER_CM  # Convert pixels to cm
#             # More accurate weight estimation based on fruit type and dimensions
#             volume_estimate = width_cm ** 3  # Simple approximation
            
#             # Different fruits have different densities and shapes
#             if label == "apple":
#                 estimated_weight_g = volume_estimate * 0.8
#             elif label == "banana":
#                 estimated_weight_g = volume_estimate * 0.6
#             elif label == "orange":
#                 estimated_weight_g = volume_estimate * 0.9
#             elif label == "grapes":
#                 estimated_weight_g = volume_estimate * 0.7
#             elif label == "pineapple":
#                 estimated_weight_g = volume_estimate * 1.0
#             elif label == "watermelon":
#                 estimated_weight_g = volume_estimate * 0.95
#             else:
#                 estimated_weight_g = volume_estimate * 0.8  # Default
            
#             # Calculate calories based on weight
#             if label in fruit_calories:
#                 calorie_estimate = (fruit_calories[label] / 100) * estimated_weight_g
#                 results.append({
#                     "label": label, 
#                     "confidence": confidence,
#                     "width_cm": width_cm, 
#                     "estimated_weight_g": estimated_weight_g,
#                     "calories": calorie_estimate
#                 })
#             else:
#                 results.append({
#                     "label": label, 
#                     "confidence": confidence,
#                     "error": "Calorie information not available"
#                 })
#         else:
#             results.append({
#                 "label": label, 
#                 "confidence": confidence,
#                 "error": "No cell phone detected for scale reference"
#             })
    
#     return results

# # Function to visualize results
# def draw_boxes(image, detections, calorie_results):
#     image_copy = image.copy()  # Ensure original image is not modified
    
#     # Create a mapping from label to calorie result
#     calorie_map = {}
#     for result in calorie_results:
#         if "label" in result:
#             calorie_map[result["label"]] = result
    
#     for label, confidence, (x1, y1, x2, y2) in detections:
#         if label == "cell_phone":
#             # Draw reference object in blue
#             color = (255, 0, 0)  # Blue in RGB
#             text = f"Phone (reference): {confidence:.2f}"
#         else:
#             # Draw fruits in green
#             color = (0, 255, 0)  # Green in RGB
            
#             # Add calorie information if available
#             if label in calorie_map and "calories" in calorie_map[label]:
#                 calories = calorie_map[label]["calories"]
#                 weight = calorie_map[label]["estimated_weight_g"]
#                 text = f"{label}: {confidence:.2f} | {calories:.1f} kcal ({weight:.0f}g)"
#             else:
#                 text = f"{label}: {confidence:.2f}"
        
#         # Draw the bounding box
#         cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
#         # Add text with a background for better visibility
#         text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#         cv2.rectangle(image_copy, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
#         cv2.putText(image_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
#     return image_copy

# # Streamlit UI
# def run():
#     st.title("Fruit Detection and Calorie Estimation 🍏🍌🍊")
    
#     st.write("""
#     Upload an image containing fruits to detect them and estimate calories.
#     Include a cell phone in the image as a size reference for accurate measurements.
#     """)
    
#     st.info("The system uses an average cell phone width (7.5 cm) as a reference. For best results, make sure the entire width of the phone is visible in the image.")
    
#     img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
#     if img_file is not None:
#         image = Image.open(img_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         with st.spinner("Detecting fruits..."):
#             # Detect objects
#             detections, image_rgb = detect_fruits(image)
            
#             # Check if a cell phone was detected
#             phone_detected = any(label == "cell_phone" for label, _, _ in detections)
            
#             if not phone_detected:
#                 st.warning("⚠️ No cell phone detected in the image. Calorie estimates will not be accurate without a reference object.")
            
#             # Estimate calories
#             calorie_results = estimate_calories(detections)
            
#             # Draw bounding boxes
#             image_with_boxes = draw_boxes(image_rgb, detections, calorie_results)
            
#             # Display results
#             st.image(image_with_boxes, caption="Detected Fruits with Calories", use_column_width=True)
            
#             # Display detailed results
#             if calorie_results:
#                 st.subheader("Detailed Results:")
#                 for result in calorie_results:
#                     if "calories" in result:
#                         st.write(f"**{result['label'].capitalize()}**:")
#                         st.write(f"- Confidence: {result['confidence']:.2f}")
#                         st.write(f"- Estimated width: {result['width_cm']:.1f} cm")
#                         st.write(f"- Estimated weight: {result['estimated_weight_g']:.0f} g")
#                         st.write(f"- Estimated calories: {result['calories']:.1f} kcal")
#                         st.write("---")
#                     elif "error" in result:
#                         st.write(f"**{result['label'].capitalize()}**: {result['error']}")
#                         st.write("---")
#             else:
#                 st.warning("No fruits detected or unable to estimate calories.")

# if __name__ == "__main__":
#     run()





# Better Version


import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter

# Load YOLO model
model = YOLO("yolov8l.pt")  # Use fine-tuned model if trained

# Define class labels
fruit_labels = ["apple", "banana", "orange", "grapes", "pineapple", "watermelon"]

# Function to process image and detect objects
def detect_fruits(image):
    # Convert PIL Image to OpenCV format (BGR) if needed
    if isinstance(image, np.ndarray):
        # If already NumPy array, ensure correct format
        image_rgb = image.copy()
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_rgb
    else:
        # Convert PIL Image to NumPy array
        image_rgb = np.array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    results = model(image_bgr)  # Run YOLOv8 on image
    detections = []
    
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])  # Get bounding box coordinates
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            
            # Check if the detected class is in our fruit list or is a reference object
            if cls_id == 67:  # 67 is cell phone in COCO dataset
                label = "cell_phone"  # More explicit label
                detections.append((label, conf, (x1, y1, x2, y2)))
            elif 0 <= cls_id < 80:  # COCO dataset has 80 classes
                coco_names = model.names
                coco_label = coco_names[cls_id].lower()
                
                # Map COCO fruit classes to our fruit_labels if they match
                if coco_label in fruit_labels:
                    detections.append((coco_label, conf, (x1, y1, x2, y2)))
    
    return detections, image_rgb  # Return the RGB image for display

# Known object dimensions (in cm)
REFERENCE_WIDTH_CM = 7.5  # Average cell phone width in cm (corrected)
PIXELS_PER_CM = None  # To be calculated

def estimate_calories(detections):
    global PIXELS_PER_CM
    
    reference_box = None
    fruit_boxes = []
    
    for label, confidence, (x1, y1, x2, y2) in detections:
        if label.lower() == "cell_phone":  # Updated to match the new label
            reference_box = (x2 - x1)  # Width of reference object
        else:
            fruit_boxes.append((label, confidence, x1, y1, x2, y2))
    
    if reference_box:
        PIXELS_PER_CM = reference_box / REFERENCE_WIDTH_CM  # Compute scale
    
    # Estimating fruit size and calories (per 100g)
    fruit_calories = {
        "apple": 52, 
        "banana": 89, 
        "orange": 47, 
        "grapes": 69, 
        "pineapple": 50, 
        "watermelon": 30
    }
    
    results = []
    
    for label, confidence, x1, y1, x2, y2 in fruit_boxes:
        if PIXELS_PER_CM:
            width_cm = (x2 - x1) / PIXELS_PER_CM  # Convert pixels to cm
            # More accurate weight estimation based on fruit type and dimensions
            volume_estimate = width_cm ** 3  # Simple approximation
            
            # Different fruits have different densities and shapes
            if label == "apple":
                estimated_weight_g = volume_estimate * 0.8
            elif label == "banana":
                estimated_weight_g = volume_estimate * 0.6
            elif label == "orange":
                estimated_weight_g = volume_estimate * 0.9
            elif label == "grapes":
                estimated_weight_g = volume_estimate * 0.7
            elif label == "pineapple":
                estimated_weight_g = volume_estimate * 1.0
            elif label == "watermelon":
                estimated_weight_g = volume_estimate * 0.95
            else:
                estimated_weight_g = volume_estimate * 0.8  # Default
            
            # Calculate calories based on weight
            if label in fruit_calories:
                calorie_estimate = (fruit_calories[label] / 100) * estimated_weight_g
                results.append({
                    "label": label, 
                    "confidence": confidence,
                    "width_cm": width_cm, 
                    "estimated_weight_g": estimated_weight_g,
                    "calories": calorie_estimate
                })
            else:
                results.append({
                    "label": label, 
                    "confidence": confidence,
                    "error": "Calorie information not available"
                })
        else:
            results.append({
                "label": label, 
                "confidence": confidence,
                "error": "No cell phone detected for scale reference"
            })
    
    return results

# Function to visualize results
def draw_boxes(image, detections, calorie_results):
    image_copy = image.copy()  # Ensure original image is not modified
    
    # Create a mapping from label to calorie result
    calorie_map = {}
    for result in calorie_results:
        if "label" in result:
            calorie_map[result["label"]] = result
    
    for label, confidence, (x1, y1, x2, y2) in detections:
        if label == "cell_phone":
            # Draw reference object in blue
            color = (255, 0, 0)  # Blue in RGB
            text = f"Phone (reference): {confidence:.2f}"
        else:
            # Draw fruits in green
            color = (0, 255, 0)  # Green in RGB
            
            # Add calorie information if available
            if label in calorie_map and "calories" in calorie_map[label]:
                calories = calorie_map[label]["calories"]
                weight = calorie_map[label]["estimated_weight_g"]
                text = f"{label}: {confidence:.2f} | {calories:.1f} kcal ({weight:.0f}g)"
            else:
                text = f"{label}: {confidence:.2f}"
        
        # Draw the bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Add text with a background for better visibility
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image_copy, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image_copy

# Function to count fruits by type
def count_fruits(detections):
    fruit_counter = Counter()
    
    for label, _, _ in detections:
        if label != "cell_phone":  # Don't count the reference object
            fruit_counter[label] += 1
    
    return fruit_counter

# Streamlit UI
def run():
    st.title("Fruit Detection and Calorie Estimation 🍏🍌🍊")
    
    st.write("""
    Upload an image containing fruits to detect them and estimate calories.
    Include a cell phone in the image as a size reference for accurate measurements.
    """)
    
    st.info("The system uses an average cell phone width (7.5 cm) as a reference. For best results, make sure the entire width of the phone is visible in the image.")
    
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Detecting fruits..."):
            # Detect objects
            detections, image_rgb = detect_fruits(image)
            
            # Count fruits by type
            fruit_counts = count_fruits(detections)
            
            # Check if a cell phone was detected
            phone_detected = any(label == "cell_phone" for label, _, _ in detections)
            
            if not phone_detected:
                st.warning("⚠️ No cell phone detected in the image. Calorie estimates will not be accurate without a reference object.")
            
            # Estimate calories
            calorie_results = estimate_calories(detections)
            
            # Draw bounding boxes
            image_with_boxes = draw_boxes(image_rgb, detections, calorie_results)
            
            # Display results
            st.image(image_with_boxes, caption="Detected Fruits with Calories", use_column_width=True)
            
            # Display fruit count summary
            if fruit_counts:
                st.subheader("Fruit Count Summary:")
                for fruit, count in fruit_counts.items():
                    st.write(f"**{fruit.capitalize()}**: {count} detected")
                st.write("---")
            
            # Calculate total calories per fruit type
            if calorie_results:
                st.subheader("Calorie Summary by Fruit Type:")
                calorie_by_type = {}
                
                for result in calorie_results:
                    if "calories" in result:
                        fruit_type = result["label"]
                        if fruit_type not in calorie_by_type:
                            calorie_by_type[fruit_type] = {
                                "count": 0,
                                "total_calories": 0,
                                "total_weight": 0
                            }
                        
                        calorie_by_type[fruit_type]["count"] += 1
                        calorie_by_type[fruit_type]["total_calories"] += result["calories"]
                        calorie_by_type[fruit_type]["total_weight"] += result["estimated_weight_g"]
                
                for fruit_type, data in calorie_by_type.items():
                    st.write(f"**{fruit_type.capitalize()}** ({data['count']} detected):")
                    st.write(f"- Total calories: {data['total_calories']:.1f} kcal")
                    st.write(f"- Total estimated weight: {data['total_weight']:.0f} g")
                    st.write(f"- Average calories per item: {data['total_calories']/data['count']:.1f} kcal")
                    st.write("---")
            
            # Display detailed results
            if calorie_results:
                st.subheader("Detailed Results (Individual Items):")
                for i, result in enumerate(calorie_results):
                    if "calories" in result:
                        st.write(f"**{result['label'].capitalize()} #{i+1}**:")
                        st.write(f"- Confidence: {result['confidence']:.2f}")
                        st.write(f"- Estimated width: {result['width_cm']:.1f} cm")
                        st.write(f"- Estimated weight: {result['estimated_weight_g']:.0f} g")
                        st.write(f"- Estimated calories: {result['calories']:.1f} kcal")
                        st.write("---")
                    elif "error" in result:
                        st.write(f"**{result['label'].capitalize()} #{i+1}**: {result['error']}")
                        st.write("---")
            else:
                st.warning("No fruits detected or unable to estimate calories.")

if __name__ == "__main__":
    run()