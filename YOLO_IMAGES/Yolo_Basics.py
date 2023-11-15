# from ultralytics import YOLO
# import cv2

# model = YOLO('yolov8n.pt')

# #output_path = "output_image.png"

# results = model("Images/6.png", show=True)

# #results = model("Images/1.png" , show = True)

# cv2.waitKey(0)

from ultralytics import YOLO
import cv2

model = YOLO('../YOLO_WEIGHT/yolov8n.pt')

# Provide the path where you want to save the output image
output_path = "output_image.png"

results = model("../Images/13.png", show=True, save=True)  # Use save=True

# cv2.imshow("Annotated Image", results.imgs[0])  # Uncomment if you want to display the image

# Wait for a key press
cv2.waitKey(0)

# Save the annotated image
if results and results.save(output_path):
    print(f"Output image saved at {output_path}")
