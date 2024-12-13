from ultralytics import YOLO 

model = YOLO('yolov8x')
#For video
result = model.track('input_videos/input_video.mp4',conf=0.2, save=True)
# For image
#result = model.track('input_videos/image.png',conf=0.2, save=True)


print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)