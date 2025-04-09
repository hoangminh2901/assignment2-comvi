import cv2
import datetime

# Open both cameras (usually 0 and 1 for built-in and external)
cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

# Set resolution (optional)
frame_width = int(cam0.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam0.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0

# Define the codec and create VideoWriter objects
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out0 = cv2.VideoWriter('cam3_output.avi', fourcc, fps, (frame_width, frame_height))
out1 = cv2.VideoWriter('cam4_output.avi', fourcc, fps, (frame_width, frame_height))

# Check if both cameras are opened
if not cam0.isOpened() or not cam1.isOpened():
    print("One or both cameras could not be opened.")
    cam0.release()
    cam1.release()
    out0.release()
    out1.release()
    exit()

print("Recording started. Press 'q' to stop.")

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if ret0 and ret1:
        # Optional: timestamp
        # timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # cv2.putText(frame0, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # cv2.putText(frame1, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        out0.write(frame0)
        out1.write(frame1)

        # Show the frames
        cv2.imshow('Camera 0', frame0)
        cv2.imshow('Camera 1', frame1)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to grab frames.")
        break

# Release everything
cam0.release()
cam1.release()
out0.release()
out1.release()
cv2.destroyAllWindows()
