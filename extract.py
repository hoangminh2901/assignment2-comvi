import cv2

# Open both videos
cap_left = cv2.VideoCapture('cam3_output.avi')
cap_right = cv2.VideoCapture('cam4_output.avi')

# Set the time in seconds where you want to extract the frame
time_in_seconds = 5  # for example, 3 seconds into the video

# Get frame rate (FPS) to calculate frame number
fps_left = cap_left.get(cv2.CAP_PROP_FPS)
fps_right = cap_right.get(cv2.CAP_PROP_FPS)

# Compute the frame number for both videos
frame_number_left = int(fps_left * time_in_seconds)
frame_number_right = int(fps_right * time_in_seconds)

# Set the video position to that frame
cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_number_left)
cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_number_right)

# Read the frame
ret_left, frame_left = cap_left.read()
ret_right, frame_right = cap_right.read()

if ret_left and ret_right:
    cv2.imwrite('frame_left.jpg', frame_left)
    cv2.imwrite('frame_right.jpg', frame_right)
    print("Frames extracted successfully.")
else:
    print("Failed to read one or both frames.")

# Release the video objects
cap_left.release()
cap_right.release()
