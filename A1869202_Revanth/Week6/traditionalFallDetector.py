import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)  # Change this if using a different camera source

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# For morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

previous_heights = {}
consecutive_frames = {}

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Applying Gaussian Blur
    fgmask = fgbg.apply(blurred_frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 8000:  # Increased threshold for contour area
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect fall
        current_topmost = y
        if id(contour) in previous_heights:
            if previous_heights[id(contour)] - current_topmost > 50:  # Adjusted the fall threshold
                consecutive_frames[id(contour)] = consecutive_frames.get(id(contour), 0) + 1
                if consecutive_frames[id(contour)] > 3:  
                    cv2.putText(frame, 'Fall detected!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                consecutive_frames[id(contour)] = 0

        previous_heights[id(contour)] = current_topmost

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = 1.0 / elapsed_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
