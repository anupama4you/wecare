# sudo apt-get install libopencv-dev python3-opencv
# sudo apt-get install python3-tk
# sudo apt install zlib1g-dev libjpeg-dev libpng-dev
# pip3 install Pillow
# sudo apt-get install python3-pil.imagetk

import cv2
import os
from tkinter import *
from PIL import Image, ImageTk

root = Tk()
root.title("Face Data Collection GUI")

# Variables for user input and count display
user_var = StringVar()
count_var = StringVar()
count_var.set("Frames Captured: 0")

def start_capture():
    count = 0
    base_dir = "collected_faces"
    user_folder = os.path.join(base_dir, user_var.get())  # Create a sub-folder under 'collected_faces' with the username

    # Create the directories if they don't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame for tkinter and update the image label
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame_rgb)
        image_tk = ImageTk.PhotoImage(image=im_pil)
        cam_label.imgtk = image_tk  # prevent GC on `imgtk`
        cam_label.config(image=image_tk)

        # Save the image
        img_name = os.path.join(user_folder, "{}.png".format(count))
        cv2.imwrite(img_name, frame)
        count += 1
        count_var.set(f"Frames Captured: {count}")

        if count >= 101:
            break

        root.update_idletasks()  # Refresh GUI
        root.update()  # Wait for a while before capturing the next frame

cap = cv2.VideoCapture(0)

# GUI Layout
frame_left = Frame(root)
frame_left.pack(side=LEFT, padx=10, pady=10)

frame_right = Frame(root)
frame_right.pack(side=RIGHT, padx=10, pady=10)

# Camera display on the left
cam_label = Label(frame_left)
cam_label.pack(pady=20)

# Elements on the right
Label(frame_right, text="Enter UserID or Name:").pack(pady=20)
Entry(frame_right, textvariable=user_var).pack(pady=20)
Button(frame_right, text="Start Capture", command=start_capture).pack(pady=20)
Label(frame_right, textvariable=count_var).pack(pady=20)
Button(frame_right, text="Finish", command=root.destroy).pack(pady=20)

root.mainloop()

cap.release()
