# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import math
from typing import List, Tuple

import cv2
from data import Person
import numpy as np

# map edges to a RGB color
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}

# A list of distictive colors
COLOR_LIST = [
    (47, 79, 79),
    (139, 69, 19),
    (0, 128, 0),
    (0, 0, 139),
    (255, 0, 0),
    (255, 215, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (30, 144, 255),
    (255, 228, 181),
    (255, 105, 180),
]

def calculate_CoG(total_x, total_y, num_points):
    """Calculate the Center of Gravity (CoG) using accumulated values."""
    return (total_x // num_points, total_y // num_points) if num_points else (0, 0)

def visualize(
    image: np.ndarray,
    list_persons: List[Person],
    keypoint_color: Tuple[int, ...] = None,
    keypoint_threshold: float = 0.10,
    instance_threshold: float = 0.1,
) -> np.ndarray:
  """Draws landmarks and edges on the input image and return it.

  Args:
    image: The input RGB image.
    list_persons: The list of all "Person" entities to be visualize.
    keypoint_color: the colors in which the landmarks should be plotted.
    keypoint_threshold: minimum confidence score for a keypoint to be drawn.
    instance_threshold: minimum confidence score for a person to be drawn.

  Returns:
    Image with keypoints and edges.
  """
  # Divide keypoints into upper, lower, and whole body
  upper_body_parts = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER","LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"]
  lower_body_parts = ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
  prev_CoGWhole_y = 0 # Setting initial COG of whole body to be 0 -> Y-coordinate because we dont want initially system when starts, it feels already patient is falling.
  prev_CoGUpper_y = 0

  for person in list_persons:
    if person.score < instance_threshold:
      continue
    
    # Initial values
    total_x_whole, total_y_whole, count_whole = 0, 0, 0
    total_x_upper, total_y_upper, count_upper = 0, 0, 0
    total_x_lower, total_y_lower, count_lower = 0, 0, 0

    keypoints = person.keypoints
    #print("KP:",keypoints)
    bounding_box = person.bounding_box
    #print("KP:",bounding_box)

    # Assign a color to visualize keypoints.
    if keypoint_color is None:
      if person.id is None:
        # If there's no person id, which means no tracker is enabled, use
        # a default color.
        person_color = (0, 255, 0)
      else:
        # If there's a person id, use different color for each person.
        person_color = COLOR_LIST[person.id % len(COLOR_LIST)]
    else:
      person_color = keypoint_color

    # Draw all the landmarks
    #print("Len(kp):",len(keypoints))
    for i in range(len(keypoints)):
      if keypoints[i].score >= keypoint_threshold:
        cv2.circle(image, keypoints[i].coordinate, 2, person_color, 4)
        #print("id:", i,"coord:",keypoints[i].coordinate, " score:", keypoints[i].score)
        
        # Add the coordinate values directly to the accumulators to find COG
        total_x_whole += keypoints[i].coordinate.x
        total_y_whole += keypoints[i].coordinate.y
        count_whole += 1
        
        if keypoints[i].body_part.name in upper_body_parts:
          # print("In Upper:", keypoints[i].body_part.name)
          total_x_upper += keypoints[i].coordinate.x
          total_y_upper += keypoints[i].coordinate.y
          count_upper += 1
        elif keypoints[i].body_part.name in lower_body_parts:
          # print("In Lower:", keypoints[i].body_part.name)
          total_x_lower += keypoints[i].coordinate.x
          total_y_lower += keypoints[i].coordinate.y
          count_lower += 1
    
    # Calculate CoG for the respective accumulators
    CoG_whole = calculate_CoG(total_x_whole, total_y_whole, count_whole)
    CoG_upper = calculate_CoG(total_x_upper, total_y_upper, count_upper)
    CoG_lower = calculate_CoG(total_x_lower, total_y_lower, count_lower)
    # print(f"CoG Whole Body: {CoG_whole}, CoG Upper Body: {CoG_upper}, CoG Lower Body: {CoG_lower}")
    
    cv2.circle(image, CoG_whole, 10, (50, 50, 200), -1)  # Darker red for whole body
    cv2.putText(image, "W "+str(CoG_whole), (CoG_whole[0] + 15, CoG_whole[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1, cv2.LINE_AA)

    cv2.circle(image, CoG_upper, 10, (50, 200, 50), -1)  # Darker green for upper body
    cv2.putText(image, "U "+str(CoG_upper), (CoG_upper[0] + 15, CoG_upper[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1, cv2.LINE_AA)

    cv2.circle(image, CoG_lower, 10, (200, 50, 50), -1)  # Darker blue for lower body
    cv2.putText(image, "L "+str(CoG_lower), (CoG_lower[0] + 15, CoG_lower[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1, cv2.LINE_AA)

    # Draw all the edges
    for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (keypoints[edge_pair[0]].score > keypoint_threshold and
          keypoints[edge_pair[1]].score > keypoint_threshold):
        cv2.line(image, keypoints[edge_pair[0]].coordinate,
                 keypoints[edge_pair[1]].coordinate, edge_color, 2)

    # Draw bounding_box with multipose
    if bounding_box is not None:
      start_point = bounding_box.start_point
      end_point = bounding_box.end_point
      cv2.rectangle(image, start_point, end_point, person_color, 2)

      # Fall Detection
      """
      Fall Detection: 
        * If height/Y-coordinate of "Whole Body COG" is lesser than previous frame "Whole Body COG", then yes there are chances he is falling.
        * If distance b/w Y-coordinates of COG of Upper Body and Lower body should be lower than X-coordinates distance.
      """
      # print("Whole:", ((CoG_whole[1] * image.shape[0])-(prev_CoGWhole_y)), "Upper:", ((CoG_upper[1] * image.shape[0])-(prev_CoGUpper_y)))
      if (((CoG_whole[1] * image.shape[0])-(prev_CoGWhole_y)>82000) and ((CoG_upper[1] * image.shape[0])-(prev_CoGUpper_y)>80000)):
        # print("--------------- Fall Detected ---------------")
        cv2.putText(image, "Fall Detected", start_point, cv2.FONT_HERSHEY_PLAIN, 2, (144, 12, 63), 2)
      prev_CoGWhole_y = (CoG_whole[1] * image.shape[0])
      prev_CoGUpper_y = (CoG_upper[1] * image.shape[0])

      # Draw id text when tracker is enabled for MoveNet MultiPose model.
      # (id = None when using single pose model or when tracker is None)
      if person.id:
        id_text = 'id = ' + str(person.id)
        cv2.putText(image, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 1)

  return image


def keep_aspect_ratio_resizer(
    image: np.ndarray, target_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
  """Resizes the image.

  The function resizes the image such that its longer side matches the required
  target_size while keeping the image aspect ratio. Note that the resizes image
  is padded such that both height and width are a multiple of 32, which is
  required by the model. See
  https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1 for more
  detail.

  Args:
    image: The input RGB image as a numpy array of shape [height, width, 3].
    target_size: Desired size that the image should be resize to.

  Returns:
    image: The resized image.
    (target_height, target_width): The actual image size after resize.

  """
  height, width, _ = image.shape
  if height > width:
    scale = float(target_size / height)
    target_height = target_size
    scaled_width = math.ceil(width * scale)
    image = cv2.resize(image, (scaled_width, target_height))
    target_width = int(math.ceil(scaled_width / 32) * 32)
  else:
    scale = float(target_size / width)
    target_width = target_size
    scaled_height = math.ceil(height * scale)
    image = cv2.resize(image, (target_width, scaled_height))
    target_height = int(math.ceil(scaled_height / 32) * 32)

  padding_top, padding_left = 0, 0
  padding_bottom = target_height - image.shape[0]
  padding_right = target_width - image.shape[1]
  # add padding to image
  image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left,
                             padding_right, cv2.BORDER_CONSTANT)
  return image, (target_height, target_width)
