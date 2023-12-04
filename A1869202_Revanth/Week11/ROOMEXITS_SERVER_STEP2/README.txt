# This is step where, we run the model to make predictions (Already models link is given in slack. USE ONLY .tflite file)
# NO NEED OF TRAINING MODEL ON SERVER. I GAVE ALREADY A TRAINED MODEL.

# Now run this FACES_2_ROOMEXIT.PY to make predictions and update results to db. 
# command: python Faces_2_RoomExit.py

# When you run this 3 things are happening inside: 
> loading faces from database, 
> processing room exit algorithm, 
> pushing back results to db under same database url, but under different child.
# Please, just look the code, can be easily understood, i wrote comments even for small things as well! 
