1. In this folder, our aim is to convert given patients videos, to convert to patients images, then from patients images we will extract faces and we will crop them to face and will save them.
2. Thats it!, we dont need to split into train/test (this was used for me to train & test models, thats how .h5 model is generated from training), and we dont need to convert to numpy arrays here. in this step.

3. Run with specific command: 
> python PreprocessingData.py -v Patients_Videos -i Patients_Images -fc 20 -s 96 -tm 0
> Patients_Videos: means location of folder from where the code should load videos from.
> Patients_Images keep this as it was, because in other parts of code, we kept this location as default to load Patients_Images and run other processing steps. If changed, then you need to pass this as argument in upcomming steps.
> FC: No of frames: 20 are enough from video.
> S: shape of image, required will be 96 * 96 * 96.
> tm: 0 is to pass, this helps to not to run traintest splits, numpy array conversions.