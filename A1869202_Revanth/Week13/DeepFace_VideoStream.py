from deepface import DeepFace

DeepFace.stream(db_path = "Patients_Images", model_name = "Facenet", enable_face_analysis = False, time_threshold=1, detector_backend="mediapipe")