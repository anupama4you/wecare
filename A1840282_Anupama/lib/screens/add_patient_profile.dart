import 'dart:ffi';
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:get/get.dart';
import 'package:get/get_core/src/get_main.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:uuid/uuid.dart';
import 'package:video_thumbnail/video_thumbnail.dart';
import 'package:watchdog_correct/classes/caregiver_class.dart';
import 'package:watchdog_correct/classes/patient_class.dart';
import 'package:watchdog_correct/reusable_widgets/reusable_widget.dart';
import 'package:watchdog_correct/reusable_widgets/user_profile_provider.dart';
import 'package:watchdog_correct/reusable_widgets/validation_utils.dart';
import 'package:watchdog_correct/screens/home_screen.dart';
import 'package:watchdog_correct/screens/upload_form.dart';
import 'package:watchdog_correct/utils/color_utils.dart';
import 'package:http/http.dart' as http;
import 'package:multiple_image_camera/camera_file.dart';
import 'package:multiple_image_camera/multiple_image_camera.dart';
import 'package:video_player/video_player.dart';
import 'dart:isolate';


class PatientProfileScreen extends StatefulWidget {
  @override
  _PatientProfileScreenState createState() => _PatientProfileScreenState();
}

class _PatientProfileScreenState extends State<PatientProfileScreen> {
  List<MediaModel> images = [];
  List<String> base64Images = [];

  TextEditingController _firstNameTextController = TextEditingController();
  TextEditingController _lastNameTextController = TextEditingController();
  TextEditingController _bedNoTextController = TextEditingController();
  TextEditingController _roomNoTextController = TextEditingController();
  TextEditingController _assignedCaregiverTextController = TextEditingController();
  late CaregiverProfile assginedCaregiver;
  bool _allowedInBed = true;
  bool _allowedInRoom = true;

  // Editing state
  bool isEditing = false;
  bool isObscurePassword = true;
  bool isVideoCaptured = false;
  bool isVideoLoading = false;

  PickedFile? _imageFile = null;
  final ImagePicker _picker = ImagePicker();
  late final File videoFile;
  late VideoPlayerController _videoPlayerController;
  String videoURL = '';
  String patientId = '';
  String? thumbnailPath = '';
  String videoDownloadURL = '';
  XFile? cameraVideoFile;

  @override
  void initState() {
    super.initState();
    patientId = Uuid().v4();
    _videoPlayerController = VideoPlayerController.asset('')
      ..initialize().then((_) {
        // Ensure the first frame is shown after the video is initialized.
        setState(() {});
      });
  }

  Future<void> showInstructionsDialog() async {
    return showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text("Video Capture Instructions", style: TextStyle(fontSize: 16)),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "1. Patient should sit steadily under good lighting.",
                style: TextStyle(fontSize: 14),
              ),
              Text(
                "2. Hold the device horizontally for best results.",
                style: TextStyle(fontSize: 14),
              ),
              Text(
                "3. Keep the subject's face in focus.",
                style: TextStyle(fontSize: 14),
              ),
              Text(
                "4. Capture a 20 seconds video of all angles of the face.",
                style: TextStyle(fontSize: 14),
              ),
              Text(
                "5. Press the record button to start capturing.",
                style: TextStyle(fontSize: 14),
              ),
              Text(
                "6. Always check the video preview before proceeding.",
                style: TextStyle(fontSize: 14),
              ),
            ],
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text("Got it", style: TextStyle(fontSize: 16)),
            ),
          ],
        );
      },
    );
  }


  Future<String?> captureFirstFrame(XFile videoFile) async {
    final thumbnailPath = await VideoThumbnail.thumbnailFile(
      video: videoFile.path,
      imageFormat: ImageFormat.JPEG, // You can change the format to JPEG if needed
      quality: 100, // Set the quality of the captured image
    );
    return thumbnailPath;
  }


  Future getVideoFile(ImageSource source) async {
    final ImagePicker picker = ImagePicker();
    try {
      XFile? pickedVideo;
      if (source == ImageSource.camera) {
        pickedVideo = await picker.pickVideo(
          source: ImageSource.camera,
          preferredCameraDevice: CameraDevice.rear,
          maxDuration: Duration(seconds: 20),
        );
      } else if (source == ImageSource.gallery) {
        pickedVideo = await picker.pickVideo(source: ImageSource.gallery);
      }
      EasyLoading.show(status: 'loading...');
      if (pickedVideo != null) {
        final localVideoFile = File(pickedVideo.path);
        initializeVideoController(localVideoFile);
        setState(() {
          cameraVideoFile = pickedVideo;
          videoURL = pickedVideo!.path;
          isVideoCaptured = true; // Video has been captured or selected
        });
        // Upload the video to Firebase or perform other actions here.
      } else {
        EasyLoading.showError('No video picked');
      }
    } catch (e) {
      EasyLoading.showError('Error picking the video');
    }
    EasyLoading.dismiss();
  }


  void initializeVideoController(File videoFile) {
    _videoPlayerController = VideoPlayerController.file(videoFile)
      ..initialize().then((_) {
        // Ensure the first frame is shown after the video is initialized.
        setState(() {});
      });
  }

  Future uploadVideoToFirebase() async {
    EasyLoading.show(status: 'uploading the video...');
    thumbnailPath = await captureFirstFrame(cameraVideoFile!);
    // print(images);
    // assigning a random patient id
    try {
      final Reference storageReference =
      FirebaseStorage.instance.ref().child('patients/${_roomNoTextController.text}').child(DateTime.now().toString() + '.mp4');
      final UploadTask uploadTask = storageReference.putFile(File(cameraVideoFile!.path));
      final TaskSnapshot storageTaskSnapshot = await uploadTask.whenComplete(() => null);
      videoDownloadURL = await storageTaskSnapshot.ref.getDownloadURL();
    } catch (e) {
      print('Error uploading video to Firebase Storage: $e');
      EasyLoading.showError('Error uploading video');
    }
    EasyLoading.dismiss();
  }

  Future<bool> checkRoomIdExists() async {
    EasyLoading.show(status: 'Room availability checking...');
    String roomId = _roomNoTextController.text.trim();
    if (roomId.isEmpty) {
      EasyLoading.showError('Please enter a room ID');
      return false;
    }

    try {
      // Query Firestore to check if the room ID already exists
      QuerySnapshot querySnapshot = await FirebaseFirestore.instance
          .collection('patients')
          .where('roomNum', isEqualTo: roomId)
          .get();

      if (querySnapshot.docs.isNotEmpty) {
        EasyLoading.showError('There is already a patient in the room');
        return false;
      } else {
        // Room ID does not exist, you can proceed with your operation
        EasyLoading.showSuccess('Room is available');
        return true;
        // Add your logic here if the room ID is available
      }
    } catch (error) {
      print('Error checking room ID: $error');
      EasyLoading.showError('Failed to check room ID');
      return false;
    } finally {
      EasyLoading.dismiss();
    }
  }

  Future<void> addProfileData() async {
    if(await checkRoomIdExists() == false) {
      EasyLoading.dismiss();
      return;
    }
    EasyLoading.show(status: 'loading...');

    var imageURLs = [];
    Reference referenceRoot = FirebaseStorage.instance.ref();
    Reference referenceDirImages = referenceRoot.child('patients/${_roomNoTextController.text}');
    try {
      await uploadVideoToFirebase();
      EasyLoading.show(status: 'uploading photos...');
      String uniqueFileName = DateTime.now().microsecondsSinceEpoch.toString();
      Reference referenceThumbnailToUpload = referenceDirImages.child('$uniqueFileName.jpg');
      await referenceThumbnailToUpload.putFile(File(thumbnailPath!));
      imageURLs.add(await referenceThumbnailToUpload.getDownloadURL());
      imageURLs.add(videoDownloadURL);
    } catch (error) {
      EasyLoading.showError('Upload error');
      EasyLoading.dismiss();
      return
      print(error);
    }

    EasyLoading.show(status: 'creating patient...');
    final Map<String, dynamic> profileAdd = {
      'firstName': _firstNameTextController.text,
      'lastName': _lastNameTextController.text,
      'careGiverId': FirebaseAuth.instance.currentUser?.uid,
      'allowedInBed': _allowedInBed,
      'allowedInRoom': _allowedInRoom,
      'imageUrls': imageURLs,
      'roomNum': _roomNoTextController.text,
      'bedNum': _bedNoTextController.text
      // Add other fields you want to update
    };

    // Construct the Firebase Realtime Database endpoint URL
    var databaseUrl =
        'https://australia-southeast1-watchdog-gamma.cloudfunctions.net/app/patients';

    try {
      final response = await http.post(
        Uri.parse(databaseUrl),
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: json.encode(profileAdd),
      );

      if (response.statusCode == 200) {
        // Profile data updated successfully
        print(json.encode(profileAdd));
        EasyLoading.showSuccess('Patient successfully created!');
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => HomeScreen()),
        );
      } else {
        EasyLoading.showError('Failed to add patient data');
        EasyLoading.dismiss();
      }
    } catch (error) {
      // Handle the error
      print('Error adding patient data: $error');
      EasyLoading.showError('Failed to pass patient data');
    }
    EasyLoading.dismiss();
  }

  static Future<void> addProfileDataInBackground(Map<String, dynamic> message) async {
    final SendPort sendPort = message['sendPort'];
    final Map<String, dynamic> profileData = message['profileData'];
    final XFile videoFile = message['video'];
    await Firebase.initializeApp();

    String? thumbnailPath = '';
    String? videoDownloadURL = '';

    // EasyLoading.show(status: 'loading...');

    var imageURLs = [];
    Reference referenceRoot = FirebaseStorage.instance.ref();
    Reference referenceDirImages = referenceRoot.child('patients/${profileData['roomNum']}');
    try {
      // --- image upload --- //
      thumbnailPath = await VideoThumbnail.thumbnailFile(
        video: videoFile.path,
        imageFormat: ImageFormat.JPEG, // You can change the format to JPEG if needed
        quality: 100, // Set the quality of the captured image
      );
      // print(images);
      // assigning a random patient id
      try {
        final Reference storageReference =
        FirebaseStorage.instance.ref().child('patients/${profileData['roomNum']}').child(DateTime.now().toString() + '.mp4');
        final UploadTask uploadTask = storageReference.putFile(File(videoFile!.path));
        final TaskSnapshot storageTaskSnapshot = await uploadTask.whenComplete(() => null);
        videoDownloadURL = await storageTaskSnapshot.ref.getDownloadURL();
      } catch (e) {
        print('Error uploading video to Firebase Storage: $e');
        // EasyLoading.showError('Error uploading video');
      }
      // --- image upload --- //

      // --- video upload --- //
      // EasyLoading.show(status: 'uploading video...');
      String uniqueFileName = DateTime.now().microsecondsSinceEpoch.toString();
      Reference referenceThumbnailToUpload = referenceDirImages.child('$uniqueFileName.jpg');
      await referenceThumbnailToUpload.putFile(File(thumbnailPath!));
      imageURLs.add(await referenceThumbnailToUpload.getDownloadURL());
      imageURLs.add(videoDownloadURL);
    } catch (error) {
      // EasyLoading.showError('Upload error');
      // EasyLoading.dismiss();
    return
      print(error);
    }
    // --- image upload --- //

    // --- database update --- //
    print(profileData);
    // var databaseUrl =
    //     'https://australia-southeast1-watchdog-gamma.cloudfunctions.net/app/patients';
    //
    // try {
    //   final response = await http.post(
    //     Uri.parse(databaseUrl),
    //     headers: <String, String>{
    //       'Content-Type': 'application/json; charset=UTF-8',
    //     },
    //     body: json.encode(profileData),
    //   );
    //
    //   print(json.encode(profileData));
    //
    //   if (response.statusCode == 200) {
    //     // Profile data updated successfully
    //     print('Profile data updated successfully');
    //     print(json.encode(profileData));
    //     EasyLoading.showSuccess('Patient successfully created!');
    //   } else {
    //     // Handle the error
    //     print('Failed to add profile data. Status code: ${response.body}');
    //     EasyLoading.showError('Failed to add patient data');
    //   }
    // } catch (error) {
    //   // Handle the error
    //   print('Error adding patient data: $error');
    //   EasyLoading.showError('Failed to pass patient data');
    // }
    // --- database update --- //

    // After the processing is complete, you can send the result back
    sendPort.send('Processing complete');
  }

  void runInIsolate() async {
    final ReceivePort receivePort = ReceivePort();

    final Map<String, dynamic> profileData = {
      'id': patientId,
      'firstName': _firstNameTextController.text,
      'lastName': _lastNameTextController.text,
      'careGiverId': FirebaseAuth.instance.currentUser?.uid,
      'allowedInBed': _allowedInBed,
      'allowedInRoom': _allowedInRoom,
      'roomNum': _roomNoTextController.text,
      'bedNum': _bedNoTextController.text
      // Add other fields you want to update
    };

    // Spawn the isolate
    await Isolate.spawn(
      addProfileDataInBackground,
      {'profileData': profileData, 'video': cameraVideoFile, 'sendPort': receivePort.sendPort},
    );

    // Listen for a message from the isolate
    receivePort.listen((data) {
      print('Isolate message: $data');
      if (data == 'Processing complete') {
        EasyLoading.dismiss();
        // Perform any actions that need to be done after processing is complete
      }
    });
  }

  @override
  void dispose() {
    _videoPlayerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Add Patient'),
      ),
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        decoration: BoxDecoration(gradient: backgroundGradient()),
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  ElevatedButton(
                    child: const Text("Capture Video"),
                    onPressed: () async {
                      await showInstructionsDialog(); // Show the instructions dialog
                      getVideoFile(ImageSource.camera);
                    },
                  ),
                  SizedBox(width: 10), // Adjust the size according to your needs
                  ElevatedButton(
                    child: const Text("Upload Video"),
                    onPressed: () async {
                      getVideoFile(ImageSource.gallery);
                    },
                  ),
                ],
              ),
                // video preview
              // Display the picked video if available

              // Display the picked video if available
              if (isVideoCaptured)
                if (_videoPlayerController != null)
                  AspectRatio(
                    aspectRatio: _videoPlayerController!.value.aspectRatio,
                    child: Stack(
                      alignment: Alignment.center,
                      children: <Widget>[
                        VideoPlayer(_videoPlayerController!),
                        Center(
                          child: ElevatedButton(
                            onPressed: () {
                              setState(() {
                                if (_videoPlayerController!.value.isPlaying) {
                                  _videoPlayerController!.pause();
                                } else {
                                  _videoPlayerController!.play();
                                }
                              });
                            },
                            child: Icon(
                              _videoPlayerController!.value.isPlaying
                                  ? Icons.pause
                                  : Icons.play_arrow,
                              size: 50,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

              Padding(
                padding: const EdgeInsets.all(25.0),
                child: Container(
                  child: GridView.builder(
                    shrinkWrap: true,
                    gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 5,
                    ),
                    itemCount: images.length,
                    itemBuilder: (context, index) {
                      return Container(
                        width: 100,
                        height: 100,
                        child: Image.file(
                          File(images[index].file.path),
                          fit: BoxFit.cover,
                        ),
                      );
                    },
                  ),
                ),
              ),
              SizedBox(height: 30),
              buildTextField('First Name', 'First Name', _firstNameTextController),
              buildTextField('Last Name', 'Last Name', _lastNameTextController),
              buildTextFieldForNumbers('Room Number', 'Room Number', _roomNoTextController),
              buildTextFieldForNumbers('Bed Number', 'Bed Number', _bedNoTextController),
              CheckboxListTile(
                title: Text('Must be in Bed'),
                value: _allowedInBed,
                onChanged: (newValue) {
                  setState(() {
                    _allowedInBed = newValue ?? false;
                  });
                },
              ),
              CheckboxListTile(
                title: Text('Must be in Room'),
                value: _allowedInRoom,
                onChanged: (newValue) {
                  setState(() {
                    _allowedInRoom = newValue ?? false;
                  });
                },
              ),
              SizedBox(height: 20),
              Center(
                child: ElevatedButton(
                  onPressed: () {
                    addProfileData();
                  },
                  child: Text('Add Patient'),
                ),
              ),
              SizedBox(height: 50),
            ],
          ),
        ),
      ),
    );
  }

  Widget buildTextField(
      String labelText,
      String placeholder,
      TextEditingController controller,
      ) {
    return Padding(
      padding: EdgeInsets.only(bottom: 30),
      child: TextFormField(
        controller: controller,
        enabled: true,
        decoration: InputDecoration(
          contentPadding: EdgeInsets.only(bottom: 5),
          labelText: labelText,
          labelStyle: TextStyle(
            color: Colors.white70,
          ),
          floatingLabelBehavior: FloatingLabelBehavior.always,
          hintStyle: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
      ),
    );
  }

  Widget buildTextFieldForNumbers(
      String labelText,
      String placeholder,
      TextEditingController controller,
      ) {
    return Padding(
      padding: EdgeInsets.only(bottom: 30),
      child: TextFormField(
        controller: controller,
        enabled: true,
        keyboardType: TextInputType.number,
        inputFormatters: <TextInputFormatter>[
          FilteringTextInputFormatter.digitsOnly,
        ],
        decoration: InputDecoration(
          contentPadding: EdgeInsets.only(bottom: 5),
          labelText: labelText,
          labelStyle: TextStyle(
            color: Colors.white70,
          ),
          floatingLabelBehavior: FloatingLabelBehavior.always,
          hintStyle: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
      ),
    );
  }

  // Widget bottomSheet() {
  //   return Container(
  //     height: 100.0,
  //     width: MediaQuery.of(context).size.width,
  //     margin: EdgeInsets.symmetric(
  //       horizontal: 20,
  //       vertical: 20,
  //     ),
  //     child: Column(
  //       children: <Widget>[
  //         Text(
  //           "Choose Profile photo",
  //           style: TextStyle(
  //             fontSize: 20.0,
  //           ),
  //         ),
  //         SizedBox(
  //           height: 20,
  //         ),
  //         Row(mainAxisAlignment: MainAxisAlignment.center, children: <Widget>[
  //           ElevatedButton.icon(
  //             icon: Icon(Icons.camera),
  //             onPressed: () {
  //               takePhoto(ImageSource.camera);
  //             },
  //             label: Text("Camera"),
  //           ),
  //           ElevatedButton.icon(
  //             icon: Icon(Icons.image),
  //             onPressed: () {
  //               takePhoto(ImageSource.gallery);
  //             },
  //             label: Text("Gallery"),
  //           ),
  //         ])
  //       ],
  //     ),
  //   );
  // }
}
