import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:video_player/video_player.dart';
import 'package:watchdog_correct/classes/caregiver_class.dart';
import 'package:watchdog_correct/classes/patient_class.dart';
import 'package:watchdog_correct/reusable_widgets/reusable_widget.dart';
import 'package:watchdog_correct/reusable_widgets/user_profile_provider.dart';
import 'package:watchdog_correct/reusable_widgets/validation_utils.dart';
import 'package:watchdog_correct/screens/home_screen.dart';
import 'package:watchdog_correct/utils/color_utils.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:firebase_database/firebase_database.dart';

import 'caregiver_profile_view.dart';

class PatientProfileScreenView extends StatefulWidget {
  final Map<String, dynamic> patientData;

  PatientProfileScreenView({required this.patientData});

  @override
  _ProfileScreenState createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<PatientProfileScreenView> {

  DatabaseReference _database = FirebaseDatabase.instance.ref();

  bool bedExitted = false;
  bool fallDetected = false;
  VideoPlayerController? _videoController;

  TextEditingController _firstNameTextController = TextEditingController();
  TextEditingController _lastNameTextController = TextEditingController();
  TextEditingController _bedNumTextController = TextEditingController();
  TextEditingController _roomNumTextController = TextEditingController();

  bool allowedInBed = false;
  bool allowedInRoom = false;

  // Editing state
  bool isEditing = false;

  PickedFile? _imageFile = null;
  final ImagePicker _picker = ImagePicker();
  List<dynamic> imageReferences = [];

  @override
  void initState() {
    super.initState();
    // Load patient data when the widget is initialized
    loadPatientData();
    initializeVideoPlayer();
  }

  Future<void> initializeVideoPlayer() async {
    final videoUrl = imageReferences.isNotEmpty ? imageReferences[1] : '';
    if (videoUrl.isNotEmpty) {
      _videoController = VideoPlayerController.networkUrl(Uri.parse(videoUrl));
      await _videoController!.initialize();
      _videoController!.setLooping(true); // Optionally loop the video
      setState(() {});
    }
  }

  Future<void> loadPatientData() async {
    final patientData = widget.patientData;
    if (patientData != null) {
      _firstNameTextController.text = patientData['firstName'] ?? '';
      _lastNameTextController.text = patientData['lastName'] ?? '';
      _bedNumTextController.text = patientData['bedNum'] ?? '';
      _roomNumTextController.text = patientData['roomNum'] ?? '';
      allowedInBed = patientData['allowedInBed'] ?? false;
      allowedInRoom = patientData['allowedInRoom'] ?? false;
      imageReferences = patientData['imageUrls'] ?? [];
      // Load other data as needed
    }
  }

  void takePhoto(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    setState(() {
      if (pickedFile != null) {
        _imageFile = PickedFile(pickedFile.path);
      } else {
        _imageFile = null;
      }
    });

    Navigator.pop(context);
  }

  Future<void> updateProfileData() async {
    EasyLoading.show(status: 'loading...');

      final Map<String, dynamic> profileUpdate = {
        'firstName': _firstNameTextController.text,
        'lastName': _lastNameTextController.text,
        'bedNum': _bedNumTextController.text,
        'roomNum': _roomNumTextController.text,
        'allowedInBed': allowedInBed,
        'allowedInRoom': allowedInRoom,
      };

      // Construct the Firebase Realtime Database endpoint URL
      final databaseUrl = 'https://australia-southeast1-watchdog-gamma.cloudfunctions.net/app/patients/${widget.patientData['id']}';
      print(json.encode(profileUpdate));
      try {
        final response = await http.put(
          Uri.parse(databaseUrl),
          headers: <String, String>{
            'Content-Type': 'application/json',
          },
          body: json.encode(profileUpdate),
        );

        if (response.statusCode == 200) {
          EasyLoading.showSuccess('Profile successfully updated!');
        } else {
          // Handle the error
          print('Failed to update profile data. Status code: ${response.body}');
          EasyLoading.showError('Failed to update profile data');
        }
      } catch (error) {
        // Handle the error
        print('Error updating profile data: $error');
      }
      EasyLoading.dismiss();
  }

  Future<void> deletePatient() async {
    try {
      EasyLoading.show(status: 'loading...');
      EasyLoading.show(status: 'deleting videos');
      Reference referenceRoot = FirebaseStorage.instance.ref();
      Reference referenceDirImages = referenceRoot.child('patients/${widget.patientData['roomNum']}');
      final result = await referenceDirImages.listAll();
      for (final item in result.items) {
        await item.delete();
      }

      print(widget.patientData['id']);
      EasyLoading.show(status: 'loading...');
      final response = await http.delete(
        Uri.parse('https://australia-southeast1-watchdog-gamma.cloudfunctions.net/app/patients/${widget.patientData['id']}'), // Replace with your API URL
        headers: <String, String>{
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        EasyLoading.showSuccess('Patient removed successfully!');
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => HomeScreen()),
        );
      } else {
        print(response.body);
        EasyLoading.showError('Failed to delete patient. Status code: ${response.statusCode}');
      }
    } catch (error) {
      print('Error deleting patient: $error');
    }
    EasyLoading.dismiss();
  }

  @override
  void dispose() {
    _videoController?.dispose(); // Dispose of the video controller
    super.dispose();
  }


  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(
        title: Text('Edit Profile'),
        leading: IconButton(
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => HomeScreen()),
            );
          },
          icon: Icon(
            Icons.arrow_back,
            color: Colors.black,
          ),
        ),
        actions: [
          IconButton(
            onPressed: () {
              setState(() {
                // Toggle the editing state
                isEditing = !isEditing;
                if (isEditing) {
                  // Enable editing mode - populate text fields with data
                  loadPatientData();
                }
              });
            },
            icon: Icon(
              isEditing ? Icons.save : Icons.edit,
              color: Colors.black,
            ),
          ),
        ],
      ),
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        decoration: BoxDecoration(gradient: backgroundGradient()),
        padding: EdgeInsets.only(left: 15, top: 20, right: 15),
        child: GestureDetector(
          onTap: () {
            FocusScope.of(context).unfocus();
          },
          child: ListView(
            children: [
              // Video player widget
              if (_videoController != null)
                AspectRatio(
                  aspectRatio: _videoController!.value.aspectRatio,
                  child: Stack(
                    alignment: Alignment.center,
                    children: <Widget>[
                      VideoPlayer(_videoController!),
                      Center(
                        child: ElevatedButton(
                          onPressed: () {
                            setState(() {
                              if (_videoController!.value.isPlaying) {
                                _videoController!.pause();
                              } else {
                                _videoController!.play();
                              }
                            });
                          },
                          child: Icon(
                            _videoController!.value.isPlaying
                                ? Icons.pause
                                : Icons.play_arrow,
                            size: 50,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              // ...
              SizedBox(height: 30),
              buildTextField(
                "First Name",
                _firstNameTextController.text ?? '',
                false,
                _firstNameTextController,
                    (value) {
                  if (isEditing) {
                    if (value!.isEmpty) {
                      return "Please enter your first name";
                    }
                  }
                  return null;
                },
              ),
              const SizedBox(
                height: 20,
              ),
              buildTextField(
                "Last Name",
                _lastNameTextController.text ?? '',
                false,
                _lastNameTextController,
                    (value) {
                  if (isEditing) {
                    if (value!.isEmpty) {
                      return "Please enter your last name";
                    }
                  }
                  return null;
                },
              ),
              const SizedBox(
                height: 20,
              ),
              buildTextField(
                "Room Number",
                _roomNumTextController.text ?? '',
                false,
                _roomNumTextController,
                    (value) {},
              ),
              const SizedBox(
                height: 20,
              ),
              buildTextField(
                "Bed Number",
                _bedNumTextController.text ?? '',
                false,
                _bedNumTextController,
                    (value) {},
              ),
              const SizedBox(
                height: 20,
              ),
              buildSwitch(
                "Must be in Bed",
                allowedInBed,
                    (value) {
                  if (isEditing) {
                    setState(() {
                      allowedInBed = value;
                    });
                  }
                },
              ),
              buildSwitch(
                "Must be in Room",
                allowedInRoom,
                    (value) {
                  if (isEditing) {
                    setState(() {
                      allowedInRoom = value;
                    });
                  }
                },
              ),
              SizedBox(height: 30),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  OutlinedButton(
                    onPressed: () {
                      if (isEditing) {
                        // Handle cancel button when in editing mode
                        setState(() {
                          isEditing = false;
                        });
                        // Reload the profile data to discard changes
                        loadPatientData();
                      }
                    },
                    child: const Text(
                      "CANCEL",
                      style: TextStyle(
                        fontSize: 15,
                        letterSpacing: 2,
                        color: Colors.white,
                      ),
                    ),
                    style: OutlinedButton.styleFrom(
                      padding: EdgeInsets.symmetric(horizontal: 10),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      if (isEditing) {
                        // Handle save button when in editing mode
                        // Save the edited profile data
                        // You can access the entered values using the text controllers
                        updateProfileData();
                        setState(() {
                          isEditing = false;
                        });
                      } else {
                        // Handle edit button
                        setState(() {
                          isEditing = true;
                        });
                      }
                    },
                    child: Text(
                      isEditing ? "SAVE" : "EDIT",
                      style: TextStyle(
                        fontSize: 15,
                        letterSpacing: 2,
                        color: Colors.white,
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      padding: EdgeInsets.symmetric(horizontal: 50),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                  ),
                ],
              ),
              // Add the "DELETE PATIENT" button
              if (isEditing)
                ElevatedButton(
                  onPressed: () {
                    // Show confirmation dialog before deleting
                    showDialog(
                      context: context,
                      builder: (BuildContext context) {
                        return AlertDialog(
                          title: Text("Confirm Deletion"),
                          content: Text("Are you sure you want to delete this patient? This action cannot be undone."),
                          actions: [
                            TextButton(
                              onPressed: () {
                                Navigator.of(context).pop(); // Close the dialog
                              },
                              child: Text("Cancel"),
                            ),
                            TextButton(
                              onPressed: () {
                                // Delete the patient when confirmed
                                deletePatient();
                                Navigator.of(context).pop(); // Close the dialog
                              },
                              child: Text(
                                "Delete",
                                style: TextStyle(color: Colors.red),
                              ),
                            ),
                          ],
                        );
                      },
                    );
                  },
                  child: Text(
                    "DELETE PATIENT",
                    style: TextStyle(
                      fontSize: 15,
                      letterSpacing: 2,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    onPrimary: Colors.red, // Reddish color for the button
                  ),
                ),
              SizedBox(
                height: 20,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget buildTextField(
      String labelText,
      String placeholder,
      bool isPasswordTextField,
      TextEditingController controller,
      String? Function(String?)? validator,
      ) {
    return Padding(
      padding: EdgeInsets.only(bottom: 30),
      child: TextFormField(
        controller: controller,
        obscureText: isPasswordTextField,
        enabled: isEditing,
        decoration: InputDecoration(
          contentPadding: EdgeInsets.only(bottom: 5),
          labelText: labelText,
          labelStyle: TextStyle(
            color: Colors.white70,
          ),
          floatingLabelBehavior: FloatingLabelBehavior.always,
          hintText: isEditing ? placeholder : '',
          hintStyle: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        validator: validator,
      ),
    );
  }

  Widget bottomSheet() {
    return Container(
      height: 100.0,
      width: MediaQuery.of(context).size.width,
      margin: EdgeInsets.symmetric(
        horizontal: 20,
        vertical: 20,
      ),
      child: Column(
        children: <Widget>[
          Text(
            "Choose Profile photo",
            style: TextStyle(
              fontSize: 20.0,
            ),
          ),
          SizedBox(
            height: 20,
          ),
          Row(mainAxisAlignment: MainAxisAlignment.center, children: <Widget>[
            ElevatedButton.icon(
              icon: Icon(Icons.camera),
              onPressed: () {
                takePhoto(ImageSource.camera);
              },
              label: Text("Camera"),
            ),
            ElevatedButton.icon(
              icon: Icon(Icons.image),
              onPressed: () {
                takePhoto(ImageSource.gallery);
              },
              label: Text("Gallery"),
            ),
          ])
        ],
      ),
    );
  }

  Widget buildSwitch(
      String label,
      bool value,
      Function(bool) onChanged,
      ) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 10),
      child: Row(
        children: [
          Text(
            label,
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          Spacer(),
          Switch(
            value: value,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }
}
