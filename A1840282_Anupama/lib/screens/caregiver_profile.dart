import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:watchdog_correct/classes/caregiver_class.dart';
import 'package:watchdog_correct/classes/patient_class.dart';
import 'package:watchdog_correct/reusable_widgets/reusable_widget.dart';
import 'package:watchdog_correct/reusable_widgets/user_profile_provider.dart';
import 'package:watchdog_correct/reusable_widgets/validation_utils.dart';
import 'package:watchdog_correct/screens/home_screen.dart';
import 'package:watchdog_correct/utils/color_utils.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

import 'caregiver_profile_view.dart';

class ProfileScreen extends StatefulWidget {
  @override
  _ProfileScreenState createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  TextEditingController _firstNameTextController = TextEditingController();
  TextEditingController _lastNameTextController = TextEditingController();
  TextEditingController _userNameTextController = TextEditingController();
  TextEditingController _phoneNoTextController = TextEditingController();
  TextEditingController _assignedPatientsTextController = TextEditingController();
  List<dynamic> assignedPatients = [];

  // Editing state
  bool isEditing = false;
  bool isObscurePassword = true;
  String imageURL = '';
  late Map<String, dynamic> userData;

  PickedFile? _imageFile = null;
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    loadData();
  }

  void loadData() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      final doc = await FirebaseFirestore.instance.collection('user').doc(user.uid).get();
      if (doc.exists) {
         userData = doc.data() as Map<String, dynamic>;
        _firstNameTextController.text = userData['firstName'] ?? '';
        _lastNameTextController.text = userData['lastName'] ?? '';
        _userNameTextController.text = userData['username'] ?? '';
        _phoneNoTextController.text = userData['phone'] ?? '';
        _assignedPatientsTextController.text = (userData['assignedPatients'] as List<dynamic>?)?.join(', ') ?? '';
        setState(() {
          imageURL = userData['imageUrl'] ?? '';
        });
      }
    }
  }

  Future<void> updateProfileData() async {
    EasyLoading.show(status: 'loading...');

    String? imageUrl;

    // Check if a new image was selected and upload it
    if (_imageFile != null) {
      try {
        Reference referenceRoot = FirebaseStorage.instance.ref();
        String uniqueFileName = DateTime.now().microsecondsSinceEpoch.toString();
        Reference referenceImageToUpload = referenceRoot.child('caregiver_images/${_firstNameTextController.text}/$uniqueFileName');

        // Upload the image
        TaskSnapshot uploadTask = await referenceImageToUpload.putFile(File(_imageFile!.path));

        // Get the download URL
        imageUrl = await uploadTask.ref.getDownloadURL();
      } catch (e) {
        EasyLoading.showError('Failed to upload image');
        return;
      }
    }

    // Prepare the data to update
    Map<String, dynamic> profileUpdate = {
      if (_firstNameTextController.text.isNotEmpty) 'firstName': _firstNameTextController.text,
      if (_lastNameTextController.text.isNotEmpty) 'lastName': _lastNameTextController.text,
      if (_userNameTextController.text.isNotEmpty) 'username': _userNameTextController.text,
      if (_phoneNoTextController.text.isNotEmpty) 'phone': _phoneNoTextController.text,
      if (imageUrl != null) 'imageUrl': imageUrl,
      // Add other fields as needed
    };

    if (profileUpdate.isEmpty) {
      EasyLoading.showInfo('No changes to save');
      return;
    }

    try {
      FirebaseFirestore.instance
          .collection('user')
          .doc(FirebaseAuth.instance.currentUser?.uid)
          .update(profileUpdate)
          .then((_) {
        EasyLoading.showSuccess('Profile successfully updated!');
      })
          .catchError((error) {
        print('Failed to update profile data: $error');
        EasyLoading.showError('Failed to update profile data');
      });
    } finally {
      EasyLoading.dismiss();
    }
  }


  List<PatientSelect> allPatients = [
    PatientSelect(name: 'Patient 1'),
    PatientSelect(name: 'Patient 2'),
    PatientSelect(name: 'Patient 3'),
    // Add more patients as needed
  ];

  void takePhoto(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    setState(() {
      if (pickedFile != null) {
        _imageFile = PickedFile(pickedFile.path);
      } else {
        _imageFile = null;
      }
    });

    print('selected photo: ${_imageFile?.path}');
    Navigator.pop(context);
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
              });
            },
            icon: Icon(
              isEditing ? Icons.save : Icons.edit,
              color: Colors.black,
            ),
          ),
        ],
      ),
      body: StreamBuilder<DocumentSnapshot>(
        stream: FirebaseFirestore.instance
            .collection('user')
            .doc(FirebaseAuth.instance.currentUser?.uid)
            .snapshots(),
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            return Center(child: CircularProgressIndicator()); // Loading indicator
          }

          return Container(
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
                  Center(
                    child: Stack(
                      children: [
                        Container(
                          width: 130,
                          height: 130,
                          decoration: BoxDecoration(
                            border: Border.all(width: 4, color: Colors.white),
                            boxShadow: [
                              BoxShadow(
                                spreadRadius: 2,
                                blurRadius: 10,
                                color: Colors.black.withOpacity(0.1),
                              ),
                            ],
                            shape: BoxShape.circle,
                            image: DecorationImage(
                              fit: BoxFit.cover,
                              image: _imageFile == null
                                  ? NetworkImage(imageURL) as ImageProvider
                                  : FileImage(File(_imageFile?.path ?? '')),
                            ),
                          ),
                        ),
                        Positioned(
                          bottom: 0,
                          right: 0,
                          child: Container(
                            height: 40,
                            width: 40,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(width: 4, color: Colors.white),
                              color: Colors.blue,
                            ),
                            child: InkWell(
                              onTap: () {
                                if (isEditing) {
                                  showModalBottomSheet(
                                    context: context,
                                    builder: ((builder) => bottomSheet()),
                                  );
                                }
                              },
                              child: Icon(
                                Icons.edit,
                                color: Colors.white,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  SizedBox(height: 10), // Add some spacing between the profile picture and the email
                  Center(
                    child: Text(
                      userData['email'], // Replace with the actual primary email
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.white
                      ),
                    ),
                  ),
                  SizedBox(height: 30),
                  buildTextField(
                    "First Name",
                    userData['firstName'] ?? '',
                    false,
                    _firstNameTextController,
                        (value) {
                      if (isEditing && value!.isEmpty) {
                        return "Please enter your first name";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  buildTextField(
                    "Last Name",
                    userData['lastName'] ?? '',
                    false,
                    _lastNameTextController,
                        (value) {
                      if (isEditing && value!.isEmpty) {
                        return "Please enter your last name";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  buildTextField(
                    "User Name",
                    userData['username'] ?? '',
                    false,
                    _userNameTextController,
                        (value) {
                      if (isEditing && value!.isEmpty) {
                        return "Please enter a username";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  buildTextField(
                    "Phone No",
                    userData['phone'] ?? '',
                    false,
                    _phoneNoTextController,
                        (value) {
                      if (isEditing && value!.isEmpty) {
                        return "Please enter phone no";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
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
                          padding: EdgeInsets.symmetric(horizontal: 50),
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
                  )
                ],
              ),
            ),
          );
        },
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
        style: TextStyle(color: Colors.white),
        decoration: InputDecoration(
          contentPadding: EdgeInsets.only(bottom: 5),
          labelText: labelText,
          labelStyle: TextStyle(
            color: Colors.white70,
          ),
          floatingLabelBehavior: FloatingLabelBehavior.always,
          hintText: isEditing ? placeholder : placeholder,
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
}
