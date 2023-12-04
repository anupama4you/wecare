import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:watchdog_correct/reusable_widgets/reusable_widget.dart';
import 'package:watchdog_correct/reusable_widgets/text_box.dart';
import 'package:watchdog_correct/screens/caregiver_profile.dart';
import 'package:watchdog_correct/screens/home_screen.dart';
import 'package:watchdog_correct/screens/signin_screen.dart';

import '../classes/caregiver_class.dart';
import '../classes/patient_class.dart';
import '../reusable_widgets/user_profile_provider.dart';
import '../utils/color_utils.dart';


class ProfileScreenView extends StatefulWidget {
  @override
  _ProfileScreenState createState() => _ProfileScreenState();

}

class _ProfileScreenState extends State<ProfileScreenView> {

  late CaregiverProfile caregiver = CaregiverProfile(
      id: '',
      username: '',
      firstName: '',
      lastName: '',
      email: '',
      phone: '',
      assignedPatients: []
  );

  // Editing state
  bool isEditing = false;
  bool isObscurePassword = true;



  PickedFile? _imageFile = null;
  final ImagePicker _picker = ImagePicker();

  List<PatientSelect> assignedPatients = [
    PatientSelect(name: 'Kaifeng'),
    PatientSelect(name: 'ooom'),
    PatientSelect(name: 'Revaan'),
    // Add more patients as needed
  ];

  @override
  void initState() {
    super.initState();
    // print(context.watch<UserProfileProvider>().cachedProfile?.firstName);
  }

  @override
  Widget build(BuildContext context) {

    final cachedProfile = context.watch<UserProfileProvider>().cachedProfile;

    return Scaffold(
      appBar: AppBar(
          title: Text('Profile'),
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
              )),
          actions: [
            IconButton(
                onPressed: (){ },
                icon: const Icon(
                  Icons.settings,
                  color: Colors.black,
                ))
          ]),
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
                                color: Colors.black.withOpacity(0.1)
                            )
                          ],
                          shape: BoxShape.circle,
                          image: DecorationImage(
                            fit: BoxFit.cover,
                            image: _imageFile == null
                                ? AssetImage("assets/images/empty-dp.png") as ImageProvider<Object>
                                : FileImage(File(_imageFile?.path ?? '')),
                          )
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 30),
              buildTextFieldViewValue("First Name", cachedProfile?.firstName ?? ''),
              buildTextFieldViewValue("Last Name", cachedProfile?.lastName ?? ''),
              buildTextFieldViewValue("Username", cachedProfile?.username ?? ''),
              // buildTextFieldViewValue("Password", "******"),
              buildTextFieldViewValue("Email", cachedProfile?.email ?? ''),
              buildTextFieldViewValue("Phone No", cachedProfile?.phone ?? ''),
              buildTextFieldViewValue("Assigned Patients", cachedProfile?.assignedPatients.toString() ?? ''),
              // Add the ListView.builder for the assigned patients
              // ListView.builder(
              //   shrinkWrap: true, // To allow the ListView to take the necessary height
              //   itemCount: assignedPatients.length,
              //   itemBuilder: (context, index) {
              //     final patient = assignedPatients[index];
              //     return buildTextField("Address", patient.name, false);
              //   },
              // ),
              SizedBox(height: 30),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  OutlinedButton(
                    onPressed: () {
                      FirebaseAuth.instance.signOut();
                      Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SignInScreen()),
                      );
                    },
                    child: const Text("LOGOUT",
                        style: TextStyle(
                            fontSize: 15,
                            letterSpacing: 2,
                            color: Colors.redAccent
                        )),
                    style: OutlinedButton.styleFrom(
                        padding: EdgeInsets.symmetric(horizontal: 50),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20))
                    ),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => ProfileScreen())
                      );
                    },
                    child: Text("EDIT", style: TextStyle(
                        fontSize: 15,
                        letterSpacing: 2,
                        color: Colors.white
                    )),
                    style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        padding: EdgeInsets.symmetric(horizontal: 50),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20))
                    ),
                  )
                ],
              )
            ],
          ),
        ),

      ),
    );
  }

}
