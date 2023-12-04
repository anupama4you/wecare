import 'dart:convert';

import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:provider/provider.dart';
import 'package:watchdog_correct/classes/patient_class.dart';
import 'package:watchdog_correct/reusable_widgets/drawer.dart';
import 'package:watchdog_correct/screens/add_patient_profile.dart';
import 'package:watchdog_correct/screens/caregiver_profile.dart';
import 'package:watchdog_correct/screens/caregiver_profile_view.dart';
import 'package:watchdog_correct/screens/patient_profile.dart';
import 'package:watchdog_correct/screens/signin_screen.dart';
import 'package:watchdog_correct/reusable_widgets/app_bar.dart';
import 'package:watchdog_correct/reusable_widgets/patient_card.dart';
import 'package:watchdog_correct/reusable_widgets/user_profile_provider.dart';
import 'package:watchdog_correct/utils/color_utils.dart';
import 'package:http/http.dart' as http;

import '../classes/caregiver_class.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _selectedSortOption = 'Default';

  @override
  void initState() {
    super.initState();
    subscribeToFCMTopic();
  }

  void subscribeToFCMTopic() {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      final topicName = "careGiverId-${user?.uid}";
      FirebaseMessaging.instance.subscribeToTopic(topicName);
      print("Subscribed to FCM topic: $topicName");
    }
  }

  Future<void> unsubscribeFromFCMTopic() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      final topicName = "careGiverId-${user.uid}";
      FirebaseMessaging.instance.unsubscribeFromTopic(topicName);
      print("Unsubscribed from FCM topic: $topicName");
    }
  }


  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: MyAppBar(),
      drawer: MyDrawer(
        onProfileTap: () {
          Navigator.pop(context);
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => ProfileScreen()),
          );
        },
        onLogoutTap: () {
          unsubscribeFromFCMTopic();
          FirebaseAuth.instance.signOut();
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => SignInScreen()),
          );
        },
      ),
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        decoration: BoxDecoration(gradient: backgroundGradient()),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        'Welcome, ${FirebaseAuth.instance.currentUser!.email}',
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    ),
                    ElevatedButton(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context) => PatientProfileScreen()),
                        );
                      },
                      child: Text('Add Patients'),
                    ),
                  ],
                ),
              ),

               Column(
                  children: [
                    StreamBuilder<QuerySnapshot>(
                      stream: FirebaseFirestore.instance
                          .collection('patients') // Replace with your collection name
                          .where('careGiverId', isEqualTo: FirebaseAuth.instance.currentUser?.uid) // Replace with the caregiver ID you want to filter by
                          .snapshots(),
                      builder: (context, snapshot) {
                        if (!snapshot.hasData) {
                          return Center(child: CircularProgressIndicator()); // Loading indicator
                        }
                        final patientsData = snapshot.data!.docs;

                        return ListView.builder(
                          shrinkWrap: true,
                          itemCount: patientsData.length,
                          itemBuilder: (context, index) {
                            final patientData = patientsData[index].data() as Map<String, dynamic>;
                            return PatientCard(
                              id: patientData['id'],
                              firstName: patientData['firstName'],
                              lastName: patientData['lastName'],
                              imageUrls: patientData['imageUrls'],
                              allowedInRoom: patientData['allowedInRoom'],
                              allowedInBed: patientData['allowedInBed'],
                              careGiverId: patientData['careGiverId'],
                              bedNum: patientData['bedNum'],
                              roomNum: patientData['roomNum'],
                            );
                          },
                        );
                      },
                    ),
                  ],
                ),
            ],
          ),
      ),
    ));
  }
}
