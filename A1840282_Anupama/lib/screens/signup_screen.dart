import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
  import 'package:http/http.dart' as http;
  import 'package:watchdog_correct/reusable_widgets/validation_utils.dart';
  import 'dart:convert';

import '../reusable_widgets/reusable_widget.dart';
import '../utils/color_utils.dart';
import 'home_screen.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({Key? key}) : super(key: key);

  @override
  _SignUpScreenState createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  TextEditingController _passwordTextController = TextEditingController();
  TextEditingController _emailTextController = TextEditingController();
  TextEditingController _userNameTextController = TextEditingController();
  TextEditingController _firstNameTextController = TextEditingController();
  TextEditingController _lastNameTextController = TextEditingController();

  bool _isLoading = false; // Added state variable for loading indicator

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          "Sign Up",
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
        ),
      ),
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        decoration: BoxDecoration(gradient: backgroundGradient()),
        child: SingleChildScrollView(
          child: Padding(
            padding: EdgeInsets.fromLTRB(20, 120, 20, 0),
            child: Form(
              child: Column(
                children: <Widget>[
                  const SizedBox(
                    height: 20,
                  ),
                  ReusableTextField(
                    text: "Enter First Name",
                    icon: Icons.person_outline,
                    isPasswordType: false,
                    controller: _firstNameTextController,
                    validator: (value) {
                      if (value!.isEmpty) {
                        return "Please enter your first name";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  ReusableTextField(
                    text: "Enter Last Name",
                    icon: Icons.person_outline,
                    isPasswordType: false,
                    controller: _lastNameTextController,
                    validator: (value) {
                      if (value!.isEmpty) {
                        return "Please enter your last name";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  ReusableTextField(
                    text: "Enter User Name",
                    icon: Icons.person_outline,
                    isPasswordType: false,
                    controller: _userNameTextController,
                    validator: (value) {
                      if (value!.isEmpty) {
                        return "Please enter a username";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  ReusableTextField(
                    text: "Enter Email Id",
                    icon: Icons.email_outlined,
                    isPasswordType: false,
                    controller: _emailTextController,
                    validator: (value) {
                      if (value!.isEmpty) {
                        return "Please enter an email";
                      }
                      if (!ValidationUtils.isValidEmail(value)) {
                        return "Please enter a valid email";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  ReusableTextField(
                    text: "Enter Password",
                    icon: Icons.lock_outlined,
                    isPasswordType: true,
                    controller: _passwordTextController,
                    validator: (value) {
                      if (value!.isEmpty) {
                        return "Please enter a password";
                      }
                      if (!ValidationUtils.isValidPassword(value)) {
                        return "Password must be at least 6 characters long and contain at least one letter and one digit";
                      }
                      return null;
                    },
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  _isLoading // Show loading indicator if _isLoading is true
                      ? CircularProgressIndicator()
                      : firebaseUIButton(context, "Sign up", () async {
                    setState(() {
                      _isLoading = true; // Show loader when the button is pressed
                    });
                    EasyLoading.show(status: 'loading...');
                    try {
                      await FirebaseAuth.instance
                          .createUserWithEmailAndPassword(
                          email: _emailTextController.text,
                          password: _passwordTextController.text)
                          .then((value) async {

                        final Map<String, dynamic> userData = {
                          'userId':
                          FirebaseAuth.instance.currentUser?.uid,
                          'username': _userNameTextController.text,
                          'email': _emailTextController.text,
                          'firstName': _firstNameTextController.text,
                          'lastName': _lastNameTextController.text,
                          'phone': '',
                          'assignedPatients': [],
                          'imageUrl': ''
                        };

                        final jsonData = json.encode(userData,
                            toEncodable: (dynamic value) {
                              if (value == null) {
                                return null;
                              }
                              return value;
                            });

                        final response = await http.post(
                          Uri.parse(
                              'https://australia-southeast1-watchdog-gamma.cloudfunctions.net/app/caregivers'),
                          headers: {'Content-Type': 'application/json'},
                          body: jsonData,
                        );

                        print('json-data ${jsonData}');

                        if (response.statusCode == 200) {
                          ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                  content: Text('Created New Account')));
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (context) => HomeScreen()),
                          );
                        } else {
                          print(response.body);
                          ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                  content: Text('Failed to add user')));
                        }
                      });
                    } catch (error) {
                      if (error is FirebaseAuthException) {
                        String errorMessage = '';

                        if (error.code == 'email-already-in-use') {
                          errorMessage =
                          'The email address is already in use by another account.';
                        } else if (error.code == 'weak-password') {
                          errorMessage = 'The password is too weak.';
                        } else if (error.code == 'invalid-email') {
                          errorMessage = 'Invalid email address.';
                        } else {
                          errorMessage =
                          'An error occurred: ${error.message}';
                        }

                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text(errorMessage)),
                        );
                      } else {
                        print("Error: $error");
                      }
                    } finally {
                      setState(() {
                        _isLoading = false; // Hide loader when done
                      });
                      EasyLoading.dismiss(); // Dismiss loading indicator
                    }
                  })
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
