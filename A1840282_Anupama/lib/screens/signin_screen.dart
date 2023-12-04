import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:watchdog_correct/reusable_widgets/reusable_widget.dart';
import 'package:watchdog_correct/reusable_widgets/validation_utils.dart';
import 'package:watchdog_correct/screens/signup_screen.dart';
import 'package:watchdog_correct/utils/color_utils.dart';

import 'home_screen.dart';

class SignInScreen extends StatefulWidget {
  const SignInScreen({Key? key}) : super(key: key);

  @override
  State<SignInScreen> createState() => _SignInScreenState();
}

class _SignInScreenState extends State<SignInScreen> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _passwordTextController = TextEditingController();
  final TextEditingController _emailTextController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Form(
        key: _formKey,
        child: Container(
          width: MediaQuery.of(context).size.width,
          height: MediaQuery.of(context).size.height,
          decoration: BoxDecoration(gradient: backgroundGradient()),
          child: SingleChildScrollView(
            child: Padding(
              padding: EdgeInsets.fromLTRB(
                  5, MediaQuery.of(context).size.height * 0.2, 5, 0),
              child: Column(
                children: <Widget>[
                  logoWidget("assets/images/wecare-logo.png"),
                  // const SizedBox(
                  //   height: 10,
                  // ),
                  ReusableTextField(
                    text: "Enter Email",
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
                    icon: Icons.lock_outline,
                    isPasswordType: true,
                    controller: _passwordTextController,
                    validator: (value) {
                      if (value!.isEmpty) {
                        return "Please enter a password";
                      }
                      if (!ValidationUtils.isValidPassword(value)) {
                        return "Please enter a valid password";
                      }
                      return null;
                    },
                  ),

                  const SizedBox(
                    height: 20,
                  ),
                  firebaseUIButton(context, "Sign In", () async {
                    if (_formKey.currentState!.validate()) {
                      // Form is valid, show loading indicator
                      EasyLoading.show(status: 'loading...');

                      try {
                        final userCredential = await FirebaseAuth.instance
                            .signInWithEmailAndPassword(
                            email: _emailTextController.text,
                            password: _passwordTextController.text);

                        // Hide loading indicator when login is successful
                        EasyLoading.dismiss();
                        EasyLoading.showSuccess('Successfully signed in!');

                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => HomeScreen()),
                        );
                      } on FirebaseAuthException catch (e) {
                        // Hide loading indicator when there is an error
                        EasyLoading.dismiss();

                        String errorMessage = 'An error occurred, please try again.';
                        if (e.code == 'invalid-email') {
                          errorMessage = 'Invalid email address.';
                        } else if (e.code == 'emailAlreadyExists') {
                          errorMessage = 'User already exists.';
                        }
                        else if (e.code == 'user-not-found' || e.code == 'wrong-password') {
                          errorMessage = 'Invalid email or password.';
                        }

                        // Show error message to the user
                        EasyLoading.showError(errorMessage);
                      } catch (error) {
                        print("Error: $error");
                      }
                    }
                  }),
                  signUpOption()
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Row signUpOption() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Text("Don't have an account?", style: TextStyle(color: Colors.white70)),
        GestureDetector(
          onTap: () {
            Navigator.push(context,
                MaterialPageRoute(builder: (context) => const SignUpScreen()));
          },
          child: const Text(
            " Sign Up",
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
          ),
        ),
        const SizedBox(
          height: 10,
        ),
      ],
    );
  }
}
