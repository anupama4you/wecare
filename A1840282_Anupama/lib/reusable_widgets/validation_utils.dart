import 'dart:core';

class ValidationUtils {
  static bool isValidEmail(String email) {
    final emailRegExp =
    RegExp(r'^[\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)+$');
    return emailRegExp.hasMatch(email);
  }

  static bool isValidPassword(String password) {
    final passwordRegExp =
    RegExp(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{6,}$');
    return passwordRegExp.hasMatch(password);
  }
}
