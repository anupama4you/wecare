import 'package:flutter/material.dart';

import '../classes/caregiver_class.dart';

class UserProfileProvider with ChangeNotifier {
  CaregiverProfile? _cachedProfile;

  CaregiverProfile? get cachedProfile => _cachedProfile;

  void setCachedProfile(CaregiverProfile profile) {
    _cachedProfile = profile;
    notifyListeners();
  }

  void updateCachedProfile(CaregiverProfile updatedProfile) {
    _cachedProfile = updatedProfile;
    notifyListeners();
  }
}
