class CaregiverProfile {
  final String id;
  final String username;
  final String firstName;
  final String lastName;
  final String email;
  final String phone;
  final List<dynamic> assignedPatients;

  CaregiverProfile({
    required this.id,
    required this.username,
    required this.firstName,
    required this.lastName,
    required this.email,
    required this.phone,
    required this.assignedPatients,
  });
}