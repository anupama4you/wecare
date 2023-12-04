class PatientSelect {
  final String name;
  bool isSelected;

  PatientSelect({required this.name, this.isSelected = false});
}

class Patient {
  String id;
  String firstName;
  String lastName;
  bool allowedInBed;
  bool allowedInRoom;
  String careGiverId;

  Patient({
    required this.id,
    required this.firstName,
    required this.lastName,
    required this.allowedInBed,
    required this.allowedInRoom,
    required this.careGiverId,
  });
}
