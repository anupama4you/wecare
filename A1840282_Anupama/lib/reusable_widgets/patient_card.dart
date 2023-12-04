import 'package:flutter/material.dart';
import 'package:watchdog_correct/screens/add_patient_profile.dart';
import 'package:watchdog_correct/screens/patient_profile.dart';

class PatientCard extends StatefulWidget {
  final bool allowedInBed;
  final bool allowedInRoom;
  final String careGiverId;
  final String firstName;
  final String id;
  final List<dynamic> imageUrls;
  final String lastName;
  final String bedNum;
  final String roomNum;

  const PatientCard({
    required this.allowedInBed,
    required this.allowedInRoom,
    required this.careGiverId,
    required this.firstName,
    required this.id,
    required this.imageUrls,
    required this.lastName,
    required this.bedNum,
    required this.roomNum
  });


  @override
  _PatientCardState createState() => _PatientCardState();
}

class _PatientCardState extends State<PatientCard> {
  List<bool> _isSelected = [];

  @override
  void initState() {
    super.initState();
    _isSelected = [widget.allowedInRoom, widget.allowedInBed];
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        // Navigate to the patient details screen and pass the patient data
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (context) => PatientProfileScreenView(
              patientData: {
                'allowedInBed': widget.allowedInBed,
                'allowedInRoom': widget.allowedInRoom,
                'careGiverId': widget.careGiverId,
                'firstName': widget.firstName,
                'id': widget.id,
                'imageUrls': widget.imageUrls,
                'lastName': widget.lastName,
                'bedNum': widget.bedNum,
                'roomNum': widget.roomNum
              },
            ),
          ),
        );
      },
      child: Card(
        elevation: 4,
        margin: EdgeInsets.all(16),
        child: ListTile(
          leading: CircleAvatar(
            backgroundImage: NetworkImage(widget.imageUrls[0]),
          ),
          title: Text('${widget.firstName} ${widget.lastName}'),
          subtitle: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Bed: ${widget.bedNum}, Room: ${widget.roomNum}'),
              // Text('Age: ${widget.age}'),
              Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween, // Adjust alignment as needed
                  children: [
                    Row(
                      children: [
                        Icon(
                          widget.allowedInRoom ? Icons.check_circle : Icons.cancel,
                          color: widget.allowedInRoom ? Colors.green : Colors.red,
                        ),
                        SizedBox(width: 4),
                        Text('In Room'),
                      ],
                    ),
                    Row(
                      children: [
                        Icon(
                          widget.allowedInBed ? Icons.check_circle : Icons.cancel,
                          color: widget.allowedInBed ? Colors.green : Colors.red,
                        ),
                        SizedBox(width: 4),
                        Text('In Bed'),
                      ],
                    ),
                  ]
              )
            ],
          ),
        ),
      )
    );
  }

  Widget _buildToggleButton(String label, IconData icon, int index) {
    return Container(
      margin: EdgeInsets.symmetric(horizontal: 4), // Add horizontal margin
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: _isSelected[index] ? Colors.blue : Colors.black), // Change icon color based on selected state
          SizedBox(width: 4), // Add small spacing between icon and text
          Text(
            label,
            style: TextStyle(
              color: _isSelected[index] ? Colors.blue : Colors.black, // Change text color based on selected state
              fontWeight: _isSelected[index] ? FontWeight.bold : FontWeight.normal, // Bold font weight when selected
            ),
          ),
        ],
      ),
    );
  }

}
