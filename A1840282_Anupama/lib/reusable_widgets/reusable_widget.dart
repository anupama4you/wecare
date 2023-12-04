import 'package:flutter/material.dart';

Image logoWidget(String imageName) {
  return Image.asset(
    imageName,
    fit: BoxFit.fitWidth,
    width: 240,
    height: 240,
    color: Colors.white,
  );
}

Image logoWidgetDrawer(String imageName) {
  return Image.asset(
    imageName,
    fit: BoxFit.fitWidth,
    width: 180,
    height: 180,
    color: Colors.black87,
  );
}


class ReusableTextField extends StatefulWidget {
  final String text;
  final IconData icon;
  final bool isPasswordType;
  final TextEditingController controller;
  final String? Function(String?)? validator;

  ReusableTextField({
    required this.text,
    required this.icon,
    required this.isPasswordType,
    required this.controller,
    this.validator,
  });

  @override
  _ReusableTextFieldState createState() => _ReusableTextFieldState();
}

class _ReusableTextFieldState extends State<ReusableTextField> {
  bool _isPasswordVisible = false;

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      controller: widget.controller,
      obscureText: widget.isPasswordType && !_isPasswordVisible,
      enableSuggestions: !widget.isPasswordType,
      autocorrect: !widget.isPasswordType,
      cursorColor: Colors.white,
      style: TextStyle(color: Colors.white.withOpacity(0.9)),
      decoration: InputDecoration(
        prefixIcon: Icon(
          widget.icon,
          color: Colors.white70,
        ),
        labelText: widget.text,
        labelStyle: TextStyle(color: Colors.white.withOpacity(0.9)),
        filled: true,
        floatingLabelBehavior: FloatingLabelBehavior.never,
        fillColor: Colors.white.withOpacity(0.3),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(30.0),
          borderSide: const BorderSide(width: 0, style: BorderStyle.none),
        ),
        suffixIcon: widget.isPasswordType
            ? IconButton(
          onPressed: () {
            setState(() {
              _isPasswordVisible = !_isPasswordVisible;
            });
          },
          icon: Icon(
            _isPasswordVisible
                ? Icons.visibility
                : Icons.visibility_off,
            color: Colors.white70,
          ),
        )
            : null,
      ),
      keyboardType: widget.isPasswordType
          ? TextInputType.visiblePassword
          : TextInputType.emailAddress,
      validator: widget.validator,
    );
  }
}



Container firebaseUIButton(BuildContext context, String title, Function onTap) {
  return Container(
    width: MediaQuery.of(context).size.width,
    height: 50,
    margin: const EdgeInsets.fromLTRB(0, 10, 0, 20),
    decoration: BoxDecoration(borderRadius: BorderRadius.circular(90)),
    child: ElevatedButton(
      onPressed: () {
        onTap();
      },
      child: Text(
        title,
        style: const TextStyle(
            color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 16),
      ),
      style: ButtonStyle(
          backgroundColor: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.pressed)) {
              return Colors.black26;
            }
            return Colors.white;
          }),
          shape: MaterialStateProperty.all<RoundedRectangleBorder>(
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)))),
    ),
  );
}

Widget buildTextFieldViewValue(String labelText, String fieldValue) {
  return Padding(
    padding: EdgeInsets.only(bottom: 30),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          labelText,
          style: TextStyle(
            color: Colors.white70,
            fontSize: 14,
          ),
        ),
        SizedBox(height: 5),
        Text(
          fieldValue,
          style: TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    ),
  );
}