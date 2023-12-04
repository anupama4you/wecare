import 'package:flutter/material.dart';

hexStringToColor(String hexColor) {
  hexColor = hexColor.toUpperCase().replaceAll("#", "");
  if (hexColor.length == 6) {
    hexColor = "FF" + hexColor;
  }
  return Color(int.parse(hexColor, radix: 16));
}

backgroundGradient() {
  return LinearGradient(colors: [
    hexStringToColor("#32b18e"),
    hexStringToColor("#008488"),
    hexStringToColor("#255667")
  ], begin: Alignment.topCenter, end: Alignment.bottomCenter);
}