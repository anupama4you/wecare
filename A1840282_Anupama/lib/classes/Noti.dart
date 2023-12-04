import 'dart:convert';

import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_tts/flutter_tts.dart';

class NotificationService {
  static final FlutterLocalNotificationsPlugin notificationsPlugin =
  FlutterLocalNotificationsPlugin();

  static void initialize() {

    const InitializationSettings initializationSettings = InitializationSettings(
      android: AndroidInitializationSettings('@mipmap/ic_launcher'),
    );

    notificationsPlugin.initialize(initializationSettings);
  }

  static void showNotification(String message, String messageType) async{
    try {
      final FlutterTts flutterTts = FlutterTts();
      final id = DateTime.now().microsecondsSinceEpoch ~/ 1000;
      const NotificationDetails notificationDetails = NotificationDetails(
        android: AndroidNotificationDetails(
            "pushnotification",
            "pushnotificationchannel",

            // playSound: true,
            // sound: RawResourceAndroidNotificationSound('urgent'),
            importance: Importance.max,
            priority: Priority.high,
        )
      );

      // Map<String, dynamic> jsonData = json.decode(message.data['patientInfo']);
      // print(json.decode(message.data.toString()));

      flutterTts.speak(message);

      await notificationsPlugin.show(
          1,
          messageType,
          message,
          notificationDetails);

    } on Exception catch (e) {
      print(e);
    }
  }
}


