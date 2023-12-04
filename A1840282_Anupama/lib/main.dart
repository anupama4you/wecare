import 'dart:convert';
import 'dart:isolate';
import 'dart:ui';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:watchdog_correct/classes/Noti.dart';
import 'package:watchdog_correct/classes/notification_provider.dart';
import 'package:watchdog_correct/reusable_widgets/user_profile_provider.dart';
import 'package:watchdog_correct/screens/home_screen.dart';
import 'package:watchdog_correct/screens/signin_screen.dart';
import 'package:provider/provider.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:firebase_messaging/firebase_messaging.dart';

final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
FlutterLocalNotificationsPlugin();

final notificationsProvider = NotificationsProvider();
bool fallDetected = false;
bool bedExited = false;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  NotificationService.initialize();

  // Initialize Firebase Cloud Messaging
  FirebaseMessaging.onMessage.listen((RemoteMessage message) async {
    print("FCM Message Data (opened app): ${message.data}");

    // Check global variables before showing the notification
    if (!fallDetected && !bedExited) {
      fallDetected = true;
      bedExited = true;

      // Show notification
      NotificationService.showNotification(
          message.data['response'], message.data['responseType']);

      // Wait for 20 seconds
      await Future.delayed(const Duration(seconds: 20));

      // Reset global variables
      fallDetected = false;
      bedExited = false;
    }
  });


  FirebaseMessaging.onMessageOpenedApp.listen((RemoteMessage message) {
    print("FCM Message Data (opened app): ${message.data['response']}");
  });

  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

  runApp(
    ChangeNotifierProvider.value(
      value: notificationsProvider,
      child: MyApp(),
    ),
  );
}

Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  NotificationService.showNotification(message.data['response'], message.data['responseType']);
}

class MyApp extends StatelessWidget {

  MyApp({Key ? key}) : super(key: key);
  final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();



  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {

    return MaterialApp(
      navigatorKey: navigatorKey,
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: FirebaseAuth.instance.currentUser == null ? const SignInScreen() : const HomeScreen(),
      builder: EasyLoading.init(),
    );
  }
}
