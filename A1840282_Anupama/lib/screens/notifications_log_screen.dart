import 'package:flutter/material.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:provider/provider.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:watchdog_correct/classes/notification_provider.dart';

import '../utils/color_utils.dart';

class NotificationsScreen extends StatelessWidget {
  const NotificationsScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final notificationsProvider = Provider.of<NotificationsProvider>(context);
    final user = FirebaseAuth.instance.currentUser;

    String formatTimeDifference(Duration difference) {
      if (difference.inDays > 0) {
        return '${difference.inDays} days ago';
      } else if (difference.inHours > 0) {
        return '${difference.inHours} hours ago';
      } else if (difference.inMinutes > 0) {
        return '${difference.inMinutes} min ago';
      } else {
        return '${difference.inSeconds} sec ago';
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Notifications',
          style: TextStyle(
            fontSize: 24,
          ),
        ),
        elevation: 0,
      ),
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        decoration: BoxDecoration(gradient: backgroundGradient()),
        child: StreamBuilder(
          stream: FirebaseFirestore.instance
              .doc('notification/${user?.uid}')
              .collection('logs')
              .orderBy('timestamp', descending: true)
              .snapshots(),
          builder: (context, snapshot) {
            if (!snapshot.hasData) {
              EasyLoading.show(status: 'loading...');
              return Container();
            }
            final notifications = snapshot.data?.docs;
            EasyLoading.dismiss();
            return Padding(
              padding: const EdgeInsets.all(16.0),
              child: ListView.builder(
                itemCount: notifications?.length,
                itemBuilder: (context, index) {
                  final notification = notifications?[index];
                  final notificationData = notification!.data();

                  final timestamp = notificationData['timestamp'] as Timestamp;
                  final timeDifference = DateTime.now().difference(timestamp.toDate());

                  // Check if the notification is within the last 10 minutes
                  final isRecent = timeDifference.inMinutes <= 10;

                  return Card(
                      elevation: 2,
                      margin: EdgeInsets.symmetric(vertical: 8.0),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12.0),
                      ),
                      child: Column(
                        children: [
                      ListTile(
                      title: Text(
                      notification['notification_msg']['responseType'],
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 18,
                          color: isRecent ? Colors.red : Colors.black,
                        ),
                      ),
                      subtitle: Text(
                        notification['notification_msg']['response'],
                        style: TextStyle(
                          fontSize: 16,
                        ),
                      ),
                      ),
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                  child: Text(
                  formatTimeDifference(timeDifference),
                  style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey,
                  ),
                    ),
                    ),
                  ],
                  ),
                  );
                },
              ),
            );
          },
        ),
      ),
    );
  }
}
