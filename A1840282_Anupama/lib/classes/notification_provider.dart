import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class NotificationFormat {
  final String title;
  final String body;

  NotificationFormat({required this.title, required this.body});
}

class NotificationsProvider extends ChangeNotifier {
  List<NotificationFormat> _notifications = [];

  List<NotificationFormat> get notifications => _notifications;

  void addNotification(NotificationFormat notification) {
    _notifications.add(notification);
    notifyListeners();
  }
}
