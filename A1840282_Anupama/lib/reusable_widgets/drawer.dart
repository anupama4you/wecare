import 'package:flutter/material.dart';
import 'package:watchdog_correct/reusable_widgets/list_tile.dart';
import 'package:watchdog_correct/reusable_widgets/reusable_widget.dart';

class MyDrawer extends StatelessWidget {
  final void Function()? onProfileTap;
  final void Function()? onLogoutTap;
  const MyDrawer({
    super.key,
    required this.onProfileTap,
    required this.onLogoutTap
  });

  @override
  Widget build(BuildContext context) {
    return Drawer(
      // backgroundColor: Colors.grey,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          //header
          Column(
            children: [
              DrawerHeader(
                  child: logoWidgetDrawer("assets/images/wecare-logo.png")
              ),

              //home
              DrawerListTile(
                icon: Icons.home,
                text: 'H O M E',
                onTap: ()=> Navigator.pop(context),
              ),

              //profile list
              DrawerListTile(
                icon: Icons.person,
                text: 'P R O F I L E',
                onTap: onProfileTap,
              ),
            ],
          ),

          //logout
          Padding(
            padding: const EdgeInsets.only(bottom: 25.0),
            child: DrawerListTile(
              icon: Icons.logout,
              text: 'L O G O U T',
              onTap: onLogoutTap,
            ),
          ),
        ],
      ),
    );
  }
}
    
    