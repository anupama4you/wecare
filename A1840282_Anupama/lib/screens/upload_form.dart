import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

class uploadForm extends StatefulWidget {

  final File videoFile;
  final String videoPath;

  uploadForm({required this.videoFile, required this.videoPath});

  @override
  State<uploadForm> createState() => _uploadFormState();
}

class _uploadFormState extends State<uploadForm> {
  VideoPlayerController? playerController;

  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    setState(() {
      playerController = VideoPlayerController.file(widget.videoFile);
    });

    playerController!.initialize();
    playerController!.play();
    playerController!.setVolume(2);
    playerController!.setLooping(true);
  }

  @override
  void dispose() {
    // TODO: implement dispose
    super.dispose();
    playerController!.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        child: Column(
          children: [
            //display video player
            SizedBox(
              width: MediaQuery.of(context).size.width,
              height: MediaQuery.of(context).size.height/ 1.6,
              child: VideoPlayer(playerController!),
            )

          ],
        ),
      ),
    );
  }
}
