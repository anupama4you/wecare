import * as functions from "firebase-functions";
import axios from "axios";
import {debug} from "firebase-functions/logger";

const link = "https://watchdog-gamma.ts.r.appspot.com/preprocess";
export const onStorageChange = functions.region("australia-southeast1")
  .storage
  .bucket("watchdog-gamma.appspot.com")
  .object()
  .onFinalize(async (object) => {
    const filePath = object.name;
    const contentType = object.contentType;

    if (filePath && filePath.startsWith("patients/")) {
      // Check if the file is a video
      const parts = filePath.split("/");
      const roomNum = parts[1];
      console.log(filePath);
      debug("filepath:"+filePath);
      debug("Object:"+object);
      debug("contentType"+contentType);
      if (filePath.endsWith(".mp4")) {
        // Make an HTTP request to your App Engine application
        await axios.post(link, {
          filePath: filePath,
          roomNum: roomNum,
        });
      } else {
        console.log("Uploaded file is not a video.");
      }
    }
    return null;
  });


