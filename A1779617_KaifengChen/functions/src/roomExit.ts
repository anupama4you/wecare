// import {realtimeDb} from "./config/firebase";
import * as functions from "firebase-functions";
import axios from "axios";

const link="http://35.244.89.9:5000/update";
export const onPatientImageChange = functions.region("australia-southeast1")
  .storage
  .bucket("watchdog-gamma.appspot.com")
  .object()
  .onFinalize(async (object) => {
    const filePath = object.name;
    if (filePath && filePath.startsWith("Patients_Images/")) {
      try {
        await axios.get(link);
      } catch (error) {
        console.error(`Failed to make GET request: ${error}`);
      }
    }
    return null;
  });

