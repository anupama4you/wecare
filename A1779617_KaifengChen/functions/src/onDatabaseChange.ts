/* eslint-disable max-len */
import {db, messaging} from "./config/firebase";
import * as functions from "firebase-functions";


interface DataType {
    bedExitted: boolean;
    fallDetected: boolean;
  }
interface Payload {
      response: string;
      responseType:string
  }
interface patientInfo{
    firstName: string,
    lastName: string,
    imageUrls:string[],
    careGiverId:string,
    allowedInBed:boolean,
    allowedInRoom:boolean,
    roomNum:string,
    bedNum:string,
}

export const onDatabaseChange = functions
  .region("australia-southeast1")
  .database
  .instance("falldetection-and-bedexits")
  .ref("/BedExits_and_FallDetections/{roomid}")
  .onWrite(async (change, context) => {
    try {
      const roomId = context.params.roomid;
      const afterData = change.after.val() as DataType || {};
      const beforeData = change.before.val() as DataType||{};


      // Query Firestore for patients with the same roomid
      const patientInfo = await getPatientInfo(roomId);

      if (!patientInfo) {
        console.log(`No patient found with room ID: ${roomId}`);
        return null;
      }
      console.log(afterData);

      const {careGiverId, firstName, roomNum, allowedInBed} = patientInfo;
      const {fallDetected: fallDetectedAfter, bedExitted: bedExittedAfter} = afterData;
      const {fallDetected: fallDetectedBefore, bedExitted: bedExittedBefore}=beforeData;
      let payload: Payload = {response: "", responseType: ""};
      console.log("allowedInBed "+allowedInBed);
      console.log("bedExitted " + bedExittedAfter);
      const bedExitted = bedExittedAfter==true&&bedExittedBefore!==bedExittedAfter;
      if (allowedInBed) {
        if (bedExitted) {
          payload = {
            response: `patient ${firstName} is in Room No. ${roomNum} is out of bed`,
            responseType: "Bed alert!",
          };
        }
      }
      if (fallDetectedAfter) {
        payload = {
          response: `patient ${firstName} is in Room No. ${roomNum} is fallen`,
          responseType: "Fall alert!",
        };
      }

      //  else {
      //   payload = {
      //     response: `patient ${firstName} is in Room No. ${roomNum} is safe on bed`,
      //     responseType: "Bed alert!",
      //   };
      // }
      if ((fallDetectedBefore==false&&fallDetectedAfter)||(allowedInBed&&bedExitted)) {
        console.log(payload);
        const entryId = db.collection("notification").doc().id;
        const documentPath = `notification/${careGiverId}/logs/${entryId}`;
        await sendFCMNotification(careGiverId, payload, patientInfo);
        await updateFirestoreDocument(documentPath, careGiverId, payload, entryId);
      }
      return null;
    } catch (error) {
      console.error("Error in onDatabaseChange:", error);
      return null;
    }
  });

/**
   * Retrieves patient information based on room ID.
   *
   * @param {string} roomId - The room ID to search for.
   * @return {Promise<object|null>} - A Promise that resolves to patient information or null if not found.
   */
async function getPatientInfo(roomId: string): Promise<patientInfo | null> {
  const patientQuery = await db.collection("patients")
    .where("roomNum", "==", roomId).get();
  if (!patientQuery.empty) {
    const patientDoc = patientQuery.docs[0];
    return patientDoc.data() as patientInfo;
  }
  return null;
}


/**
   * Sends an FCM notification.
   *
   * @param {string} careGiverId - The caregiver's ID.
   * @param {object} payload - The notification payload.
   * @param {patientInfo} patientInfo -the patient information
   * @return {Promise<void>} - A Promise that resolves when the notification is sent.
   */
async function sendFCMNotification(careGiverId: string, payload: Payload, patientInfo:patientInfo) {
  const fcmPayload={
    data: {response: payload.response, responseType: payload.responseType, patientInfo: JSON.stringify(patientInfo)},
  };
  await messaging.sendToTopic(`careGiverId-${careGiverId}`, fcmPayload);
}

/**
   * Updates a Firestore document.
   *
   * @param {string} documentPath - The path to the Firestore document.
   * @param {string} careGiverId - The caregiver's ID.
   * @param {object} data - The data to update the document with.
   * @param {string} entryId - the unique id the entry.
   * @return {Promise<void>} - A Promise that resolves when the document is updated.
   */
async function updateFirestoreDocument(documentPath: string, careGiverId: string, data: any, entryId:string) {
  const docRef = db.doc(documentPath);
  await docRef.set({
    id: entryId,
    timestamp: new Date(),
    user_id: careGiverId,
    notification_msg: data,
  });
}
