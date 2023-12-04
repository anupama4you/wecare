/* eslint-disable camelcase */
/* eslint-disable max-len */
import {db, messaging} from "./config/firebase";
import * as functions from "firebase-functions";

interface DataType {
    Prev_Prev_distance: number;
    Prev_distance:number,
    Room_Exited:false;
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

export const onRoomExitChange = functions
  .region("australia-southeast1")
  .database
  .instance("roomexitdetection")
  .ref("/ROOMEXITS/{roomid}")
  .onWrite(async (change, context) => {
    try {
      const roomId = context.params.roomid;
      const afterData = change.after.val() as DataType || {};
      const beforeData = change.before.val() as DataType||{};
      const {Room_Exited: RoomExitedBefore} = beforeData;
      const {Room_Exited: RoomExitedAfter} = afterData;
      // Query Firestore for patients with the same roomid
      const patientInfo = await getPatientInfo(roomId);

      if (!patientInfo) {
        console.log(`No patient found with room ID: ${roomId}`);
        return null;
      }
      const {careGiverId, firstName, roomNum, allowedInRoom} = patientInfo;

      let payload: Payload = {response: "", responseType: ""};
      if (allowedInRoom && (RoomExitedBefore !== RoomExitedAfter)) {
        if (RoomExitedAfter) {
          payload = {
            response: `patient ${firstName} is out of Room No. ${roomNum}`,
            responseType: "roomOut alert!",
          };
          const entryId = db.collection("notification").doc().id;
          const documentPath = `notification/${careGiverId}/logs/${entryId}`;
          await sendFCMNotification(careGiverId, payload, patientInfo);
          await updateFirestoreDocument(documentPath, careGiverId, payload, entryId);
        }
      }
    } catch (error) {
      console.error("Error in onDatabaseChange:", error);
    }
    return null;
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
