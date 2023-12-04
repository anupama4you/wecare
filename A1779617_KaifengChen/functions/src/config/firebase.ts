import * as functions from "firebase-functions";
import * as admin from "firebase-admin";

admin.initializeApp({
  credential: admin.credential.cert({
    privateKey: functions.config().private.key.replace(/\\n/g, "\n"),
    projectId: functions.config().project.id,
    clientEmail: functions.config().client.email,
  }),
  databaseURL: "https://falldetection-and-bedexits.firebaseio.com",
  storageBucket: "watchdog-gamma.appspot.com",
});

const db = admin.firestore();
const realtimeDb = admin.database();
const messaging = admin.messaging();
const bucket = admin.storage();

export {admin, db, realtimeDb, messaging, bucket};
