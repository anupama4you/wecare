// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAQM60D2p1I2JkAM7Ff3soPFSNXOuJ_WhE",
  authDomain: "watchdog-d9ca3.firebaseapp.com",
  databaseURL:
    "https://watchdog-d9ca3-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "watchdog-d9ca3",
  storageBucket: "watchdog-d9ca3.appspot.com",
  messagingSenderId: "638990071663",
  appId: "1:638990071663:web:4f6a7f638605a5b1a61a9a",
  measurementId: "G-RJYH133LFC",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
