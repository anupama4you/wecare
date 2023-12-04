/* eslint-disable max-len */
import * as functions from "firebase-functions";
import * as express from "express";
import Routes from "./routes";
import {onDatabaseChange} from "./onDatabaseChange";
import {onStorageChange} from "./onStorageChange";
import {onPatientImageChange} from "./roomExit";
import {onRoomExitChange} from "./onRoomExitChange";
const watchdogGamma = express();

watchdogGamma.use(Routes);

exports.app = functions.region("australia-southeast1").
  https.onRequest(watchdogGamma);

exports.onDatabaseChange = onDatabaseChange;

exports.onStorageChange = onStorageChange;

exports.onPatientImageChange=onPatientImageChange;

exports.onRoomExitChange=onRoomExitChange;
