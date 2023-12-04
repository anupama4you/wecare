import {Router} from "express";
import {addCaregiver, deleteCaregiver, getCaregiver, updateCaregiver}
  from "./Controllers/caregiversController";
import {addPatients, deletePatients, getPatients, getPatientsList,
  updatePatients} from
  "./Controllers/patientController";

// eslint-disable-next-line new-cap
const router = Router();
// careGivers
router.post("/caregivers/", addCaregiver);
router.put("/caregivers/:id", updateCaregiver);
router.get("/caregivers/:id", getCaregiver);
router.delete("/caregivers/:id", deleteCaregiver);

// patientss
router.post("/patients/", addPatients);
router.put("/patients/:id", updatePatients);
router.get("/patients/:id", getPatients);
router.delete("/patients/:id", deletePatients);
router.get("/getpatientlist", getPatientsList);
// router.get("/getpatientrealtimeinfo/:id", getPatientsRealtimeInfo);

export default router;
