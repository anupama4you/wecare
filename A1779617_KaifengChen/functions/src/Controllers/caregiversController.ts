import {Response} from "firebase-functions/v1";
import {db} from "../config/firebase";

/*
caregivers
*/
type EntryType = {
    userId:string,
    username:string,
    firstName: string,
    lastName: string,
    email: string,
    phone: string,
    assignedPatients:string[]
    imageUrl:string,
}

type Request ={
    body: EntryType,
    params:{id:string}
}

const addCaregiver = async (req: Request, res: Response) => {
  const {userId, username, firstName, lastName, email, phone, imageUrl,
    assignedPatients = []} = req.body;

  try {
    await db.runTransaction(async (t) => {
      const entryRef = db.collection("user").doc(userId);
      const entryObject = {
        id: userId,
        firstName,
        lastName,
        email,
        phone,
        username,
        assignedPatients,
        imageUrl,
      };

      // Add the new caregiver ID to each of the assigned patients
      for (const patientId of assignedPatients) {
        const patientRef = db.collection("patients").doc(patientId);
        const patientDoc = await t.get(patientRef);

        if (patientDoc.exists) {
          const patientData = patientDoc.data() || {};
          patientData.careGiverId = userId; // Updating the careGiverId

          await t.update(patientRef, patientData);
        }
      }

      // Add the new caregiver
      await t.set(entryRef, entryObject);
    });

    res.status(200).send({
      status: "success",
      message: "Caregiver added successfully",
    });
  } catch (error) {
    if (error instanceof Error) {
      res.status(500).json(error.message);
    } else {
      res.status(500).json("An unexpected error occurred");
    }
  }
};


const updateCaregiver = async (req: Request, res: Response) => {
  const {id}=req.params;
  const {username, firstName, lastName, email, phone,
    assignedPatients, imageUrl} = req.body;

  try {
    await db.runTransaction(async (t) => {
      const entryRef = db.collection("user").doc(id);
      const entryDoc = await t.get(entryRef);

      if (!entryDoc.exists) {
        throw new Error("No such caregiver!");
      }

      const currentData = entryDoc.data() || {};
      const newAssignedPatients = assignedPatients ||
        currentData.assignedPatients || [];

      const entryObject = {
        id,
        username: username || currentData.username,
        firstName: firstName || currentData.firstName,
        lastName: lastName || currentData.lastName,
        email: email || currentData.email,
        phone: phone || currentData.phone,
        assignedPatients: newAssignedPatients,
        imageUrl: imageUrl || currentData.imageUrl,
      };

      // Update the caregiver ID for each of the newly assigned patients
      for (const patientId of newAssignedPatients) {
        const patientRef = db.collection("patients").doc(patientId);
        const patientDoc = await t.get(patientRef);

        if (patientDoc.exists) {
          const patientData = patientDoc.data() || {};
          patientData.careGiverId = id; // Updating the careGiverId

          await t.update(patientRef, patientData);
        }
      }

      // Update the caregiver
      await t.set(entryRef, entryObject);
    });

    res.status(200).json({
      status: "success",
      message: "Caregiver updated successfully",
    });
  } catch (error) {
    if (error instanceof Error) {
      res.status(500).json(error.message);
    } else {
      res.status(500).json("An unexpected error occurred");
    }
  }
};


const deleteCaregiver = async (req: Request, res: Response): Promise<void> => {
  const {id} = req.params;

  try {
    await db.runTransaction(async (t) => {
      const entryRef = db.collection("user").doc(id);
      const entryDoc = await t.get(entryRef);

      if (!entryDoc.exists) {
        // eslint-disable-next-line max-len
        throw new Error("Caregiver does not exist. Maybe it's already deleted.");
      }

      const assignedPatients = entryDoc.data()?.assignedPatients || [];

      for (const patientId of assignedPatients) {
        const patientRef = db.collection("patients").doc(patientId);
        const patientDoc = await t.get(patientRef);

        if (patientDoc.exists) {
          const patientData = patientDoc.data() || {};
          patientData.careGiverId = null; // Removing the careGiverId

          // Update the patient's data
          await t.update(patientRef, patientData);
        }
      }

      // Delete the caregiver
      await t.delete(entryRef);
    });

    res.status(200).json({
      status: "success",
      message: "Caregiver deleted successfully",
    });
  } catch (error) {
    if (error instanceof Error) {
      res.status(500).json(error.message);
    } else {
      res.status(500).json("An unexpected error occurred");
    }
  }
};


const getCaregiver = async (req: Request, res: Response): Promise<void> => {
  const {id} = req.params;
  try {
    const entry = db.collection("user").doc(id);
    const data = await entry.get().catch((error) => {
      res.status(400).json({
        status: "error",
        message: error.message,
      });
      throw new Error("Error getting entry"); // Halting execution
    });

    res.status(200).json({
      status: "success",
      message: "entry retrieved successfully",
      data: data.data(), // Assuming you want the data of the document
    });
  } catch (error) {
    if (error instanceof Error) {
      res.status(500).json(error.message);
    } else {
      res.status(500).json("An unexpected error occurred");
    }
  }
};


export {deleteCaregiver, addCaregiver, updateCaregiver, getCaregiver};
