/* Adapted by Brandyn White for Opennpy */
/*******************************************************************************
*                                                                              *
*   PrimeSense NITE 1.3 - Players Sample                                       *
*   Copyright (C) 2010 PrimeSense Ltd.                                         *
*                                                                              *
*******************************************************************************/

#include <XnOpenNI.h>
#include <XnCodecIDs.h>
#include <XnCppWrapper.h>
#include "tracker.h"
#include "opennpy_aux.h"

using namespace xn;

extern Context context;
extern DepthGenerator depthGenerator;
extern ImageGenerator imageGenerator;
extern UserGenerator userGenerator;
extern DepthMetaData depthData; 
extern ImageMetaData imageData;
extern SceneMetaData sceneData;

XnUserID num_players = 0;
XnBool is_calibrated = FALSE;
int tracker_inited = 0;
extern int initialized;

int GetSkel(XnUserID user, XnSkeletonJointPosition joints[24], XnPoint3D proj_joints[24])
{
    if (!userGenerator.GetSkeletonCap().IsCalibrated(user)) {
        printf("not calibrated\n");
        return 1;
    }
    if (!userGenerator.GetSkeletonCap().IsTracking(user)) {
        printf("not tracked\n");
        return 1;
    }
    for (int i = 0; i < 24; ++i) {
        userGenerator.GetSkeletonCap().GetSkeletonJointPosition(user, (XnSkeletonJoint)i, joints[i]);
        proj_joints[i] = joints[i].position;
    }
    depthGenerator.ConvertRealWorldToProjective(24, proj_joints, proj_joints);
    return 0;
}

void PrintSkel(XnSkeletonJointPosition joints[24]) {
    for (int i = 0; i < 24; ++i)
        printf("[%d] %f %f %f\n", i, joints[i].position.X, joints[i].position.Y, joints[i].position.Z);
}

XnBool AssignPlayer(XnUserID user)
{
    if (num_players != 0)
        return FALSE;

    XnPoint3D com;
    userGenerator.GetCoM(user, com);
    if (com.Z == 0)
        return FALSE;

    printf("Matching for existing calibration\n");
    userGenerator.GetSkeletonCap().LoadCalibrationData(user, 0);
    userGenerator.GetSkeletonCap().StartTracking(user);
    num_players = user;
    return TRUE;

}
void XN_CALLBACK_TYPE NewUser(UserGenerator& generator, XnUserID user, void* pCookie)
{
    if (!is_calibrated) // check on player0 is enough
    {
        printf("Look for pose\n");
        userGenerator.GetPoseDetectionCap().StartPoseDetection("Psi", user);
        return;
    }

    AssignPlayer(user);
}
void FindPlayer()
{
    if (num_players != 0)
    {
        return;
    }
    XnUserID aUsers[20];
    XnUInt16 nUsers = 20;
    userGenerator.GetUsers(aUsers, nUsers);

    for (int i = 0; i < nUsers; ++i)
    {
        if (AssignPlayer(aUsers[i]))
            return;
    }
}
void LostPlayer()
{
    num_players = 0;
    FindPlayer();

}
void XN_CALLBACK_TYPE LostUser(UserGenerator& generator, XnUserID user, void* pCookie)
{
    printf("Lost user %d\n", user);
    if (num_players == user)
    {
        LostPlayer();
    }
}
void XN_CALLBACK_TYPE PoseDetected(PoseDetectionCapability& pose, const XnChar* strPose, XnUserID user, void* cxt)
{
    printf("Found pose \"%s\" for user %d\n", strPose, user);
    userGenerator.GetSkeletonCap().RequestCalibration(user, TRUE);
    userGenerator.GetPoseDetectionCap().StopPoseDetection(user);
}

void XN_CALLBACK_TYPE CalibrationStarted(SkeletonCapability& skeleton, XnUserID user, void* cxt)
{
    printf("Calibration started\n");
}

void XN_CALLBACK_TYPE CalibrationEnded(SkeletonCapability& skeleton, XnUserID user, XnBool bSuccess, void* cxt)
{
    printf("Calibration done [%d] %ssuccessfully\n", user, bSuccess?"":"un");
    if (bSuccess)
    {
        if (!is_calibrated)
        {
            userGenerator.GetSkeletonCap().SaveCalibrationData(user, 0);
            num_players = user;
            userGenerator.GetSkeletonCap().StartTracking(user);
            is_calibrated = TRUE;
        }

        XnUserID aUsers[10];
        XnUInt16 nUsers = 10;
        userGenerator.GetUsers(aUsers, nUsers);
        for (int i = 0; i < nUsers; ++i)
            userGenerator.GetPoseDetectionCap().StopPoseDetection(aUsers[i]);
    }
}

void XN_CALLBACK_TYPE CalibrationCompleted(SkeletonCapability& skeleton, XnUserID user, XnCalibrationStatus eStatus, void* cxt)
{
    printf("Calibration done [%d] %ssuccessfully\n", user, (eStatus == XN_CALIBRATION_STATUS_OK)?"":"un");
    if (eStatus == XN_CALIBRATION_STATUS_OK)
    {
        if (!is_calibrated)
        {
            userGenerator.GetSkeletonCap().SaveCalibrationData(user, 0);
            num_players = user;
            userGenerator.GetSkeletonCap().StartTracking(user);
            is_calibrated = TRUE;
        }

        XnUserID aUsers[10];
        XnUInt16 nUsers = 10;
        userGenerator.GetUsers(aUsers, nUsers);
        for (int i = 0; i < nUsers; ++i)
            userGenerator.GetPoseDetectionCap().StopPoseDetection(aUsers[i]);
    }
}

#define CHECK_RC(rc, what)											\
	if (rc != XN_STATUS_OK)											\
	{																\
		printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
		return rc;													\
	}

int read_data(XnSkeletonJointPosition joints[24], XnPoint3D proj_joints[24]) {
    // Update internal buffers
    context.WaitAndUpdateAll();
    depthGenerator.GetMetaData(depthData);
    imageGenerator.GetMetaData(imageData);
    userGenerator.GetUserPixels(0, sceneData);
    if (num_players != 0)
    {
        XnPoint3D com;
        userGenerator.GetCoM(num_players, com);
        if (com.Z == 0)
        {
            num_players = 0;
            FindPlayer();
        }
    }
    return GetSkel(num_players, joints, proj_joints);
}


int init_tracker()
{
    XnStatus rc = XN_STATUS_OK;
    EnumerationErrors errors;
    if (!initialized)
        opennpy_init();
    
    if (!userGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON) ||
            !userGenerator.IsCapabilitySupported(XN_CAPABILITY_POSE_DETECTION))
    {
        printf("User generator doesn't support either skeleton or pose detection.\n");
        return XN_STATUS_ERROR;
    }

    userGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);

    rc = context.StartGeneratingAll();
    CHECK_RC(rc, "StartGenerating");

    XnCallbackHandle hUserCBs, hCalibrationStartCB, hCalibrationCompleteCB, hPoseCBs;
    userGenerator.RegisterUserCallbacks(NewUser, LostUser, NULL, hUserCBs);
    rc = userGenerator.GetSkeletonCap().RegisterToCalibrationStart(CalibrationStarted, NULL, hCalibrationStartCB);
    CHECK_RC(rc, "Register to calbiration start");
    rc = userGenerator.GetSkeletonCap().RegisterToCalibrationComplete(CalibrationCompleted, NULL, hCalibrationCompleteCB);
    CHECK_RC(rc, "Register to calibration complete");
    rc = userGenerator.GetPoseDetectionCap().RegisterToPoseDetected(PoseDetected, NULL, hPoseCBs);
    CHECK_RC(rc, "Register to pose detected");
    tracker_inited = 1;
}

int get_joints(double *out_joints, double *out_proj_joints, double *out_conf, uint16_t **depth_data,
               uint8_t **image_data, uint16_t **scene_data)
{
    XnSkeletonJointPosition joints[24];
    XnPoint3D proj_joints[24];
    if (!tracker_inited)
        init_tracker();
    int out_val = read_data(joints, proj_joints);
    //if (!out_val)
    //    PrintSkel(joints);
    if (!out_val)
        for (int i = 0; i < 24; ++i, out_joints += 3, out_proj_joints += 2, ++out_conf) {
            out_joints[0] = joints[i].position.X;
            out_joints[1] = joints[i].position.Y;
            out_joints[2] = joints[i].position.Z;
            out_proj_joints[0] = proj_joints[i].X;
            out_proj_joints[1] = proj_joints[i].Y;
            *out_conf = joints[i].fConfidence;
        }
    *depth_data = (uint16_t*) depthData.Data();
    *image_data = (uint8_t*) imageData.Data();
    *scene_data = (uint16_t*) sceneData.Data();
    return out_val;
}
