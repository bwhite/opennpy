#include <XnCppWrapper.h> 
#include <cstdio>
#include <cstdlib>
#include "opennpy_aux.h"

using namespace xn; 

#define CHECK_RC(rc, what)											\
	if (rc != XN_STATUS_OK)											\
	{																\
		printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
		return rc;													\
	}

Context context;
DepthGenerator depthGenerator;
ImageGenerator imageGenerator;
UserGenerator userGenerator;

DepthMetaData depthData; 
ImageMetaData imageData;
int initialized = 0;
int user_initialized = 0;
int g_nPlayer = 0;

void XN_CALLBACK_TYPE NewUser(xn::UserGenerator& generator, XnUserID user, void* pCookie)
{
/*
	if (!g_bCalibrated) // check on player0 is enough
	{
		printf("Look for pose\n");
		g_UserGenerator.GetPoseDetectionCap().StartPoseDetection("Psi", user);
		return;
	}

	AssignPlayer(user);
// 	if (g_nPlayer == 0)
// 	{
// 		printf("Assigned user\n");
// 		g_UserGenerator.GetSkeletonCap().LoadCalibrationData(user, 0);
// 		g_UserGenerator.GetSkeletonCap().StartTracking(user);
// 		g_nPlayer = user;
// 	}
*/
}

void PrintSkel(XnUserID user)
{
    if (!userGenerator.GetSkeletonCap().IsCalibrated(user)) {
        printf("not calibrated\n");
        return;
    }
    if (!userGenerator.GetSkeletonCap().IsTracking(user)) {
        printf("not tracked\n");
        return;
    }
    XnSkeletonJointPosition joint;
    userGenerator.GetSkeletonCap().GetSkeletonJointPosition(user, XN_SKEL_HEAD, joint);
    printf("%f %f %f\n", joint.position.X, joint.position.Y, joint.position.Z);
}

XnBool AssignPlayer(XnUserID user)
{
	if (g_nPlayer != 0)
		return FALSE;

	XnPoint3D com;
	userGenerator.GetCoM(user, com);
	if (com.Z == 0)
		return FALSE;

	printf("Matching for existing calibration\n");
	userGenerator.GetSkeletonCap().LoadCalibrationData(user, 0);
	userGenerator.GetSkeletonCap().StartTracking(user);
	g_nPlayer = user;
        PrintSkel(user);
	return TRUE;

}



void FindPlayer()
{
    printf("FindPlayer\n");

	if (g_nPlayer != 0)
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
    printf("LostPlayer\n");

	g_nPlayer = 0;
	FindPlayer();

}
void XN_CALLBACK_TYPE LostUser(xn::UserGenerator& generator, XnUserID user, void* pCookie)
{
	printf("Lost user %d\n", user);
	if (g_nPlayer == user)
	{
		LostPlayer();
	}

}
void XN_CALLBACK_TYPE PoseDetected(xn::PoseDetectionCapability& pose, const XnChar* strPose, XnUserID user, void* cxt)
{
    printf("PoseDetected\n");
/*
	printf("Found pose \"%s\" for user %d\n", strPose, user);
	g_UserGenerator.GetSkeletonCap().RequestCalibration(user, TRUE);
	g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(user);
*/
}

void XN_CALLBACK_TYPE CalibrationStarted(xn::SkeletonCapability& skeleton, XnUserID user, void* cxt)
{
	printf("Calibration started\n");
}

void XN_CALLBACK_TYPE CalibrationEnded(xn::SkeletonCapability& skeleton, XnUserID user, XnBool bSuccess, void* cxt)
{
    printf("CalibrationEnded\n");
/*
	printf("Calibration done [%d] %ssuccessfully\n", user, bSuccess?"":"un");
	if (bSuccess)
	{
		if (!g_bCalibrated)
		{
			g_UserGenerator.GetSkeletonCap().SaveCalibrationData(user, 0);
			g_nPlayer = user;
			g_UserGenerator.GetSkeletonCap().StartTracking(user);
			g_bCalibrated = TRUE;
		}

		XnUserID aUsers[10];
		XnUInt16 nUsers = 10;
		g_UserGenerator.GetUsers(aUsers, nUsers);
		for (int i = 0; i < nUsers; ++i)
			g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(aUsers[i]);
	}
*/
}

void XN_CALLBACK_TYPE CalibrationCompleted(xn::SkeletonCapability& skeleton, XnUserID user, XnCalibrationStatus eStatus, void* cxt)
{
    printf("CalibrationCompleted\n");
/*
	printf("Calibration done [%d] %ssuccessfully\n", user, (eStatus == XN_CALIBRATION_STATUS_OK)?"":"un");
	if (eStatus == XN_CALIBRATION_STATUS_OK)
	{
		if (!g_bCalibrated)
		{
			g_UserGenerator.GetSkeletonCap().SaveCalibrationData(user, 0);
			g_nPlayer = user;
			g_UserGenerator.GetSkeletonCap().StartTracking(user);
			g_bCalibrated = TRUE;
		}

		XnUserID aUsers[10];
		XnUInt16 nUsers = 10;
		g_UserGenerator.GetUsers(aUsers, nUsers);
		for (int i = 0; i < nUsers; ++i)
			g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(aUsers[i]);
	}
*/
}

int opennpy_user_init(void) {
    //if (!initialized)
    //    opennpy_init();
    printf("User Init\n");
    XnStatus nRetVal = XN_STATUS_OK;
    nRetVal = userGenerator.Create(context);
    CHECK_RC(nRetVal, "Create user generator");
    userGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);

    nRetVal = context.StartGeneratingAll();
    CHECK_RC(nRetVal, "StartGeneratingAll");

    XnCallbackHandle hUserCBs, hCalibrationStartCB, hCalibrationCompleteCB, hPoseCBs;
    userGenerator.RegisterUserCallbacks(NewUser, LostUser, NULL, hUserCBs);
    nRetVal = userGenerator.GetSkeletonCap().RegisterToCalibrationStart(CalibrationStarted, NULL, hCalibrationStartCB);
    CHECK_RC(nRetVal, "Register to calibration start");
    nRetVal = userGenerator.GetSkeletonCap().RegisterToCalibrationComplete(CalibrationCompleted, NULL, hCalibrationCompleteCB);
    CHECK_RC(nRetVal, "Register to calibration complete");
    nRetVal = userGenerator.GetPoseDetectionCap().RegisterToPoseDetected(PoseDetected, NULL, hPoseCBs);
    CHECK_RC(nRetVal, "Register to pose detected");
    
}

int opennpy_init(void)
{
    XnStatus nRetVal = XN_STATUS_OK;
    nRetVal = context.Init();

    CHECK_RC(nRetVal, "Initialize context");
    nRetVal = depthGenerator.Create(context);
    CHECK_RC(nRetVal, "Create depth generator");
    nRetVal = imageGenerator.Create(context);
    CHECK_RC(nRetVal, "Create image generator");
    XnMapOutputMode mapMode; 
    mapMode.nXRes = 640;
    mapMode.nYRes = 480;
    mapMode.nFPS = 30; 
    depthGenerator.SetMapOutputMode(mapMode); 
    imageGenerator.SetMapOutputMode(mapMode); 
    opennpy_user_init();

    initialized = 1;
    return 0;
}

uint8_t *opennpy_sync_get_video(void)
{
    if (!initialized)
        opennpy_init();
    context.WaitOneUpdateAll(imageGenerator);
    imageGenerator.GetMetaData(imageData);
    return (uint8_t *)imageData.Data();
}

uint16_t *opennpy_sync_get_depth(void)
{
    if (!initialized)
        opennpy_init();
    FindPlayer();
    context.WaitOneUpdateAll(depthGenerator);
    depthGenerator.GetMetaData(depthData); 
    return (uint16_t *)depthData.Data();
}

void opennpy_shutdown(void) {
   context.Shutdown(); 
   initialized = 0;
}

void opennpy_align_depth_to_rgb(void) {
    if (!initialized)
        opennpy_init();
    depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);
}

/*
int main(void) { 
    XnStatus nRetVal = XN_STATUS_OK;    
    printf("generating\n"); 

    for (int i = 0; i < 5; i++) {
        unsigned short *depth_ptr = sync_get_depth();
        unsigned char *rgb_ptr = sync_get_video();
        FILE * fp = fopen("out.pgm", "w");
	fprintf(fp, "P5 %d %d 65535\n", 640, 480);
        fwrite(depth_ptr, 640*480*2, 1, fp);
        fclose(fp);
        fp = fopen("out.ppm", "w");
	fprintf(fp, "P6 %d %d 255\n", 640, 480);
        fwrite(rgb_ptr, 640*480*3, 1, fp);
        fclose(fp);
        printf("Here\n");
        align_depth_to_rgb();
    }
    // context.StopGeneratingAll(); ... does not work sometimes for me, so just kill it :-/ 
 
    return 0;
} 
*/
