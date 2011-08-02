#include <XnCppWrapper.h> 
#include <cstdio>
#include <cstdlib>
#include <cmath>
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
SceneMetaData sceneData;
int initialized = 0;

int opennpy_init(void)
{
    XnStatus nRetVal = XN_STATUS_OK;
    nRetVal = context.Init();

    CHECK_RC(nRetVal, "Initialize context");
    nRetVal = depthGenerator.Create(context);
    CHECK_RC(nRetVal, "Create depth generator");
    nRetVal = imageGenerator.Create(context);
    CHECK_RC(nRetVal, "Create image generator");
    nRetVal = userGenerator.Create(context);
    CHECK_RC(nRetVal, "Create user generator");
    XnMapOutputMode mapMode; 
    mapMode.nXRes = 640;
    mapMode.nYRes = 480;
    mapMode.nFPS = 30; 
    depthGenerator.SetMapOutputMode(mapMode); 
    imageGenerator.SetMapOutputMode(mapMode); 
    nRetVal = context.StartGeneratingAll();
    CHECK_RC(nRetVal, "StartGeneratingAll");
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
    //depthGenerator.GetAlternativeViewPointCap().SetViewPoint(imageGenerator);
    imageGenerator.GetAlternativeViewPointCap().SetViewPoint(depthGenerator);
    context.WaitOneUpdateAll(imageGenerator); 
    depthGenerator.WaitAndUpdateData(); 
}

void opennpy_get_fov(double *hfov, double *vfov) {
    if (!initialized)
        opennpy_init();
    XnFieldOfView fov;
    depthGenerator.GetFieldOfView(fov);
    *hfov = fov.fHFOV;
    *vfov = fov.fVFOV;
}

void opennpy_depth_to_3d(uint16_t *depth, double *world, double hfov, double vfov) {
    double z;
    int x, y;
    const double x2z = tan(hfov / 2) * 2;
    const double y2z = tan(vfov / 2) * 2;
    for (y = 0; y < 480; ++y)
        for (x = 0; x < 640; ++x, world += 3) {
            z = depth[y * 640 + x];
            world[0] = ((x / 640.) - .5) * x2z * z;
            world[1] = (.5 - (y / 480.)) * y2z * z;
            world[2] = z;
        }
}
