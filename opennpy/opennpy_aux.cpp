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

DepthMetaData depthData; 
ImageMetaData imageData;
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
