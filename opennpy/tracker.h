#ifndef TRACKER_H
#define TRACKER_H
#ifdef __cplusplus
extern "C" {
#endif
    #include <stdint.h>
    int get_joints(double *out_joints, double *out_proj_joints, double *out_conf, uint16_t **depth_data,
                   uint8_t **image_data, uint16_t **scene_data);
#ifdef __cplusplus
}
#endif
#endif
