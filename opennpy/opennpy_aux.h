#ifndef OPENNPY_AUX_H
#define OPENNPY_AUX_H

#ifdef __cplusplus
extern "C" {
#endif
    #include <stdint.h>
    int opennpy_init(void);
    uint8_t *opennpy_sync_get_video(void);
    uint16_t *opennpy_sync_get_depth(void);
    void opennpy_shutdown(void);
    void opennpy_align_depth_to_rgb(void);
    void opennpy_depth_to_3d(uint16_t *depth, double *world);
#ifdef __cplusplus
}
#endif
#endif
