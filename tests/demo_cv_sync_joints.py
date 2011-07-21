#!/usr/bin/env python
import opennpy
import cv
import frame_convert
import time
import numpy as np
opennpy.align_depth_to_rgb()
#cv.NamedWindow('Depth')
cv.NamedWindow('Video')
print('Press ESC in window to stop')


def get_depth():
    return frame_convert.pretty_depth_cv(opennpy.sync_get_depth()[0])


def get_video():
    return frame_convert.video_cv(opennpy.sync_get_video()[0])

import pprint
st = time.time()
while 1:
    if time.time() - st < .2:
        continue
    st = time.time()
    joints = opennpy.sync_get_joints()
    if not joints:
        vid = opennpy.sync_get_video()[0]
    else:
        vid = joints['image']
        x, y = (joints['scene'] == 0).nonzero()
        vid[x, y, :] = 0
    vid = frame_convert.video_cv(vid)
    if joints:
        for x in joints['joints'][0].values():
            if x['world'][2] == 0. or x['conf'] < .5:
                continue
            try:
                cv.Circle(vid, tuple(map(int, x['image'])), 5, (0, 255, 0), -1)
            except:
                continue
    cv.ShowImage('Video', vid)
    if cv.WaitKey(10) == 27:
        break
