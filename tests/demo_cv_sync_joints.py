#!/usr/bin/env python
import opennpy
import cv
import frame_convert
import time
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
    #cv.ShowImage('Depth', get_depth())
    vid = get_video()
    #vid = get_depth()
    if joints:
        pprint.pprint(joints)
        for x, y in joints[0]['image_joints'].items():
            if joints[0]['world_joints'][x][2] == 0.:
                continue
            try:
                cv.Circle(vid, tuple(map(int, y)), 5, (0, 0, 255), -1)
            except:
                continue
    cv.ShowImage('Video', vid)
    if cv.WaitKey(10) == 27:
        break
