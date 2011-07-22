#!/usr/bin/env python
import opennpy
import cv
import frame_convert

cv.NamedWindow('Depth')
cv.NamedWindow('Video')
print('Press ESC in window to stop')


def get_depth():
    return frame_convert.pretty_depth_cv(opennpy.sync_get_depth()[0])


def get_video():
    return frame_convert.video_cv(opennpy.sync_get_video()[0])


while 1:
    cv.ShowImage('Depth', get_depth())
    cv.ShowImage('Video', get_video())
    print(opennpy.get_fov())
    if cv.WaitKey(10) == 27:
        break
