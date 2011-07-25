#!/usr/bin/env python
# (C) Copyright 2010 Brandyn A. White
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Openni wrapper that provides a libfreenect-esque interface"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np
cimport numpy as np

cdef extern from "numpy/arrayobject.h":
    void import_array()
    cdef object PyArray_SimpleNewFromData(int nd, np.npy_intp *dims,
                                           int typenum, void *data)

cdef extern from "opennpy_aux.h":
    int opennpy_init()
    void *opennpy_sync_get_video()
    void *opennpy_sync_get_depth()
    void opennpy_shutdown()
    void opennpy_align_depth_to_rgb()
    void opennpy_get_fov(double *hfov, double *vfov)
    void opennpy_depth_to_3d(np.uint16_t *depth, double *world, double hfov, double vfov)

cdef extern from "tracker.h":
    int get_joints(np.float64_t *out_joints, np.float64_t *out_proj_joints, np.float64_t *out_conf, void **depth_data,
                   void **image_data, void **scene_data)

import_array()
timestamp = 0
cdef np.npy_intp ddims[2]
cdef np.npy_intp vdims[3]
ddims[0], ddims[1]  = 480, 640
vdims[0], vdims[1], vdims[2]  = 480, 640, 3
JOINT_LABELS = ['head', 'neck', 'torso', 'waist',
                'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand', 'left_fingertip',
                'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand', 'right_fingertip',
                'left_hip', 'left_knee', 'left_ankle', 'left_foot',
                'right_hip', 'right_knee', 'right_ankle', 'right_foot']

def sync_get_joints():
    cdef np.ndarray joints = np.zeros((24, 3))
    cdef np.ndarray proj_joints = np.zeros((24, 2))
    cdef np.ndarray conf = np.zeros(24)
    cdef void *depthp
    cdef void *imagep
    cdef void *scenep
    if not get_joints(<np.float64_t *>joints.data, <np.float64_t *>proj_joints.data, <np.float64_t *>conf.data,
                      &depthp, &imagep, &scenep):
        player_joints = dict([(l, {'world': x, 'image': y, 'conf': z})
                         for l, (x, y, z) in zip(JOINT_LABELS, zip(joints, proj_joints, conf))])
        image = PyArray_SimpleNewFromData(3, vdims, np.NPY_UINT8, imagep).copy()
        depth = PyArray_SimpleNewFromData(2, ddims, np.NPY_UINT16, depthp).copy()
        scene = PyArray_SimpleNewFromData(2, ddims, np.NPY_UINT16, scenep).copy()
        return {'joints': {0: player_joints},
                'depth': depth, 'image': image, 'scene': scene}

def sync_get_video():
    global timestamp
    cdef void *data = opennpy_sync_get_video()
    if not data:
        return
    timestamp += 1
    return PyArray_SimpleNewFromData(3, vdims, np.NPY_UINT8, data).copy(), timestamp

def sync_get_depth():
    global timestamp
    cdef void *data = opennpy_sync_get_depth()
    if not data:
        return
    timestamp += 1
    return PyArray_SimpleNewFromData(2, ddims, np.NPY_UINT16, data).copy(), timestamp

def sync_stop():
    opennpy_shutdown()

def align_depth_to_rgb():
    opennpy_align_depth_to_rgb()

def depth_to_3d(np.ndarray[np.uint16_t, ndim=2, mode='c'] depth, hfov=1.0144686707507438, vfov=0.78980943449644714):
    cdef np.ndarray world = np.zeros((480, 640, 3))
    opennpy_depth_to_3d(<np.uint16_t *>depth.data, <np.float64_t *>world.data, hfov, vfov)
    return world

def get_fov():
    cdef double hfov, vfov
    opennpy_get_fov(&hfov, &vfov)
    return hfov, vfov
