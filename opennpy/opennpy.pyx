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

import_array()
timestamp = 0
cdef np.npy_intp ddims[2]
cdef np.npy_intp vdims[3]
ddims[0], ddims[1]  = 480, 640
vdims[0], vdims[1], vdims[2]  = 480, 640, 3

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
