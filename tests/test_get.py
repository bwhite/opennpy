import opennpy

a = opennpy.sync_get_depth()
b = opennpy.sync_get_video()
opennpy.align_depth_to_rgb()
a = opennpy.sync_get_depth()
b = opennpy.sync_get_video()
