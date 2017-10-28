# -*- coding: utf-8 -*-

from vpython import *

root = sphere(pos=vector(0,0,0), radius=0.005, color=color.red)
x_axis = cylinder(pos = root.pos, axis= vec(20,0,0) - root.pos, radius = 0.05)
y_axis = cylinder(pos = root.pos, axis= vec(0,20,0) - root.pos, radius = 0.05)
z_axis = cylinder(pos = root.pos, axis= vec(0,0,20) - root.pos, radius = 0.05)

corner_marker = sphere(pos=vector(0,0,0), radius=0.5, color=color.red)
marker_x_mid = sphere(pos=vector(7,0,0), radius=0.5, color=color.yellow)
marker_x_end = sphere(pos=vector(10,0,0), radius=0.5, color=color.yellow)
orient_marker_x = sphere(pos=vector(2.5,0,0), radius=0.5, color=color.green)

arm_x = cylinder(pos = corner_marker.pos, axis=marker_x_end.pos - corner_marker.pos, radius = 0.2)

marker_y_mid = sphere(pos=vector(0,5,0), radius=0.5, color=color.cyan)
marker_y_end = sphere(pos=vector(0,10,0), radius=0.5, color=color.cyan)
orient_marker_y = sphere(pos=vector(0,8,0), radius=0.5, color=color.green)
arm_y = cylinder(pos = corner_marker.pos, axis=marker_y_end.pos - corner_marker.pos, radius = 0.2)

tracker = compound([arm_x, arm_y, corner_marker, marker_x_mid, marker_x_end, marker_y_mid, marker_y_end, orient_marker_x, orient_marker_y])
deltat = 0.05
roll = 0
scene.autoscale = False
while roll < 5:
    rate(5)
    tracker.rotate(angle=20, axis = vector(0,0,5), origin = vector(0,0,0))
    roll += deltat

pitch = 0
while pitch < 5:
    rate(5)
    tracker.rotate(angle=30, axis = vector(5,0,0), origin = vector(0,0,0))
    pitch += deltat
yaw = 0
while yaw < 5:
    rate(5)
    tracker.rotate(angle=40, axis = vector(0,5,0), origin = vector(0,0,0))
    yaw += deltat
    