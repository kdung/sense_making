#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:25:41 2017

@author: sophie
"""
from vpython import *

ball = sphere(pos=vector(-5,0,0), radius=0.5, color=color.cyan)
wallR = box(pos=vector(6,0,0), size=vector(0.2,12,12), color=color.green) 
wallL = box(pos = vector(-11,0,0), size=vector(0.2,12,12), color=color.green)
ball.velocity = vector(25,0,0)
vscale = 0.1
varr = arrow(pos = ball.pos, axis = vscale * ball.velocity, color=color.yellow)
deltat = 0.005
t = 0
scene.autoscale = False
while t < 30:
    rate(100)
    if ball.pos.x > wallR.pos.x - 0.2 or ball.pos.x < wallL.pos.x:
        ball.velocity.x = -ball.velocity.x
    ball.pos = ball.pos + ball.velocity*deltat
    varr.pos = ball.pos
    varr.axis = vscale * ball.velocity
    t = t + deltat