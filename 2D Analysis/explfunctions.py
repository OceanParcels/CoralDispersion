# -*- coding: utf-8 -*-
"""
Exploratory functions

Created on Thu Nov 21 12:44:03 2019

@author: reint fischer
"""

def deleteparticle(particle,fieldset,time):
    print('Particle '+str(particle.id)+' has died at t = '+str(time))
    particle.delete()
