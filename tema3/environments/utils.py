#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:51:17 2018

@author: juangabriel
"""

import gym
import cv2
import numpy as np

#Para los environments de OpenAI
#Este método será por si es necesario redimensionar una instancia o una observación
#Es decir, rescalara valores en caso de que sea necesario
class ResizeReshapeFrames(gym.ObservationWrapper):
    def __init__(self, environment):
        super(ResizeReshapeFrames, self).__init__(environment)
        if len(self.observation_space.shape) == 3:  # Si = 3 tenemos 3 canales => son imágenes
            self.desired_width = 84
            self.desired_height = 84
            self.desired_channels = self.observation_space.shape[2]   
            #La imagen viene en C x H x W =>convertimos a la conversión estandar de pytorch
            self.observation_space = gym.spaces.Box(0,255, (self.desired_channels, self.desired_height, self.desired_width), dtype = np.uint8)
            #dtype = np.uint8, ya que son valores discretos entre 0 y 255
            
    
    def observation(self, obs):
        if len(obs.shape) == 3:     # Si = 3 tenemos 3 canales => son imágenes
            obs = cv2.resize(obs, (self.desired_width, self.desired_height)) #redimensionamos
            #convertimos a la conversión estandar de pytorch C x H x W. C: channels, H: height, W:width
            if obs.shape[2] < obs.shape[0]:
                obs = np.reshape(obs, (obs.shape[2], obs.shape[1], obs.shape[0]))
        return obs