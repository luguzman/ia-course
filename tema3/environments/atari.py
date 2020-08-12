#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:20:42 2018

@author: juangabriel
"""

import gym
import atari_py
import numpy as np
import random
import cv2
from collections import deque
from gym.spaces.box import Box

def get_games_list():
    return atari_py.list_games()


def make_env(env_id, env_conf):
    env = gym.make(env_id)
    if 'NoFrameskip' in env_id:
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max = 30)
        env = MaxAndSkipEnv(env, skip = env_conf['skip_rate'])
    
    if env_conf['episodic_life']:
        env = EpisodicLifeEnv(env)
        
    try: 
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env) # Intenta hacer el disparo
    except AttributeError:
        pass
    
    env = AtariRescale(env, env_conf['useful_region'])
    
    if env_conf['normalize_observation']:
        env = NormalizedEnv(env)
    
        
    env = FrameStack(env, env_conf['num_frames_to_stack'])
    
    if env_conf['clip_reward']:
        env = ClipReward(env)
        
    return env
    
# Redimensionamos laimagen de la atari    
def process_frame_84(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"]+160, :160] #Creamos una matriz con el tamaño deseado
    frame = frame.mean(2)   #Sacamos la media de los 3 canales de color para procesarlo a escala de grises
    #convertimos a float para reducir el espacio de almacenamiento ya que un float32 (32 bits) 
    #ocupa menos que su version analoga de enteros de 8 bits, ya que un float32 se puede alamacenar 
    #en 1 byte y 1 entero de 8 bits se necesitan 4 bytes para cada pixel
    frame = frame.astype(np.float32)
    #Re-escalamos la imagen:
    frame *= 1.0/255.0
    frame = cv2.resize(frame, (84, conf["dimension2"])) #Redimensionamos las imagenes aqui y en lo siguiente
    #Nota: si no hicieramos esta reducción estaríamos hablando de 500 GB en memoria 
    frame = cv2.resize(frame, (84,84))  #Es importante que la matriz sea cuadrada por la operación de convolución
    frame = np.reshape(frame, [1,84,84])    # Procesamos la info
    return frame

class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        # creamos una caja de 0 a 1 con dimension (1,84,84)
        self.observation_space = Box(0, 255, [1,84,84], dtype = np.uint8)  
        self.conf = env_conf
        
    def observation(self, observation):
        return process_frame_84(observation, self.conf)

# Tipificación de los datos
class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env = None):
        gym.ObservationWrapper.__init__(self, env)
        self.mean = 0
        self.std = 0
        self.alpha = 0.9999     #Para los calculos internos
        self.num_steps = 0
        
    def observation(self, observation):
        self.num_steps +=1
        self.mean = self.mean * self.alpha + observation.mean() * (1-self.alpha) #recalculamos la media
        self.std = self.std * self.alpha + observation.std() * (1-self.alpha)
        #En principio esta media y esta desviación son sesgadas
        
        unbiased_mean = self.mean / (1-pow(self.alpha, self.num_steps)) #esto nos da un valor de la media sin sesgo
        unbiased_std = self.std / (1-pow(self.alpha, self.num_steps))   #esto nos da un valor de la desviación estandar sin sesgo
        
        return (observation - unbiased_mean) / (unbiased_std + 1e-8) 
        #El + 1e-8 es para el caso en que la std sea 0 para que no nos cause problemas dividir entre 0

class ClipReward(gym.RewardWrapper):    #cortar las reecompensas para que estas sean 1,0,-1
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
    def reward(self, reward):
        return np.sign(reward)
#Nota: Cabe decir que esto de recortar la reecompensa es muy bueno para el juego de atari, sin embargo,
#no siempre es la mejor opción 
    
#La siguiente clase es apra que el agente empiece desde nuevos puntos aleatoriamente
class NoopResetEnv(gym.Wrapper):    #Tiene lugar cuando se resetea el entorno
    def __init__(self, env, noop_max = 30): #tomaremos un numero aleatorio de valores
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"     #la condición para seguir adelante 
        #es que exista una acción programada por el developer que se encargue de no llevar una 
        #accion contra el entorno, ie, que se eset quieto
        
    def reset(self):   
        self.env.reset()
        noops = random.randrange(1, self.noop_max +1)   
        assert noops > 0
        observation = None
        for _ in range(noops):
            observation, _, done, _ = self.env.step(self.noop_action)   #la reecompensa y el next_setp nos da igual
        return observation
    
    def step(self, action):
        return self.env.step(action)
    
class FireResetEnv(gym.Wrapper):        #Pulsara el boton de fire si esta disponible
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE" 
        assert len(env.unwrapped.get_action_meanings()) >= 3        
        #Las dos asersiones anteriromes son más que nada que el fire no es un boton de disparo como tal
        #sino que es el boton de inicio de la partida 
        
    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs
    
    def step(self, action):
        return self.env.step(action)

#La siguiente clase es para que en caso de que se pierda una vida no resetear el environment y hacerlo 
#solo cuando hay un game over ya que el sino el agente interpretará como malo el perder una vida
class EpisodicLifeEnv(gym.Wrapper):         
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.has_really_died = False            
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.has_really_died = False
        lives = info['ale.lives']
        if lives < self.lives and lives > 0:          #si me quedan menos vidas que las disponibles y lives >0
            done = True
            self.has_really_died = True               #entonces realmente habremos terminado
        self.lives = lives                            #devolvemos las 4 vidas
        return obs, reward, done, info
    
    #Sí murio realmente se resetea el environment sino continuar 
    def reset(self):
        if self.has_really_died is False:
            obs = self.env.reset()
            self.lives = 0
        else:
            obs,_,_,info = self.env.step(0)
            self.lives = info['ale.lives']
        return obs
    
#Tantos pasos como indique la variable skip devuelve el mejor valor. 
#Es decir, acelereamos el proceso de aprendizaje
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env = None, skip = 4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = deque(maxlen=2)  #Es una ayuda que nos ayudara a saltar los frames que no quiero procesar
        self._skip = skip           #número de frames a saltar
        
        
    def step(self, action):     
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)        #Añadimos la obs actual
            total_reward += reward
            if done:                        #si se acaba antes de que finalice el bucle que termine
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis = 0)    #Me quedo con la mejor de todas las obs
        return max_frame, total_reward, done, info
    
    #Limpiamos todo el buffer y regresamos la primera obs
    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
    
#algoritmo para apilar los útloms k frames
#AYUDA EN TERMINOS DE EFICICIECDIA EN RAM
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen = k) #creamos un array vacio de longitud k
        shape = env.observation_space.shape #shape para crear nuestro espacio
        self.observation_space = Box(low = 0, high = 255, shape = (shape[0] * k, shape[1], shape[2]), dtype = np.uint8)
        #Creamo un box con el valor minimo y maximo de pixel, le damos el shape
        #k: para almacenar tantos canales de color por el no. de frames que deseo stackear 
        #shape[1]: es la anchura que se mantendrá constante
        #shape[2]: es la altura que se mantendrá constante

    
    def reset(self):    #vaciar el entorno
        obs = self.env.reset()
        for _ in range(self.k):     #tantas veces como indique le parametro k
            self.frames.append(obs) #Añanadimos la misma obs k vecess
        return self.get_obs()


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)       #Añadimos la última obs
        return self.get_obs(), reward, done, info
    
     
    def get_obs(self):
        assert len(self.frames) == self.k       #sino no podría ser posible realizarlo
        return LazyFrames(list(self.frames))    #le pasamos una lista de frames
        #LazyFrames: reduce el tamaño del buffer

#LazyFrames: reduce el tamaño del buffer   
class LazyFrames(object):
    def __init__(self, frames):
        self.frames = frames
        self.out = None
        
        
    def _force(self):
        if self.out is None:        # si todavia no ha sido configurada la variable out
            self.out = np.concatenate(self.frames, axis = 0)  #concatenaremos cada uno de los frames
            self.frames = None      # vaciamos la memoria intermedia (nunca tendremas más de k frames)
        return self.out
    
    def __array__(self, dtype = None):  #definira lo que es el array
        out = self._force()
        if dtype is not None:           
            out = out.astype(dtype)     #hacemos el casting al tipo de dato en cuestión
        return out
    
    def __len__(self):
        return len(self._force())       #longitud del objeto out
    
    def __getitem__(self, i):
        return self._force()[i]         #devuelve el bjeto i-ésimo del out

