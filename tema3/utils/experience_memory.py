#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:48:47 2018

@author: juangabriel
"""

from collections import namedtuple
import random

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])

class ExperienceMemory(object):
    """
    Un buffer que simula la memoria, experiencia del agente
    """
    def __init__(self, capacity = int(1e6)):
        """
        :param capacity: Capacidad total de la memoria cíclica (número máximo de experiencias almacenables)
        :return:
        """
        self.capacity = capacity
        self.memory_idx = 0 #identificador que sabe la experiencia actual (ie en la que me encuentro)
        self.memory = []
        
    def sample(self, batch_size):
        """
        :param batch_size: Tamaño de la memoria a recuperar
        :return: Una muestra aleatoria del tamaño batch_size de experiencias de la memoria
        """
        assert batch_size <= self.get_size(), "El tamaño de la muestra es superior a la memoria disponible"
        return random.sample(self.memory, batch_size)
    
    def get_size(self):
        """
        :return: Número de experiencias almacenadas en memoria
        """
        return len(self.memory)
    
    def store(self, exp):
        """
        :param experience: Objeto experiencia a ser almacenado en memoria
        :return:
        """
        self.memory.insert(self.memory_idx % self.capacity, exp)
        self.memory_idx += 1

        """
        Nota: 'self.memory_idx % self.capacity' lo que hace es indicar la posición en donde se insertará
            exp. Lo que haces es asignarlo un indicador dentro del array entre 0 - 999 999 dado que 
            capacity = 1e6. Posteriormente sí memory_idx es = 1e6, notar que la posición en donde se 
            almacenará exp dentro del array 'memory' será 0 (ie reemplazará el valor que ya estaba ahí)
            y si fuera memory_idx es = 1e6 + 1 la posición en donde se almacenará exp será 1 y así
            sucesivamente. Entonces aquí es donde hacemos que nuestro buffer sea cíclico haciendo que 
            guarde la info más reciente."""