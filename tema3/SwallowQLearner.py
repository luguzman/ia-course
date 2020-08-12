#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:13:57 2018

@author: juangabriel
"""

import torch
import numpy as np
from libs.perceptron import SLP
from utils.decay_schedule import LinearDecaySchedule
import random
import gym
# importamos el buffer que creamos para simular la memoria, experiencia del agente y la clase de Experience
from utils.experience_memory import ExperienceMemory, Experience

MAX_NUM_EPISODES = 100000
STEPS_PER_EPISODE = 300


class SwallowQLearner(object):
    def __init__(self, environment, learning_rate = 0.005, gamma = 0.98):
        self.obs_shape = environment.observation_space.shape
        
        self.action_shape = environment.action_space.n
        self.Q = SLP(self.obs_shape, self.action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = learning_rate)
        
        self.gamma = gamma
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min, 
                                                 max_steps = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q
        
        # Inicializamos el buffer
        self.memory = ExperienceMemory(capacity = int(1e5))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
         
    def get_action(self, obs):
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())   
        self.step_num += 1 ##EN EL VIDEO SE NOS OLVIDÓ SUBIR EL STEP EN UNA UNIDAD
        return action
        
        
    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
        
    def replay_experience(self, batch_size):
        """
        Vuelve a jugar usando la experiencia aleatoria almacenada
        :param batch_size: Tamaño de la muestra a tomar de la memoria
        :return: 
        """
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)   
      
    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiences: fragmento de recuerdos anteriores
        :return: 
        """
        batch_xp = Experience(*zip(*experiences))
        # El * indica que hay una referencia (no le paso directamente el valor sino la referencia ya que 
        # extraereos varios valores)
        obs_batch = np.array(batch_xp.obs)
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)
        
        td_target = reward_batch + ~done_batch * \
                    np.tile(self.gamma, len(next_obs_batch)) * \
                    self.Q(next_obs_batch).detach().max(1)[0].data.numpy() # .numpy: Lo que sucede es que estamos multiplicando 3 arrays de Numpy por un tensor de Torch y al parecer eso no es posible en estas versiones por lo que debes convertir ese tensor en un array de Numpy
        td_target = torch.from_numpy(td_target) # dado que estamos trabajando con tensores debemos convertir ese array en un Tensor
        #np.tile: creamos un array repitiendo el valor de gamma len(next_obs_batch) veces
        # ~done_batch = No done_batch = 0: significa En caso de que sea True = ~done_batch = 1 y
        # denota el fin de un episodio si es false indica que hay otra experiencia, ie otra observación
        # es la misma función que teníamos en la función learn solo que lo hemos pasado a vectorial (a vectores)
        # por lo tanto esta denotación matricial no evita meter if por si ha o no acabado el episodio
        
        td_target = td_target.to(self.device)
        action_idx = torch.from_numpy(action_batch).to(self.device)
        td_error = torch.nn.functional.mse_loss(
                self.Q(obs_batch).gather(1, action_idx.view(-1,1).long()), # .long: view(-1,1) -> retorna un Tensor y gather no espera un Tensor, esta esperando un LongTensor
                # gather (1:estamos diciendo que uniremos por fila) en la posición -> action_idx 
                # dentro de todo el conjunto de observaciones -> view(-1,1)
                # ie juntamos todas las acciones que coinciden con las acciones determinadas de cuando
                # estuvimos en ese punto
                td_target.float().unsqueeze(1))
        
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_optimizer.step()
        
        
        
        
        
    
if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    agent = SwallowQLearner(environment)
    first_episode = True
    episode_rewards = list()
    for episode in range(MAX_NUM_EPISODES):
        obs = environment.reset()
        total_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
            #environment.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)
            # Para guardar un objeto experience con los 5 datos de obs, action, etc
            agent.memory.store(Experience(obs, action, reward, next_obs, done))
            agent.learn(obs, action, reward, next_obs)
            
            obs = next_obs
            total_reward += reward
            
            if done is True:
                if first_episode:
                    max_reward = total_reward
                    first_episode = False
                episode_rewards.append(total_reward)
                if total_reward > max_reward:
                    max_reward = total_reward
                print("\nEpisodio#{} finalizado con {} iteraciones. Recompensa = {}, Recompensa media = {}, Mejor recompensa = {}".
                      format(episode, step+1, total_reward, np.mean(episode_rewards), max_reward))
                # Denotamos el mínimode iteraciones para poder empezar a hacer uso de la memoria
                if agent.memory.get_size()>100:
                    agent.replay_experience(32) # Aquí denotamos que recuerde 32 (aleatorias) de los 100
                break
    environment.close()
            
            
            
            
            
            
            
            
