#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:00:35 2018

@author: juangabriel
"""
import torch
import random
import gym
import numpy as np

from datetime import datetime
#importamos argparse para denotar argumentos en consola cuando inicialicemos la ejecución de un fichero
from argparse import ArgumentParser 

from libs.perceptron import SLP
from libs.cnn import CNN

from utils.decay_schedule import LinearDecaySchedule
from utils.experience_memory import ExperienceMemory, Experience
from utils.params_manager import ParamsManager

import environments.atari as Atari
import environments.utils as env_utils

#Importamos tensorboardX para ver como el agente progresa o si se ha estancado, inclusive para tunear los
#hiperparámetros en caso de que no tenga un buen desarrollo. Guardaremos los resultados
from tensorboardX import SummaryWriter

## Parseador de Argumentos
args = ArgumentParser("DeepQLearning")  #DeepQLearning: nombre del fichero
args.add_argument("--params-file", help = "Path del fichero JSON de parámetros. El valor por defecto es parameters.json",
                  default="parameters.json", metavar = "PFILE") #metavar: donde se guardaran los valores
#Nota: aunque este argumento lo estemos guardando como params-file se guardará como params-file ya que python no guarda los "-"
args.add_argument("--env", help = "Entorno de ID de Atari disponible en OpenAI Gym. El valor por defecto será SeaquestNoFrameskip-v4",
                  default = "SeaquestNoFrameskip-v4", metavar="ENV")
args.add_argument("--gpu-id", help = "ID de la GPU a utilizar, por defecto 0", default = 0, type = int, metavar = "GPU_ID")
#Nota: aunque este argumento lo estemos guardando como gpu-id se guardará como gpu_id ya que python no guarda los "-"
args.add_argument("--test", help = "Modo de testing para jugar sin aprender. Por defecto está desactivado", 
                  action = "store_true", default = False)
args.add_argument("--render", help = "Renderiza el entorno en pantalla. Desactivado por defecto", action="store_true", default=False)
args.add_argument("--record", help = "Almacena videos y estados de la performance del agente", action="store_true", default=False)
args.add_argument("--output-dir", help = "Directorio para almacenar los outputs. Por defecto = ./trained_models/results",
                  default = "./trained_models/results")
#Una vez escritos todos, se parsean asi mismos y quedan listos para ser utilizados
args = args.parse_args()    


# Parámetros globales
manager = ParamsManager(args.params_file)

# Ficheros de logs acerca de la configuración de las ejecuciones
summary_filename_prefix = manager.get_agent_params()['summary_filename_prefix']
summary_filename = summary_filename_prefix + args.env + datetime.now().strftime("%y-%m-%d-%H-%M")

## Summary Writer de TensorBoardX
writer = SummaryWriter(summary_filename)

manager.export_agent_params(summary_filename + "/"+"agent_params.json")
manager.export_environment_params(summary_filename + "/"+"environment_params.json")



#Contador global de ejecuciones
global_step_num = 0

# Habilitar entrenamiento por gráfica o CPU
use_cuda = manager.get_agent_params()['use_cuda']
device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")

# Habilitar la semilla aleatoria para poder reproducir el experimento a posteriori
seed = manager.get_agent_params()['seed']
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)




class DeepQLearner(object):
    def __init__(self, obs_shape, action_shape, params):
       
        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['learning_rate']
        self.best_mean_reward = -float("inf")
        self.best_reward = -float("inf")
        self.training_steps_completed = 0
        self.action_shape = action_shape
        
        if len(obs_shape)  == 1: ## Solo tenemos una dimensión del espacio de observaciones
            self.DQN = SLP
        elif len(obs_shape) == 3: ## El estado de observaciones es una imagen/3D
            self.DQN = CNN
            
        self.Q = self.DQN(obs_shape, action_shape, device).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.learning_rate)
        
        if self.params['use_target_network']:
            self.Q_target = self.DQN(obs_shape, action_shape, device).to(device)
           
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = self.params['epsilon_max']
        self.epsilon_min = self.params['epsilon_min']
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min, 
                                                 max_steps = self.params['epsilon_decay_final_step']) 
                                                 # Podríamos tomar la mitad de todos lo steps que se daran con: .5*max_num_episodes*steps_per_episode
        self.step_num = 0
        
        self.memory = ExperienceMemory(capacity = int(self.params['experience_memory_size']))
        
         
    def get_action(self, obs):
        #lo convertimos a numpy ya que si es una imagen lo que se insert puede ser un frame
        obs = np.array(obs) 
        # escalamos los valores para que queden entre 0 y 1 por el maximo que es 255 para enteros de 
        # 8 bits que es como se guardan las imagenes
        obs = obs / 255.0
        if len(obs.shape) == 3: # tenemos una imagen
            if obs.shape[2] < obs.shape[0]: # WxHxC -> C x H x W
                obs = obs.reshape(obs.shape[2], obs.shape[1], obs.shape[0])
            obs = np.expand_dims(obs, 0)   
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):
        #Registramos los valores en tensorboardX
        #Le pasamos donde lo vamos a guardar, el objeto a guardar y el número de iteración
        writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)  
        self.step_num +=1
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            #not self.params["test"]: será para denotar al modelo por consola si el usuario
            #desea pasar por la fase de entrenamiento o si directamente ir a mejorar las acciones
            #es decir, sin pasar por las desiciones aleatorias.
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())   
        return action
        
        
    def learn(self, obs, action, reward, next_obs, done):
        if done: # si hemos acabado
            td_target = reward + 0.0 #el target se queda igual
        else: 
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        #Los siguientes pasos son para actualizar la red neuronal
        self.Q_optimizer.zero_grad()
        td_error.backward()
        #Registramos los valores en tensorboardX
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()
        
    def replay_experience(self, batch_size = None):
        """
        Vuelve a jugar usando la experiencia aleatoria almacenada
        :param batch_size: Tamaño de la muestra a tomar de la memoria
        :return: 
        """
        #Si se especifica un batch_size por consola tomará ese y si no el denotado en params
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)   
        self.training_steps_completed += 1
      
    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiences: fragmento de recuerdos anteriores
        :return: 
        """
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)/255.0    #Escalamos las observaciones
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        
        if self.params["clip_reward"]:  #Si tengo que cortar las recompesas (quitaremos los negativos)
            reward_batch = np.sign(reward_batch)    #me quedo en el signo (+/-)
        next_obs_batch = np.array(batch_xp.next_obs)/255.0   #Escalamos las observaciones
        done_batch = np.array(batch_xp.done)
        
        
        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_frequency'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())  
                #Lo que estamos denotando en las 3 ultímas lineas de código es que si usamos target_network
                #siempre siempre iremos actualizaremos con la misma formula del target y que cada
                #2000 iteraciones carguemos una copia del diccionario de estados de la red sirve como
                #backup (no necesariamente incrementa la eficiencia) y haremos un td_target "diferente"
            td_target = reward_batch + ~done_batch *\
                        np.tile(self.gamma, len(next_obs_batch)) * \
                        self.Q_target(next_obs_batch).max(1)[0].data
        #np.tile: creamos un array repitiendo el valor de gamma len(next_obs_batch) veces
        # ~done_batch = No done_batch = 0: significa En caso de que sea True = ~done_batch = 1 y
        # denota el fin de un episodio si es false indica que hay otra experiencia, ie otra observación
        # es la misma función que teníamos en la función learn solo que lo hemos pasado a vectorial (a vectores)
        # por lo tanto esta denotación matricial no evita meter if por si ha o no acabado el episodio
        else: 
            td_target = reward_batch + ~done_batch * \
                        np.tile(self.gamma, len(next_obs_batch)) * \
                        self.Q(next_obs_batch).detach().max(1)[0].data
        
        td_target = td_target.to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = torch.nn.functional.mse_loss(
                self.Q(obs_batch).gather(1, action_idx.view(-1,1)),
                td_target.float().unsqueeze(1))
        #Actualizamos los valores de la red neuronal:
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_optimizer.step()
        
    def save(self, env_name):  # this wiil be to save an environment
        file_name = self.params['save_dir']+"DQL_"+env_name+".ptm"
        agent_state = {"Q": self.Q.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(agent_state, file_name)
        print("Estado del agente guardado en : ", file_name)
        
        
    def load(self, env_name): # this wiil be to load an environment
        file_name = self.params['load_dir']+"DQL_"+env_name+".ptm"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        #lambda: decalara variables que tomar valor en cualquier lugar
        # storage: nos sirve para referenciar la propia localización en donde se encuentra el archivo
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Cargado del modelo Q desde", file_name,
              "que hasta el momento tiene una mejor recompensa media de: ",self.best_mean_reward,
              " y una recompensa máxima de: ", self.best_reward)
        
    
if __name__ == "__main__":
    env_conf = manager.get_environment_params()
    env_conf["env_name"] = args.env
    
    if args.test:
    # Si episodic_life  = False => cuando acaba el juego es el fin del episodio en lugar de que se acabe 
    # al final de cada vida
        env_conf["episodic_life"] = False   
    reward_type = "LIFE" if env_conf["episodic_life"] else "GAME"
    #Nota: Entonces las recompesas que se reportaran serán a lo largo de toda su vida o de todo el juego
    
    custom_region_available = False     #Por defecto supondremos que quiere la configuración estándar
    #En el caso de que haya escrito una en consola:
    for key, value in env_conf["useful_region"].items():
        if key in args.env:     #Si la clave se encuentra dentro de la variable environment 
            env_conf["useful_region"] = value       #Le añadimos en useful_region al valor que nos haya asignado
            custom_region_available = True          #Ahora entonces hay una region personalizada
            break
    if custom_region_available is not True:         #Si no hay una región personalizada
        env_conf["useful_region"] = env_conf["useful_region"]["Default"]    #Dimos una conf por defecto al inicio
    print("Configuración a utilizar:", env_conf)
    
    atari_env = False           # Partimos del valor atari = False
    for game in Atari.get_games_list():     # por cada juego de la lista de juegos de atari
        if game.replace("_", "") in args.env.lower():       #si alguno coincide o se encuentra dentro de la lista de argumentos
            atari_env = True
    
    if atari_env:
        environment = Atari.make_env(args.env, env_conf)    #establecemos la configuracón y creamos el entorno de atari
    else:   #si no es Atari
        environment = env_utils.ResizeReshapeFrames(gym.make(args.env)) 
        
    obs_shape = environment.observation_space.shape #Obtenemos el shape de las posibles acciones del environment
    action_shape = environment.action_space.n   #Número de acciones
    agent_params = manager.get_agent_params()
    agent_params["test"] = args.test            
    agent_params["clip_reward"] = env_conf["clip_reward"]
    agent = DeepQLearner(obs_shape, action_shape, agent_params) 
    
    episode_rewards = list()
    previous_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0
    if agent_params['load_trained_model']:  #cargamos un modelo previamente entrnado en caso de true
        try:
            agent.load(env_conf['env_name'])
            previous_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            print("ERROR: no existe ningún modelo entrenado para este entorno. Empezamos desde cero")

    #contador para saber en que episodio vamos 
    episode = 0
    while global_step_num < agent_params['max_training_steps']: 
        obs = environment.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done:     #Mientras no hayamso terminado
            if env_conf['render'] or args.render:   #env_conf['render'] o el usuario lo denota por consola
                environment.render()        #para renderizarlo por pantalla
            
            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)
            #Guardamos toda la info actual (obs, action, reward, next_obs, done)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))
            
            obs = next_obs
            total_reward += reward
            step += 1
            global_step_num += 1
            
            if done is True:
                episode += 1
                episode_rewards.append(total_reward)
            
                if total_reward > agent.best_reward:
                    agent.best_reward = total_reward
                
                if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew: 
                    num_improved_episodes_before_checkpoint += 1
                
                #Si hemos mejorado más de agent_params['save_freq'] veces, persistimos la info
                if num_improved_episodes_before_checkpoint >= agent_params['save_freq']:
                    previous_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_conf['env_name'])
                    num_improved_episodes_before_checkpoint = 0
                
                print("\n Episodio #{} finalizado con {} iteraciones. Con {} estados: recompensa = {}, recompensa media = {:.2f}, mejor recompensa = {}".
                      format(episode, step+1, reward_type, total_reward, np.mean(episode_rewards), agent.best_reward))
                
                #Registramos la reecompensa, la media y máxima reecompensa
                writer.add_scalar("main/ep_reward", total_reward, global_step_num)
                writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), global_step_num)
                writer.add_scalar("main/max_ep_reward", agent.best_reward, global_step_num)
                
                # replay_start_size: inica el tamañode memoria de inicio
                # si el tamaño de la memoria supera 2 veces el replay_start_size
                if agent.memory.get_size() >= 2*agent_params['replay_start_size'] and not args.test:
                    agent.replay_experience()   
                    
                break       #saldra con la ejecución actualdel primer bucle
            
    environment.close()
    writer.close()       #Denotamos que cierre la ejecución del algoritmo 