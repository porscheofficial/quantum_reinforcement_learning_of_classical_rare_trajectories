import learn
import models_Q as model
import plot
import pandas as pd
import os
import numpy as np
from configparser import ConfigParser



# Load config files
actor_file = "config_files/models/Qactor.ini"
critic_file = "config_files/models/Qcritic.ini"
task_file = "config_files/learning_task/task.ini"
config = ConfigParser()
config.read([actor_file, critic_file, task_file])



# Create paths for saving
simulation_name= "Quantum AC"
path_dir = simulation_name
path = os.getcwd()+"/Simulations/" + path_dir

counter = 1
while os.path.exists(path):
        path = os.getcwd()+"/Simulations/" + path_dir +" "+str(counter)
        counter += 1
try:
    os.mkdir(path, 0o777)
except:
    print("problem with file creation")

# Create directory to save files
csv_path= path + "/CSVs"
os.mkdir(csv_path, 0o777)
csv_path_traj_learn= path + "/CSVs/Trajectories during learning"
os.mkdir(csv_path_traj_learn, 0o777)
csv_path_model= path + "/CSVs/Learning parameters and model Data"
os.mkdir(csv_path_model, 0o777)

plots_path= path + "/Plots"
os.mkdir(plots_path, 0o777)

model_path= path + "/Models"
os.mkdir(model_path, 0o777)
critic_path= path + "/Models/Critic"
os.mkdir(critic_path, 0o777)
actor_path= path + "/Models/Actor"
os.mkdir(actor_path, 0o777)



# Create models for learning
ac_model = model.create_ac_model(config, actor_path, critic_path)

# Learn with actor-critic implementation
ac_batch_avg, ac_batch_rare_count, ac_return_per_episode, rare_dif_counts, losses = learn.learn_batched(path, config, ac_model)

# Generate trajectories after learning with policy of AC model
ac_return_per_episode_trained, rare_dif_counts_trained  = learn.generate_trajectories(path, config, ac_model)



# Save implementation data
header = [
    "Implementation", "T", "X", "RW probs", "b", "s", "Episode count", "Batch size", 
    "Actor layer count ", "Actor qubit count", "Actor noise", "Beta", 
    "Learning rates actor (input, variational, output)", "Actor parameter count", 
    "Critic layer count ", "Critic qubit count", "Critic noise", 
    "Learning rates critic (input, variational, output)", "Critic parameter count", 
    "Returns per episode", "Returns per batch", "Rare trajectories per batch", 
    "Returns per episode after training", "Count of different rare trajectories during learning", 
    "Count of different rare trajectories generated after learning", "Actor loss", "Critic loss"
]

data = []

a_trainableParams = np.sum([np.prod(v.get_shape()) for v in ac_model["actor"].trainable_weights])
a_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in ac_model["actor"].non_trainable_weights])

c_trainableParams = np.sum([np.prod(v.get_shape()) for v in ac_model["critic"].trainable_weights])
c_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in ac_model["critic"].non_trainable_weights])

param_count_a = a_trainableParams + a_nonTrainableParams
param_count_c = c_trainableParams + c_nonTrainableParams

data.append([
    config.get("agent","rl_class"), config.get("environment","T"), config.get("environment","X"), 
    config.get("random_walker","rw_probs"), config.get("reward","b"), config.get("reward","s"), 
    config.get("episodes","episodes"), config.get("episodes","batch_size"), 
    config.get("actor_circuit","layers"), config.get("actor_circuit","qubits"), 
    config.get("actor_circuit","noise"), config.get("actor_learning_rates","beta"), 
    [
        config.get("actor_learning_rates","a_in"), config.get("actor_learning_rates","a_var"), 
        config.get("actor_learning_rates","a_out")
    ], 
    param_count_a, config.get("critic_circuit","layers"), config.get("critic_circuit","qubits"), 
    config.get("critic_circuit","noise"), 
    [
        config.get("critic_learning_rates","c_in"), config.get("critic_learning_rates","c_var"), 
        config.get("critic_learning_rates","c_out")
    ], 
    param_count_c, ac_return_per_episode, ac_batch_avg, ac_batch_rare_count, 
    ac_return_per_episode_trained, rare_dif_counts, rare_dif_counts_trained, 
    losses.actor_loss, losses.critic_loss
])

data_csv = pd.DataFrame(data, columns=header)
data_csv.to_csv(csv_path_model+"/AC_Implementation_Data.csv", index=False)



# Plot results
episodes = range(config.getint("environment","T"))
batches = range(len(ac_batch_avg))
name_trained = "After Learning"
plot_counter = 1

# Plot returns per episode
plot.plot_return_per_episode(path, str(plot_counter), ac_return_per_episode)
plot_counter += 1

# Plot avarage return per batch
plot.plot_avg_return_per_batch(path, str(plot_counter), ac_batch_avg)
plot_counter += 1

# Plot count of different rare trajectories generated per episode
plot.plot_dif_rare_per_episode(path, str(plot_counter), rare_dif_counts)
plot_counter += 1

# Plot count of rare trajectory per batch and probability of generating rare trajectory in batch
plot.plot_rare_count_per_batch(path, plot_counter, ac_batch_rare_count, config.getint("episodes","batch_size"))
plot_counter += 2

# Plot actor loss
plot.plot_actor_loss(path, str(plot_counter), losses.actor_loss)
plot_counter += 1

# Plot actor loss
plot.plot_critic_loss(path, str(plot_counter), losses.critic_loss)
plot_counter += 1

# Plot trajectories generated during learning
plot.plot_trajectories_learning(path, str(plot_counter), config.getint("episodes","episodes"))
plot_counter += 1

# Plot heatmaps of probability of going up/down acording to policy
plot.plot_final_policy(plots_path, str(plot_counter),config.getint("environment","T"), ac_model["actor"],config.get("agent","type"))
plot_counter += 1

# Plot value function of model
plot.plot_final_value_function(plots_path, str(plot_counter),config.getint("environment","T"), ac_model["critic"],config.get("agent","type"))
plot_counter += 1

# Plot generated trajectories after learning
plot.plot_trajectories_after_learning(path, name_trained, config.getint("episodes","episodes_trained"))

print("Done :)")
