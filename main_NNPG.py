import learn
import models_NN as model
import plot
import pandas as pd
import os
import numpy as np
from configparser import ConfigParser



# Load config files
model_file = "config_files/models/NNactor.ini"
task_file = "config_files/learning_task/task.ini"
config = ConfigParser()
config.read([model_file, task_file])




# Create paths for saving
simulation_name= "NN PG"
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

pg_path= path + "/Model"
os.mkdir(pg_path, 0o777)



# Create models for learning
pg_model = model.create_pg_model(config, pg_path)

# Learn with policy gradient implementation
pg_batch_avg, pg_batch_rare_count, pg_return_per_episode, rare_dif_counts, loss = learn.learn_batched(path, config, pg_model)

# Generate trajectories after learning with policy of PG model
pg_return_per_episode_trained, rare_dif_counts_trained = learn.generate_trajectories(path, config, pg_model)


# Save implementation data
header = [
    "Implementation", "T", "X", "RW probs", "Episode count", "s", "b", "beta", 
    "Batch size", "Param Count Layer 1", "Param Count Layer 2", "Parameter count total", 
    "Returns per episode", "Avarage returns per batch", "Rare trajectories per batch", 
    "Returns per episode after training", "Count of different rare trajectories learning", 
    "Count of different rare trajectories generated", "Actor loss"
]
data = []

pg_trainableParams = np.sum([np.prod(v.get_shape()) for v in pg_model["actor"].trainable_weights])
pg_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in pg_model["actor"].non_trainable_weights])

pg_param_count = pg_trainableParams + pg_nonTrainableParams

data.append([
    "PG", config.get("environment","T"), config.get("environment","X"), 
    config.get("random_walker","rw_probs"), config.get("episodes","episodes"), 
    config.get("reward","s"), config.get("reward","b"), config.get("actor_learning_rates","beta"), 
    config.get("episodes","batch_size"), config.get("actor_network","n_param1"), 
    config.get("actor_network","n_param2"), pg_param_count, pg_return_per_episode, 
    pg_batch_avg, pg_batch_rare_count, pg_return_per_episode_trained, rare_dif_counts, 
    rare_dif_counts_trained, loss.actor_loss
])

data_csv = pd.DataFrame(data, columns=header)
data_csv.to_csv(csv_path_model+"/Implementation_data.csv", index=False)



# Plot results
episodes = range(config.getint("episodes","episodes"))
batches = range(len(pg_batch_avg))
name_trained = "After Learning"
plot_counter = 1

# Plot returns per episode
plot.plot_return_per_episode(path, str(plot_counter), pg_return_per_episode)
plot_counter += 1

# Plot avarage return per batch
plot.plot_avg_return_per_batch(path, str(plot_counter), pg_batch_avg)
plot_counter += 1

# Plot count of rare trajectory per batch and probability of generating rare trajectory in batch
plot.plot_rare_count_per_batch(path, plot_counter, pg_batch_rare_count, config.getint("episodes","batch_size"))
plot_counter += 2

# Plot actor loss
plot.plot_actor_loss(path, str(plot_counter), loss.actor_loss)
plot_counter += 1

# Plot trajectories generated during learning
plot.plot_trajectories_learning(path, str(plot_counter), config.getint("episodes","episodes"))
plot_counter += 1

# Plot heatmaps of probability of going up/down acording to policy
plot.plot_final_policy(plots_path, str(plot_counter),config.getint("environment","T"), pg_model["actor"], config.get("agent","type"))
plot_counter += 1

# Plot generated trajectories after learning
plot.plot_trajectories_after_learning(path, name_trained, config.getint("episodes","episodes_trained"))

print(pg_model["actor"].summary())
print("Done :)")
