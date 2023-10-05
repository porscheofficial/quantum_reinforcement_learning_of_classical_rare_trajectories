import tensorflow as tf
import cirq
import numpy as np
import math
import pylab
import pandas as pd
import os
from configparser import ConfigParser
from tensorflow.python.keras import backend
from io import StringIO
import sys



class Trajectory:
    def __init__(self, start_state):
        self.current_state = start_state
        self.states = [start_state]
        self.actions = []
        self.rewards_kl = [0]
        self.rewards = []
        self.traj_return = 0
        self.rare = False
        self.probs = []
        self.values = []



class Losses:
    def __init__(self):
        self.actor_loss = []
        self.critic_loss = []



def acumulate_rewards(rewards):
    """ Compute the acumulated rewards.

        Input
        -----
        rewards: [float]
            a list with all rewards of a trajectory

        Output
        ------
        rewards_acumulated: [float]
            list of acumulated rewards where
                - last entry only includes its own reward;
                - second to last entry includes its reward plus the reward of the last entry;
                - ...
                - first entry is the sum of all rewards in the trajectory (=return)
    """
    rewards_acumulated = []

    acumulation = 0

    for i in range(len(rewards)-1, 0, -1):
        acumulation += rewards[i]
        rewards_acumulated.insert(0,acumulation)

    return rewards_acumulated



def calculate_reward_kl(action_probability,action,rw_probs):
    """ Calculates the kl regularization part of the reward for each step in an episode.

        Input
        -----
        action probability: float
            action probability as calculated by policy
        action: int
            action taken
        rw_probs: [float]
            random walker"s original action probabilities

        Output
        ------
        reward_kl: float
            kl regularization term of reward.
    """
    reward_kl =  - math.log(action_probability/rw_probs[action])

    return reward_kl



def interact(action, action_probability, state, rw_probs):
    """ Performs a interaction with the environment

        Input
        -----
        action probability: float
            action probability as calculated by policy
        action: int
            action taken
        state: [int]
            current state of environment
        rw_probs: [float]
            random walker"s original action probabilities

        Output
        ------
        next_state: [int]
            next state after applying chosen action
        reward_kl: float
            kl regularization term of reward.
    """
    reward_kl = calculate_reward_kl(action_probability,action,rw_probs)

    next_state = [state[0]+1, state[1] + (2*action-1)]

    return next_state, reward_kl



def compute_rewards(traj, T, X, b, s):
    """ Computes rewards for each step in an episode and total retun of trajectory.

        Input
        -----
        traj : Trajectory
            trajectory for which rewards are to be computed
        T : int
            final endtime for trajectory
        X : int
            final endpoint for rare trajectory
        b : float
            parameter determining wether the dynamics of a random walk bridge (b=0) or a random walk excursion (b>0) is to be learned
        s : float
            parameter giving the rate at which normal trajectories are suppressed

        Output
        ------
        rewards : [float]
            a list with all rewards per step in the trajectory traj
    """
    rewards = []
    rare = False


    if traj.states[T][1]==X:
         rare = True

    for i, state in enumerate(traj.states):

        delta = 0

        if i==T:
            delta = 1

        reward_kl = traj.rewards_kl[i]
        x = state[1]

        H = 1

        if x < 0:
            H = 0
            if b > 0:
                rare = False

        d = abs(X-x)

        D_x = ((d)**2) * delta + b*(1-H)

        reward_t = ((-1) * s * D_x) + reward_kl

        rewards.append(reward_t)

    traj.traj_return = sum(rewards)
    traj.rewards = rewards
    traj.rare = rare

    return rewards



def step(traj, model, actions, rw_probs, agent_type, rl_type):
    """ Computes a step (action) taken in an episode.

        Input
        -----
        traj : Trajectory
            trajectory for which rewards are to be computed
        model : dict
            reinforcement learning model for calculating the policy
        actions : [int]
            list of possible actions the actor can take
        rw_probs : [float]
            random walker"s original action probabilities
        agent_type : str
            information wether the agent is a quantum or a NN agent
        rl_type : str
            information wether the agent is a policy gradient or an actor-critic agent
    """
    # Preprocess state
    state = np.array(traj.current_state)
    state = tf.convert_to_tensor([state])

    # Compute policy
    if agent_type == "NN":
        action_probability = model["actor"](state)
    if agent_type == "PQC":
        action_probability = model["actor"]([state])

    policy = action_probability.numpy()[0]

    action = np.random.choice(len(actions), p=policy)

    # Perform action
    next_state, reward_kl = interact(action, policy[action], traj.current_state, rw_probs)

    # Compute value
    if rl_type == "AC":
        if agent_type == "NN":
            value = model["critic"](state)
        if agent_type == "PQC":
            value = model["critic"]([state])
        traj.values.append(value.numpy()[0][0])

    # Update
    traj.current_state = next_state
    traj.states.append(next_state)
    traj.rewards_kl.append(reward_kl)
    traj.actions.append(action)
    traj.probs.append(policy[action])

    return



@tf.function
def policy_gradient_update(states, rewards, actions, model, batch_size, loss):
    """ Function that updates the quantum policy gradient model parameters after each batch.

        Input
        -----
        states : [[int]]
            tensor containing all states of a batch of trajectories
        rewards : [float]
            tensor containing all rewards of a batch of trajectories
        actions : [int]
            tensor containing all actions taken in a batch of trajectories
        model : dict
            quantum policy gradient model
        batch_size : int
            batch size
        loss : dict
            dictionary to save policy gradient loss during learning
    """
    with tf.GradientTape() as tape_actor:
        tape_actor.watch(model["actor"].trainable_variables)

        logits = model["actor"]([states])

        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)

        loss_actor = - tf.math.reduce_sum (rewards * log_probs) / batch_size

    grads_actor = tape_actor.gradient(loss_actor, model["actor"].trainable_variables)
    for optimizer, w in zip([model["op_in"],model["op_var"],model["op_out"]], [model["w_in"],model["w_var"],model["w_out"]]):
        optimizer.apply_gradients([(grads_actor[w], model["actor"].trainable_variables[w])])

    # save loss
    loss.actor_loss.append(loss_actor.numpy())



@tf.function
def NN_policy_gradient_update(states, rewards, actions, model, batch_size, loss):
    """ Function that updates the NN policy gradient model parameters after each batch.

        Input
        -----
        states : [[int]]
            tensor containing all states of a batch of trajectories
        rewards : [float]
            tensor containing all rewards of a batch of trajectories
        actions : [int]
            tensor containing all actions taken in a batch of trajectories
        model : dict
            NN policy gradient model
        batch_size : int
            batch size
        loss : dict
            dictionary to save policy gradient loss during learning
    """
    with tf.GradientTape() as tape_actor:
        tape_actor.watch(model["actor"].trainable_variables)

        logits = model["actor"](states)

        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)

        loss_actor = (tf.math.reduce_sum(- rewards * log_probs)) / batch_size

    grads_actor = tape_actor.gradient(loss_actor, model["actor"].trainable_variables)
    model["actor_op"].apply_gradients(zip(grads_actor, model["actor"].trainable_variables))

    # save loss
    loss.actor_loss.append(loss_actor.numpy())



@tf.function
def reinforce_update(states, next_states, rewards, actions, models, batch_size, losses):
    """ Function that updates the quantum policies and quantum value function approximator parameters after each batch.

        Input
        -----
        states : [[int]]
            tensor containing all states of a batch of trajectories
        next_states : [[int]]
            tensor containing all next states of a batch of trajectories
        rewards : [float]
            tensor containing all rewards of a batch of trajectories
        actions : [int]
            tensor containing all actions taken in a batch of trajectories
        model : dict
            quantum actor-critic model
        batch_size : int
            batch size
        losses : dict
            dictionary to save actor loss and critic loss during learning
    """
    with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
        tape_actor.watch(models["actor"].trainable_variables)
        tape_critic.watch(models["critic"].trainable_variables)

        logits = models["actor"]([states])
        values = models["critic"]([states])

        # Gather next state values
        values_next = models["critic"]([next_states])

        # calculate TD-error
        td = tf.stop_gradient(values_next + rewards - values)

        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)

        loss_actor = - tf.math.reduce_sum(td * log_probs)/batch_size

        loss_critic = - tf.math.reduce_sum(td * values)/batch_size

    # get gradients actor
    grads_actor = tape_actor.gradient(loss_actor, models["actor"].trainable_variables)
    # optimize actor parameters
    for optimizer, w in zip([models["op_in_a"],models["op_var_a"],models["op_out_a"]], [models["w_in_a"],models["w_var_a"],models["w_out_a"]]):
        optimizer.apply_gradients([(grads_actor[w], models["actor"].trainable_variables[w])])

    # get gradient critic
    grads_critic = tape_critic.gradient(loss_critic, models["critic"].trainable_variables)
    # optimize critic parameters
    for optimizer, w in zip([models["op_in_c"],models["op_var_c"],models["op_out_c"]], [models["w_in_c"],models["w_var_c"],models["w_out_c"]]):
        optimizer.apply_gradients([(grads_critic[w], models["critic"].trainable_variables[w])])

    # save losses
    losses.critic_loss.append(loss_critic.numpy())
    losses.actor_loss.append(loss_actor.numpy())



@tf.function
def NN_reinforce_update(states, next_states, rewards, actions, models, batch_size, losses):
    """ Function that updates the NN policies and NN value function approximator"s parameters after each batch.

        Input
        -----
        states : [[int]]
            tensor containing all states of a batch of trajectories
        next_states : [[int]]
            tensor containing all next states of a batch of trajectories
        rewards : [float]
            tensor containing all rewards of a batch of trajectories
        actions : [int]
            tensor containing all actions taken in a batch of trajectories
        model : dict
            quantum actor-critic model
        batch_size : int
            batch size
        losses : dict
            dictionary to save actor loss and critic loss during learning
    """
    with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
        tape_actor.watch(models["actor"].trainable_variables)
        tape_critic.watch(models["critic"].trainable_variables)

        logits = models["actor"](states)
        values = models["critic"](states)

        # Gather next state values
        values_next = models["critic"](next_states)

        td = tf.stop_gradient(values_next + rewards - values)

        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)

        loss_actor = - tf.math.reduce_sum(td * log_probs)/batch_size

        loss_critic = - tf.math.reduce_sum(td * values)/batch_size

    # get gradients actor
    grads_actor = tape_actor.gradient(loss_actor, models["actor"].trainable_variables)
    models["actor_op"].apply_gradients(zip(grads_actor, models["actor"].trainable_variables))

    # get gradient critic
    grads_critic = tape_critic.gradient(loss_critic, models["critic"].trainable_variables)
    models["critic_op"].apply_gradients(zip(grads_critic, models["critic"].trainable_variables))

    # save losses
    losses.critic_loss.append(loss_critic.numpy())
    losses.actor_loss.append(loss_actor.numpy())



def learn_batched(path, config, models):
    """ Performs the learning of a rare dynamic.

        Input
        -----
        path : str
            path where learning results and models are to be saved
        config : configparser.ConfigParser
            configuration settings for all parameters of the learning task
        models: dict
            reinforcement learning model to be used for learning task

        Output
        ------
        batch_avgs : [float]
            list of mean return per batch
        batch_rare_counts : [int]
            list of count of rare trajectories per batch
        return_per_episode : [float]
            list of return per episode
        rare_dif_counts : [int]
            ist of accumulated count of rare trajectories per episode
        losses : dict
            dictionary with reinforcement learning model losses during learning
    """
    # Prepare CSV file to save data in it during learning
    header = ["n", "Batch id","Rare trajectory?","Return","Time","Positions", "Probabilities", "KL-Regularization","Values"]
    data = []
    file_counter = 0
    file_path = path+"/CSVs/Trajectories during learning/"+str(file_counter)+".csv"

    # Create new CSV file every 5000 trajectories to prevent CSV file of getting to big:
    new_file = 5000

    # Return of all trajectories generated during learning
    return_per_episode = []

    # Avarage returns per batch
    batch_avgs = []

    # Rare trajectory count per batch
    batch_rare_counts = []

    # Rare trajectory count total during learning
    rare_count_total = 0

    # Count different rare trajectories during learning
    rare_dif_counts = []
    rare_dif_count = 0
    rare_dif = {}

    # Save Losses
    losses = Losses()

    # # Counter to stop after 100 consecutive rare trajectories generated
    # count_rare_folge = 0

    # Counter for saving model weights every now and then
    save_model_weights = 20
    save_model_weights_counter = 1

    tf.config.run_functions_eagerly(True)

    for batch in range(int(config.getint("episodes","episodes")/config.getint("episodes","batch_size"))):

        # Init counter for rare trajectories generated in batch
        rare_count_batch = 0
        returns_batch = []

        # Init batch data arrays for updating trainable params
        rewards_batch = []
        states_batch = []
        next_states_batch = []
        action_batch = []
        episode_returns = []
        rare_count_batch = 0

        # Init trajectories
        start_state = config.get("random_walker","start_state").split(",")
        start_state = [int(i) for i in start_state]
        trajectories = [Trajectory(start_state) for _ in range(config.getint("episodes","batch_size"))]

        for i in range(config.getint("episodes","batch_size")):

            traj = trajectories[i]

            # Generate trajectories
            for t in range(config.getint("environment","T")):

                actions = config.get("environment","actions").split(",")
                actions = [int(i) for i in actions]
                rw_probs = config.get("random_walker","rw_probs").split(",")
                rw_probs = [float(i) for i in rw_probs]
                step(traj, models, actions, rw_probs, config.get("agent","type"), config.get("agent","rl_class"))

            # Weigh trajectories acording to rareness using a specified weight function
            rewards = compute_rewards(traj, config.getint("environment","T"), config.getint("environment","X"), config.getfloat("reward","b"), config.getfloat("reward","s"))

            if config.get("agent","RL_class")== "PG":
                rewards_batch.extend(acumulate_rewards(rewards))
            else:
                rewards_batch.extend(rewards[1:])

            # Gather episode states, actions and rewards for updating training params
            states_batch.extend(traj.states[:-1])
            next_states_batch.extend(traj.states[1:])
            action_batch.extend(traj.actions)
            returns_batch.append(traj.traj_return)

            # Count rare trajectories generated in batch for plotting purposes and print return for learning monitoring purpose
            if traj.rare == True:
                rare_count_batch +=1
                rare_count_total +=1
                # count_rare_folge +=1
                if str(traj.states) not in rare_dif:
                    rare_dif[str(traj.states)] = 1
                    rare_dif_count += 1
                else:
                    rare_dif.update({str(traj.states): rare_dif[str(traj.states)]+1})
            # else:
            #     count_rare_folge = 0

            rare_dif_counts.append(rare_dif_count)

            # Collect returns of trajectories during learning and per batch for plotting purposes
            return_per_episode.append(traj.traj_return)
            returns_batch.append(traj.traj_return)

            # Create new CSV file to save trajectory data during learning for later plotting
            if ( (batch*config.getint("episodes","batch_size")) + i)%new_file==0 and ( (batch*config.getint("episodes","batch_size")) + i)!=0:
                #Save CSV file:
                data_csv = pd.DataFrame(data, columns=header)
                file_path= path+"/CSVs/Trajectories during learning/AC,"+str(file_counter)+".csv"
                data_csv.to_csv(file_path, index=False)
                file_counter+=1
                data=[]

            traj_times, traj_pos = zip(*(traj.states))

            data.append([((batch*config.getint("episodes","batch_size")) + i), batch, traj.rare, traj.traj_return, traj_times, traj_pos, traj.probs, traj.rewards_kl, traj.values])

        # # Stop learning after 100 consecutive rare trajectories
        # if count_rare_folge >= 100:
        #     break

        # Update model parameters
        if config.get("agent","RL_class")== "PG":

            states_batch = tf.convert_to_tensor(states_batch)
            rewards_batch = tf.convert_to_tensor(rewards_batch)
            id_action_pairs_batch = [[i,a] for i, a in enumerate(action_batch)]
            id_action_pairs_batch = tf.convert_to_tensor(id_action_pairs_batch)

            # Update model parameters.
            if config.get("agent","type")== "PQC":
                policy_gradient_update(states_batch, rewards_batch, id_action_pairs_batch, models, config.getint("episodes","batch_size"), losses)

            elif config.get("agent","type")== "NN":
                NN_policy_gradient_update(states_batch, rewards_batch, id_action_pairs_batch, models, config.getint("episodes","batch_size"), losses)

        elif config.get("agent","RL_class")== "AC":

            rewards_batch = tf.convert_to_tensor(rewards_batch)
            states_batch = tf.convert_to_tensor(states_batch)
            next_states_batch = tf.convert_to_tensor(next_states_batch)
            id_action_pairs_batch = [[i,a] for i, a in enumerate(action_batch)]
            id_action_pairs_batch = tf.convert_to_tensor(id_action_pairs_batch)

            # Update model parameters.
            if config.get("agent","type")== "PQC":
                reinforce_update(states_batch, next_states_batch, rewards_batch, id_action_pairs_batch,models, config.getint("episodes","batch_size"), losses)

            elif config.get("agent","type")== "NN":
                NN_reinforce_update(states_batch, next_states_batch, rewards_batch, id_action_pairs_batch,models, config.getint("episodes","batch_size"), losses)

        # Gather some batch observables for plotting
        avg_return_batch = np.mean(returns_batch)
        batch_avgs.append(avg_return_batch)
        batch_rare_counts.append(rare_count_batch)

        # Print batch avarage for monitoring purpose
        print("Finished episode", (batch + 1) * config.getint("episodes","batch_size"), ", Average return per episode: ", avg_return_batch, ", rare count: ", rare_count_batch, "dif rare count:", rare_dif_count)

        # Save model weights during training, save after save_model_weights-number of batches passed
        save_model_weights_counter += 1
        if save_model_weights_counter > save_model_weights:
            print("saving during training")
            models["actor"].save_weights(models["actor_path"]+"last_checkpoint")
            if config.get("agent","RL_class")=="AC":
                models["critic"].save_weights(models["critic_path"]+"last_checkpoint")
            save_model_weights_counter = 0

    # Save final model weights
    print("saving")
    models["actor"].save_weights(models["actor_path"]+"last_checkpoint")
    if config.get("agent","RL_class")=="AC":
        models["critic"].save_weights(models["critic_path"]+"last_checkpoint")

    #Save CSV file:
    data_csv = pd.DataFrame(data, columns=header)
    file_path= path+"/CSVs/Trajectories during learning/AC,"+str(file_counter)+".csv"
    data_csv.to_csv(file_path, index=False)

    return batch_avgs, batch_rare_counts, return_per_episode, rare_dif_counts, losses



def generate_trajectories(path, config, model):
    """ Generate trajectories using trained policy of given model.

        Input
        -----
        path : str
            path where learning results and models are to be saved
        config : configparser.ConfigParser
            configuration settings for all parameters of the learning task
        model: dict
            reinforcement learning model to be used for learning task

        Output
        ------
        return_per_episode : [float]
            list of return per episode
        rare_dif_counts : [int]
            ist of accumulated count of rare trajectories per episode
    """
    csv_path_traj_gen= path + "/CSVs/Trajectories after learning"
    os.mkdir(csv_path_traj_gen, 0o777)

    # Prepare CSV file to save data in it during learning
    header = ["n","Rare trajectory?","Return","Time","Positions"]
    data = []
    file_counter = 0
    file_path = csv_path_traj_gen+"/"+str(file_counter)+".csv"

    # Create new CSV file every 5000 trajectories to prevent CSV file of getting to big:
    new_file = 5000

    # Return of all trajectories generated during learning
    return_per_episode = []

    # Rare trajectory count total during learning
    rare_count_total = 0

    # Count different rare trajectories
    rare_dif_counts = []
    rare_dif_count = 0
    rare_dif = {}

    # Init trajectories
    start_state = config.get("random_walker","start_state").split(",")
    start_state = [int(i) for i in start_state]
    trajectories = [Trajectory(start_state) for _ in range(config.getint("episodes","episodes_trained"))]

    for i, traj in enumerate(trajectories):

        # Generate trajectory
        for t in range(config.getint("environment","T")):

            actions = config.get("environment","actions").split(",")
            start_state = [int(i) for i in actions]
            rw_probs = config.get("random_walker","rw_probs").split(",")
            rw_probs = [float(i) for i in rw_probs]
            step(traj, model, actions, rw_probs, config.get("agent","type"), config.get("agent","rl_class"))

        # Weigh trajectories acording to rareness using a specified weight function
        rewards = compute_rewards(traj, config.getint("environment","T"), config.getint("environment","X"), config.getfloat("reward","b"), config.getfloat("reward","s"))

        # Count rare trajectories generated in batch for plotting purposes and print return for learning monitoring purpose
        if traj.rare == True:
            print("             ",i, " return: ", str(traj.traj_return), " Rare Trajectory!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            rare_count_total +=1
            if str(traj.states) not in rare_dif:
                rare_dif[str(traj.states)] = 1
                rare_dif_count += 1
            else:
                rare_dif.update({str(traj.states): rare_dif[str(traj.states)]+1})
        else:
            print("             ",i, " return: ", str(traj.traj_return))

        rare_dif_counts.append(rare_dif_count)

        # Collect returns of trajectories and per batch for plotting purposes
        return_per_episode.append(traj.traj_return)

        # Create new CSV file to save trajectory data during generation for later plotting
        if i%new_file==0 and i!=0:
            #Save CSV file:
            data_csv = pd.DataFrame(data, columns=header)
            file_path= csv_path_traj_ge+"/"+str(file_counter)+".csv"
            data_csv.to_csv(file_path, index=False)
            file_counter+=1
            data=[]

        traj_times, traj_pos = zip(*(traj.states))

        data.append([i, traj.rare, traj.traj_return, traj_times, traj_pos])

    # Save CSV file:
    data_csv = pd.DataFrame(data, columns=header)
    file_path= csv_path_traj_gen+"/"+str(file_counter)+".csv"
    data_csv.to_csv(file_path, index=False)

    return return_per_episode, rare_dif_counts
