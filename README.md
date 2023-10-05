# quantum_reinforcement_learning_of_classical_rare_trajectories
Quantum Reinforcement Learning of Classical Rare Trajectories
This repository contains the code used to generate the results of the following publication (ToDo:add link to paper). The code includes four different reinforcement learning implementations to learn rare dynamics of the random walker model.

## Abstract
Rare events are crucial essential for understanding the behavior of systems like non-equilibrium and industrial systems. The effective search for such rare dynamics is frequently the subject of research. To this end, specific methods are required for generating rare events and sampling their statistics in an automated and statistically meaningful way. With the advent of quantum computing and its potential advantages over classical computing for certain applications, the question arises whether quantum computers could also provide an advantage in the generation of rare dynamics. In this article, we propose approaches to quantum reinforcement learning (QRL) for studying rare dynamics of time-dependent systems and  investigate their benefits over classical approaches based on neural networks. We make use of variational quantum algorithms, which are currently a prominent method for making use of state-of-the-art noisy intermediate-scale quantum computers. We demonstrate that our QRL framework can learn and generate the rare dynamics of random walks. Furthermore, we are able to show better learning behavior with fewer parameters compared to the classical approaches. This is the first investigation of QRL applied to the statistics of rare events and suggests that QRL is a viable method to study rare dynamics.

## Installation
The TensorFlow and TensorFlow Quantum packages are required to run this code. See the installation commands below.\
A complete list on all requirements is provided in the 'requirements.txt' file.

```bash
pip install tensorflow=2.4.1
```

```bash
pip install tensorflow-quantum
```

Tutorials on how to install Tensorflow Quantum can be found online (https://www.tensorflow.org/quantum/install).

## Usage
You can configure the input parameters for the reinforcement learning agents in the config_files and then run the 'main'-scrips to train a new reinforcement learning agent.

Run
* main_QAC.py for a PQC-based actor-critic agent.
* main_QPG.py for a PQC-based policy gradient agent.
* main_NNAC.py for a NN-based actor-critic agent.
* main_NNPG.py for a NN-based policy gradient agent.

When you run the script, you will constantly get prints on the simulation process. After the learning process has finished, the code also automatically generates plots of the learning process.

Each script generates a folder in the 'Simulations'-folder with all the data generated during the script run. The resulting folder contains the following sub-folders:
* 'CSVs' folder: contains a CSV file with the implementation settings (all defined input parameters from the configuration files) and agent performance (returns per episode, returns per batch, etc.). Contains CSV files with all trajectories generated during learning. Contains CSV files with all trajectories generated with the trained agent.
* 'Models' folder: contains all model parameters. These files are crucial for loading the agent afterward.
* 'Plots' folder: All plots generated after learning will be saved here. After learning, the code automatically provides plots on the learning performance, policy, and value function.

The other scripts provide the functions for building the reinforcement learning agents, learning, and plotting.
* models_Q.py: contains the classes and functions to build a reinforcement learning agent with parameterized quantum circuits (PQC).
* models_NN.py: contains the classes and functions to build a reinforcement learning agent with classical neural networks.
* learn.py: contains the classes and functions for learning the rare dynamics of a random walker with either a policy gradient or an actor-critic approach.
* plot.py: contains all functions for generating the plots.

## Authors and acknowledgment

Laura Ohff

Alissa Wilms

## License

See [LICENSE](./LICENSE.md).