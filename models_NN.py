import tensorflow as tf
from tensorflow.keras import backend as K


def sinAct(x):
    """Compute sinus activation function."""
    return K.sin(x)



class model_Critic_NN(tf.keras.Model):
  """Critic neural network for approximating a value function."""

  def __init__(self, n_param1, n_param2):
    super().__init__()

    self.d1 = tf.keras.layers.Dense(n_param1,activation="relu")
    self.d2 = tf.keras.layers.Dense(n_param2,activation="relu")
    self.value = tf.keras.layers.Dense(1)

  def call(self, inputs):
    x = tf.convert_to_tensor(inputs)
    x = self.d1(x)
    x = self.d2(x)
    x = self.value(x)
    return x



class model_Actor_NN(tf.keras.Model):
    """Actor neural network for computing a policy."""

    def __init__(self, n_param1, n_param2, n_actions):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(n_param1,activation=sinAct)
        self.d2 = tf.keras.layers.Dense(n_param2,activation=sinAct)
        self.out = tf.keras.layers.Dense(n_actions, activation = "softmax")

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x



def create_pg_model(config, pg_path):
    """ Creates a NN-based policy gradient model with optimizers to be used for a learning task.

        Input
        -----
        config : configparser.ConfigParser
            configuration settings for all parameters of the learning task
        pg_path : str
            path where model is to be saved

        Output
        ------
        pg_model : dict
            dictionary containing reinforcement learning model and optimizers
    """

    # Init PG model for learning
    actor = model_Actor_NN(config.getint("actor_network","n_param1"), config.getint("actor_network","n_param2"), len(config.get("environment","actions").split(",")))

    # Init PG optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.getfloat("actor_learning_rates","beta"))

    # Define dictionary for policy gradient model
    pg_model = {
        "actor": actor,
        "actor_op": optimizer,
        "actor_path": pg_path+"/"
        }

    return pg_model



def create_ac_model(config, actor_path, critic_path):
    """Creates a NN-based actor-critic model with optimizers to be used for a learning task.

        Input
        -----
        config : configparser.ConfigParser
            configuration settings for all parameters of the learning task
        actor_path : str
            path where the actor quantum model is to be saved
        critic_path : str
            path where the critic quantum model is to be saved

        Output
        ------
        ac_model : dict
            dictionary containing reinforcement learning models and optimizers
    """

    # Init NN model for learning
    model_critic = model_Critic_NN(config.getint("critic_network","n_param1"), config.getint("critic_network","n_param2"))
    model_actor = model_Actor_NN(config.getint("actor_network","n_param1"), config.getint("actor_network","n_param2"), len(config.get("environment","actions").split(",")))

    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=config.getfloat("critic_learning_rates","alpha"))
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=config.getfloat("actor_learning_rates","beta"))

    # Define dictionary for Actor-Critic models
    ac_model = {
        "critic": model_critic,
        "critic_op": critic_optimizer,
        "critic_path": critic_path+"/",

        "actor": model_actor,
        "actor_op": actor_optimizer,
        "actor_path": actor_path+"/"
        }

    return ac_model
