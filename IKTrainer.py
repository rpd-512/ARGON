import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import random
import copy
import time

class Robot:
    def __init__(self, name, dh_param, csv_set):
        self.name = name
        self.dh_param = dh_param
        self.csv_set = csv_set
        self.dof = len(dh_param)

        X_pos = csv_set[["target_pos_x", "target_pos_y", "target_pos_z"]].values
        y_angles = csv_set[[f"final_ang_{i}" for i in range(1, self.dof+1)]].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_pos, y_angles, test_size=0.2, random_state=42
        )
        
    @staticmethod
    def load_dh_parameters(yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)['dh_parameters']

    def dh_transform_tf(self, a, alpha, d, theta):
        ca = tf.constant(np.cos(np.radians(alpha)), dtype=tf.float32)
        sa = tf.constant(np.sin(np.radians(alpha)), dtype=tf.float32)
        ct = tf.cos(theta)
        st = tf.sin(theta)

        row1 = tf.stack([ct, -st * ca,  st * sa, a * ct], axis=1)
        row2 = tf.stack([st,  ct * ca, -ct * sa, a * st], axis=1)
        row3 = tf.stack([tf.zeros_like(theta), sa * tf.ones_like(theta), ca * tf.ones_like(theta), d * tf.ones_like(theta)], axis=1)
        row4 = tf.stack([tf.zeros_like(theta), tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)], axis=1)

        return tf.stack([row1, row2, row3, row4], axis=1)
    
    @tf.function
    def forward_kinematics_batch(self, joint_angles):
        batch_size = tf.shape(joint_angles)[0]
        T = tf.eye(4, batch_shape=[batch_size])
        for i, param in enumerate(self.dh_param):
            a, alpha, d = param['a'], param['alpha'], param['d']
            theta_i = joint_angles[:, i]
            T_i = self.dh_transform_tf(a, alpha, d, theta_i)
            T = tf.matmul(T, T_i)
        return T[:, :3, 3]

class NeuralTrainer:
    def __init__(self, robot, epoch=100, batch_size=512, gamma=0.5, alpha=0.001, layer_config=[[512, 'relu'], [512, 'relu'], [128, 'relu'], [64, 'relu']]):
        self.robot = robot
        self.gamma = gamma
        self.epoch = epoch
        self.batch_size = batch_size
        self.layer_config = layer_config
        self.alpha = alpha

    def fk_positional_loss(self, y_true, y_pred):
        pos_true = self.robot.forward_kinematics_batch(y_true)
        pos_pred = self.robot.forward_kinematics_batch(y_pred)
        return tf.reduce_mean(tf.norm(pos_true - pos_pred, axis=1))

    def composite_loss(self, y_true, y_pred):
        fk_loss = self.fk_positional_loss(y_true, y_pred)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return self.gamma * fk_loss + (1 - self.gamma) * mse_loss

    def make_model(self):
        ik_model = models.Sequential()
        ik_model.add(layers.Input(shape=(3,)))
        for units, activation in self.layer_config:
            ik_model.add(layers.Dense(units, activation=activation))
        ik_model.add(layers.Dense(self.robot.dof))
        optimizer = Adam(learning_rate=self.alpha)
        ik_model.compile(optimizer=optimizer, loss=self.composite_loss)
        return ik_model
    
    def train(self):
        self.model = self.make_model()
        
        history = self.model.fit(
            self.robot.X_train, self.robot.y_train,
            validation_data=(self.robot.X_test, self.robot.y_test),
            epochs=self.epoch,
            batch_size=self.batch_size,
            verbose=2
        )
        return self.model, history

    def evaluate(self):
        start_time = time.perf_counter()
        y_pred = self.model.predict(self.robot.X_test, verbose=0)
        end_time = time.perf_counter()

        y_true = self.robot.y_test

        angular_mse = tf.reduce_mean(tf.square(y_true - y_pred)).numpy()

        pos_true = self.robot.forward_kinematics_batch(tf.convert_to_tensor(y_true, dtype=tf.float32))
        pos_pred = self.robot.forward_kinematics_batch(tf.convert_to_tensor(y_pred, dtype=tf.float32))

        pos_errors = tf.norm(pos_true - pos_pred, axis=1).numpy()
        positional_mean = np.mean(pos_errors)

        total_time = end_time - start_time
        time_per_prediction = total_time / len(self.robot.X_test) * 1e6
        layer_count = sum(layer[0] for layer in self.layer_config)
        return {
            "angular_mse": float(angular_mse),
            "positional_mean": float(positional_mean),
            "prediction_time": float(time_per_prediction),
            "layer_size": layer_count
        }

    def save_model(self, filename = "ik_model.keras"):
        self.model.save(filename)
    def clone(self):
        return NeuralTrainer(
            robot=self.robot,
            epoch=self.epoch,
            batch_size=self.batch_size,
            gamma=self.gamma,
            alpha=self.alpha,
            layer_config=[layer[:] for layer in self.layer_config]  # deep copy of layer config
        )
    def __matmul__(self, other):
        if not isinstance(other, NeuralTrainer):
            raise TypeError("Can only crossover with another NeuralTrainer")

        # --- Epoch, batch, gamma, alpha crossover ---
        new_batch_size = random.choice([self.batch_size, other.batch_size])
        new_gamma = random.choice([self.gamma, other.gamma])
        new_alpha = random.choice([self.alpha, other.alpha])

        # --- Custom crossover for variable-sized layer config ---
        lc1 = self.layer_config
        lc2 = other.layer_config

        min_len = min(len(lc1), len(lc2))
        max_len = max(len(lc1), len(lc2))

        # Recombine common indices
        new_layer_config = []
        for i in range(min_len):
            chosen = random.choice([lc1[i], lc2[i]])
            new_layer_config.append(copy.deepcopy(chosen))

        # Handle extra layers from the longer parent
        longer = lc1 if len(lc1) > len(lc2) else lc2
        if len(longer) > min_len:
            extra_layers = longer[min_len:]
            # With some chance, mutate (shuffle, drop, or slightly modify these layers)
            for layer in extra_layers:
                if random.random() < 0.5:
                    mutated = copy.deepcopy(layer)
                    # mutate number of neurons or activation
                    if isinstance(mutated[0], int):
                        mutated[0] = max(1, int(mutated[0] * random.uniform(0.8, 1.2)))
                    if isinstance(mutated[1], str) and random.random() < 0.3:
                        mutated[1] = random.choice(
                            [
                                'relu',       # 0
                                'sigmoid',    # 1
                                'tanh',       # 2
                                'softmax',    # 3
                                'elu',        # 4
                                'selu',       # 5
                                'softplus',   # 6
                                'softsign',   # 7
                                'swish'       # 8
                            ]
                        )
                    new_layer_config.append(mutated)

        # --- Mutation (optional step) ---
        if random.random() < 0.2:  # mutation rate
            idx = random.randint(0, len(new_layer_config) - 1)
            if isinstance(new_layer_config[idx][0], int):
                new_layer_config[idx][0] = max(1, int(new_layer_config[idx][0] * random.uniform(0.7, 1.3)))

        return NeuralTrainer(
            robot=self.robot,
            epoch=self.epoch,
            batch_size=new_batch_size,
            gamma=new_gamma,
            alpha=new_alpha,
            layer_config=new_layer_config
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        summary = f"🧠 NeuralTrainer Summary\n"
        summary += f"  Epochs:         {self.epoch}\n"
        summary += f"  Batch Size:     {self.batch_size}\n"
        summary += f"  Learning Rate:  {self.alpha:.5f}\n"
        summary += f"  Gamma (FK Mix): {self.gamma:.2f}\n"
        summary += f"  Layers:\n"
        for i, (units, act) in enumerate(self.layer_config):
            summary += f"    Layer {i+1}: {units} neurons, Activation: {act}\n"
        summary += f"  Output Dim:     {self.robot.dof} (from robot DOF)\n"
        summary += f"  Robot:          {self.robot.name}\n"
        return summary


if(__name__ == "__main__"):
    yaml_path = "dh_parameters/kuka_youbot.yaml"
    csv_path="datasets/kuka_youbot.csv"

    csv_s = pd.read_csv(csv_path)
    dh_p = Robot.load_dh_parameters(yaml_path)

    r = Robot("kuka_youbot", dh_p, csv_s)

    mtrain = NeuralTrainer(
            robot=r, 
            epoch=20,
            batch_size=64,
            gamma=0.01,
            alpha=0.001,
            layer_config=[[512, 'relu'], [512, 'relu'], [128, 'relu'], [64, 'relu']]
        )

    mtrain.train()
    print(mtrain.evaluate())
    mtrain.save_model()