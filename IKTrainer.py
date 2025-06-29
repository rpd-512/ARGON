import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import random
import copy

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
            verbose=1
        )
        return self.model, history

    def evaluate_error(self):
        y_pred = self.model.predict(self.robot.X_test, verbose=0)
        y_true = self.robot.y_test

        angular_mse = tf.reduce_mean(tf.square(y_true - y_pred)).numpy()

        pos_true = self.robot.forward_kinematics_batch(tf.convert_to_tensor(y_true, dtype=tf.float32))
        pos_pred = self.robot.forward_kinematics_batch(tf.convert_to_tensor(y_pred, dtype=tf.float32))

        pos_errors = tf.norm(pos_true - pos_pred, axis=1).numpy()

        positional_mean = np.mean(pos_errors)
       
        return {
            "angular_mse": float(angular_mse),
            "positional_mean": float(positional_mean)
        }
    
    def save_model(self, filename = "ik_model.keras"):
        self.model.save(filename)

    def __matmul__(self, other):
        if not isinstance(other, NeuralTrainer):
            raise TypeError("Can only crossover with another NeuralTrainer")
    
        new_layer_config = []
        for layer1, layer2 in zip(self.layer_config, other.layer_config):
            chosen_layer = random.choice([layer1, layer2])
            new_layer_config.append(copy.deepcopy(chosen_layer))

        new_gamma = random.choice([self.gamma, other.gamma])
        new_alpha = random.choice([self.alpha, other.alpha])
        new_batch_size = random.choice([self.batch_size, other.batch_size])

        return NeuralTrainer(
            robot=self.robot,
            epoch=self.epoch,
            batch_size=new_batch_size,
            gamma=new_gamma,
            alpha=new_alpha,
            layer_config=new_layer_config
        )


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
    print(mtrain.evaluate_error())
    mtrain.save_model()