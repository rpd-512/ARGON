from ik_trainer import Robot, NeuralTrainer
import pandas as pd
import numpy as np
import matplotlib as plt

class GeneticAlgorithm:
    def __init__(self, pop, itr, robot):
        self.population_size = pop
        self.iteration_count = itr
        self.robot = robot

    def generate_population(self):
        pass

    def evolve(self):
        pass
    
    def plot_line(self):
        pass

if(__name__ == "__main__"):
    yaml_path = "dh_parameters/kuka_youbot.yaml"
    csv_path="datasets/kuka_youbot.csv"

    csv_s = pd.read_csv(csv_path)
    dh_p = Robot.load_dh_parameters(yaml_path)

    robot = Robot("kuka_youbot", dh_p, csv_s)

    pop = 20
    itr = 15

    genetic_algo = GeneticAlgorithm(pop, itr, robot)