from IKTrainer import Robot, NeuralTrainer
import pandas as pd
import numpy as np
import matplotlib as plt
import random

class GeneticAlgorithm:
    def __init__(self, pop, itr, robot):
        self.population_size = pop
        self.iteration_count = itr
        self.robot = robot
        self.population_array = []

    def generate_population(self):
        act_func = [
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
        for _ in range(self.population_size):
            num_layers = random.randint(2, 6)
            self.population_array.append(
                NeuralTrainer(
                    robot=self.robot,
                    epoch=20,
                    batch_size=random.choice([32, 64, 128, 256, 512]),
                    gamma=random.uniform(0, 1),
                    alpha=random.uniform(0, 0.01),
                    layer_config = [
                        [random.choice([16, 32, 64, 128, 256, 512, 1024]), random.choice(act_func)]
                        for _ in range(num_layers)
                    ]
                )
            )

    def fitness(self, neural_net):
        neural_net.train()
        eval = neural_net.evaluate()
        e_alph = eval['angular_mse']
        e_pi = eval['positional_mean']
        e_tau = eval['prediction_time']
        e_lambda = eval['layer_size']
        print(eval)
        return (1)*e_alph + (1)*e_pi  - (1)*e_tau - (1)*e_lambda

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
    genetic_algo.generate_population()

    arr = genetic_algo.population_array
    print("\nParent 1")
    print(arr[0])
    print("\nParent 2")
    print(arr[1])
    print("\nCrossover")
    print(arr[0]@arr[1])

    print("\n\nEvaluating Parent 1")
    genetic_algo.fitness(arr[0])
    print("\n\nEvaluating Parent 2")
    genetic_algo.fitness(arr[1])
    print("\n\nEvaluating Child")
    genetic_algo.fitness(arr[0]@arr[1])