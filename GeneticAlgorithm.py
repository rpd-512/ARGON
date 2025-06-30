from IKTrainer import Robot, NeuralTrainer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from copy import deepcopy

class GeneticAlgorithm:
    def __init__(self, pop, itr, robot):
        self.population_size = pop
        self.iteration_count = itr
        self.robot = robot
        self.population_array = []
        self.plot_array = [[],[]]
        self.fitness_cache = {}
    
    def fitness(self, neural_net):
        key = neural_net.hash_key()

        if key in self.fitness_cache:
            print("\n\n\nMEMO USED\n\n\n")
            return self.fitness_cache[key]
        neural_net.train()
        
        evaluation = neural_net.evaluate()

        e_alpha = evaluation['angular_mse']
        e_pi = evaluation['positional_mean']
        e_tau = evaluation['prediction_time']
        e_lambda = evaluation['layer_size']
        
        print(evaluation)

        w_alpha = 1.0     # angular MSE weight
        w_pi = 0.25        # positional error weight
        w_tau = 0.1       # prediction time weight
        w_lambda = 0.01    # network size weight

        fit = (
            w_alpha * e_alpha +
            w_pi * e_pi +
            w_tau * e_tau +
            w_lambda * e_lambda
        )
        return [fit,e_alpha,e_pi,e_tau,e_lambda]  # lower is better

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
            neur_trainer = NeuralTrainer(
                robot=self.robot,
                epoch=1,
                batch_size=random.choice([32, 64, 128, 256, 512]),
                gamma=random.uniform(0, 1),
                alpha=random.uniform(0, 0.01),
                layer_config = [
                    [random.choice([16, 32, 64, 128, 256, 512, 1024]), random.choice(act_func)]
                    for _ in range(num_layers)
                ]
            )
            fit = self.fitness(neur_trainer)
            self.population_array.append([fit[0],fit[1:],neur_trainer])

    def print_through_generation(self, filename="output_file/generation_log.txt"):
        with open(filename, "w") as f:
            for i, nn in enumerate(self.plot_array[6]):
                f.write(f"=== Generation {i} ===\n")
                f.write(str(nn))  # __str__ prints the NeuralTrainer summary
                f.write("\n\n")


    def evolve(self):
        self.plot_array = [[], [], [], [], [], [], []]
        elite_val = self.population_size // 2
        cross_value = 0.75

        for generation in tqdm(range(self.iteration_count)):
            # Sort by fitness (lower is better)
            self.population_array.sort(key=lambda x: x[0])

            # Log current best
            self.plot_array[0].append(generation)
            self.plot_array[1].append(self.population_array[0][0])
            self.plot_array[2].append(self.population_array[0][1][0])
            self.plot_array[3].append(self.population_array[0][1][1])
            self.plot_array[4].append(self.population_array[0][1][2])
            self.plot_array[5].append(self.population_array[0][1][3])
            self.plot_array[6].append(self.population_array[0][2])

            self.population_array[0][2].save_model("gen_"+str(generation)+"_best.keras")
            # === Select elites (clone to avoid mutation)
            elite_count = random.randint(1, elite_val)
            elite_parents = [
                [entry[0], entry[1][:], entry[2].clone()]
                for entry in self.population_array[:elite_count]
            ]

            # === Create new population (start with elites)
            new_population = elite_parents[:]

            # === Reproduce rest
            while len(new_population) < self.population_size:
                parent1 = random.choice(self.population_array)[2].clone()
                parent2 = random.choice(self.population_array)[2].clone()

                # crossover
                if random.uniform(0, 1) < cross_value:
                    try:
                        child = parent1 @ parent2
                    except:
                        child = parent1  # fallback if crossover not defined

                else:
                    child = parent1

                # fitness evaluation
                fit = self.fitness(child)
                new_population.append([fit[0], fit[1:], child])

            # Update population
            self.population_array = new_population

    def plot_line(self):
        metrics = {
            "Fitness": self.plot_array[1],
            "Angular MSE": self.plot_array[2],
            "Positional Error": self.plot_array[3],
            "Prediction Time": self.plot_array[4],
            "Model Size": self.plot_array[5]
        }

        for title, values in metrics.items():
            plt.figure(figsize=(15, 8))
            plt.plot(self.plot_array[0], values, color='r', label='Genetic Algorithm')
            plt.xlabel("Iteration Number")
            plt.ylabel("Value")
            plt.title(title)
            plt.legend()
            plt.savefig(f"output_file/graphs/{title.replace(' ', '_')}.png")
            plt.close()



if(__name__ == "__main__"):
    yaml_path = "dh_parameters/kuka_youbot.yaml"
    csv_path="datasets/kuka_youbot.csv"

    csv_s = pd.read_csv(csv_path)
    dh_p = Robot.load_dh_parameters(yaml_path)

    robot = Robot("kuka_youbot", dh_p, csv_s)

    pop = 20
    itr = 20

    genetic_algo = GeneticAlgorithm(pop, itr, robot)
    genetic_algo.generate_population()
    genetic_algo.evolve()
    genetic_algo.print_through_generation()
    genetic_algo.plot_line()
    genetic_algo.population_array.sort()
    genetic_algo.population_array[0][2].save_model("output_files/overall_best.keras")