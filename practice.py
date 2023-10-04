
__author__ = "Marion Millard-Grelet"
__organization__ = "COSC343, University of Otago"
__email__ = "milma737@student.otago.ac.nz"
import numpy as np

agentName = "steve"
trainingSchedule = [("random_agent.py", 20), ("self", 20)]
# trainingSchedule = [("random_agent.py", 0)]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.chromosome_length = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size

    def forward(self, inputs, chromosome):
        w1 = chromosome[:self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
        b1 = chromosome[self.input_size * self.hidden_size:self.input_size * self.hidden_size + self.hidden_size]
        w2 = chromosome[
             self.input_size * self.hidden_size + self.hidden_size:self.input_size * self.hidden_size + self.hidden_size + self.hidden_size * self.output_size].reshape(
            self.hidden_size, self.output_size)
        b2 = chromosome[self.input_size * self.hidden_size + self.hidden_size + self.hidden_size * self.output_size:]
        z1 = np.dot(inputs, w1) + b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        return z2

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

class Cleaner:
    CHROMOSOME_LENGTH = 256

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        self.nn = NeuralNetwork(nPercepts, 10, nActions)
        self.chromosome = np.random.uniform(-1, 1, self.nn.chromosome_length)
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns

    def AgentFunction(self, percepts):
        visual, energy, bin, fails = percepts
        combined_percepts = np.concatenate((visual[:, :, 0].flatten(), visual[:, :, 1].flatten(),
                                            visual[:, :, 2].flatten(), visual[:, :, 3].flatten(),
                                            [energy], [bin], [fails]))
        scores = self.nn.forward(combined_percepts, self.chromosome)

        return scores

def evalFitness(population):
    N = len(population)
    fitness = np.zeros((N))
    for n, cleaner in enumerate(population):
        fitness[n] = cleaner.game_stats['cleaned']
    return fitness

def dynamic_tournament_size(fitness):
    diversity = np.std(fitness)
    min_tournament_size = 2
    max_tournament_size = 5
    if diversity > 0.5:
        return min_tournament_size
    elif diversity < 0.2:
        return max_tournament_size
    else:
        return int(min_tournament_size + (max_tournament_size - min_tournament_size) * (0.5 - diversity) / 0.3)

def tournament_selection(population, fitness):
    tournament_size = dynamic_tournament_size(fitness)
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    selected_fitness = [fitness[i] for i in selected_indices]
    best_index = selected_indices[np.argmax(selected_fitness)]
    return population[best_index]

def one_point_crossover(parent1_chromosome, parent2_chromosome):
    crossover_point = np.random.randint(1, len(parent1_chromosome))
    child_chromosome = np.concatenate((parent1_chromosome[:crossover_point],
                                       parent2_chromosome[crossover_point:]))
    return child_chromosome

def adaptive_mutate(chromosome, current_fitness, previous_fitness=None, base_mutation_rate=0.02):
    if previous_fitness is None:
        mutation_rate = base_mutation_rate
    elif current_fitness <= previous_fitness:
        mutation_rate = min(base_mutation_rate + 0.01, 0.1)
    else:
        mutation_rate = max(base_mutation_rate - 0.01, 0.01)
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] += np.random.uniform(-0.1, 0.1)
            chromosome[i] = np.clip(chromosome[i], -1, 1)

def newGeneration(old_population, previous_avg_fitness=None):
    N = len(old_population)
    gridSize = old_population[0].gridSize
    nPercepts = old_population[0].nPercepts
    nActions = old_population[0].nActions
    maxTurns = old_population[0].maxTurns

    fitness = evalFitness(old_population)
    sorted_indices = np.argsort(fitness)[::-1]
    sorted_population = [old_population[i] for i in sorted_indices]

    elitism_size = int(0.2 * N)
    new_population = sorted_population[:elitism_size]

    while len(new_population) < N:
        parent1 = tournament_selection(sorted_population, fitness)
        parent2 = tournament_selection(sorted_population, fitness)

        child = Cleaner(nPercepts, nActions, gridSize, maxTurns)
        child.chromosome = one_point_crossover(parent1.chromosome, parent2.chromosome)

        current_avg_fitness = np.mean(fitness)
        adaptive_mutate(child.chromosome, current_avg_fitness, previous_avg_fitness)

        new_population.append(child)

    avg_fitness = np.mean(fitness)
    return (new_population, avg_fitness)
