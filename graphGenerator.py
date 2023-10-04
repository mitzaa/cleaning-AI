import matplotlib.pyplot as plt

# Given data
generations = [i for i in range(1, 51)]
average_fitness = [5.37, 5.47, 5.54, 5.6, 5.65, 5.67, 5.71, 5.73, 5.75, 5.76, 5.78, 5.79, 5.8, 5.81, 5.82, 5.83, 5.83, 5.84, 5.85, 5.85, 5.86, 5.86, 5.87, 5.87, 5.88, 5.88, 5.88, 5.89, 5.89, 5.89, 5.89, 5.9, 5.9, 5.9, 5.91, 5.91, 5.91, 5.91, 5.91, 5.92, 5.92, 5.92, 5.92, 5.92, 5.92, 5.93, 5.93, 5.93, 5.93, 5.93]

# Plotting the data
plt.figure(figsize=(14, 7))
plt.plot(generations, average_fitness, marker='o', linestyle='-', color='b')
plt.title('Average Fitness across Generations')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.grid(True)
plt.tight_layout()
plt.show()

