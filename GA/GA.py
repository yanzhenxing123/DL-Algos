"""
@Time : 2023/10/24 19:33
@Author : yanzx
@Description :
"""
import random


def fitness(queens):
    """
    定义八皇后问题的适应度函数
    :param queens:
    :return:
    """
    n = len(queens)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if queens[i] == queens[j] or abs(queens[i] - queens[j]) == j - i:
                conflicts += 1
    return -conflicts  # 最小化冲突次数


def initialize_population(pop_size, n):
    """
    初始化种群
    :param pop_size:
    :param n:
    :return:
    """
    population = []
    for _ in range(pop_size):
        individual = list(range(n))
        random.shuffle(individual)
        population.append(individual)
    return population


def select(population, fitness_values):
    """
    选择操作（轮盘赌选择）
    :param population:
    :param fitness_values:
    :return:
    """
    total_fitness = sum(fitness_values)
    normalized_fitness = [f / total_fitness for f in fitness_values]
    selected = random.choices(population, weights=normalized_fitness, k=len(population))
    return selected


def crossover(parent1, parent2):
    """
    交叉操作（部分映射交叉）
    :param parent1:
    :param parent2:
    :return:
    """
    n = len(parent1)
    start = random.randint(0, n - 2)
    end = random.randint(start + 1, n - 1)
    child = [-1] * n
    for i in range(start, end + 1):
        child[i] = parent1[i]
    for i in range(n):
        if child[i] == -1:
            for gene in parent2:
                if gene not in child:
                    child[i] = gene
                    break
    return child


def mutate(individual, mutation_rate):
    """
    变异
    :param individual:
    :param mutation_rate:
    :return:
    """
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]


# 主函数：遗传算法的迭代
def genetic_algorithm(pop_size, generations, n, mutation_rate):
    population = initialize_population(pop_size, n)
    for generation in range(generations):
        fitness_values = [fitness(individual) for individual in population]
        selected_population = select(population, fitness_values)
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.extend([child, parent1])
        population = new_population
        best_individual = max(population, key=fitness)
        print(f"Generation {generation + 1}: Best individual = {best_individual}, Fitness = {fitness(best_individual)}")


if __name__ == "__main__":
    population_size = 100
    num_generations = 100
    board_size = 8
    mutation_prob = 0.1
    genetic_algorithm(population_size, num_generations, board_size, mutation_prob)
