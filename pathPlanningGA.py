# Description: This program uses a genetic algorithm to find the shortest path between a set of waypoints
from numpy import random as rand
from matplotlib import pyplot as plt
import random

import pygame
import fuzzyLogic as fl


class PathFinder:
    def __init__(self, populationSize, generations, waypoints, crossoverRate = 0.7, mutationRate=0.001, selectionMethod='elitism', crossoverMethod = 'ordered', isCoverage=False) -> None:
        self.populationSize = populationSize 
        self.generations = generations 
        self.crossoverRate = crossoverRate 
        self.mutationRate = mutationRate 
        self.waypoints = waypoints
        self.selectionMethod = selectionMethod
        self.crossoverMethod = crossoverMethod
        self.chromosomeLength = len(waypoints) 
        self.best_path = None
        self.population = []
        self.fuzzyLogic = fl.FuzzyLogic()
        self.prevFitness = 0
        self.currFitness = 0
        self.isCoverage = isCoverage

        # Create the paths
        for _ in range(self.populationSize):
            # Create a copy of the waypoints
            path = self.waypoints.copy()

            # Randomize the order of the waypoints
            rand.shuffle(path)

            # Add the path to the population
            self.population.append(path)

    def encodeChromosome(self, path):
        # Initialize the binary string
        binary_string = ''
        # Iterate over the waypoints in the path
        for waypoint in path:
            if type(waypoint) is float:
                continue
            # Find the index of the waypoint in the list of all waypoints
            index = self.waypoints.index(waypoint)

            # Convert the index to binary and pad it with zeros to make it of fixed length
            binary_code = format(index, '0' + str(self.chromosomeLength) + 'b')

            # Add the binary code to the binary string
            binary_string += binary_code

        return binary_string
    def decodeChromosome(self, binary_string):
        # Initialize the path
        path = []

        # Split the binary string into chunks
        for i in range(0, len(binary_string), self.chromosomeLength):
            binary_code = binary_string[i:i + self.chromosomeLength]

            # Convert the binary code back to an index
            index = int(binary_code, 2)

            # Get the corresponding waypoint
            waypoint = self.waypoints[index]

            # Add the waypoint to the path
            path.append(waypoint)

        return path
    def fitnessFunc(self, individual):
        try:
            if(self.isCoverage is False):
                # Calculate the total distance of the path
                total_distance = 0
                for i in range(len(individual) - 1):
                    if(type(individual[i])!=tuple or type(individual[i+1])!=tuple):
                        continue
                    waypoint1 = individual[i]
                    waypoint2 = individual[i + 1]
                    distance = round((((waypoint2[0] - waypoint1[0]) ** 2 + (waypoint2[1] - waypoint1[1]) ** 2) ** 0.5), 0)
                    total_distance += distance

                # Return the negative of the total distance (because shorter distances are better)
                return -total_distance
            elif(self.isCoverage is True):
                # Calculate the total distance of the path
                total_distance = 0
                for i in range(len(individual) - 1):
                    if(type(individual[i])!=tuple or type(individual[i+1])!=tuple):
                        continue
                    waypoint1 = individual[i]
                    waypoint2 = individual[i + 1]
                    distance = round((((waypoint2[0] - waypoint1[0]) ** 2 + (waypoint2[1] - waypoint1[1]) ** 2) ** 0.5), 0)
                    total_distance += distance

                # Calculate the number of unique points covered by the individual
                unique_points = len(set(individual))

                # The fitness is a weighted sum of total_distance and unique_points
                weight_distance = 0.5  # adjust this weight as needed
                weight_unique_points = 1  # adjust this weight as needed
                fitness = weight_distance * -total_distance + weight_unique_points * unique_points

                return fitness
            else:
                raise TypeError
        except TypeError:
            print("isCoverage must be a boolean")
            return exit()
            
    def evaluate(self):
        # Iterate over the population
        fitnessArray = []
        for individual in self.population:
            # Encode the chromosome
            binary_string = self.encodeChromosome(individual)

            # Decode the chromosome
            decoded_path = self.decodeChromosome(binary_string)

            # Calculate the fitness of the individual
            fitness = self.fitnessFunc(decoded_path) 

            # Add the fitness to the individual
            individual.append(fitness)
            fitnessArray.append(fitness)
        self.prevFitness = self.currFitness
        self.currFitness = sum(fitnessArray)/len(fitnessArray)
        return fitnessArray
    def selection(self):
        # Implement elitism
        try:
            match self.selectionMethod:
                case 'elitism':
                    # Select the top 10% of the population
                    if(len(self.population) < 20):
                        topPopulation = 5
                    else:
                        topPopulation = 10
                    sorted_population = sorted(self.population, key=self.fitnessFunc, reverse=True)
                    return sorted_population[:len(sorted_population) // topPopulation]
                case 'roulette':
                    # Select two parents based on the roulette wheel selection method
                    fitness_values = [self.fitnessFunc(individual) for individual in self.population]
                    total_fitness = sum(fitness_values)
                    probabilities = [fitness / total_fitness for fitness in fitness_values]
                    return random.choices(self.population, weights=probabilities, k=2)
                case 'tournament': 
                    # Select two parents based on the tournament selection method
                    tournament_size = len(self.population) // 10
                    tournament1 = random.sample(self.population, tournament_size)
                    tournament2 = random.sample(self.population, tournament_size)
                    parent1 = max(tournament1, key=self.fitnessFunc)
                    parent2 = max(tournament2, key=self.fitnessFunc)
                    return parent1, parent2
                case _:
                    raise RuntimeError
        except RuntimeError: 
            print('Selection method not implemented')
            return exit()
    def crossover(self): 
        try:
            match(self.crossoverMethod):
                case 'ordered':
                    # Perform ordered crossover
                    # Select two parents
                    parents = self.selection()
                    if(len(parents) == 2):
                        parent1 = parents[0]
                        parent1 = parent1[:-1]
                        parent2 = parents[1]
                        parent2 = parent2[:-1]
                    else:
                        parent1 = random.choice(parents)
                        parent1 = parent1[:-1]
                        parent2 = random.choice(parents)
                        parent2 = parent2[:-1]
                        if(parent2==parent1):
                            while(parent2==parent1):
                                parent2 = random.choice(parents)
                            parent2 = parent2[:-1]                            
                    # Randomly select a slice of parent1
                    start = rand.randint(1, len(parent1))
                    end = rand.randint(start, len(parent1))
                    child = [None]*len(parent1)
                    child[start:end] = parent1[start:end]

                    # Fill the remaining positions with the genes from parent2 in the order they appear in parent2
                    pointer = end
                    for gene in parent2:
                        if gene not in child:
                            if pointer >= len(parent1):
                                pointer = 0
                            child[pointer] = gene
                            pointer += 1
                    return child
                case 'partially_mapped':
                    #TODO: Implement partially mapped crossover
                    pass
                case 'cycle':
                    #TODO: Implement cycle crossover
                    pass
                case 'edge_recombination':
                    #TODO: Implement edge recombination crossover
                    pass
                case 'uniform':
                    #TODO: Implement uniform crossover
                    pass
                case 'linear':
                    #TODO: Implement linear crossover
                    pass
                case 'single_arithmetic':
                    #TODO: Implement single arithmetic crossover
                    pass
                case 'single_point':
                    #TODO: Implement single point crossover
                    pass
                case 'two_point':
                    #TODO: Implement two point crossover
                    pass
                case 'half_uniform':
                    #TODO: Implement half uniform crossover
                    pass
                case 'uniform_mask':
                    #TODO: Implement uniform mask crossover
                    pass
                case 'three_parent':
                    #TODO: Implement three parent crossover
                    pass
                case _:
                    raise RuntimeError
        except RuntimeError:
            print('Crossover method not implemented')
            return exit()
    def mutate(self, individual): #Two point swap
        # Randomly select two indices
        index1, index2 = random.sample(range(len(individual)-1), 2)

        # Swap the waypoints at these indices
        individual[index1], individual[index2] = individual[index2], individual[index1]

        # Replace the individual in the population
        #self.population[self.population.index(individual)] = individual
        return individual
    def normalizeFitness(self, minFitness, maxFitness, fitnessVal):
        # Min-max normalization
        if maxFitness == minFitness:
            return 0
        return (fitnessVal - minFitness)/(maxFitness - minFitness)

    def run(self, callback=None):
        # Run the genetic algorithm for the specified number of generations
        #Population is initialized in the constructor
        
        # Initialize lists to store the average and best fitness
        avg_fitnesses = []
        best_fitnesses = []
        all_solutions = []

        for generation in range(self.generations):
            #print('Initial Population:', self.population)
            print('Population size:', len(self.population))
            print('Cross-over rate:', self.crossoverRate)
            print('Mutation rate:', self.mutationRate)
            print('Selection method:', self.selectionMethod)
            print('Crossover method:', self.crossoverMethod)
            print('Generation:', generation + 1)
            # Evaluate the population
            popFit = self.evaluate() #[self.fitnessFunc(individual) for individual in self.population]
            self.population = sorted(self.population, key=lambda individual: self.fitnessFunc(individual), reverse=True) #[x for _, x in sorted(zip(popFit, self.population), key=lambda pair: pair[0], reverse=True)]
            print('Population Fitness:', sum(popFit)/len(popFit))
            newPopulation = []
            # Call the defuzzify method of the fuzzy logic class
            #all_solutions.append(max(self.population, key=self.fitnessFunc))

            self.prevFitness = self.normalizeFitness(min(popFit), max(popFit), self.prevFitness)
            self.currFitness = self.normalizeFitness(min(popFit), max(popFit), self.currFitness)
            self.mutationRate, self.crossoverRate = (self.fuzzyLogic.defuzzify(self.prevFitness, self.currFitness)[0], 
                                                     self.fuzzyLogic.defuzzify(self.prevFitness, self.currFitness)[1])

            # Determine the number of elites
            num_elites = int(self.populationSize * 0.02)  # 2% of the population

            newPopulation = []

            # Perform selection, crossover, and mutation to fill the rest of the new population
            while len(newPopulation) < self.populationSize:
                # if len(newPopulation) == 0:
                #     newPopulation = self.population[:num_elites+1]
                newIndividual = self.crossover()
            
                if(rand.random() < self.mutationRate):
                    newIndividual = self.mutate(newIndividual)
                
                newPopulation.append(newIndividual)

            # Get the current best path
            self.best_path = max(self.population, key=self.fitnessFunc)

            # Call the callback function with the current best path
            if callback:
                callback(self.best_path)
            
            self.population = newPopulation

            # Calculate the fitness of all individuals
            fitnesses = [self.fitnessFunc(individual) for individual in self.population]

            # Calculate and store the average and best fitness
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness = max(fitnesses)
            avg_fitnesses.append(avg_fitness)
            best_fitnesses.append(best_fitness)
        # Print the average and best fitness over generations
        print('Average fitness over generations:', avg_fitnesses)
        print('Best fitness over generations:', best_fitnesses)

        # Print the best path
        best_path = max(self.population, key=self.fitnessFunc) 
        print('Best path:', best_path , '\nDistance:', self.fitnessFunc(best_path))
        return avg_fitnesses, best_fitnesses, all_solutions
    
    def plot(self, avgFitnesses, bestFitnesses):
        # Calculate the inverse of the best and average fitnesses
        #bestFitnesses = [-x for x in bestFitnesses]
        #avgFitnesses = [-x for x in avgFitnesses]

        # Create a list of generations
        generations = range(1, len(bestFitnesses) + 1)

        # Plot the inverse of the best and average fitnesses
        plt.plot(generations, bestFitnesses, label='Best Fitness')
        plt.plot(generations, avgFitnesses, label='Average Fitness')

        # Add labels and a legend
        plt.xlabel('Generation')
        plt.ylabel('Fitness Level')
        plt.legend()

        # Show the plot
        plt.show()

    def drawMap(self, waypoints, bestPath, mapWidth, mapHeight):
        # Initialize Pygame
        pygame.init()

        backgroundColour = (144, 238, 144) #Light green
        waypointColour = (255, 255, 255) #White
        pathColour = (255, 0, 0) #Red
        borderColour = (222, 184, 135) #Light brown 
        waypointSize = 5

        # Create a Pygame window
        window = pygame.display.set_mode((mapWidth, mapHeight))  
        window.fill(backgroundColour)

        # Calculate the minimum and maximum x and y coordinates of the waypoints
        min_x = min(waypoint[0] for waypoint in waypoints) - waypointSize*2
        max_x = max(waypoint[0] for waypoint in waypoints) + waypointSize*2
        min_y = min(waypoint[1] for waypoint in waypoints) - waypointSize*2
        max_y = max(waypoint[1] for waypoint in waypoints) + waypointSize*2

        # Draw a border around the waypoints
        pygame.draw.rect(window, borderColour, pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y), 2)

        # Draw the waypoints
        for waypoint in waypoints:
            pygame.draw.circle(window, waypointColour, waypoint, waypointSize) 

        # Calculate the number of line segments to draw
        num_segments = len(bestPath) - 1 if len(bestPath) % 2 == 0 else len(bestPath) - 2

        # Draw the best path
        for i in range(num_segments):
            if type(bestPath[i]) is float or type(bestPath[i+1]) is float:
                break
            pygame.draw.line(window, pathColour, waypoints[waypoints.index(bestPath[i])], waypoints[waypoints.index(bestPath[i + 1])], 3)

        # Update the display
        pygame.display.update()

        # Wait for the user to close the window
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

    
def runGA():
    # Initialize an empty list to store the waypoints
    waypoints = []

    # Define the dimensions of the map
    map_width = 300
    map_height = 300   

    # Define the dimensions of the area where the waypoints will be generated
    area_width = (map_width//10)//2  # Size of Plot
    area_height = (map_height//10)//2  # Size of Plot

    # Calculate the offset to center the waypoints in the map
    offset_x = (map_width)//2 - area_width*5
    offset_y = (map_height)//2 -area_height*5

    # Generate a calculated # of random waypoints
    for _ in range(area_height*area_width): 
        # Generate a random x and y coordinate
        x = random.randint(0, area_width)*10  # X coordinate * size
        y = random.randint(0, area_height)*10  # Y coordinate * size

        # Adjust the coordinates to center the waypoints in the map
        x += offset_x
        y += offset_y
        
        if(x,y) in waypoints:
            continue

        

        # Add the waypoint to the list
        waypoints.append((x, y))
    popSize = 100
    generations = 25
    crossoverRate = 0.7
    mutationRate = 0.002
    crossoverMethod = 'ordered'
    selectionMethod = 'elitism'
    isCoverage = True
    #mutationMethod = '2p swap'

    # Create an instance of the class
    pathFinder = PathFinder(popSize, generations, waypoints, crossoverRate, mutationRate, selectionMethod, crossoverMethod, isCoverage)  

    fitnesses = pathFinder.run()
    avgFitnesses = fitnesses[0]
    bestFitnesses = fitnesses[1]
    allSolutions = fitnesses[2]

    pathFinder.plot(avgFitnesses, bestFitnesses)
    pathFinder.drawMap(waypoints, pathFinder.best_path, map_width, map_height)

runGA()