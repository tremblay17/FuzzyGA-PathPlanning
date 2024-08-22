# Genetic Algorithm for Coverage Path Planning
This repository contains a Python script that defines a fitness function for path optimization. The fitness function is designed to find the most unique paths with the shortest distance.

### Code Explanation
The script contains a function named ```fitnessFunc``` which calculates the fitness of a given path (individual). The fitness is a weighted sum of the total distance covered by the path and the number of unique points covered by the path. It takes the negative total distance so the goal is still maximizing the shortest distance

Here's a brief explanation of how the function works:

1. Calculate the total distance of the path: The function iterates over each pair of consecutive waypoints in the individual. It calculates the Euclidean distance between the two waypoints and adds it to the total distance. If either waypoint is not a tuple, it skips to the next pair.

2. Calculate the number of unique points covered by the path: The function calculates the number of unique points in the individual by converting the list of waypoints to a set and getting its length.

3. Calculate the fitness: The fitness is a weighted sum of the total distance and the number of unique points. The weights for the total distance and the number of unique points can be adjusted based on how much importance you want to give to each factor.

### Mutation
Mutation is a genetic operator used to maintain genetic diversity from one generation of a population to the next. It alters one or more gene values in a chromosome from its initial state. In mutation, the solution may change entirely from the previous solution. Hence GA can come to a better solution by using mutation.

Here's a simple pseudocode for mutation:

1. For each gene in the individual:
    - Generate a random number
    - If the random number is less than the mutation rate, change the gene

This algoritm implements several other types of mutation that can be selected.

### Crossover
Crossover is another genetic operator used to vary the programming of chromosomes from one generation to the next. It is a process of taking more than one parent solutions and producing a child solution from them.

Here's a simple pseudocode for a one-point crossover:

1. Choose a random crossover point
2. The new individual (offspring) is created by taking all genes from the first parent up to the crossover point, then all genes from the second parent after that.

This algoritm implements several other types of crossovers that can be selected.

### Usage
To use the ```fitnessFunc``` function, you need to pass in an individual, which is a list of waypoints. Each waypoint is a tuple containing the coordinates of the point.

```
individual = [(0, 0), (1, 1), (2, 2), (1, 1), (0, 0)]
fitness = fitnessFunc(individual)
```

In the above example, the ```fitnessFunc``` function calculates the fitness of the path defined by the ```individual``` list.

### Customization
You can adjust the weights ```weight_distance``` and ```weight_unique_points``` in the ```fitnessFunc``` function based on how much importance you want to give to the total distance and the number of unique points, respectively.

# 
# Fuzzy Logic

This Python code is implementing a fuzzy logic system using the ```skfuzzy``` library. Fuzzy logic is a form of many-valued logic in which the truth values of variables may be any real number between 0 and 1. It is employed to handle the concept of partial truth, where the truth value may range between completely true and completely false.

The ```fuzzySets``` function is defining the universe of discourse for four variables: ```prevFitness```, ```currFitness```, ```mutationRate```, and ```crossoverRate```. The universe of discourse is the range of all possible values that a variable can take. Here, it's defined as a range from 0 to 1 with a step of 0.01. ```prevFitness``` and ```currFitness``` are defined as ```Antecedent``` (input) variables, while ```mutationRate``` and ```crossoverRate``` are defined as ```Consequent``` (output) variables.

The membership functions for these variables are defined using the ```automf``` function, which automatically defines membership functions. The membership functions are named 'low', 'medium', and 'high', which means each variable can take a low, medium, or high value based on its membership function.

The ```fuzzyRules``` function is defining the fuzzy rules for the system. Fuzzy rules are if-then rules that are used to determine the output of a fuzzy system. Here, nine rules are defined based on the combinations of 'low', 'medium', and 'high' values of ```prevFitness``` and ```currFitness```. The output of each rule is a tuple of 'low', 'medium', or 'high' values for ```mutationRate``` and ```crossoverRate```.

The rules are then used to create a control system, which is a system that manages the operations of the other parts of a system using control loops. A simulation of the control system is also created, which can be used to simulate the behavior of the system for different inputs.

