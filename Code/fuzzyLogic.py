from matplotlib import pyplot as plt
from skfuzzy import control as ctrl
import numpy as np

class FuzzyLogic:
    def __init__(self) -> None:
        self.prevFitness = None
        self.currFitness = None 
        self.mutationRate = None
        self.crossoverRate = None
        self.simulation = None
    def fuzzySets(self):
        # Define the universe of discourse
        self.prevFitness = ctrl.Antecedent(np.arange(0, 1, 0.01), 'prevFitness')
        self.currFitness = ctrl.Antecedent(np.arange(0, 1, 0.01), 'currFitness')
        self.mutationRate = ctrl.Consequent(np.arange(0, 1, 0.001), 'mutationRate')
        self.crossoverRate = ctrl.Consequent(np.arange(0, 1, 0.1), 'crossoverRate')

        # Define the membership functions
        names = ['low', 'medium', 'high']
        for var in [self.prevFitness, self.currFitness, self.mutationRate, self.crossoverRate]:
            var.automf(names=names)

    def fuzzyRules(self):
        # Define the fuzzy rules
        rule1 = ctrl.Rule(self.prevFitness['low'] & self.currFitness['low'], 
                        (self.mutationRate['high'], self.crossoverRate['low']))
        rule2 = ctrl.Rule(self.prevFitness['low'] & self.currFitness['medium'], 
                        (self.mutationRate['medium'], self.crossoverRate['low']))
        rule3 = ctrl.Rule(self.prevFitness['low'] & self.currFitness['high'], 
                        (self.mutationRate['low'], self.crossoverRate['medium']))
        rule4 = ctrl.Rule(self.prevFitness['medium'] & self.currFitness['low'], 
                        (self.mutationRate['high'], self.crossoverRate['medium']))
        rule5 = ctrl.Rule(self.prevFitness['medium'] & self.currFitness['medium'], 
                        (self.mutationRate['medium'], self.crossoverRate['medium']))
        rule6 = ctrl.Rule(self.prevFitness['medium'] & self.currFitness['high'], 
                        (self.mutationRate['low'], self.crossoverRate['high']))
        rule7 = ctrl.Rule(self.prevFitness['high'] & self.currFitness['low'], 
                        (self.mutationRate['medium'], self.crossoverRate['high']))
        rule8 = ctrl.Rule(self.prevFitness['high'] & self.currFitness['medium'], 
                        (self.mutationRate['low'], self.crossoverRate['high']))
        rule9 = ctrl.Rule(self.prevFitness['high'] & self.currFitness['high'], 
                        (self.mutationRate['low'], self.crossoverRate['high']))

        # Create the control system with these rules
        self.controlSystem = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

        # Create the simulation
        self.simulation = ctrl.ControlSystemSimulation(self.controlSystem)
    def defuzzify(self, prev_fitness_value, curr_fitness_value):
        self.fuzzySets()
        self.fuzzyRules()
        # Set the input values
        self.simulation.input['prevFitness'] = prev_fitness_value
        self.simulation.input['currFitness'] = curr_fitness_value

        # Compute the output values
        self.simulation.compute()

        # Get the crisp values
        crisp_mutation_rate = self.simulation.output['mutationRate']
        crisp_crossover_rate = self.simulation.output['crossoverRate']
        return crisp_mutation_rate, crisp_crossover_rate
    def view(self):
        # View the membership function
        self.prevFitness.view()
        self.currFitness.view()
        self.mutationRate.view()
        self.crossoverRate.view()

        plt.show()

def testRun():
    fsets = FuzzyLogic()
    fsets.fuzzySets()
    fsets.fuzzyRules()
    fsets.view()
