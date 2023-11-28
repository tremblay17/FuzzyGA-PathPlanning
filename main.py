import sys, getopt
import pathPlanningGA
import random
import env

#TODO: Make main.py work instead of running pathPlanningGA.py
def runGA(populationSize, mutationProb, crossoverProb, numOfGenerations, map_width, map_height):
    # Initialize an empty list to store the waypoints
    waypoints = []

    # Define the dimensions of the map
    map_width = map_width
    map_height = map_height

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
    popSize = populationSize
    generations = numOfGenerations
    crossoverRate = crossoverProb
    mutationRate = mutationProb
    crossoverMethod = 'ordered'
    selectionMethod = 'tournament'
    #mutationMethod = '2p swap'

    # Create an instance of the class
    pathFinder = pathPlanningGA.PathFinder(popSize, generations, waypoints, crossoverRate, mutationRate, selectionMethod, crossoverMethod)  

    fitnesses = pathFinder.run()
    avgFitnesses = fitnesses[0]
    bestFitnesses = fitnesses[1]

    pathFinder.plot(avgFitnesses, bestFitnesses)
    pathFinder.drawMap(waypoints, pathFinder.best_path, map_width, map_height)

def main(argv):
    populationSize = 25
    mutationProb = 0.001
    crossoverProb = 0.7
    numOfGenerations = 100
    map_width=400 
    map_height=400

    opts, args = getopt.getopt(argv,"h:p:m:c:g:w:h:",["population=","mutationrate=",
                                                  "crossoverrate=","generations=", "width=", "height="])

    for opt, arg in opts:
        if opt == '-h':
            print("Usage: main.py -p <populationSize> -m <mutationProb> -c <crossoverProb> -g <numOfGenerations>")
            return 1
        elif opt in ("-p", "--population"):
            populationSize = int(arg)
        elif opt in ("-m", "--mutationrate"):
            mutationProb = float(arg)
        elif opt in ("-c", "--crossoverrate"):
            crossoverProb = float(arg)
        elif opt in ("-g", "--generations"):
            numOfGenerations = int(arg)
        elif opt in ("-w", "--width"):
            map_width = int(arg)
        elif opt in ("-h", "--height"):
            map_height = int(arg)
        else:
            print("Usage: main.py -p <populationSize> -m <mutationProb> -c <crossoverProb> -g <numOfGenerations>")
            return 1
    runGA(populationSize, mutationProb, crossoverProb, numOfGenerations, map_width, map_height)
    
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])