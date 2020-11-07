import matplotlib.pyplot as plt
import matplotlib
 
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
 
plt.ion()
def plot_durations(episode_scores):
    plt.figure(2)
    plt.clf()
    scores_t = T.tensor(episode_scores, dtype=T.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    # plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = T.cat((T.zeros(99), means))
        plt.plot(means.numpy())
 
    plt.pause(0.1)  # pause a bit so that plots are updated
    if is_ipython:
#         display.clear_output(wait=True)
        display.display(plt.gcf())
# from dqn import *
import time
# import time, threading


import pymunk
from pymunk.vec2d import Vec2d
 

 
def newcircle(space, x,y,r):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (x,y)
    shape = pymunk.Circle(body,r)
    shape.elasticity =1
    shape.friction=1
    space.add(body,shape)
    return shape
    
 
 
def newship(space, size, mass,username,structure,colour): 
    body = pymunk.Body()
    space.add(body)
 
    body.position = Vec2d(1000,460)
    # body.position = Vec2d(0,0)
    
    
    
    shape = pymunk.Poly.create_box(body, (size, size), 0.0)
    shape.filter = pymunk.ShapeFilter(group=1)
    shape.elasticity=0
    shape.friction=1
    shape.mass = mass
    shape.friction = 0.7
    shape.username = username
    shape.structure = structure
    shape.input = [False,False,False,False,False]
    shape.positionPrevx=1000
    shape.positionPrevy=460
    shape.done=False
    
    space.add(shape)
    
    return shape
 
    
    
    
    
import math
 
 
 
def getInputs(ship):
    p = ship.body
    
    
    distance = math.hypot(p.position.x-1000,p.position.y-1000)
    movement = math.hypot(p.position.x-ship.positionPrevx,p.position.y-ship.positionPrevy)
    angle = p.angle
 
    
    angledif = math.atan2(p.position.x-1000,p.position.y-1000) - angle
    moveangle = math.atan2(p.velocity.y, p.velocity.x)
    
#     regularisation : make inputs between -1 and 1
 
    distance = distance / 5000
    movement = movement / 10 
    angledif = math.fmod(angledif, (math.pi*2))/math.pi/2
    moveangle = math.fmod(moveangle, (math.pi*2))/math.pi/2
    ship.positionPrevx = p.position.x
    ship.positionPrevy = p.position.y
    
    
 
    
    return [distance,movement,angledif,moveangle]
    # return [p.position.x,p.position.y,p.velocity.x,p.velocity.y,math.fmod(p.angle, (math.pi*2))/math.pi/2,math.fmod(p.angular_velocity, (math.pi*2))/math.pi/2]
 
 
# global scores
# global eps_history
# scores, eps_history = [], []

# ships["AI"] = newbox(space,80,100,"OrbitAI",struct,"AI")

global mindistance
mindistance = 455
# global observation
 
# observation = getInputs()
    
 
global steps
steps = 0
def calculatereward(AIship):
    global steps
    global mindistance
    done=False
    
    distance = math.hypot(AIship.body.position.x-1000,AIship.body.position.y-1000)
    reward = distance/10000
#     reward = 1
    
    movement = math.hypot(AIship.body.position.x-AIship.positionPrevx,AIship.body.position.y-AIship.positionPrevy)

    # print(movement)
    if distance > 5000 or distance < mindistance or (movement<0.001 and steps >=100):
        done = True
        # if (distance > 5000):
        #     print('Distance')
        # if (distance < mindistance):
        #     print('min distance')
        # if (movement<0.001 and steps >=100):
        #     print('movement')
        reward = -1000
    return reward,done
global episode
episode=0
 
score = 0
 
 
actionlist = {0:[False,True,True,False,False],
    1:[True,False,True,False,False],
    2:[False,False,True,False,False],
    3:[True,False,False,False,False],
    4:[False,False,False,False,False],
    5:[False,True,False,False,False],
    6:[True,False,False,True,False],
    7:[False,True,False,True,False],
    8:[False,False,False,True,False],
  }
global planets
global space
global manualplanets
global ships
def gameinit():
    global ships
    global space
    space = pymunk.Space()  
    space.gravity = 0,0
    ships = []
    global manualplanets
    manualplanets = [{'x':1000,'y':1000,'r':500}]
    
    global planets
    planets = []
    for planet in manualplanets:
        planets.append(newcircle(space,planet['x'],planet['y'],planet['r']))
    
 
def update():
    global manualplanets
    global space
    global ships
    for key,ship in enumerate(ships):
        # print(ships[key].input)
        left,right,up,down,spacepress = ships[key].input
        if up:
            ships[key].body.apply_impulse_at_local_point((0,-300), (0,0))
        if left:
            ships[key].body.angular_velocity += -0.01
        else:
            if right:
                ships[key].body.angular_velocity += 0.01
        
        for planet in planets:
            # print(ships[key])
            distance = math.hypot(planet.body.position.x-ships[key].body.position.x,planet.body.position.y-ships[key].body.position.y)
            direction = math.atan2(planet.body.position.y-ships[key].body.position.y,planet.body.position.y-ships[key].body.position.x) +math.pi/2
            force = planet.area/10 / ((distance)**2)
 
            # print([distance,direction,force])
            
            ships[key].body.velocity += Vec2d(math.sin(direction)*force ,-math.cos(direction)*force)
        if spacepress:
            # ships[key].body.position = Vec2d(1000,460)
            ships[key].body.position = Vec2d(1000,460)
            ships[key].body.velocity = Vec2d(0,0)
            ships[key].body.angular_velocity = 0
            ships[key].body.angle = 0
        
        # if ships[key].body.angular_velocity > 0.3:
        #     ships[key].body.angular_velocity = 0.3
        # else:
        #     if ships[key].body.angular_velocity < -0.3:
        #         ships[key].body.angular_velocity = -0.3
    space.step(1)
 
import pygame
import pymunk.pygame_util   
global scalediv
global shipimage
scalediv = 20
shipimage = pygame.image.load('./static/spaceship.png')
shipimage = pygame.transform.smoothscale(shipimage, (int(80 /scalediv), int(80/scalediv)))
def pyinit():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((1000,540))
    clock = pygame.time.Clock()
    # global draw_options
    # draw_options = pymunk.pygame_util.DrawOptions(screen)
def pyrender():
    global shipimage
    global screen
    global ships
    global scalediv
    centrex = 500
    centrey = 200
    artmode = True
    if not artmode:
        screen.fill(pygame.color.THECOLORS["black"])

    for planet in planets:
        pygame.draw.circle(screen,(100,100,100),(int(planet.body.position.x/scalediv)+centrex,int(planet.body.position.y/scalediv+centrey)),int(planet.radius/scalediv))
    for ship in ships:
        tempimage = pygame.transform.rotate(shipimage,ship.body.angle)
        screen.blit(shipimage, (int(ship.body.position.x/scalediv +centrex ),int(ship.body.position.y/scalediv)+centrey))

    pygame.display.flip()
 

import os
import time
import neat
import visualize
import pickle
global gen
gen = 0    
def run_car(genomes, config):
    
    gameinit()
    global space
    global mindistance
    global ships
    global stats
    global generation
    global steps
    # Init NEAT
    nets = []
    ships = []
    struct = {0:{0:'bullet'}}
    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        # Init my cars
        ships.append(newship(space,80,100,"OrbitAI",struct,"AI"))

    # Init my game
    pyinit()


    # Main loop
    
    generation += 1
    
    if generation % 10 == 0:
      # print(p)
      visualize.plot_stats(statistics=stats)
      # ualize.plot_stats()
    
    mindistance = 455
    
    steps = 0
    maxstep = 300_000
    forever = False
    while forever or steps <= maxstep:

        if steps == maxstep:
            finished = 10_000
            print('Finished!')
        else:
            finished = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    screen.fill(pygame.color.THECOLORS["black"])
        pyrender()

        # Input my data and get result from network
        for index, ship in enumerate(ships):
            output = nets[index].activate(getInputs(ship))
            i = output.index(max(output))
            ship.input = actionlist[i]

        # Update car and fitness
        remain_ships = 0
        
        update()
        for i, ship in enumerate(ships):
            reward,done = calculatereward(ship)
            ship.done = done
            if not done:
                remain_ships += 1
#                 car.update(map)
                genomes[i][1].fitness += reward + finished
            else:
                space.remove(ship)
                nets.pop(ships.index(ship))
                genomes.pop(ships.index(ship))
                ships.pop(ships.index(ship))
        # print(remain_ships)
                
        
        # check
#         print(remain_ships)
        if remain_ships == 0:
            break
        
        if mindistance<=2000:
            mindistance += 0.4
        steps +=1

        
        
    

global p
def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy ship.
    :param config_file: location of config file
    :return: None
    """
    print('Started')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    global stats
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(100))

    # Run for up to 50 generations.
#     winner = p.run(eval_genomes, 9999999999999)
    winner = p.run(run_car, 999999999999999)
    
    
    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

             
 
if __name__ == '__main__':
#     env = gym.make('LunarLander-v2')
 

    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-feedforward.txt')
    generation = 0
    run('config-feedforward.txt')
   
 
