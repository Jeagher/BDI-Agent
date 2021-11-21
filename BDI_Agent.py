import numpy as np
import time
from copy import deepcopy
from tkinter import *

##### Fire Fighter simulation in a open environnement with BDI Agents
# Fire Fighters BDI Agents patrol in the forest and have to extinguish fire, to do so they have acces to water in their Base 
# And in fire Trucks that they can freely moove on the map to reduce the agent travel time between fire and water
# Fire Truck can contain up to X units of water, it can be refilled at the Base, N agents are needed to moove it 
# Agent also have to take a break to eat ans sleep every Y actions during a T period of time
# Agent can communicate using walkie-talkie

##### Map Representation
# '-' : Noting, 'O' : Obstacle, 'B' : Fire Fighter Base, 'H' : Heat Point (A special obstacle that can easily ignite his surrounding, invisible to the agents), 
# 'T' : Fire Truck (Truck that agent can moove to have easier acces to water), 'F' : Fire, 'A' Agent(s)

# An obstacle, a Heat point and the base can't be set on fire, also agents and truck can't moove into fire. 
# If an agent accidently happen to be in a fire cell, he has to flee to the base to get cure and rest 
# If an agent is in a fire cell and can't escape (4 directions are in fire), he will die within Z iterations if not helped (not implemented yet)  


NA = 2  # Number of Agents 
NH = 5 # Number of Heat points
NO = 20 # Number of obstacles 
NT = 0 # Number of Trucks

IT = 1000 # Number of environnement iteration

HEIGHT = 20 # Map Height
LENGHT = 20 # Map Lenght

MAX_SUPPLY = 20 # Truck Max water supply
MAX_ENERGY = 100 # Agent maximum energy, need to rest before going back to work
MAX_BRANCH_LENGHT = 10 # Max lenght of an exploration branch 
DIVISION_ITER = 2 # Number of child node of each node in the exploration algorithm 
VISION_RANGE = 1 # The agent can see up to 1 cell 
IGNITION_PROBA = 0.05 # Heat Points ignition probability
PROPAGATION_PROBA = 0.01 # Fire propagation probability

KEEP_INTENTION_PROBA = 0 # Probability to keep the current intention, a high probability will mean a reactive agent but a loss in persistance

# Map Area class
class MapArea():
    def __init__(self,lenght,height):
        self.height = height
        self.lenght = lenght
        self.map_area = []
        self.map_last_seen = [] 
        self.base_pos = (self.height//2,self.lenght//2)
        self.fixed_elt_pos = [(self.height//2,self.lenght//2)] # Only the base position is registered at the begining 
        self.initialise_map_area()

    # At the begining, the agents known nothing of the environnement expect the fact that it is a height*lenght are and the base position
    # A map_last_seen exist to help agent exploration and keep track of when a place has last been seen. 
    def initialise_map_area(self):
        area_grid = np.full((self.height, self.lenght),' ')
        area_grid[self.height//2,self.lenght//2] = 'B'
        last_seen_grid = np.full((self.height, self.lenght), 50)
        last_seen_grid[self.height//2,self.lenght//2] = 0
        self.map_area = area_grid
        self.map_last_seen = last_seen_grid
    
    def update(self,update_list):
        for elt in update_list :
            (x,y),sigle = elt
            self.map_area[y,x] = sigle
            self.map_last_seen[y,x] = 0
            if sigle in ['O','H'] and (x,y) not in self.fixed_elt_pos : # if it is an obstacle
                self.fixed_elt_pos.append((x,y))

    # Get all neighbour cell not in the non_valid list
    def get_valid_neighbour_cells(self,pos,non_valid=[]):
        x,y = pos
        possible_neighbour = [(i,j) for i in range(-1,2,1) for j in range(-1,2,1) if (i,j) != (0,0)] # 8 directions 
        valid_neighbour = []
        for i,j in possible_neighbour :
            if x+i >= 0 and x+i < self.height and y+j >= 0 and y+j < self.lenght and self.map_area[y+j,x+i] not in non_valid : # If (x+i,y+j) in bound and a 'valid' cell
                valid_neighbour.append((x+i,x+j))
        return valid_neighbour

    def get_cell(self, pos, type='area'):
        x,y = pos
        if x>=0 and x< self.lenght and y>= 0 and y<self.height : # (x,y) is in map boundaries
            if type == 'area' :
                return self.map_area[y,x]
            else :
                return self.map_last_seen[y,x]
        else : 
            return None

    def get_fire_cells(self):
        fire_pos_list = []
        for x in range(self.height):
            for y in range(self.lenght):
                if self.map_area[y,x] == 'F' : # There is a fire in this cell 
                    fire_pos_list.append((x,y))
        return fire_pos_list

    # At the end of a iteration, add +1 value for all cell that can change
    def new_day(self):
        ind = [1 if (x,y) not in self.fixed_elt_pos else 0 for x in range(self.lenght) for y in range(self.height)] 
        self.map_last_seen += np.array(ind).reshape((self.lenght,self.height)) 

# Class Environnement
class Environnement():
    def __init__(self, height, lenght):
        self.map_area = MapArea(lenght,height)
        self.height = height
        self.lenght = lenght
        self.heat_points_pos = []
        self.obstacles_pos = []
        self.base_pos =  ()
        self.agents_pos = {}
        self.trucks_pos = {}
        self.create_env(NA,NH,NO,NT)

    # No Fire Trucks in V1
    def create_env(self, nA, nH, nO, nT):
        # Map initialisation with nO obstacles, nH Heat points, nT trucks next to tha base and the base
        grid = self.map_area.map_area
        lock_ind = [self.lenght*(self.height//2 - 2 + j) + (self.lenght//2 - 2) + i for i in range(5) for j in range(5)] # Select the ind of a 5*5 square around the base 
        truck_ind = np.random.choice(lock_ind, size=NT, replace=False) # Randomly select nT elements in the locked area
        # grid.ravel()[truck_ind] = np.array(['T' for i in range(nT)]) # Place the trucks in the locked area
        ind = np.array([i for i in range(self.height*self.lenght) if i not in lock_ind]) # Select all possibles indices for Obstacles and Heat Points
        np.random.shuffle(ind) # Shuffle the list 
        ind = ind[:nO+nH] # Select only nO+nH elements 

        # Modify the array returned by ravel which is a view of the original array, place Obstacles and Heat Points
        obs_ind = ind[:nO]
        heat_ind = ind[-nH:]
        grid.ravel()[obs_ind] = np.array(['O' for i in range(nO)])
        grid.ravel()[heat_ind] = np.array(['H' for j in range(nH)])

        # Initialise Environement variables 
        self.map_area.map_area =  grid
        base_pos = (self.lenght//2,self.height//2) # The base is in the middle of the map 
        self.base_pos = base_pos 
        self.heat_points_pos = [(i//self.lenght,i%self.lenght) for i in heat_ind]
        self.obstacles_pos = [(i//self.lenght,i%self.lenght) for i in obs_ind]
        truck_ind = [(i//self.lenght,i%self.lenght) for i in truck_ind]
        self.trucks_pos = {i:truck_ind[i] for i in range(len(truck_ind))}
        self.agents_pos = {i:(base_pos) for i in range(nA)} # All agents start in the base

    def set_fire(self,pos):
        x,y = pos
        if self.map_area.get_cell(pos,'area') in ['A',' '] :
            self.map_area.map_area[y,x] = 'F'
            return True
        return False

    def extinguish_fire(self,pos):
        x,y = pos
        if self.map_area.get_cell(pos,'area') != 'F' : 
            return False
        self.map_area.map_area[y,x] = ' '
        return True

    def spread_fire(self):
        fire_pos_ind = []
        for y in range(self.lenght) :
            for x in range(self.height):
                if self.map_area.map_area[y,x] == 'F' : # If there is a fire 
                    fire_pos_ind.append((x,y))
        
        # Get all none obstacle/base/heat point cell which as a fire cell as neighbour
        valid_neighbour = []
        for x,y in fire_pos_ind :
            valid_neighbour += self.map_area.get_valid_neighbour_cells((x,y),['O','H','B','F'])
        # Start a fire in this case as fire propagation phenomenom with a Propagation probability
        for xn,yn in valid_neighbour :
            if np.random.random() < PROPAGATION_PROBA :
                self.set_fire((xn,yn))

    # Heat points are points were a fire can start 
    def heat_point_ignition(self):
        ignite_neighbour = []
        for x,y in self.heat_points_pos :
            if np.random.random() < IGNITION_PROBA :
                ignite_neighbour = self.map_area.get_valid_neighbour_cells((x,y),['O','H','B','F'])
        for x,y in ignite_neighbour :
            self.set_fire((x,y))

    # Update Agent/truck position when he mooves
    def update_agent_pos(self,id,pos,prev_pos):
        x,y = pos
        x_prev,y_prev = prev_pos
        self.agents_pos[id] = pos
        if pos != self.base_pos : # The agent can go into the base, but the base will still be displayed 
            self.map_area.map_area[y,x] = 'A'
        
        keep = False
        # If at least an agent is on the previous cell if it is a truck only one truck can be on a cell 
        if prev_pos != self.base_pos : 
            for id in self.agents_pos.keys():
                if self.agents_pos[id] == prev_pos : 
                    keep = True
                    break
        if keep == False :
            self.map_area.map_area[y_prev][x_prev] = ' '

    # Display on the IPython Console    
    def representation(self):
        rep = ""
        for y in range(self.height):
            for x in range(self.lenght):
                rep += self.map_area.get_cell((x,y))
            rep += ' \n'
        print(rep)
        print('\n \n')

# class FireTruck():
#     def __init__(self,env,id):
#         self.pos = ()
#         self.water_supply = 0
#         self.env = env
#         self.id = id

#     def refill_water(self):
#         if abs(self.pos[0] - LENGHT//2) in [0,1] and abs(self.pos[1]-HEIGHT//2) in [0,1] : # Then the truck is next to the base 
#             self.water_supply = MAX_SUPPLY
#             return True
#         return False 

#     def init_pos(self):
#         self.pos = self.env.trucks_pos[self.id]

#     def water_taken(self):
#         return min(1,max(self.water_supply-1,0))

#     def check_water_supply(self):
#         return self.water_supply 

#     def update_pos(self,new_pos):
#         self.env.update_agent_pos(self.id,new_pos,self.pos)
#         self.pos = new_pos 

# Node object Used in path planning algorithm and exploration algorithm
class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
        self.exploration_score = 0
        self.in_sight=[]


# Each agent has is own beliefs of the are but when an agent update the map, he will tell the other using his walkie-talkie to update it as well
# It is more efficient computationly speaking to use a shared map object than creating a object for every agent and update it every time
# But the idea is the same. The map is part of every agent personnal beliefs 

# Desire class
class Desire():
    def __init__(self,priority_value,intention,sub_desire=None):
        self.priority_value = priority_value
        self.sub_desire = sub_desire
        self.intention = intention

# Intetion class
class Intention():
    def __init__(self,action,location):
        self.action = action
        self.location = location

# Agent class
class Agent():
    # The agent has at the begining an empty map of the area that will be be update with his perception function, when uptdated, the agent tells the other what he knows
    # So only one shared map is needed it becomes part of his beliefs base and will be uptdated during the process 
    # The agent has a bag that can contain 1 watter unit used to extinguish the fire in one cell of the map 

    def __init__(self, env, id, map_area):
        # Agent Beliefs
        self.energy = MAX_ENERGY
        self.vision_range = VISION_RANGE
        self.max_branch_lenght = MAX_BRANCH_LENGHT
        self.pos = ()
        self.map_area = map_area
        self.has_water = False
        self.id = id
        self.env = env
        self.rest_time = 0
        self.partner_current_intention = {}  
        # Agent Desires
        self.desire_base = []
        # Agent Intentions 
        self.intention_list = []

        self.init_pos()

    def init_pos(self):
        self.pos = self.env.agents_pos[self.id]
    
    def update_pos(self,new_pos):
        self.env.update_agent_pos(self.id,new_pos,self.pos)
        self.pos = new_pos 
    
    def vision(self,pos,map_type,eye_sight=True): # Map type 'area' or 'last_seen'
        current_area_state = []
        # An Obstacle vision management need to be done if vision management is higher than 1  
        visible_cells = [(pos[0]+i,pos[1]+j) for i in range(-self.vision_range,self.vision_range+1,1) for j in range(-self.vision_range,self.vision_range+1,1)]
        for x,y in visible_cells :
            if eye_sight == True : # Look with his eyes at his surroundings, in the env
                cell = self.env.map_area.get_cell((x,y),map_type)
            else : # check in the agent map what he can discover
                cell = self.map_area.get_cell((x,y),map_type)
            if cell != None :
                current_area_state.append(((x,y),cell))
        return current_area_state

    # Save insight last seen indices ect, ect, ... 
    def exploration_cell_score(self,pos,path_insight): # Idea of what will discover the agent by going in a designated cell 
        current_state = self.vision(pos,'last_seen',False)
        insight_cells = [x[0] for x in current_state]
        scores = [x[1] for x in current_state]
        new_cells = [(pos,scores[i]) for i,pos in enumerate(insight_cells) if pos not in path_insight]
        if len(new_cells) > 0 :
            return [x[0] for x in new_cells], sum([x[1] for x in new_cells]) # Return new insight cells from path and the associated exploration score 
        else :
            return [], 0 # Nothing new going there
        
    # Exploration is done using Rapidly-exploring Random Trees analog ideas 
    def explore(self,pos): # Explore the map to keep an updated map_area, and updated belief base will result in better decision making 
        
        start_node = Node(None, pos)
        parent_list=[start_node]
        # Loop until you reach max branch lenght
        for i in range(self.max_branch_lenght) :
            child_list = []
             # Generate DIVISION ITER number of children for each child node, 1 Node is the one with highest exploration score and the other are selected randomly 
            for parent_node in parent_list : 
                valid_neighbour_nodes = self.map_area.get_valid_neighbour_cells(parent_node.position,['O','F','H'])
                child_score_list = []
                # Get all possible children of this node, store its insight cells, it exploration score and it pos 
                for node_pos in valid_neighbour_nodes :
                    insight_cells, exploration_score = self.exploration_cell_score(node_pos,parent_node.in_sight)
                    child_score_list.append((node_pos,insight_cells,exploration_score))
                best_child = max(child_score_list, key=lambda x:x[2])
                new_node = Node(parent_node,best_child[0])
                new_node.in_sight = best_child[1] + parent_node.in_sight
                new_node.exploration_score = best_child[2] + parent_node.exploration_score
                child_list.append(new_node)
                child_score_list.remove(best_child)
                # Randomly add the other nodes 
                for i in range(DIVISION_ITER-1):
                    if len(child_score_list) > 0 :
                        np.random.shuffle(child_score_list)
                        new_node = Node(parent_node,child_score_list[0][0])
                        new_node.in_sight = child_score_list[0][1]
                        new_node.exploration_score = child_score_list[0][2] + parent_node.exploration_score
                        child_list.append(new_node)
                        child_score_list.pop(0)
                    else:
                        break  
            parent_list = child_list  

        # Get the branch with the best exploration score
        best_last_node = max(parent_list, key= lambda x:x.exploration_score) # At the end of the MAX_BRANCH_LENGHT ITER all last nodes are in the parent_list
        path = []
        current_node = best_last_node
        while current_node is not None :
            path.append(current_node.position)
            current_node = current_node.parent
        return path[::-1] # Return reversed path
    
    def take_water(self,pos):
        if self.map_area.get_cell(pos,'area') == 'B':
            self.has_water = True
            return True
        return False

    def extinguish_fire(self,pos):
        if self.has_water == False :
            return False
        self.map_area.update([pos,' '])
        self.env.extinguish_fire(pos)
        return True

    def euclidian_distance(self,pos,dest):
        return (pos[0]-dest[0])**2 + (pos[1]-dest[1])**2

    # Return the cell that minimise the distance between the current pos and the destination
    def minimise_dist_to_cell(self,pos,dest):
        valid_neighbour_cells = self.map_area.get_valid_neighbour_cells(pos,['O','F','H']) # Get all valid moove cell of the position
        if len(valid_neighbour_cells) > 0 : # At least a moove is possible 
            dist_list = [(i,self.euclidian_distance(elt,dest)) for i,elt in enumerate(valid_neighbour_cells)] # Create a list with tuple (i,dist)
            min_cell = min(dist_list, key=lambda x:x[1]) # Get the min dist and the associated indice
            return valid_neighbour_cells[min_cell[0]] 
        return None

    # A* path planning algorithm, the algorithm use f = g + h as a way to calculate a node cost. f is the node cost,
    # g the start node and the current node and h the estimated diatance between the end_node and the current node  
    def a_star_algorithm(self,pos,dest):
        # Create start/end_node and initialize both open and closed list
        start_node = Node(None, pos)
        end_node = Node(None, dest)
        open_list = [start_node]
        closed_list = []

        # Loop until you find the destination (early return) or if the destination can't be reached (len(open_list)=0)
        while len(open_list) > 0:
            
            # Get the current node, node with the lowest f score in the open_list meaning with the highest chance to get us to the goal 
            current_node = open_list[0]
            current_index = 0
            for index, node in enumerate(open_list):
                if node.f < current_node.f:
                    current_node = node
                    current_index = index
            open_list.pop(current_index) # Pop current node of the open list and add it to the closed list, he will be examined 
            closed_list.append(current_node)
            # Goal found, return the path 
            if current_node.position == end_node.position:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # Return reversed path

            # Generate children of this node
            children = []
            valid_neighbour_nodes = self.map_area.get_valid_neighbour_cells(current_node.position,['O','F','H'])
            for node_pos in valid_neighbour_nodes :
                new_node = Node(current_node,node_pos)
                children.append(new_node)
           
            # Calculate the f,g and h values of children if it has not been calculated yet
            for child in children :
                if child not in closed_list and not(child in open_list and child.g > max([node.g for node in open_list])):
                    child.g = current_node.g + 1
                    child.h = self.euclidian_distance(child.position,end_node.position)
                    child.f = child.g + child.h
                    open_list.append(child)
        
        # If no path can be found 
        return None

    # The agent will use the A* path planning algorithm to get to the destination, return a bool to indicate if the agent has mooved 
    # if no path can be found, he will go to the valid cell that minimise the distance between him and his destination
    def moove_to_cell(self,dest_pos):
        path = self.a_star_algorithm(self.pos,dest_pos)
        if path != None :
            new_cell_pos = path[1] # A* algorithm return a list with the path pos starting by the start pos
            self.update_pos(new_cell_pos)
            return True
        else :
            new_cell_pos = self.minimise_dist_to_cell(self.pos,dest_pos)
            if new_cell_pos != None : 
                self.update_pos(new_cell_pos)
                return True
        return False

    # Return True if the agent has ended is rest time, else False, each call update the rest_time
    def rest(self):
        if self.rest_time == 10 :
            self.rest_time = 0 
            self.energy = MAX_ENERGY
            return True
        self.rest_time += 1
        return False
    
    # The agent check his surroundings and update his map 
    def perception(self):
        current_area_state = self.vision(self.pos,'area')
        self.map_area.update(current_area_state)

    # Check beliefs to update the desire base 
    def update_desire_base(self):
        
        # Recalculate the desire base, priority values, easier to do it this way than update every desire in base 
        self.desire_base = []

        # If the agent energy is lower than the max dimension of the map, the agent will run out of energy, it wants to rest
        if self.energy < max(self.map_area.lenght,self.map_area.height) :  
            if self.pos != self.map_area.base_pos : # If the agent is not at the base, he will want to go to the base 
                sub_desire = Desire(1000,Intention('go_to',self.map_area.base_pos))
            else : 
                sub_desire = None # Else he rests to the base 
            desire = Desire(1000,Intention('rest',self.map_area.base_pos),sub_desire)
            self.desire_base.append(desire)

        # Look for fire in map area
        fire_pos_list = self.map_area.get_fire_cells()
        if len(fire_pos_list) > 0 :
            for fire_pos in fire_pos_list :
                fire_neighbour_pos = self.map_area.get_valid_neighbour_cells(fire_pos,['F','O','H'])
                if len(fire_neighbour_pos) > 0: # If the agent think he can have acces to the fire 
                    priority_value = 100 - np.sqrt(self.euclidian_distance(self.pos,fire_pos))
                    if self.has_water == False :
                        sub_sub_sub_desire = Desire(priority_value,Intention('go_to',self.map_area.base_pos))
                        sub_sub_desire = Desire(priority_value,Intention('take_water',self.map_area.base_pos),sub_sub_sub_desire)
                    else :
                        sub_sub_desire = None
                    if self.pos in fire_neighbour_pos and self.has_water == True :
                        sub_desire = None
                    else :
                        sub_desire = Desire(priority_value,Intention('go_to',fire_neighbour_pos[0]),sub_sub_desire)
                    desire = Desire(priority_value,Intention('extinguish_fire',fire_pos),sub_desire)
                    self.desire_base.append(desire)
        
        # Explore is the base agent desire but the one with the lowest priority value
        explore_path = self.explore(self.pos) # explore_path can be empty if the agent is trapped 
        priority_value = 0
        sub_desire = Desire(priority_value,Intention('explore_path',explore_path[1:]))
        desire = Desire(priority_value,Intention('explore',None),sub_desire)
        self.desire_base.append(desire)

    # Keep the current intention with a probability p or add in the intention list all intentions associated to the desire with highest priority and its subdesires 
    def intention_selection(self):

        if np.random.random() < KEEP_INTENTION_PROBA and len(self.intention_list) > 0 : # The current intention is kept
            return
        priority_desire = max([(desire,desire.priority_value) for desire in self.desire_base], key=lambda x:x[1])[0]
        self.intention_list = [priority_desire.intention]
        while priority_desire.sub_desire != None : # Add all subdesire intentions in the intention list, current intention will be the lowest desire intention 
            priority_desire = priority_desire.sub_desire
            self.intention_list.append(priority_desire.intention)

    # Select the action to do to fullfil the current intention 
    def plan_selection_execution(self):
        
        current_intention = self.intention_list[-1]
        
        if current_intention.action == 'go_to':
            location = current_intention.location
            self.moove_to_cell(location)
        elif current_intention.action == 'explore_path':
            location = current_intention.location[0]
            self.moove_to_cell(location)
        elif current_intention.action == 'explore':
            pass
        elif current_intention.action == 'take_water':
            self.take_water()
        elif current_intention.action == 'extinguish_fire':
            pass
        elif current_intention.action == 'rest':
            self.rest()


    # BDI agent reflexion process 
    def agent_thinking_process(self):
        # 1 - Update the agent belief base by perceving his environnement
        self.perception()

        # 2 - Update the desire base and prioritiy values using the belief base 
        self.update_desire_base()

        # 3 - Select the Intetion assiociated with the desire of highest priority value
        self.intention_selection()

        # 4 - Select the best plan to achieve the current intention taking into account the belief base, Execute it has well  
        self.plan_selection_execution()

        # 6 - Communicate the result of the action to the other agents (done updating shared object among agents) and update working variables 
        self.perception()
        self.energy -= 1

        # The agent dies if his energy reach 0 or if he his in a fire cell during more than 2 period of time 
        if self.energy == 0 :
            print('Agent died')

class Interface(Tk):
    def __init__(self,agent_map):
        Tk.__init__(self)
        self.map_area = agent_map
        self.width = LENGHT
        self.height = HEIGHT
        self.title = self.title("Fire Fighter Simulation")
        self.geometry = self.geometry("900x900")
        self.cell_width = int(900/self.width) -2 
        self.cell_height = int(900/self.height) -2 
        self.discovered_cells = []
        self.canva = Canvas(self,width=self.width*self.cell_width,height=self.height*self.cell_height,bg='slate grey')
        self.canva.pack()
        for i in range(1,self.width):
            self.canva.create_line(i*self.cell_width,0,i*self.cell_width,self.width*self.cell_width)
        for i in range(1,self.height):
            self.canva.create_line(0,i*self.cell_height,self.height*self.cell_height,i*self.cell_height)
        self.liste_case = [(i*self.cell_width+2,j*self.cell_height+2,(i+1)*self.cell_width-2,(j+1)*self.cell_height-2) for i in range(self.width) for j in range(self.height)]

    def display_cell(self, ligne, col, obj):
        case = ligne*self.width + col
        color_dic = {'F':'red','B':'gold','A':'cyan','H':'orange','O':'sienna', ' ' : 'white'}
        color = color_dic[obj]
        self.canva.create_rectangle(*self.liste_case[case],fill=color,outline=color)
        
    def display(self):
        for y in range(self.width):
            for x in range(self.height):
                if self.map_area.get_cell((x,y),'last_seen') == 0 and (x,y) not in self.discovered_cells: # If the cell has just been discovered 
                    self.discovered_cells.append((x,y))
                if (x,y) in self.discovered_cells :  
                    cell = self.map_area.get_cell((x,y))
                    self.display_cell(y, x, cell)

if __name__ == '__main__':
    # Create Environnement, agent shared map and agents
    np.random.seed(1)
    env = Environnement(HEIGHT,LENGHT) 
    agent_map = MapArea(LENGHT,HEIGHT)
    agent_list = [] 
    for id in range(NA):
        agent_list.append(Agent(env,id,agent_map))           

    # Create Display interface
    interface = Interface(agent_map)
    interface.display()
    interface.update_idletasks()
    interface.update()
    
    # Start Simulation 
    for iteration in range (10):
        np.random.shuffle(agent_list)
        for agent in agent_list:
            agent.agent_thinking_process()
            interface.display()
            interface.update()
        env.heat_point_ignition()
        env.spread_fire()
        agent_map.new_day()
    interface.display()
    interface.update()