from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
# os.environ["SDL_VIDEODRIVER"] = "dummy" #https://www.pygame.org/wiki/DummyVideoDriver

import numpy as np
import random
import pygame
import Box2D
from Box2D.b2 import world, circleShape, edgeShape, polygonShape, dynamicBody
from Box2D import b2DistanceJointDef, b2WeldJointDef, b2FrictionJointDef, b2FixtureDef, \
                  b2PolygonShape, b2ContactListener, b2Fixture, b2Vec2, b2RayCastOutput, b2RayCastInput, \
                  b2RayCastCallback
from scipy.misc import imresize
from PIL import Image
import cv2
from datetime import datetime
import math
from utils import bfs
import pylab as plt


LM_COLOR_LIST = [
    [45, 45, 180, 50], # blue
    [50, 150, 50, 50], # green
    [180, 45, 45, 50], # red
    [180, 180, 45, 50], # yellow
    [0.7 * 255, 0.2 * 255, 0.7 * 255, 255], # purple
    [0.2 * 255, 0.2 * 255, 0.2 * 255, 255], # black
]

ENT_COLOR_LIST = [
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [117, 216, 230, 255],
    [255, 153, 153, 255],
    # [0, 0, 255, 255],
    # [255, 255, 0, 255],
    # [0, 255, 255, 255],
    # [255, 0, 255, 255]
]

SIZE = [1 * 0.8, 1.5 * 0.8, 2 * 0.8]

DENSITY = [1, 2]

# STRENGTH = [300 / 100, 600 / 100, 150 / 100]
STRENGTH = [150 / 5, 300 / 5, 600 / 5, 450 / 4]

GOAL = [
# go to landmark
['LMA', 0, 0, +1], # 0:  agent 0 on landmark 0
['LMA', 1, 0, +1], # 1:  agent 1 on landmark 0
['LMO', 0, 0, +1], # 2:  object  on landmark 0
['LMA', 0, 1, +1], # 3:  agent 0 on landmark 1
['LMA', 1, 1, +1], # 4:  agent 1 on landmark 1
['LMO', 0, 1, +1], # 5:  object  on landmark 1
['LMA', 0, 2, +1], # 6:  agent 0 on landmark 2
['LMA', 1, 2, +1], # 7:  agent 1 on landmark 2
['LMO', 0, 2, +1], # 8:  object  on landmark 2
['LMA', 0, 3, +1], # 9:  agent 0 on landmark 3
['LMA', 1, 3, +1], # 10: agent 1 on landmark 3
['LMO', 0, 3, +1], # 11: object  on landmark 3

# negation
['LMA', 0, 0, -1], # 12: agent 0 not on landmark 0
['LMA', 1, 0, -1], # 13: agent 1 not on landmark 0
['LMO', 0, 0, -1], # 14: object  not on landmark 0
['LMA', 0, 1, -1], # 15: agent 0 not on landmark 1
['LMA', 1, 1, -1], # 16: agent 1 not on landmark 1
['LMO', 0, 1, -1], # 17: object  not on landmark 1
['LMA', 0, 2, -1], # 18: agent 0 not on landmark 2
['LMA', 1, 2, -1], # 19: agent 1 not on landmark 2
['LMO', 0, 2, -1], # 20: object  not on landmark 2
['LMA', 0, 3, -1], # 21: agent 0 not on landmark 3
['LMA', 1, 3, -1], # 22: agent 1 not on landmark 3
['LMO', 0, 3, -1], # 23: object  not on landmark 3

# touch entity
['TE', 0, 1, +1], # 24:  agent 0 touching agent 1
['TE', 0, 2, +1], # 25:  agent 0 touching object 0
['TE', 1, 2, +1], # 26:  agent 1 touching object 0

#negation
['TE', 0, 1, -1], # 27:  agent 0 not touching agent 1
['TE', 0, 2, -1], # 28:  agent 0 not touching object 0
['TE', 1, 2, -1], # 29:  agent 1 not touching object 0

# go to room
['RA', 0, 0, +1], # 30: agent 0 on room 0
['RA', 1, 0, +1], # 31: agent 1 on room 0
['RO', 0, 0, +1], # 32: object  on room 0
['RA', 0, 1, +1], # 33: agent 0 on room 1
['RA', 1, 1, +1], # 34: agent 1 on room 1
['RO', 0, 1, +1], # 35: object  on room 1
['RA', 0, 2, +1], # 36: agent 0 on room 2
['RA', 1, 2, +1], # 37: agent 1 on room 2
['RO', 0, 2, +1], # 38: object  on room 2
['RA', 0, 3, +1], # 39: agent 0 on room 3
['RA', 1, 3, +1], # 40: agent 1 on room 3
['RO', 0, 3, +1], # 41: object  on room 3

# negation
['RA', 0, 0, -1], # 42: agent 0 not on room 0
['RA', 1, 0, -1], # 43: agent 1 not on room 0
['RO', 0, 0, -1], # 44: object  not on room 0
['RA', 0, 1, -1], # 45: agent 0 not on room 1
['RA', 1, 1, -1], # 46: agent 1 not on room 1
['RO', 0, 1, -1], # 47: object  not on room 1
['RA', 0, 2, -1], # 48: agent 0 not on room 2
['RA', 1, 2, -1], # 49: agent 1 not on room 2
['RO', 0, 2, -1], # 50: object  not on room 2
['RA', 0, 3, -1], # 51: agent 0 not on room 3
['RA', 1, 3, -1], # 52: agent 1 not on room 3
['RO', 0, 3, -1], # 53: object  not on room 3

# grab object
['GE', 0, 2, +1], # 54: agent 0 grabbing object 0
['GE', 1, 2, +1], # 55: agent 1 grabbing object 0

# negation
['GE', 0, 2, -1], # 56: agent 0 Releasing object 0
['GE', 1, 2, -1], # 57: agent 1 Releasing object 0

['stop'],         # 58: stop

# protect
['LMOP', 0, 0, +1], # 59:  object on landmark 0 and away from other agents
['LMOP', 0, 1, +1], # 60:  object on landmark 1 and away from other agents
['LMOP', 0, 2, +1], # 61:  object on landmark 2 and away from other agents
['LMOP', 0, 3, +1], # 62:  object on landmark 3 and away from other agents

# no force (object)
['noforce'],      # 63: no force
]

POS = [
(16 - 8, 12 + 8), # 0
(16 - 8, 12 + 2), # 1
(16 - 8, 12 - 2), # 2
(16 - 8, 12 - 8), # 3

(16 + 8, 12 + 8), # 4
(16 + 8, 12 + 2), # 5
(16 + 8, 12 - 2), # 6
(16 + 8, 12 - 8), # 7

(16,     12 + 4), # 8
(16,     12 - 4), # 9
(16 - 2, 12 + 4), # 10
(16 - 2, 12 - 4), # 11
(16 + 2, 12 + 4), # 12
(16 + 2, 12 - 4), # 13
(16 - 2, 12 + 8), # 14
(16 - 2, 12 - 8), # 15
(16 + 2, 12 + 8), # 16
(16 + 2, 12 - 8),  # 17
(16,     12),     # 18
]

COST = [0.0, 0.1, 0.2]

WALL_SEGS = {'..........': None, 
             '.....=====': [5, 10], 
             '=====.....': [0, 5],
             '...=======': [2.5, 10], 
             '=======...': [0, 7.5],
             '==========': [0, 10]}

FRICTION = 0
LM_SIZE = 2.5

RELATION = [0, +1, -1, +2, -2]



def _my_draw_edge(edge, screen, body, fixture, color, PPM, SCREEN_HEIGHT):
    vertices = [(body.transform * v) * PPM for v in edge.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.line(screen, color, vertices[0], vertices[1], 5)
edgeShape.draw = _my_draw_edge


def _my_draw_circle(circle, screen, body, fixture, color, PPM, SCREEN_HEIGHT):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.circle(screen, color, [int(
        x) for x in position], int(circle.radius * PPM))
circleShape.draw = _my_draw_circle


def my_draw_polygon(polygon, screen, body, fixture, color, PPM, SCREEN_HEIGHT):
    # draw body (polygon)
    Vs = polygon.vertices
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)
    # draw eyes (circles)
    R = _get_dist(Vs[0], Vs[1]) * 0.15
    eye_pos = [((Vs[0][0] - Vs[1][0]) * 0.0 + Vs[1][0], (Vs[0][1] - Vs[1][1]) * 0.25 + Vs[1][1]),
               ((Vs[3][0] - Vs[2][0]) * 0.0 + Vs[2][0], (Vs[3][1] - Vs[2][1]) * 0.25 + Vs[2][1])]
    for pos in eye_pos:
        position = body.transform * pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(screen, [0, 0, 0, 255], [int(
            x) for x in position], int(R * PPM))

    R = _get_dist(Vs[0], Vs[1]) * 0.13
    for pos in eye_pos:
        position = body.transform * pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(screen, [255, 255, 255, 255], [int(
            x) for x in position], int(R * PPM))

    R = _get_dist(Vs[0], Vs[1]) * 0.1
    eye_pos = [((Vs[0][0] - Vs[1][0]) * 0.0 + Vs[1][0], (Vs[0][1] - Vs[1][1]) * 0.15 + Vs[1][1]),
               ((Vs[3][0] - Vs[2][0]) * 0.0 + Vs[2][0], (Vs[3][1] - Vs[2][1]) * 0.15 + Vs[2][1])]
    for pos in eye_pos:
        position = body.transform * pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(screen, [0, 0, 0, 255], [int(
            x) for x in position], int(R * PPM))


polygonShape.draw = my_draw_polygon


def _my_draw_patch(pos, screen, color, PPM, SCREEN_WIDTH, SCREEN_HEIGHT):
    position = [(pos[0] - 2.5) * PPM, (pos[1] + 2.5) * PPM]
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.rect(screen, color, [int(
        x) for x in position] + [int(5 * PPM), int(5 * PPM)])


def _get_world_pos(body):
    """get the position"""
    if isinstance(body.fixtures[0].shape, b2PolygonShape):
        center = (np.mean([v[0] for v in body.fixtures[0].shape.vertices]),
                  np.mean([v[1] for v in body.fixtures[0].shape.vertices]))
    else:
        center = body.fixtures[0].shape.pos
    position = body.transform * center
    return position


def _get_world_vel(body):
    """get the velocity"""
    # vel = body.transform * body.linearVelocity
    # print(body.linearVelocity, vel)
    vel = body.linearVelocity
    return (vel[0], vel[1])


def _get_pos(body):
    """get the position"""
    position = body.transform * body.fixtures[0].shape.pos
    return position


def _get_body_bound(body):
    """get the boundary of a cicle"""
    position = body.transform * body.fixtures[0].shape.pos
    radius = body.fixtures[0].shape.radius
    return (position[0] - radius, position[1] - radius,
            position[0] + radius, position[1] + radius)


def _get_door(body):
    vertices1 = [(body.transform * v)  \
                    for v in body.fixtures[0].shape.vertices]
    vertices2 = [(body.transform * v) \
                    for v in body.fixtures[-1].shape.vertices]

    return [vertices1[0], vertices2[-1]]


"""TODO: currently only consider a rectangle w/o rotation"""
def _get_room_bound(body):
    """get the boundary of a room (upper-left corner + bottom-right corner)"""
    x_list, y_list = [], []
    for fixture in body.fixtures:
        vertices = [(body.transform * v) \
                    for v in fixture.shape.vertices]
        x_list += [v[0] for v in vertices]
        y_list += [v[1] for v in vertices]
    min_x, min_y = min(x_list), min(y_list)
    max_x, max_y = max(x_list), max(y_list)
    return (min_x, min_y, max_x, max_y)


def _in_room(body, room):
    body_bound = _get_body_bound(body)
    min_x, min_y, max_x, max_y = _get_room_bound(room)
    return body_bound[2] >= min_x and body_bound[3] >= min_y and \
           body_bound[0] <= max_x and body_bound[1] <= max_y

def _point_in_room(point, room):
    min_x, min_y, max_x, max_y = _get_room_bound(room)
    return point[0] >= min_x and point[1] >= min_y and \
           point[0] <= max_x and point[1] <= max_y


def _get_obs(screen, SCREEN_WIDTH, SCREEN_HEIGHT):
    string_image = pygame.image.tostring(screen, 'RGB')
    temp_surf = pygame.image.fromstring(string_image, 
                    (SCREEN_WIDTH, SCREEN_HEIGHT),'RGB')
    return(pygame.surfarray.array3d(temp_surf))


def _get_dist(pos1, pos2):
    """get the distance between two 2D positions"""
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def _get_dist_lm(pos, lm):
    if pos[0] > lm[0] - 1.5 and pos[0] < lm[0] + 1.5:
        if pos[1] > lm[1] - 1.5 and pos[1] < lm[1] + 1.5:
            return 0
        else:
            return min(abs(pos[1] - (lm[1] - 2.5)), abs(pos[1] - (lm[1] + 2.5)))
    elif pos[1] > lm[1] - 1.5 and pos[1] < lm[1] + 1.5:
         return min(abs(pos[0] - (lm[0] - 2.5)), abs(pos[0] - (lm[0] + 2.5)))
    else:
        return _get_dist(pos, (lm[0] - 2.5, lm[1] - 2.5))


def _norm(vec):
    """L2-norm of a vector"""
    return (vec[0] ** 2 + vec[1] ** 2) ** 0.5


def _get_angle(vec1, vec2):
    """angle between 2 vectors"""
    return np.arccos(((vec1[0]*vec2[0]) + (vec1[1]*vec2[1])) / (_norm(vec1)*_norm(vec2)))


def _get_segment_intersection(pt1, pt2, ptA, ptB):
    # https://www.cs.hmc.edu/ACM/lectures/intersections.html
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
    returns a tuple: (xi, yi, valid, r, s), where
    (xi, yi) is the intersection
    r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
    s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
        valid == 0 if there are 0 or inf. intersections (invalid)
        valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001
    # the first line is pt1 + r*(pt2-pt1)
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1
    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;
    DET = (-dx1 * dy + dy1 * dx)
    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)
    # now, the determinant should be OK
    DETinv = 1.0/DET
    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x-x1) + dx * (y-y1))
    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))
    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0
    return (xi, yi, 1, r, s)


def _get_point_dist_from_seg(p1,p2,pnt):
    p1, p2, pnt = np.array(p1), np.array(p2), np.array(pnt)
    return np.abs(np.cross(p2-p1, p1-pnt) / _norm(p2-p1))


def _not_door(x,y,doors_pos):
    if x == 16:
        if y > doors_pos[0][1] or y < doors_pos[2][1]:
            return False
    if y == 12:
        if x < doors_pos[1][0] or x > doors_pos[3][0]:
            return False
    return True


def _on_room_boundary(grid_cell,room):
    min_x, min_y, max_x, max_y = _get_room_bound(room)
    return grid_cell[0] == 0 or grid_cell[1] == 0 or \
           grid_cell[1] == max_x-min_x-1 or grid_cell[1] == max_y-min_y-1



# Belief can represent an agent's belief over:
# the env state (partially observable) = location & orientation for any entity
# other agent observations = location & orientation for each entity
# goals of others = probability of each goal in GOAL
class Belief:
    def __init__(self, n_space, world_center, is_static=True):
        """ entities: agent / item / goal
        :param n_space: grid size / n_goals
        :param world_center: (16,12)
        """
        self.grid_location_belief = (1 / (n_space[0]*n_space[1])) * np.ones((n_space[0],n_space[1])) #uniform init
        self.location = None
        self.last_believed_grid_location = None
        self.last_believed_world_location = None
        self.angle = 0.0
        self.linear_velocity = (0.0, 0.0)
        self.angular_velocity = 0.0
        self.n_grid_space = n_space
        self.world_center = world_center
        self.room_limitation = self.get_room_limitation()
        self.is_static = is_static

    def get_room_limitation(self):
        room_limitation = np.ones((self.n_grid_space[0],self.n_grid_space[1]))
        #wall boundaries
        room_limitation[0,:] = 0
        room_limitation[-1,:] = 0
        room_limitation[:,0] = 0
        room_limitation[:,-1] = 0
        #maze wall boundaries
        room_limitation[9,5:15] = 0
        room_limitation[10,5:15] = 0
        room_limitation[3:15,9] = 0
        room_limitation[3:15,10] = 0
        return room_limitation

    #different update function for env and goals? - for now assume goals are known.
    #TODO use map of what I can see, estimate other's gaze.
    def update(self, field_of_view, observations=None, r=None, c=None):
        if observations is not None:
            #exact locations and angles of observations + one-hot grid
            self.location = observations[0]
            self.last_believed_grid_location = [r,c]
            self.last_believed_world_location = observations[0]
            self.angle = observations[1]
            self.linear_velocity = observations[2]
            self.angular_velocity = observations[3]
            self.grid_location_belief = np.zeros((self.n_grid_space[0],self.n_grid_space[1]))
            self.grid_location_belief[r,c] = 1 #(r*self.n_grid_space[0])+c
        else:
            self.location = None
            #item is believed to be seen when not
            if self.is_static and self.last_believed_grid_location is not None and\
                    field_of_view[self.last_believed_grid_location[0],self.last_believed_grid_location[1]] == 1:
                self.last_believed_grid_location = None
                self.last_believed_world_location = None
            #TODO other param estimations

            #entity is out of agent's field of view and not neighboring FOV
            out_of_FOV = [np.sum(field_of_view[max(r-1,0):min(r+2,self.n_grid_space[0]),
                                  max(c-1,0):min(c+2,self.n_grid_space[1])]) == 0 \
                for r in range(self.n_grid_space[0]) for c in range(self.n_grid_space[1])]
            out_of_FOV = np.array(out_of_FOV).reshape(self.n_grid_space[0],self.n_grid_space[1])
            possible_areas = out_of_FOV * self.room_limitation
            if self.last_believed_grid_location and not self.is_static:
                x, y = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
                d = np.sqrt(x*x+y*y)
                sigma, mu = 1.0, 0.0
                g = np.exp(-((d-mu)**2 / ( 2.0 * sigma**2 )))
                high_prob_area = np.zeros_like(self.grid_location_belief)
                high_prob_area[max(self.last_believed_grid_location[0]-2,0):min(self.last_believed_grid_location[0]+3,self.n_grid_space[0]),
                               max(self.last_believed_grid_location[1]-2,0):min(self.last_believed_grid_location[1]+3,self.n_grid_space[1])] = \
                    g[max(0,2-self.last_believed_grid_location[0]):min(5,2+self.n_grid_space[0]-self.last_believed_grid_location[0]),
                      max(0,2-self.last_believed_grid_location[1]):min(5,2+self.n_grid_space[1]-self.last_believed_grid_location[1])]
            else: #uniform
                high_prob_area = (1 / np.sum(possible_areas)) * possible_areas
            new_belief = (self.grid_location_belief + high_prob_area) * possible_areas
            self.grid_location_belief = new_belief / np.sum(new_belief)


    def sample_state(self):
        #if know exact entity state return it.
        #else sample location from grid and then uniformly within a cell
        if self.location: #know exact by obs
            location_estimation = self.location
            angle_estimation = self.angle
        #don't know exact and not static. or - don't know exact and static and last_believed_location is None
        elif self.location is None and (not self.is_static or self.last_believed_grid_location is None):
            grid_cells = [(r,c) for r in range(self.n_grid_space[0]) for c in range(self.n_grid_space[1])]
            grid_cell_idx = np.random.choice(np.arange(len(grid_cells)), 1, p=self.grid_location_belief.flatten())[0] #(r,c)
            grid_cell_estimation = grid_cells[grid_cell_idx]
            col_baseline = self.world_center[0] - (self.n_grid_space[0] // 2)
            row_baseline = self.world_center[1] + (self.n_grid_space[1] // 2)
            location_estimation = (np.random.uniform(col_baseline + grid_cell_estimation[1],
                                                     col_baseline + grid_cell_estimation[1] + 1),
                                   np.random.uniform(row_baseline - grid_cell_estimation[0],
                                                     row_baseline - (grid_cell_estimation[0] + 1))) #x,y
            self.last_believed_grid_location = grid_cell_estimation
            self.last_believed_world_location = location_estimation
            #positive - left turn (0 to 2*pi), negative - right turn
            #TODO use past observations
            angle_estimation = math.radians(np.random.uniform(-360,360))
        else: #static and have last_believed_location
            location_estimation = self.last_believed_world_location
            angle_estimation = self.angle
        #TODO velocity sampling
        return location_estimation, angle_estimation, self.linear_velocity, self.angular_velocity


class RayCastAnyCallback(b2RayCastCallback):
    """This callback finds any hit"""
    def __repr__(self):
        return 'Any hit'

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False
        self.point = None
        self.normal = None

    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        return 0.0

class MyContactListener(b2ContactListener):
    def agent_item_update(self,agent,item,manifold):
        agent_id = agent[0][1]
        if item[1] == agent_id: #item attached to agent
            agent[1][agent_id][item[0]] = [manifold, True, False] # [contact,is item attached to agent, does contact involve attached item]
        elif item[1] is None: # item not attached
            agent[1][agent_id][item[0]] = [manifold, False, False]
        else: #attached to other agent
            other_agent_id = 1 - agent_id
            agent[1][agent_id][item[0]] = [manifold, False, True] # is touching item that's attached to other agent
            agent[1][other_agent_id][agent[0]] = [manifold, False, True] # is agent touching through attached item

    def agent_item_remove(self,agent,item):
        agent_id = agent[0][1]
        if item[1] == agent_id: #item attached to agent
            return
        elif item[1] is None: # item not attached
            agent[1][agent_id].pop(item[0])
        else: #attached to other agent
            other_agent_id = 1 - agent_id
            agent[1][agent_id].pop(item[0])
            agent[1][other_agent_id].pop(agent[0])

    def agent_agent_update(self,agent1,agent2,manifold):
        #update both agents
        agent1[1][agent1[0][1]][agent2[0]] = [manifold, False, False]
        agent2[1][agent2[0][1]][agent1[0]] = [manifold, False, False]

    def agent_agent_remove(self,agent1,agent2):
        #update both agents
        agent1[1][agent1[0][1]].pop(agent2[0])
        agent2[1][agent2[0][1]].pop(agent1[0])

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData
        if bodyA is not None and bodyB is not None: #both not None means collision between agents and/or items
            if bodyA[0][0] == 'agent' and bodyB[0][0] == 'item':
                self.agent_item_update(bodyA, bodyB, contact.manifold)
            elif bodyA[0][0] == 'item' and bodyB[0][0] == 'agent':
                self.agent_item_update(bodyB, bodyA, contact.manifold)
            elif bodyA[0][0] == 'agent' and bodyB[0][0] == 'agent':
                self.agent_agent_update(bodyA, bodyB, contact.manifold)

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData
        if bodyA is not None and bodyB is not None:
            if bodyA[0][0] == 'agent' and bodyB[0][0] == 'item':
                self.agent_item_remove(bodyA, bodyB)
            elif bodyA[0][0] == 'item' and bodyB[0][0] == 'agent':
                self.agent_item_remove(bodyB, bodyA)
            elif bodyA[0][0] == 'agent' and bodyB[0][0] == 'agent':
                self.agent_agent_remove(bodyA, bodyB)

class Maze_v1:
    """two agents move one item to certain position"""
    def __init__(self,
                 action_type,
                 maze_sampler, 
                 goals,
                 strengths,
                 sizes,
                 densities,
                 init_positions,
                 action_space_types,
                 costs,
                 visibility,
                 num_agents=2,
                 num_items=2,
                 PPM=20.0, 
                 TARGET_FPS=60,
                 SCREEN_WIDTH=640,
                 SCREEN_HEIGHT=480,
                 TIME_STEP=5,
                 enable_renderer=True,
                 random_colors=False,
                 random_colors_agents=False):
        self.action_type = action_type
        self.maze_sampler = maze_sampler
        self.goals = list(goals)
        self.strengths = list(strengths)
        self.sizes = list(sizes)
        self.densities = list(densities)
        self.init_positions = list(init_positions)
        self.action_space_types = action_space_types
        self.costs = list(costs)
        self.visibility = list(visibility)
        self.PPM = PPM
        self.TARGET_FPS = TARGET_FPS
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.TIME_STEP = 1.0 / TARGET_FPS #* TIME_STEP
        self.NUM_STEPS_PER_TICK = TIME_STEP
        self.enable_renderer = enable_renderer
        self.obs_dim = (3, 86, 86)
        self.state_dim = 28
        self.num_agents = num_agents
        self.num_items = num_items
        self.random_colors = random_colors
        self.random_colors_agents = random_colors_agents

        self.action_space = ['turnleft', 'turnright', 'up', 'down', 'left', 'right', 'upleft', 'upright', 'downleft', 'downright', 'stop', 'noforce', 'attach', 'detach']
        # self.action_space = ['up', 'down', 'left', 'right', 'stop', 'noforce', 'attach', 'detach']
        self.action_size = len(self.action_space)
        if self.enable_renderer:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Maze_v1')
            self.clock = pygame.time.Clock()
        self.room_dim = (20, 20)
        self._max_dist = _norm((20, 20))
        self.door_length = 0
        self.clip_interval = None
        self.touch_sensor = [{} for _ in range(num_agents)]

        random.seed(1)


    def seed(self):
        random.seed(datetime.now())


    def set_clip_interval(self, clip_interval):
        self.clip_interval = list(clip_interval)


    def _get_room_id(self, pos):
        if pos[0] < 16:
            if pos[1] > 12:
                return 0
            else:
                return 3
        else:
            if pos[1] > 12:
                return 1
            else:
                return 2


    def get_action_id(self, action):
        for action_id, a in enumerate(self.action_space):
            if action == a:
                return action_id
        return 0


    def setup(self, env_id, agent_id, max_episode_length, record_path=None):
        """setup a new espisode"""
        self.env_id = env_id
        self.planning_agent = agent_id
        self.max_episode_length = max_episode_length
        self.world = world(contactListener=MyContactListener(), gravity=(0, 0), doSleep=True)
        self.room = self.world.CreateBody(position=(16, 12))
        self.room.CreateEdgeChain(
            [
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2 - self.door_length)]
        )
        self.connected = [[None] * 4 for _ in range(4)]
        self.connected[0][1] = self.connected[1][0] = 0
        self.connected[0][3] = self.connected[3][0] = 1
        self.connected[2][3] = self.connected[3][2] = 2
        self.connected[1][2] = self.connected[2][1] = 3

        self.doors_pos = [(16, self.room_dim[1] / 4 + 12),
                          (-self.room_dim[0] / 4 + 16, 12),
                          (16, -self.room_dim[1] / 4 + 12),
                          (self.room_dim[0] / 4 + 16, 12)]
        self.doors_size = [10] * 4

        # build maze 
        env_def = self.maze_sampler.get_env_def(env_id)
        for wall_id, wall in enumerate(env_def['maze_def']):
            seg = WALL_SEGS[wall]
            # print(wall_id, wall, seg, env_id, env_def['maze_def'])
            if seg:
                self.doors_size[wall_id] = 10 - abs(seg[0] - seg[1])
                if wall_id == 0:
                    if env_id == 16:
                        seg[1] -= 3
                    self.room.CreateEdgeChain([(0, seg[0]), (0, seg[1])])
                    if abs(seg[0] - seg[1]) > self.room_dim[1] * 0.5 - 1e-6:
                        self.connected[0][1] = None
                        self.connected[1][0] = None
                    else:
                        self.doors_pos[0] = (16,  12 + (self.room_dim[1] / 2 + seg[1]) / 2)
                elif wall_id == 1:
                    self.room.CreateEdgeChain([(-seg[0], 0), (-seg[1], 0)])
                    if abs(seg[0] - seg[1]) > self.room_dim[1] * 0.5 - 1e-6:
                        self.connected[0][3] = None
                        self.connected[3][0] = None
                    else:
                        self.doors_pos[1] = (16 + (-self.room_dim[0] / 2 - seg[1]) / 2, 12)

                elif wall_id == 2:
                    self.room.CreateEdgeChain([(0, -seg[0]), (0, -seg[1])])
                    if abs(seg[0] - seg[1]) > self.room_dim[1] * 0.5 - 1e-6:
                        self.connected[2][3] = None
                        self.connected[3][2] = None
                    else:
                        self.doors_pos[2] = (16, 12 + (-self.room_dim[1] / 2 - seg[1]) / 2)
                elif wall_id == 3:
                    self.room.CreateEdgeChain([(seg[0], 0), (seg[1], 0)])
                    if abs(seg[0] - seg[1]) > self.room_dim[1] * 0.5 - 1e-6:
                        self.connected[1][2] = None
                        self.connected[2][1] = None
                    else:
                        self.doors_pos[3] = (16 + (self.room_dim[0] / 2 + seg[1]) / 2, 12)

        self.agents, self.items = [], []
        self.init_agents_pos = [None] * self.num_agents
        self.init_items_pos = [None] * self.num_items
        self.trajectories = [None] * (self.num_agents + self.num_items)
        # add agents
        for agent_id in range(self.num_agents):
            x, y = POS[self.init_positions[agent_id]][0], POS[self.init_positions[agent_id]][1]
            self.init_agents_pos[agent_id] = (x, y)
            R = SIZE[self.sizes[agent_id]]
            body = self.world.CreateDynamicBody(
                position=(x, y),
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(vertices=[(-1.5 * R, -R),
                                                   (-0.5 * R, R), 
                                                   (0.5 * R, R), 
                                                   (1.5 * R, -R)]),
                    density=DENSITY[self.densities[agent_id]],
                    friction=FRICTION
                )
            )
            body.userData = [('agent',agent_id), self.touch_sensor, None] #3rd entry for attached item_id (None if not attached)
            body.field_of_view = np.zeros(self.room_dim)
            body.last_observations = {}
            body.beliefs = [Belief(n_space=self.room_dim, world_center=self.room.position, is_static=False) \
                                        for _ in range(self.num_agents)] + \
                           [Belief(n_space=self.room_dim, world_center=self.room.position) \
                                        for _ in range(self.num_items)]
            self.agents.append(body)
            self.trajectories[agent_id] = [(x, y, 0, 0)]

        # add items
        ITEM_BASE = self.num_agents
        for item_id in range(self.num_items):
            index = ITEM_BASE + item_id
            x, y = POS[self.init_positions[index]][0], POS[self.init_positions[index]][1]
            self.init_items_pos = [(x, y)]
            body = self.world.CreateDynamicBody(position=(x, y))
            body.CreateCircleFixture(radius=SIZE[self.sizes[index]], density=DENSITY[self.densities[index]], friction=FRICTION, restitution=0)
            body.userData = [('item', item_id), None] #2nd entry for attached agent_id (None if not attached)
            self.items.append(body)
            self.trajectories[index] = [(x, y, 0, 0)]

        # friction
        self.groundBody = self.world.CreateStaticBody(
            position=(16, 12),
            shapes=polygonShape(box=(self.room_dim[0],self.room_dim[1])),
        )
        for body in self.agents + self.items:
            dfn = b2FrictionJointDef(
                            bodyA=body,
                            bodyB=self.groundBody,
                            localAnchorA=(0, 0),
                            localAnchorB=(0, 0),
                            maxForce=100,
                            maxTorque=0
                        )
            self.world.CreateJoint(dfn)

        self.steps = 0
        self.running = False
        self.video = None
        if record_path:
            self.video = cv2.VideoWriter(
                record_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                30, 
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.attached = [None] * self.num_agents
        self.once_attached = [False] * self.num_agents

        self.landmark_centers = [
        (16 - self.room_dim[0] / 2 + 2.5, 12 + self.room_dim[1] / 2 - 2.5),
        (16 + self.room_dim[0] / 2 - 2.5, 12 + self.room_dim[1] / 2 - 2.5),
        (16 + self.room_dim[0] / 2 - 2.5, 12 - self.room_dim[1] / 2 + 2.5),
        (16 - self.room_dim[0] / 2 + 2.5, 12 - self.room_dim[1] / 2 + 2.5),
        ]
        self.landmark_corners = [
        (16 - self.room_dim[0] / 2 + 1.25, 12 + self.room_dim[1] / 2 - 1.25),
        (16 + self.room_dim[0] / 2 - 1.25, 12 + self.room_dim[1] / 2 - 1.25),
        (16 + self.room_dim[0] / 2 - 1.25, 12 - self.room_dim[1] / 2 + 1.25),
        (16 - self.room_dim[0] / 2 + 1.25, 12 - self.room_dim[1] / 2 + 1.25),
        ]

        if not self.random_colors:
            if not self.random_colors_agents:
                self.colors = {
                    0: (0, 0, 0, 255), # ground body
                    1: ENT_COLOR_LIST[0], # agent 1
                    2: ENT_COLOR_LIST[1], # agent 2
                    3: ENT_COLOR_LIST[2], # item 1
                    4: ENT_COLOR_LIST[3], # item 2
                }
            else:
                order = [1, 2]
                random.shuffle(order)
                self.colors = {
                    0: (0, 0, 0, 255), # ground body
                    order[0]: ENT_COLOR_LIST[0], # agent 1
                    order[1]: ENT_COLOR_LIST[1], # agent 2
                    3: ENT_COLOR_LIST[2], # item 1
                    4: ENT_COLOR_LIST[3], # item 2
                }
        else:
            order = [1, 2, 3]
            random.shuffle(order)
            self.colors = {
                0: (0, 0, 0, 255), # ground body
                order[0]: ENT_COLOR_LIST[0],
                order[1]: ENT_COLOR_LIST[1],
                order[2]: ENT_COLOR_LIST[2],
                order[3]: ENT_COLOR_LIST[3],
            }

        self.path, self.doors = {}, {}
        for room_id1 in range(4):
            for room_id2 in range(4):
                if room_id1 != room_id2:
                    self.path[(room_id1, room_id2)], self.doors[(room_id1, room_id2)] = bfs(self.connected, room_id1, room_id2)


    def start(self):
        """start the episode"""
        self.running = True
        self.repeat_actions = [None] * 2
        self.world.Step(1.0 / self.TARGET_FPS, 10, 10)
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]
        self.items_pos = [_get_world_pos(item) for item in self.items]
        self.agents_vel = [_get_world_vel(agent) for agent in self.agents]
        self.items_vel = [_get_world_vel(item) for item in self.items]
        self.actions = [None] * 2
        self.vels = [[_get_world_vel(body)] for body in self.agents]


    def reset_history(self):
        self.repeat_actions = [None] * 2
        self.actions = [None] * 2
        self.vels = [[_get_world_vel(body)] for body in self.agents]
        self.trajectories = [[(pos[0], pos[1])] for pos in self.agents_pos + self.items_pos]
        self.once_attached = [a for a in self.attached]


    def check_attachable(self, agent_id):
        """check if agent can grab an item"""
        return self.action_space_types[agent_id] < 2 and \
               self.attached[agent_id] is None and \
               _get_dist(self.agents_pos[agent_id], self.items_pos[0]) < SIZE[self.sizes[agent_id]] + SIZE[self.sizes[2]] + 0.08


    def get_action_space(self, agent_id):
        num_actions = len(self.action_space) - self.action_space_types[agent_id]
        if not self.attached[agent_id]:
            num_actions = min(num_actions, len(self.action_space) - 1)
        if GOAL[self.goals[agent_id]][0] in ['RO', 'GE', 'TE']:
            if GOAL[self.goals[agent_id]][0] == 'RO':
                item_room = self._get_room_id(self.items_pos[0])
                goal_room = GOAL[self.goals[agent_id]][2]
                if item_room != goal_room and \
                    self._is_blocked(self.doors[(item_room, goal_room)], SIZE[self.sizes[2]]):
                    num_actions = min(num_actions, len(self.action_space) - 1)
        else:
            num_actions = min(num_actions, len(self.action_space) - 2)
        return self.action_space[:num_actions]


    def send_action(self, agent_id, action):
        """send action to an agent"""
        if action is None: return
        self.repeat_actions[agent_id] = action
        if action not in (self.action_space[:-self.action_space_types[agent_id]] if self.action_space_types[agent_id] \
                            else self.action_space):
            return
        if action == 'attach': 
            head_mid_point = self.agents[agent_id].GetWorldPoint(localPoint=(0.0, SIZE[self.sizes[agent_id]]))
            tail_mid_point = self.agents[agent_id].GetWorldPoint(localPoint=(0.0, -SIZE[self.sizes[agent_id]]))
            right_mid_point = self.agents[agent_id].GetWorldPoint(localPoint=(SIZE[self.sizes[agent_id]], 0.0))
            left_mid_point = self.agents[agent_id].GetWorldPoint(localPoint=(-SIZE[self.sizes[agent_id]], 0.0))
            all_agent_pos = [head_mid_point, tail_mid_point, right_mid_point, left_mid_point]
            print(all_agent_pos)
            print(self.items_pos)
            if self.attached[agent_id] is None:
                min_dist = 1e6
                selected_item_id = None
                selected_agent_anchor_idx = None
                for item_id in range(self.num_items):
                    all_dist = [_get_dist(pos, self.items_pos[item_id]) for pos in all_agent_pos]
                    cur_dist = min(all_dist)
                    agent_anchor_idx = all_agent_pos.index(all_agent_pos[all_dist.index(cur_dist)])
                    print(cur_dist, SIZE[self.sizes[self.num_agents + item_id]] + 0.2)
                    if cur_dist < SIZE[self.sizes[self.num_agents + item_id]] + 0.5 and cur_dist < min_dist:
                        min_dist = cur_dist
                        selected_item_id = item_id
                        selected_agent_anchor_idx = agent_anchor_idx
                if selected_item_id == None:
                    print('no attach')
                    return
                f = {'spring': 0.3, 'rope': 0.1, 'rod': 100}
                d = {'spring': 0, 'rope': 0, 'rod': 0.5}
                print(selected_item_id)
                agent_size, object_size = SIZE[self.sizes[agent_id]], SIZE[self.sizes[self.num_agents + selected_item_id]]
                agent_anchors = [(0, agent_size + object_size), (0, -agent_size - object_size),
                                 (agent_size + object_size, 0), (-agent_size - object_size, 0)]
                dfn = b2WeldJointDef(
                        # frequencyHz=f['rod'],
                        # dampingRatio=d['rod'],
                        bodyA=self.agents[agent_id],
                        bodyB=self.items[selected_item_id],
                        localAnchorA=agent_anchors[selected_agent_anchor_idx],
                        localAnchorB=(0, 0)
                    )
                self.attached[agent_id] = self.world.CreateJoint(dfn)
                self.once_attached[agent_id] = True
                self.agents[agent_id].userData[2] = selected_item_id
                if ('item',selected_item_id) in self.touch_sensor[agent_id]:
                    self.touch_sensor[agent_id][('item',selected_item_id)][1] = True
                self.items[selected_item_id].userData[1] = agent_id
            return
        elif action == 'detach':
            if self.attached[agent_id] is not None:
                self.world.DestroyJoint(self.attached[agent_id])
                self.attached[agent_id] = None
                item_id = self.agents[agent_id].userData[2]
                self.items[item_id].userData[1] = None
                if ('item',item_id) in self.touch_sensor[agent_id]:
                    self.touch_sensor[agent_id][('item',item_id)][1] = False
                self.agents[agent_id].userData[2] = None
            return

        # print(agent_id, action)
        # if self.actions[agent_id] is None:
        #     self.actions[agent_id] = [action]
        # else:
        #     self.actions[agent_id].append(action)
        fx, fy = 0.0, 0.0
        df = STRENGTH[self.strengths[agent_id]] * self.NUM_STEPS_PER_TICK
        if action.startswith('turn'):
            if action == 'turnleft':
                self.agents[agent_id].ApplyTorque(df * 1.0, True)
            else:
                self.agents[agent_id].ApplyTorque(-df * 1.0, True)
            return        

        if action == 'up':
            fy += df
        elif action == 'down':
            fy -= df
        elif action == 'left':
            fx -= df
        elif action == 'right':
            fx += df
        elif action == 'upleft':
            fx -= df * 0.707
            fy += df * 0.707
        elif action == 'upright':
            fx += df * 0.707
            fy += df * 0.707
        elif action == 'downleft':
            fx -= df * 0.707
            fy -= df * 0.707
        elif action == 'downright':
            fx += df * 0.707
            fy -= df * 0.707
        elif action == 'stop':
            if self.strengths[agent_id] != 3:
                self.agents[agent_id].linearVelocity.x = 0
                self.agents[agent_id].linearVelocity.y = 0
            return
        elif action == 'noforce':
            return
        else:
            print('ERROR: invalid action!')
        if action == 'stop': print(fx, fy)
        f = self.agents[agent_id].GetWorldVector(localVector=(fx, fy))
        p = self.agents[agent_id].GetWorldPoint(localPoint=(0.0, -SIZE[self.sizes[agent_id]] / 12.0))
        self.agents[agent_id].ApplyForce(f, p, True)
        # self.agents[agent_id].ApplyLinearImpulse(f, p, True)


    def render_all(self):
        self.update_field_of_view()
        self.update_observations()
        #sample belief
        belief_states = [self.get_belief_state(0), self.get_belief_state(1)]
        # print(belief_states)
        #make sure belief is valid
        belief_states[0] = np.array(self.valid_belief(0, belief_states[0]))
        belief_states[1] = np.array(self.valid_belief(1, belief_states[1]))
        #update beliefs over locations after validity check step (might shift due to physics engine).
        for agent_id in range(self.num_agents):
            self.agents[agent_id].beliefs[1-agent_id].last_believed_world_location = \
                belief_states[agent_id][2:4]
            self.agents[agent_id].beliefs[1-agent_id].last_believed_grid_location = \
                self.world_point_to_grid_cell(belief_states[agent_id][2],
                                              belief_states[agent_id][3])
            for item_id in range(self.num_items):
                self.agents[agent_id].beliefs[2+item_id].last_believed_world_location = \
                    belief_states[agent_id][12+(2*item_id):14+(2*item_id)]
                self.agents[agent_id].beliefs[2+item_id].last_believed_grid_location = \
                    self.world_point_to_grid_cell(belief_states[agent_id][12+(2*item_id)],
                                                  belief_states[agent_id][12+(2*item_id)+1])
        #display sampled beliefs
        self._display_imagined(0, belief_states[0])
        self._display_imagined(1, belief_states[1])

        #display FOV
        self.render()


    def step(self):
        """apply one step and update the environment"""
        # print("agent positions", self.agents_pos)
        self.steps += 1
        self.world.Step(self.TIME_STEP, 10, 10)
        # self.world.ClearForces()

        # print('vel:', round(self.agents[0].angularVelocity,2), round(self.agents[1].angularVelocity,2))
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]
        self.agents_vel = [_get_world_vel(agent) for agent in self.agents]
        print(self.agents_vel)
        for agent_id, (agent_pos, agent_vel) in enumerate(zip(self.agents_pos, self.agents_vel)):
            self.trajectories[agent_id].append((agent_pos[0], agent_pos[1], 
                                                agent_vel[0], agent_vel[1]))
            self.vels[agent_id].append((agent_vel[0], agent_vel[1]))
            if self.actions[agent_id] is None:
                self.actions[agent_id] = [self.repeat_actions[agent_id]]
            else:
                self.actions[agent_id].append(self.repeat_actions[agent_id])
        self.items_pos = [_get_world_pos(item) for item in self.items]
        self.items_vel = [_get_world_vel(item) for item in self.items]
        for item_id, (item_pos, item_vel) in enumerate(zip(self.items_pos, self.items_vel)):
            self.trajectories[item_id + 2].append((item_pos[0], item_pos[1], 
                                                   item_vel[0], item_vel[1]))
        agent1beliefs_item1 = [] #TODO not general enough
        for t in range(self.NUM_STEPS_PER_TICK - 1):
            if self.video and \
            (self.clip_interval is None \
                or self.steps > self.clip_interval[0] and self.steps <= self.clip_interval[1]):
                self.render_all()
            for agent_id in range(2): #real action
                # if self.repeat_actions[agent_id] == 'stop' or t == self.NUM_STEPS_PER_TICK - 1:
                if self.repeat_actions[agent_id] not in ['attach', 'detach']:
                    self.send_action(agent_id, self.repeat_actions[agent_id])
            self.world.Step(self.TIME_STEP, 10, 10)
            if t == self.NUM_STEPS_PER_TICK - 2:
                self.world.ClearForces()
            # if self.repeat_actions[1] == 'stop':
            #     print('vel:', t, self.agents[1].linearVelocity.x, self.agents[1].linearVelocity.y)            
            self.agents_pos = [_get_world_pos(agent) for agent in self.agents]
            self.agents_vel = [_get_world_vel(agent) for agent in self.agents]
            for agent_id, (agent_pos, agent_vel) in enumerate(zip(self.agents_pos, self.agents_vel)):
                self.trajectories[agent_id].append((agent_pos[0], agent_pos[1],
                                                    agent_vel[0], agent_vel[1]))
                self.vels[agent_id].append((agent_vel[0], agent_vel[1]))
            self.items_pos = [_get_world_pos(item) for item in self.items]
            self.items_vel = [_get_world_vel(item) for item in self.items]
            for item_id, (item_pos, item_vel) in enumerate(zip(self.items_pos, self.items_vel)):
                self.trajectories[item_id + 2].append((item_pos[0], item_pos[1], 
                                                       item_vel[0], item_vel[1]))
        # agent1beliefs_item1.append(belief_states[1][14:16])
        # for item_id in range(self.num_items):
        #     self.items[item_id].linearVelocity.x = 0
        #     self.items[item_id].linearVelocity.y = 0
        # self.items_vel = [_get_world_vel(item) for item in self.items]

        for agent_id in range(self.num_agents):
            self.agents[agent_id].angularVelocity = 0
          
        self.running = not self.terminal()
        if self.enable_renderer and \
            (self.clip_interval is None \
                or self.steps > self.clip_interval[0] and self.steps <= self.clip_interval[1]):
            self.render_all()
        return agent1beliefs_item1


    # FIXME: only for two agents
    def extract_state_feature(self, x1, x2):
        points = [
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2, -self.room_dim[1] / 2)]
        points += [(p[0] - 16, p[1] - 12) for p in self.items_pos]

        p1 = (x1[0] - 16, x1[1] - 12)
        p2 = (x2[0] - 16, x2[1] - 12)
        feat = [p1[0], p1[1]]
        feat += [p1[0] - p2[0], p1[1] - p2[1]]
        for point in points:
            feat += [p1[0] - point[0], p1[1] - point[1]]
        feat += [p2[0], p2[1]]
        feat += [p2[0] - p1[0], p2[1] - p1[1]]
        for point in points:
            feat += [p2[0] - point[0], p2[1] - point[1]]
        return feat


    def get_obs(self, input_type):
        """get observation"""
        if input_type == 'image':
            return imresize(_get_obs(self.screen,
                            self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
                            (self.obs_dim[1], self.obs_dim[2])).transpose([2, 0, 1]).reshape((-1))
        else:
            #FIXME: only for 2 agents
            positions = [[x, y] for x, y in self.agents_pos]
            return [np.asarray(self.extract_state_feature(positions[0], positions[1])) 
                        for _ in range(2)]


    def get_belief_state(self, agent_id):
        """
        based on get_state, entity locations use the beliefs of the agent_id
        """
        agent_location, agent_angle, agent_linear_v, agent_angular_v = \
            self.agents[agent_id].beliefs[agent_id].sample_state() #assume agent's belief about self is true
        other_agent_location, other_agent_angle, other_agent_linear_v, other_agent_angular_v = \
            self.agents[agent_id].beliefs[1-agent_id].sample_state()
        item0_location, _, item0_linear_v, item0_angular_v = \
            self.agents[agent_id].beliefs[self.num_agents+0].sample_state()
        item1_location, _, item1_linear_v, item1_angular_v = \
            self.agents[agent_id].beliefs[self.num_agents+1].sample_state()

        #TODO what is attached used for
        return (agent_location[0], agent_location[1],
                other_agent_location[0], other_agent_location[1],
                agent_angle, other_agent_angle,
                agent_linear_v[0], agent_linear_v[1], agent_angular_v,
                other_agent_linear_v[0], other_agent_linear_v[1], other_agent_angular_v,
                item0_location[0], item0_location[1],
                item1_location[0], item1_location[1],
                item0_linear_v[0], item0_linear_v[1], item0_angular_v,
                item1_linear_v[0], item1_linear_v[1], item1_angular_v,
                self.attached[0] is not None, self.attached[1] is not None)


    def get_state(self, agent_id):
        """
        get the state of an agent
        TODO: add velocities to the state; more than 2 agents
        """
        return (self.agents_pos[agent_id][0], self.agents_pos[agent_id][1], 
                self.agents_pos[1 - agent_id][0], self.agents_pos[1 - agent_id][1], 
                self.agents[agent_id].linearVelocity.x, self.agents[agent_id].linearVelocity.y,
                self.agents[1 - agent_id].linearVelocity.x, self.agents[1 - agent_id].linearVelocity.y,
                self.items_pos[0][0], self.items_pos[0][1], self.items[0].linearVelocity.x, self.items[0].linearVelocity.y,
                self.attached[0] is not None, self.attached[1] is not None)


    def terminal(self):
        """check if the goal is achieved"""
        return False


    def get_reward(self):
        dist_to_bottom_left_corner = _get_dist(self.items_pos[0], (16 - 10 + 2, 12 - 10 + 2))
        normalized_dist = dist_to_bottom_left_corner / _get_dist((16, 12 + 4), (16 - 10 + 2, 12 - 10 + 2))
        # r = [-normalized_dist - 0.2, -normalized_dist - 0.2]
        r = [1.0 - normalized_dist, 1.0 - normalized_dist]
        return r[self.planning_agent]


    def _get_dist_room(self, pos, room_id):
        cur_room_id = self._get_room_id(pos)
        if cur_room_id != room_id:
            door_id = self.connected[cur_room_id][room_id]
            # if door_id is not None:
            #     print(pos, cur_room_id, room_id)
            if door_id is None: return self._max_dist
            return _get_dist(pos, self.doors_pos[door_id]) + _get_dist(self.doors_pos[door_id], self.landmark_corners[room_id])
        else:
            # print(pos, cur_room_id, room_id)
            return _get_dist(pos, self.landmark_corners[room_id])


    def _get_doors(self, start_room_id, path):
        cur_room_id = start_room_id
        doors = []
        for target_room_id in path:
            door_id = self.connected[cur_room_id][target_room_id]
            doors.append(door_id)
            cur_room_id = target_room_id
        return doors


    def _get_dist_pos(self, pos1, pos2):
        if self.env_id == 0:
            return _get_dist(pos1, pos2)
        room_id1 = self._get_room_id(pos1)
        room_id2 = self._get_room_id(pos2)
        if room_id1 != room_id2:
            path = self.path[(room_id1, room_id2)]
            if path is None: return self._max_dist * 10
            doors = self.doors[(room_id1, room_id2)]
            dist = _get_dist(pos1, self.doors_pos[doors[0]]) + \
                   _get_dist(pos2, self.doors_pos[doors[-1]])
            if len(doors) > 1:
                for door_id in range(0, len(doors) - 1):
                    dist += _get_dist(self.doors_pos[doors[door_id]], self.doors_pos[doors[door_id + 1]])
        else:
            dist = _get_dist(pos1, pos2)
        return dist


    def is_far(self):
        return self._get_dist_pos(self.agents_pos[0], self.agents_pos[1]) > 14.14


    def get_reward_state(self, agent_id, curr_state, action, t, T, goal_id=None):
        # next_state = self.transition(curr_state, action)
        # print(next_state)
        goal = GOAL[self.goals[agent_id]] if goal_id is None else GOAL[goal_id]
        if goal[0] == 'stop':
            return 0.0 if action == 'noforce' else -COST[self.costs[agent_id]]

        ITEMBASE = 8
        if agent_id == 0:
            agents_pos = [(curr_state[0], curr_state[1]), 
                          (curr_state[2], curr_state[3])]
        else:
            agents_pos = [(curr_state[2], curr_state[3]), 
                          (curr_state[0], curr_state[1])]
        items_pos = [(curr_state[ITEMBASE + 0], curr_state[ITEMBASE + 1])]
        attached = [curr_state[ITEMBASE + 4], curr_state[ITEMBASE + 5]]
        """TODO: RA & RO"""
        if goal[0] == 'LMA':
            dist_to_goal = self._get_dist_pos(agents_pos[goal[1]], self.landmark_centers[goal[2]])
        elif goal[0] == 'RA':
            dist_to_goal = self._get_dist_room(agents_pos[goal[1]], goal[2])
            # if agent_id == 0:
            #     print(agents_pos[goal[1]], goal[2], dist_to_goal)
        elif goal[0] == 'LMO': 
            dist_to_goal = self._get_dist_pos(items_pos[goal[1]], self.landmark_centers[goal[2]])
        elif goal[0] == 'RO': 
            dist_to_goal = self._get_dist_room(items_pos[goal[1]], goal[2])
        elif goal[0] == 'LMOP':
            dist_to_goal = self._get_dist_pos(items_pos[goal[1]], self.landmark_centers[goal[2]]) \
                        - 0.5 * self._get_dist_pos(items_pos[goal[1]], agents_pos[1 - agent_id])
        elif goal[0] == 'TE':
            if goal[2] < self.num_agents:
                dist_to_goal = self._get_dist_pos(agents_pos[goal[1]], agents_pos[goal[2]])
            else:
                dist_to_goal = self._get_dist_pos(agents_pos[goal[1]], items_pos[goal[2] - self.num_agents])
        elif goal[0] == 'GE':
            """TODO: more objects"""
            dist_to_goal = 0.0 if attached[goal[1]] else 1.0
        else:
            raise ValueError('Invalid goal!')

        normalized_dist = goal[3] * dist_to_goal / self._max_dist
        # if goal[0] == 'LMA':
        #     normalized_dist = goal[3] * dist_to_goal / self._max_dist# _get_dist_lm(self.init_agents_pos[goal[1]], self.landmark_centers[goal[2]])
        # else:
        #     normalized_dist = goal[3] * dist_to_goal / self._max_dist #_get_dist(self.init_items_pos[goal[1]], self.landmark_centers[goal[2]])
        # r = [-normalized_dist - 0.2, -normalized_dist - 0.2]
        # print(agent_id, items_pos, self.landmark_centers[goal[2]], dist_to_goal, normalized_dist)
        # # input('press any key to continue...')
        r = -normalized_dist
        cost = 0.0 if action == 'noforce' else COST[self.costs[agent_id]]
        return r - cost


    def is_terminal(self, curr_state, t):
        """check if the goal is achieved"""
        # if t == self.max_episode_length: return True
        # ITEMBASE = 8
        # items_pos = [(curr_state[ITEMBASE + 0], curr_state[ITEMBASE + 1])]
        # items_vel = [(curr_state[ITEMBASE + 2], curr_state[ITEMBASE + 3])]
        # dist_to_bottom_left_corner = _get_dist(items_pos[0], (16 - 10 + 2, 12 - 10 + 2))
        # goal_dist = dist_to_bottom_left_corner
        # d_threshold = 0.5
        # v_threshold = 0.01
        # return _norm(items_vel[0]) < v_threshold and goal_dist < d_threshold
        return False


    def _setup_tmp(self, env_id, agents_pos, agents_angle, agents_vel, items_pos, items_vel, attached):
        """setup a new espisode from beliefs. some vars used for construction are used from true env."""
        # self.tmp_world = world(contactListener=MyContactListener(), gravity=(0, 0), doSleep=True)
        self.tmp_world = world(gravity=(0, 0), doSleep=True)
        self.tmp_room = self.tmp_world.CreateBody(position=(16, 12))
        self.tmp_room.CreateEdgeChain(
            [
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2 - self.door_length)]
        )
        if self.enable_renderer:
            self.tmp_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Maze_v1 belief')
            self.tmp_clock = pygame.time.Clock()
        # build maze
        env_def = self.maze_sampler.get_env_def(env_id)
        for wall_id, wall in enumerate(env_def['maze_def']):
            seg = WALL_SEGS[wall]
            if seg:
                if wall_id == 0:
                    self.tmp_room.CreateEdgeChain([(0, seg[0]), (0, seg[1])])
                elif wall_id == 1:
                    self.tmp_room.CreateEdgeChain([(-seg[0], 0), (-seg[1], 0)])
                elif wall_id == 2:
                    self.tmp_room.CreateEdgeChain([(0, -seg[0]), (0, -seg[1])])
                elif wall_id == 3:
                    self.tmp_room.CreateEdgeChain([(seg[0], 0), (seg[1], 0)])

        self.tmp_agents, self.tmp_items = [], []
        self.tmp_trajectories = [None] * (self.num_agents + self.num_items)
        # add agents
        for agent_id in range(self.num_agents):
            R = SIZE[self.sizes[agent_id]]
            body = self.tmp_world.CreateDynamicBody(
                position=agents_pos[agent_id], angle=agents_angle[agent_id],
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(vertices=[(-1.5 * R, -R),
                                                   (-0.5 * R, R),
                                                   (0.5 * R, R),
                                                   (1.5 * R, -R)]),
                    density=DENSITY[self.densities[agent_id]],
                    friction=FRICTION
                )
            )
            body.linearVelocity.x = agents_vel[agent_id][0]
            body.linearVelocity.y = agents_vel[agent_id][1]
            body.angularVelocity = 0 #agents_vel[agent_id][2] #TODO
            #TODO should these be simulated in some way as well? (userData for sampling close entities --> collision)
            #FOV for display technicalities?
            # body.userData = [('agent',agent_id), self.touch_sensor, None] #3rd entry for attached item_id (None if not attached)
            # body.field_of_view = np.zeros(self.room_dim)
            # body.last_observations = {}
            self.tmp_agents.append(body)
            self.tmp_repeat_actions = [None] * 2
            self.tmp_actions = [None] * self.num_agents
            #TODO is this needed?
            # self.tmp_trajectories[agent_id] = [(x, y, 0, 0)]
        self.tmp_agents_pos = [_get_world_pos(agent) for agent in self.tmp_agents]
        # print('_setup_tmp tmp_agents_pos',self.tmp_agents_pos)
        self.tmp_agents_vel = [_get_world_vel(agent) for agent in self.tmp_agents]

        # add items
        ITEM_BASE = self.num_agents
        for item_id in range(self.num_items):
            index = ITEM_BASE + item_id
            body = self.tmp_world.CreateDynamicBody(position=items_pos[item_id])
            body.CreateCircleFixture(radius=SIZE[self.sizes[index]], density=DENSITY[self.densities[index]], \
                                     friction=FRICTION, restitution=0)
            body.linearVelocity.x = items_vel[item_id][0]
            body.linearVelocity.y = items_vel[item_id][1]
            # body.userData = [('item', item_id), None] #2nd entry for attached agent_id (None if not attached)
            self.tmp_items.append(body)
            # self.trajectories[index] = [(x, y, 0, 0)]
        self.tmp_items_pos = [_get_world_pos(item) for item in self.tmp_items]
        self.tmp_items_vel = [_get_world_vel(item) for item in self.tmp_items]

        # friction
        self.tmp_groundBody = self.tmp_world.CreateStaticBody(
            position=(16, 12),
            shapes=polygonShape(box=(self.room_dim[0],self.room_dim[1])),
        )
        for body in self.tmp_agents + self.tmp_items:
            dfn = b2FrictionJointDef(
                            bodyA=body,
                            bodyB=self.tmp_groundBody,
                            localAnchorA=(0, 0),
                            localAnchorB=(0, 0),
                            maxForce=100,
                            maxTorque=0
                        )
            self.tmp_world.CreateJoint(dfn)

        #TODO recording
        # self.tmp_steps = 0
        # self.tmp_running = False
        # self.tmp_video = None
        # belief_record_path = ""
        # if belief_record_path:
        #     self.video = cv2.VideoWriter(
        #         belief_record_path,
        #         cv2.VideoWriter_fourcc(*'mp4v'),
        #         30,
        #         (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        #TODO attached!
        self.tmp_attached = [None] * self.num_agents
        self.once_attached = [False] * self.num_agents



    def _send_action_tmp(self, agent_id, action):
        if action is None: return
        self.tmp_repeat_actions[agent_id] = action
        if action not in (self.action_space[:-self.action_space_types[agent_id]] if self.action_space_types[agent_id] \
                            else self.action_space):
            return
        df = STRENGTH[self.strengths[agent_id]] * self.NUM_STEPS_PER_TICK
        if action.startswith('turn'):
            if action == 'turnleft':
                self.tmp_agents[agent_id].ApplyTorque(df * 1.0, True)
            else:
                self.tmp_agents[agent_id].ApplyTorque(-df * 1.0, True)
            return
        if action == 'attach': 
            if self.tmp_attached[agent_id] is None and _get_dist(self.tmp_agents_pos[agent_id], self.tmp_items_pos[0]) < SIZE[self.sizes[agent_id]] + SIZE[self.sizes[2]] + 0.1:
                f = {'spring': 0.3, 'rope': 0.1, 'rod': 100}
                d = {'spring': 0, 'rope': 0, 'rod': 0.5}
                dfn = b2DistanceJointDef(
                        frequencyHz=f['rod'],
                        dampingRatio=d['rod'],
                        bodyA=self.tmp_agents[agent_id],
                        bodyB=self.tmp_items[0],
                        localAnchorA=(0, 0),
                        localAnchorB=(0, 0),
                    )
                self.tmp_attached[agent_id] = self.tmp_world.CreateJoint(dfn)
                # print('here!!', agent_id, self.planning_agent, self.tmp_agents_pos, self.tmp_items_pos[0], self.agents_pos)
            return
        elif action == 'detach':
            if self.tmp_attached[agent_id] is not None:
                self.tmp_world.DestroyJoint(self.tmp_attached[agent_id])
                self.tmp_attached[agent_id] = None
            return

        fx, fy = 0.0, 0.0
        df = STRENGTH[self.strengths[agent_id]] * self.NUM_STEPS_PER_TICK
        
        if action == 'up':
            fy += df
        elif action == 'down':
            fy -= df
        elif action == 'left':
            fx -= df
        elif action == 'right':
            fx += df
        elif action == 'upleft':
            fx -= df * 0.707
            fy += df * 0.707
        elif action == 'upright':
            fx += df * 0.707
            fy += df * 0.707
        elif action == 'downleft':
            fx -= df * 0.707
            fy -= df * 0.707
        elif action == 'downright':
            fx += df * 0.707
            fy -= df * 0.707
        elif action == 'stop':
            if self.strengths[agent_id] != 3:
                self.tmp_agents[agent_id].linearVelocity.x = 0
                self.tmp_agents[agent_id].linearVelocity.y = 0
            return
        elif action == 'noforce':
            return
        else:
            print('ERROR: invalid action!')
        if action == 'stop': print(fx, fy)
        f = self.tmp_agents[agent_id].GetWorldVector(localVector=(fx, fy))
        p = self.tmp_agents[agent_id].GetWorldPoint(localPoint=(0.0, 0.0))
        self.tmp_agents[agent_id].ApplyForce(f, p, True)
        # self.tmp_agents[agent_id].ApplyLinearImpulse(f, p, True)


    def _step_valid(self):
        """based on _step_tmp, move only to eliminate implausible collisions/overlap"""
        # self.tmp_steps += 1
        self.tmp_world.Step(self.TIME_STEP, 10, 10)
        # self.tmp_world.ClearForces()
        self.tmp_agents_pos = [_get_world_pos(agent) for agent in self.tmp_agents]
        self.tmp_agents_vel = [_get_world_vel(agent) for agent in self.tmp_agents]
        self.tmp_items_pos = [_get_world_pos(item) for item in self.tmp_items]
        self.tmp_items_vel = [_get_world_vel(item) for item in self.tmp_items]
        # print('_step_valid',self.tmp_items_pos)
        # for item_id, (item_pos, item_vel) in enumerate(zip(self.tmp_items_pos, self.tmp_items_vel)):
        #     self.tmp_trajectories[item_id + 2].append((item_pos[0], item_pos[1],
        #                                            item_vel[0], item_vel[1]))
        for t in range(self.NUM_STEPS_PER_TICK - 1):
            self.tmp_world.Step(self.TIME_STEP, 10, 10)
            if t == self.NUM_STEPS_PER_TICK - 2:
                self.tmp_world.ClearForces()
            # if self.repeat_actions[1] == 'stop':
            #     print('vel:', t, self.agents[1].linearVelocity.x, self.agents[1].linearVelocity.y)
            self.tmp_agents_pos = [_get_world_pos(agent) for agent in self.tmp_agents]
            self.tmp_agents_vel = [_get_world_vel(agent) for agent in self.tmp_agents]
            # for agent_id, (agent_pos, agent_vel) in enumerate(zip(self.tmp_agents_pos, self.tmp_agents_vel)):
            #     self.tmp_trajectories[agent_id].append((agent_pos[0], agent_pos[1],
            #                                         agent_vel[0], agent_vel[1]))
            #     self.tmp_vels[agent_id].append((agent_vel[0], agent_vel[1]))
            self.tmp_items_pos = [_get_world_pos(item) for item in self.tmp_items]
            self.tmp_items_vel = [_get_world_vel(item) for item in self.tmp_items]
            # print(self.tmp_items_vel)
            # for item_id, (item_pos, item_vel) in enumerate(zip(self.tmp_items_pos, self.tmp_items_vel)):
            #     self.tmp_trajectories[item_id + 2].append((item_pos[0], item_pos[1],
            #                                            item_vel[0], item_vel[1]))
        # for item_id in range(self.num_items):
        #     self.items[item_id].linearVelocity.x = 0
        #     self.items[item_id].linearVelocity.y = 0
        # self.items_vel = [_get_world_vel(item) for item in self.items]
        for agent_id in range(self.num_agents):
            self.tmp_agents[agent_id].angularVelocity = 0
        self.tmp_running = not self.terminal()


    def _step_tmp(self):
        """apply one step and update the environment"""
        # self.tmp_steps += 1
        self.tmp_world.Step(self.TIME_STEP, 10, 10)
        # self.tmp_world.ClearForces()
        self.tmp_agents_pos = [_get_world_pos(agent) for agent in self.tmp_agents]
        self.tmp_agents_vel = [_get_world_vel(agent) for agent in self.tmp_agents]
        for agent_id, (agent_pos, agent_vel) in enumerate(zip(self.tmp_agents_pos, self.tmp_agents_vel)):
            # self.tmp_trajectories[agent_id].append((agent_pos[0], agent_pos[1], agent_vel[0], agent_vel[1]))            #
            # self.tmp_vels[agent_id].append((agent_vel[0], agent_vel[1]))
            if self.tmp_actions[agent_id] is None:
                self.tmp_actions[agent_id] = [self.tmp_repeat_actions[agent_id]]
            else:
                self.tmp_actions[agent_id].append(self.tmp_repeat_actions[agent_id])

        self.tmp_items_pos = [_get_world_pos(item) for item in self.tmp_items]
        self.tmp_items_vel = [_get_world_vel(item) for item in self.tmp_items]
        # print('_step_tmp',self.tmp_items_pos)
        # for item_id, (item_pos, item_vel) in enumerate(zip(self.tmp_items_pos, self.tmp_items_vel)):
        #     self.tmp_trajectories[item_id + 2].append((item_pos[0], item_pos[1],
        #                                            item_vel[0], item_vel[1]))
        for t in range(self.NUM_STEPS_PER_TICK - 1):
            for agent_id in range(2):
                # if self.repeat_actions[agent_id] == 'stop' or t == self.NUM_STEPS_PER_TICK - 1:
                if self.tmp_repeat_actions[agent_id] not in ['attach', 'detach']:
                    self._send_action_tmp(agent_id, self.tmp_repeat_actions[agent_id])
            self.tmp_world.Step(self.TIME_STEP, 10, 10)
            if t == self.NUM_STEPS_PER_TICK - 2:
                self.tmp_world.ClearForces()
            # if self.repeat_actions[1] == 'stop':
            #     print('vel:', t, self.agents[1].linearVelocity.x, self.agents[1].linearVelocity.y)
            self.tmp_agents_pos = [_get_world_pos(agent) for agent in self.tmp_agents]
            self.tmp_agents_vel = [_get_world_vel(agent) for agent in self.tmp_agents]
            # for agent_id, (agent_pos, agent_vel) in enumerate(zip(self.tmp_agents_pos, self.tmp_agents_vel)):
            #     self.tmp_trajectories[agent_id].append((agent_pos[0], agent_pos[1],
            #                                         agent_vel[0], agent_vel[1]))
            #     self.tmp_vels[agent_id].append((agent_vel[0], agent_vel[1]))
            self.tmp_items_pos = [_get_world_pos(item) for item in self.tmp_items]
            self.tmp_items_vel = [_get_world_vel(item) for item in self.tmp_items]
            # print(self.tmp_items_vel)
            # for item_id, (item_pos, item_vel) in enumerate(zip(self.tmp_items_pos, self.tmp_items_vel)):
            #     self.tmp_trajectories[item_id + 2].append((item_pos[0], item_pos[1],
            #                                            item_vel[0], item_vel[1]))
        # for item_id in range(self.num_items):
        #     self.items[item_id].linearVelocity.x = 0
        #     self.items[item_id].linearVelocity.y = 0
        # self.items_vel = [_get_world_vel(item) for item in self.items]
        for agent_id in range(self.num_agents):
            self.tmp_agents[agent_id].angularVelocity = 0
        self.tmp_running = not self.terminal()


    def _get_state_tmp(self, agent_id):
        """
        get the state of an agent
        TODO: more than 2 agents
        """

        # return (self.agents_pos[agent_id][0], self.agents_pos[agent_id][1],
        #         self.agents_pos[1 - agent_id][0], self.agents_pos[1 - agent_id][1],
        #         self.agents[agent_id].linearVelocity.x, self.agents[agent_id].linearVelocity.y,
        #         self.agents[1 - agent_id].linearVelocity.x, self.agents[1 - agent_id].linearVelocity.y,
        #         self.items_pos[0][0], self.items_pos[0][1], self.items[0].linearVelocity.x, self.items[0].linearVelocity.y,
        #         self.attached[0] is not None, self.attached[1] is not None)

        return (self.tmp_agents_pos[agent_id][0], self.tmp_agents_pos[agent_id][1],
                self.tmp_agents_pos[1-agent_id][0], self.tmp_agents_pos[1-agent_id][1],
                self.tmp_agents[agent_id].angle, self.tmp_agents[1-agent_id].angle,
                self.tmp_agents[agent_id].linearVelocity.x, self.tmp_agents[agent_id].linearVelocity.y, self.tmp_agents[agent_id].angularVelocity,
                self.tmp_agents[1-agent_id].linearVelocity.x, self.tmp_agents[1-agent_id].linearVelocity.y, self.tmp_agents[1-agent_id].angularVelocity,
                self.tmp_items_pos[0][0], self.tmp_items_pos[0][1],
                self.tmp_items_pos[1][0], self.tmp_items_pos[1][1],
                self.tmp_items[0].linearVelocity.x, self.tmp_items[0].linearVelocity.y, self.tmp_items[0].angularVelocity,
                self.tmp_items[1].linearVelocity.x, self.tmp_items[1].linearVelocity.y, self.tmp_items[1].angularVelocity,
                self.attached[0] is not None, self.attached[1] is not None)
        

    def _display_imagined(self, agent_id, curr_state):
        """based on transition func (simulate one step)"""
        """curr_state = believed state"""
        VELBASE = 6
        ITEMBASE = 12
        if agent_id == 0:
            curr_agents_pos = [(curr_state[0], curr_state[1]), 
                               (curr_state[2], curr_state[3])]
            curr_agents_angle = [curr_state[4], curr_state[5]]
            curr_agents_vel = [[curr_state[VELBASE + 0], curr_state[VELBASE + 1], curr_state[VELBASE + 2]],
                               [curr_state[VELBASE + 3], curr_state[VELBASE + 4], curr_state[VELBASE + 5]]]
        else:
            curr_agents_pos = [(curr_state[2], curr_state[3]), 
                               (curr_state[0], curr_state[1])]
            curr_agents_angle = [curr_state[5], curr_state[4]]
            curr_agents_vel = [[curr_state[VELBASE + 3], curr_state[VELBASE + 4], curr_state[VELBASE + 5]],
                               [curr_state[VELBASE + 0], curr_state[VELBASE + 1], curr_state[VELBASE + 2]]]

        curr_items_pos = [(curr_state[ITEMBASE + 0], curr_state[ITEMBASE + 1]),
                          (curr_state[ITEMBASE + 2], curr_state[ITEMBASE + 3])]
        curr_items_vel = [[curr_state[ITEMBASE + 4], curr_state[ITEMBASE + 5], curr_state[ITEMBASE + 6]],
                          [curr_state[ITEMBASE + 7], curr_state[ITEMBASE + 8], curr_state[ITEMBASE + 9]]]

        attached = [curr_state[ITEMBASE + 10], curr_state[ITEMBASE + 11]]

        self._setup_tmp(self.env_id, curr_agents_pos, curr_agents_angle, curr_agents_vel, \
                        curr_items_pos, curr_items_vel, attached)
        self.belief_render(agent_id)


    def valid_belief(self, agent_id, curr_state, action=None):
        """simulate one step with 0 velocity to make sure belief is valid"""
        VELBASE = 6
        ITEMBASE = 12
        if agent_id == 0:
            curr_agents_pos = [(curr_state[0], curr_state[1]),
                               (curr_state[2], curr_state[3])]
            curr_agents_angle = [curr_state[4], curr_state[5]]
            curr_agents_vel = [[curr_state[VELBASE + 0], curr_state[VELBASE + 1], curr_state[VELBASE + 2]],
                               [curr_state[VELBASE + 3], curr_state[VELBASE + 4], curr_state[VELBASE + 5]]]
        else:
            curr_agents_pos = [(curr_state[2], curr_state[3]),
                               (curr_state[0], curr_state[1])]
            curr_agents_angle = [curr_state[5], curr_state[4]]
            curr_agents_vel = [[curr_state[VELBASE + 3], curr_state[VELBASE + 4], curr_state[VELBASE + 5]],
                               [curr_state[VELBASE + 0], curr_state[VELBASE + 1], curr_state[VELBASE + 2]]]

        curr_items_pos = [(curr_state[ITEMBASE + 0], curr_state[ITEMBASE + 1]),
                          (curr_state[ITEMBASE + 2], curr_state[ITEMBASE + 3])]
        curr_items_vel = [[curr_state[ITEMBASE + 4], curr_state[ITEMBASE + 5], curr_state[ITEMBASE + 6]],
                          [curr_state[ITEMBASE + 7], curr_state[ITEMBASE + 8], curr_state[ITEMBASE + 9]]]
        attached = [curr_state[ITEMBASE + 10], curr_state[ITEMBASE + 11]]
        #to eliminate movement
        curr_agents_vel, curr_items_vel = np.zeros_like(curr_agents_vel), np.zeros_like(curr_items_vel)

        self._setup_tmp(self.env_id, curr_agents_pos, curr_agents_angle, curr_agents_vel, curr_items_pos, curr_items_vel, attached)
        print('_tmp_transition ', agent_id)
        #degenerate step, just to make sure valid belief locations.
        self._step_valid()
        next_state = self._get_state_tmp(agent_id)
        return next_state


    def transition(self, agent_id, curr_state, action, nb_steps, cInit, cBase):
        """transition func (simulate one step)"""
        # VELBASE = 4
        # ITEMBASE = 8
        # if agent_id == 0:
        #     curr_agents_pos = [(curr_state[0], curr_state[1]),
        #                        (curr_state[2], curr_state[3])]
        #     curr_agents_vel = [(curr_state[VELBASE + 0], curr_state[VELBASE + 1]),
        #                        (curr_state[VELBASE + 2], curr_state[VELBASE + 3])]
        # else:
        #     curr_agents_pos = [(curr_state[2], curr_state[3]),
        #                        (curr_state[0], curr_state[1])]
        #     curr_agents_vel = [(curr_state[VELBASE + 2], curr_state[VELBASE + 3]),
        #                        (curr_state[VELBASE + 0], curr_state[VELBASE + 1])]
        # curr_items_pos = [(curr_state[ITEMBASE + 0], curr_state[ITEMBASE + 1])]
        # curr_items_vel = [(curr_state[ITEMBASE + 2], curr_state[ITEMBASE + 3])]
        # attached = [curr_state[ITEMBASE + 4], curr_state[ITEMBASE + 5]]
        #
        # curr_agents_angle = [self.agents[0].angle,self.agents[1].angle] #TODO

        VELBASE = 6
        ITEMBASE = 12
        if agent_id == 0:
            curr_agents_pos = [(curr_state[0], curr_state[1]),
                               (curr_state[2], curr_state[3])]
            curr_agents_angle = [curr_state[4], curr_state[5]]
            curr_agents_vel = [[curr_state[VELBASE + 0], curr_state[VELBASE + 1], curr_state[VELBASE + 2]],
                               [curr_state[VELBASE + 3], curr_state[VELBASE + 4], curr_state[VELBASE + 5]]]
        else:
            curr_agents_pos = [(curr_state[2], curr_state[3]),
                               (curr_state[0], curr_state[1])]
            curr_agents_angle = [curr_state[5], curr_state[4]]
            curr_agents_vel = [[curr_state[VELBASE + 3], curr_state[VELBASE + 4], curr_state[VELBASE + 5]],
                               [curr_state[VELBASE + 0], curr_state[VELBASE + 1], curr_state[VELBASE + 2]]]

        curr_items_pos = [(curr_state[ITEMBASE + 0], curr_state[ITEMBASE + 1]),
                          (curr_state[ITEMBASE + 2], curr_state[ITEMBASE + 3])]
        curr_items_vel = [[curr_state[ITEMBASE + 4], curr_state[ITEMBASE + 5], curr_state[ITEMBASE + 6]],
                          [curr_state[ITEMBASE + 7], curr_state[ITEMBASE + 8], curr_state[ITEMBASE + 9]]]
        attached = [curr_state[ITEMBASE + 10], curr_state[ITEMBASE + 11]]

        self._setup_tmp(self.env_id, curr_agents_pos, curr_agents_angle, curr_agents_vel, curr_items_pos, curr_items_vel, attached)
        self._send_action_tmp(agent_id, action)
        self._step_tmp()
        # print(curr_agents_pos, action, self.tmp_agents_pos)
        next_state = self._get_state_tmp(agent_id)
        # print(curr_state, action, next_state)
        return next_state


    def set_state(self, agents_pos, agents_vel, items_pos, items_vel, attached, steps):
        """set the state of the environment"""
        for agent_id, (pos, vel) in enumerate(zip(agents_pos, agents_vel)):
            self.world.DestroyBody(self.agents[agent_id])
            self.agents[agent_id] = self.world.CreateDynamicBody(position=(pos[0], pos[1]))
            self.agents[agent_id].CreateCircleFixture(radius=SIZE[self.sizes[agent_id]], density=DENSITY[self.densities[agent_id]], friction=FRICTION)
            self.agents[agent_id].linearVelocity.x = vel[0]
            self.agents[agent_id].linearVelocity.y = vel[1]
        self.agents_pos = [_get_world_pos(agent) for agent in self.agents]
        self.agents_vel = [_get_world_vel(agent) for agent in self.agents]

        for item_id, (pos, vel) in enumerate(zip(items_pos, items_vel)):
            self.world.DestroyBody(self.items[item_id])
            self.items[item_id] = self.world.CreateDynamicBody(position=(pos[0], pos[1]))
            self.items[item_id].CreateCircleFixture(radius=SIZE[self.sizes[self.num_agents + item_id]], density=DENSITY[self.densities[self.num_agents + item_id]], friction=FRICTION, restitution=0)
            self.items[item_id].linearVelocity.x = vel[0]
            self.items[item_id].linearVelocity.y = vel[1]
        self.items_pos = [_get_world_pos(item) for item in self.items]
        self.items_vel = [_get_world_vel(item) for item in self.items]
        
        for agent_id in range(self.num_agents):
            if self.attached[agent_id] is not None:
                self.world.DestroyJoint(self.attached[agent_id])
            if attached[agent_id] is not None:
                f = {'spring': 0.3, 'rope': 0.1, 'rod': 100}
                d = {'spring': 0, 'rope': 0, 'rod': 0.5}
                dfn = b2DistanceJointDef(
                        frequencyHz=f['rod'],
                        dampingRatio=d['rod'],
                        bodyA=self.agents[agent_id],
                        bodyB=self.items[0],
                        localAnchorA=(0, 0),
                        localAnchorB=(0, 0),
                    )
                self.attached[agent_id] = self.world.CreateJoint(dfn)

        self.steps = steps


    def belief_render(self, render_agent_id):
        """render the tmp environment, based on render"""
        colors = self.colors
        self.tmp_screen.fill((255, 255, 255, 255)) #TODO same screen?
        for lm_loc, color in zip(self.landmark_centers, LM_COLOR_LIST):
            _my_draw_patch(lm_loc, self.tmp_screen, color, self.PPM, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        for body_id, body in enumerate([self.tmp_room] + self.tmp_agents + self.tmp_items):
            if body_id and not self.visibility[body_id - 1]: continue
            for fixture in body.fixtures:
                fixture.shape.draw(self.tmp_screen, body, fixture, colors[body_id], self.PPM, self.SCREEN_HEIGHT)
        pygame.display.flip()

        cropped = pygame.display.get_surface().subsurface((120,40,400,400))
        frame = pygame.surfarray.array3d(cropped)
        frame = np.transpose(frame, (1, 0, 2))
        mask = self.agents[render_agent_id].field_of_view * 255
        mask = np.repeat(np.repeat(mask, 20, axis=1), 20, axis=0)
        mask = np.reshape(mask, mask.shape + (1,))
        mask = np.repeat(mask, 3, axis=2)
        combined = np.uint8(0.5*(frame)+0.5*(mask))
        # plt.imshow(combined)
        # plt.show()
        results_dir = 'belief_snapshots/'+str(render_agent_id)+'bel/'
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.imsave(results_dir+str(self.steps)+'agent'+str(render_agent_id)+'belief.png', combined)


    def render(self):
        """render the environment"""
        colors = self.colors

        self.screen.fill((255, 255, 255, 255))
        for lm_loc, color in zip(self.landmark_centers, LM_COLOR_LIST):
            _my_draw_patch(lm_loc, self.screen, color, self.PPM, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        for body_id, body in enumerate([self.room] + self.agents + self.items):
            if body_id and not self.visibility[body_id - 1]: continue
            for fixture in body.fixtures:
                fixture.shape.draw(self.screen, body, fixture, colors[body_id], self.PPM, self.SCREEN_HEIGHT)
        pygame.display.flip()

        # cropped = pygame.display.get_surface().subsurface((117.5,37.5,405,405))
        cropped = pygame.display.get_surface().subsurface((120,40,400,400))
        frame = pygame.surfarray.array3d(cropped)
        frame = np.transpose(frame, (1, 0, 2))
        for agent_id, agent in enumerate(self.agents):
            mask = agent.field_of_view * 255
            mask = np.repeat(np.repeat(mask, 20, axis=1), 20, axis=0)
            mask = np.reshape(mask, mask.shape + (1,))
            mask = np.repeat(mask, 3, axis=2)
            combined = np.uint8(0.5*(frame)+0.5*(mask))
            # plt.imshow(combined)
            # plt.show()
            results_dir = 'belief_snapshots/'+str(agent_id)+'FOV/'
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            plt.imsave(results_dir+str(self.steps)+'agent'+str(agent_id)+'FOV.png', combined)

        self.clock.tick(self.TARGET_FPS)
        if self.video:
            obs = _get_obs(self.screen, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            self.video.write(cv2.cvtColor(obs.transpose(1, 0, 2), 
                                            cv2.COLOR_RGB2BGR))


    def replay(self, trajectories, record_path=None, order=0):
        """replay an old espisode based on recorded trajectories"""
        self.video = None
        if record_path:
            self.video = cv2.VideoWriter(
                record_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                30, 
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.world = world(gravity=(0, 0), doSleep=True)
        self.room = self.world.CreateBody(position=(16, 12))
        self.room.CreateEdgeChain(
            [
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2,  self.room_dim[1] / 2),
             ( self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2, -self.room_dim[1] / 2),
             (-self.room_dim[0] / 2,  self.room_dim[1] / 2 - self.door_length)]
        )
        self.agents, self.items = [], []
        self.densities = [1, 1]
        T = len(trajectories[0])
        for t in range(T):
            for body in self.agents:
                self.world.DestroyBody(body)
            for body in self.items:
                self.world.DestroyBody(body)
            self.agents, self.items = [], []
            agent_indices = [0, 1] if order == 0 else [1, 0]
            for agent_id in agent_indices:
                body = self.world.CreateDynamicBody(position=(trajectories[agent_id][t][0], 
                                                              trajectories[agent_id][t][1]))
                body.CreateCircleFixture(radius=1, density=self.densities[0], friction=FRICTION)
                self.agents.append(body)
            body = self.world.CreateDynamicBody(position=(trajectories[2][t][0], 
                                                          trajectories[2][t][1]))
            body.CreateCircleFixture(radius=2, density=self.densities[0], friction=FRICTION)
            self.items.append(body)
            self.render()


    def release(self):
        """release video writer"""
        if self.video:
            # self.step()
            self.video.release()
            self.video = None


    def destroy(self):
        """destroy the environment"""
        self.world.DestroyBody(self.room)
        self.room = None
        for agent in self.agents:
            self.world.DestroyBody(agent)
        self.agents = []
        if self.enable_renderer:
            pygame.quit()


    #vector direction from start point along gaze ray, size 1
    def line_to_grid_cells(self, start_point, vector):
        max_length = (self.room_dim[0]**2 + self.room_dim[1]**2)**0.5
        #from start_point going in vector dir (should be of size 1 with orientation line) get new point
        grid_cells = []
        cur_point = start_point
        while math.hypot(cur_point[0] - start_point[0], cur_point[1] - start_point[1]) <= max_length:
            cur_cell = self.world_point_to_grid_cell(cur_point[0], cur_point[1])
            if [cur_cell[0],cur_cell[1]] not in grid_cells and \
                    (cur_cell[0] >= 0 and cur_cell[0] < self.room_dim[0] and \
                     cur_cell[1] >= 0 and cur_cell[1] < self.room_dim[1]):
                grid_cells.append([cur_cell[0],cur_cell[1]])
                cur_point = cur_point + vector
        return np.array(grid_cells)


    def set_field_of_view(self, agent):
        p1 = agent.worldCenter #body center
        Vs = agent.fixtures[0].shape.vertices
        eye_pos = [((Vs[0][0] - Vs[1][0]) * 0.3 + Vs[1][0], (Vs[0][1] - Vs[1][1]) * 0.3 + Vs[1][1]),
                   ((Vs[3][0] - Vs[2][0]) * 0.3 + Vs[2][0], (Vs[3][1] - Vs[2][1]) * 0.3 + Vs[2][1])]
        p2 = agent.GetWorldPoint(localPoint=(eye_pos[0][0],eye_pos[0][1])) #right eye
        p3 = agent.GetWorldPoint(localPoint=(0.0,eye_pos[0][1])) #middle
        cone_boundary_vec = b2Vec2(p2[0]-p1[0], p2[1]-p1[1])
        cone_mid_vec = b2Vec2(p3[0]-p1[0], p3[1]-p1[1])

        agent.field_of_view = np.zeros(self.room_dim)
        #move to center of cell world point
        row_baseline = self.room.position.y + (self.room_dim[0] // 2) - 0.5
        col_baseline = self.room.position.x - (self.room_dim[1] // 2) + 0.5
        grid_cells_world_center = [[col_baseline+c,row_baseline-r] \
                                    for r in range(self.room_dim[0]) for c in range(self.room_dim[1])] #x,y
        cone_angle = _get_angle(cone_boundary_vec, cone_mid_vec) # half angle
        in_cone_idx = [_get_angle(cone_mid_vec,(c[0]-p1[0],c[1]-p1[1])) < cone_angle \
                       for c in grid_cells_world_center]
        in_cone_y, in_cone_x = np.meshgrid(np.arange(self.room_dim[0]), np.arange(self.room_dim[1]))
        agent.field_of_view[in_cone_x.flatten()[in_cone_idx], in_cone_y.flatten()[in_cone_idx]] = 1


    def is_visable(self,cell_center,agent_id):
        collisions = self.calculate_collision(self.agents[agent_id].worldCenter,cell_center,agent_id)
        # if len(collisions) != 0:
        #     print('cell',cell_center,'col',collisions)
        return len(collisions) == 0

    def _update_FOV(self,agent_id):
        cone_idx = np.where(self.agents[agent_id].field_of_view)
        row_baseline = self.room.position.y + (self.room_dim[0] // 2) - 0.5
        col_baseline = self.room.position.x - (self.room_dim[1] // 2) + 0.5
        cone_world_center = [[col_baseline+c,row_baseline-r] for r,c in zip(cone_idx[0],cone_idx[1])] #x,y
        visable_in_cone = np.array([self.is_visable(cell_center,agent_id) for cell_center in cone_world_center]).astype(int)
        self.agents[agent_id].field_of_view[cone_idx] = visable_in_cone


    def calculate_collision(self,p1,p2,agent_id):
        collision_points = []
        #other agent (4 lines) and items (circle) --> can ask if center dist <= R and set center as collision
        other_agent_center = self.agents[1-agent_id].worldCenter
        dist = _get_point_dist_from_seg([p1[0],p1[1]], [p2[0],p2[1]], [other_agent_center[0],other_agent_center[1]])
        R = SIZE[self.sizes[1-agent_id]]
        is_on_ray = _get_dist(p1,p2) >= _get_dist(other_agent_center,p2) and \
                        _get_dist(p1, p2) >= _get_dist(other_agent_center, p1)
        if dist <= R and is_on_ray:
            collision_points.append(other_agent_center)
        #items
        for item_id in range(self.num_items):
            item_center = self.items[item_id].worldCenter
            dist = _get_point_dist_from_seg([p1[0],p1[1]], [p2[0],p2[1]], [item_center[0],item_center[1]])
            R = SIZE[self.sizes[self.num_agents+item_id]]
            is_on_ray = _get_dist(p1, p2) >= _get_dist(item_center, p2) and \
                            _get_dist(p1, p2) >= _get_dist(item_center, p1)
            if dist <= R and is_on_ray:
                collision_points.append(item_center)
        #maze walls
        room_center = self.room.position
        doors_pos = [(16, 1.4 * (self.room_dim[1] / 4) + 12), (-self.room_dim[0] / 4 + 16, 12),
                     (16, -self.room_dim[1] / 4 + 12), (self.room_dim[0] / 4 + 16, 12)]
        maze_walls = [(room_center, doors_pos[0]), (doors_pos[1], room_center),
                      (doors_pos[2], room_center), (room_center, doors_pos[3])]
        #room walls
        min_x, min_y, max_x, max_y = _get_room_bound(self.room)
        room_walls = [((min_x, min_y),(min_x, max_y)), ((max_x, min_y),(max_x, max_y)),
                      ((min_x, min_y),(max_x, min_y)), ((min_x, max_y),(max_x, max_y))]
        wall_intersections = [_get_segment_intersection(p1, p2, wall[0], wall[1]) for wall in maze_walls+room_walls]
        [collision_points.append((x,y)) for (x, y, valid, r, s) in wall_intersections if valid and r >= 0 and\
         _not_door(x,y,doors_pos) and _point_in_room((round(x,2),round(y,2)),self.room) and
         _get_dist(p1, p2) >= _get_dist((x,y), p2) and _get_dist(p1, p2) >= _get_dist((x,y), p1)]
        #return closest collision to p1
        # return collision_points[np.argmin([_get_dist(p1, col) for col in collision_points])]
        if len(collision_points) != 0:
            return [collision_points[np.argmin([_get_dist(p1, col) for col in collision_points])]]
        else:
            return collision_points


    def update_field_of_view(self):
        for agent_id, agent in enumerate(self.agents):
            self.set_field_of_view(agent)
            self._update_FOV(agent_id)

            # plt.imshow(agent.field_of_view)
            # plt.show()
            # print('over')


    def world_point_to_grid_cell(self, world_x, world_y):
        world_x, world_y = round(world_x,2), round(world_y,2)
        col_baseline = self.room.position.x - (self.room_dim[0] // 2)
        row_baseline = self.room.position.y + (self.room_dim[1] // 2)
        # print(col_baseline,row_baseline)
        grid_row, grid_col = math.ceil(row_baseline - world_y), math.ceil(world_x - col_baseline)
        if abs(world_y-row_baseline) != 0.0: #> 0.1:
            grid_row -= 1
        if abs(world_x-col_baseline) != 0.0: #> 0.1:
            grid_col -= 1
        # print(world_x, world_y, '-->',grid_row, grid_col)
        return [grid_row, grid_col]

    #update observations and beliefs
    def update_observations(self):
        for agent_id, agent in enumerate(self.agents):
            for other_agent_id, other_agent_pos in enumerate(self.agents_pos):
                world_vertices_pos = [self.agents[other_agent_id].GetWorldPoint(localPoint=v) \
                                      for v in self.agents[other_agent_id].fixtures[0].shape.vertices]
                agent_grid_cells = [self.world_point_to_grid_cell(x, y) \
                                    for x, y in world_vertices_pos + [(other_agent_pos.x, other_agent_pos.y)]]
                agent_is_observable = [agent.field_of_view[grid_cell[0],grid_cell[1]] for grid_cell in agent_grid_cells]
                #center or vertices in observed grid
                if sum(agent_is_observable) > 0:
                    if ('agent',other_agent_id) in agent.last_observations.keys():
                        agent.last_observations[('agent',other_agent_id)].append(other_agent_pos)
                    else:
                        agent.last_observations[('agent',other_agent_id)] = [other_agent_pos]
                    r, c = self.world_point_to_grid_cell(other_agent_pos.x, other_agent_pos.y)
                    agent.beliefs[other_agent_id].update(agent.field_of_view,
                                                         (other_agent_pos, self.agents[other_agent_id].angle,
                                                            [self.agents[other_agent_id].linearVelocity.x,
                                                             self.agents[other_agent_id].linearVelocity.y],
                                                            self.agents[other_agent_id].angularVelocity),
                                                         r,c)
                else:
                    agent.beliefs[other_agent_id].update(agent.field_of_view)
            for item_id, item_pos in enumerate(self.items_pos):
                cur_radius = SIZE[self.sizes[self.num_agents+item_id]]
                world_pos = [(item_pos.x, item_pos.y), (item_pos.x+cur_radius, item_pos.y),
                             (item_pos.x-cur_radius, item_pos.y), (item_pos.x, item_pos.y+cur_radius),
                             (item_pos.x, item_pos.y-cur_radius)]
                item_grid_cells = [self.world_point_to_grid_cell(x,y) for x,y in world_pos]
                item_is_observable = [agent.field_of_view[grid_cell[0],grid_cell[1]] for grid_cell in item_grid_cells]
                #perimeter or vertices in observed grid or item is attached to agent
                if sum(item_is_observable) > 0 or \
                        (('item',item_id) in self.touch_sensor[agent_id].keys() and \
                           self.touch_sensor[agent_id][('item',item_id)][1]):
                    if ('item',item_id) in agent.last_observations.keys():
                        agent.last_observations[('item',item_id)].append(item_pos)
                    else:
                        agent.last_observations[('item',item_id)] = [item_pos]
                    r, c = self.world_point_to_grid_cell(item_pos.x, item_pos.y)
                    agent.beliefs[self.num_agents+item_id].update(agent.field_of_view,
                                                                  (item_pos,self.items[item_id].angle,
                                                                     [self.items[item_id].linearVelocity.x,
                                                                      self.items[item_id].linearVelocity.y],
                                                                     self.items[item_id].angularVelocity),
                                                                  r,c)
                else:
                    agent.beliefs[self.num_agents+item_id].update(agent.field_of_view)


    def get_subgoals_goto(self, entity_id, goal_id):
        # print('subgoals_goto:', entity_id, goal_id)
        goal = GOAL[goal_id]
        if goal[0] == 'LMA':
            if self.env_id == 0:
                if self.attached[entity_id]:
                    return 56 + entity_id
            if self.get_reward_state(entity_id, self.get_state(entity_id), 'stop', None, None, goal_id) > -0.02:
                if self.attached[entity_id]:
                    return 56 + entity_id
                else:
                    return 58
        if self.env_id == 0: # when there are no obstacles
            return goal_id
        if goal[0] == 'RO':
            entity_pos = self.items_pos[goal[1]]
        else:
            if entity_id < self.num_agents:
                entity_pos = self.agents_pos[entity_id]
            else:
                entity_pos = self.items_pos[entity_id - self.num_agents]
        entity_room = self._get_room_id(entity_pos)
        if goal[0] == 'LMA':
            goal_room = self._get_room_id(self.landmark_centers[goal[2]])
        elif goal[0] in ['RA', 'RO']:
            goal_room = goal[2]
        else: # TE
            if goal[2] >= self.num_agents:
                goal_room = self._get_room_id(self.items_pos[0])
            else:
                goal_room = self._get_room_id(self.agents_pos[1 - entity_id])
        if entity_room == goal_room:
            return goal_id
        path = self.path[(entity_room, goal_room)]
        if path is None:
            return goal_id
        else:
            return 30 + path[0] * 3 + entity_id
