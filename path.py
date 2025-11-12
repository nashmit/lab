#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from matplotlib.pyplot import pause
from numpy.linalg import pinv

import time
from tools import setcubeplacement, getcubeplacement

from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from setup_meshcat import updatevisuals

rng = np.random.default_rng(123)  # optional seed

class GeneratedPath:

    def __init__(self, robot, cube, viz=None):
        self.robot = robot
        self.cube = cube
        self.G = []
        self.viz = viz
        self.dt = 1/30.0 if viz else 0

    def oMf_forFrameId(self, frameName):
        frameID = self.robot.model.getFrameId(frameName)
        return self.robot.data.oMf[frameID]

    def collision_check(self, q):
         '''Return true if q in collision, false otherwise.'''

         pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
         return pin.computeCollisions(robot.collision_model,robot.collision_data,False)

    def next_random_cube_configuration(self,
            q,
            cube_init_position, lowerTranslationOffsets, upperTranslationOffsets,
            checkcollision=True):
        
        """
        Return a random configuration, not/maybe in collision

        :param q: Some start configuration used for IK algorithm 
        :param cube_init_position: 
        :param lowerTranslationOffsets: 
        :param upperTranslationOffsets: 
        :param checkcollision: 
        :return: 
        """

        while True:
            translation_offsets = rng.uniform(low=lowerTranslationOffsets, high=upperTranslationOffsets)
            print(translation_offsets)
            offsets = pin.SE3(pin.Quaternion.Identity(),translation_offsets)

            # It also updates the cube's configuration!
            new_robot_q, success = computeqgrasppose(
                self.robot, q, self.cube, cube_init_position * offsets, viz=self.viz)

            print(success)

            if checkcollision:
                if success and not self.collision_check(new_robot_q):
                    return cube_init_position * offsets, new_robot_q, success
            else:
                return cube_init_position * offsets, new_robot_q, success
            
    def distance(self, q1, q2, norm=np.linalg.norm):
        '''Return the euclidian distance between two configurations'''
        return norm(q2 - q1)

    def distance_SE3(self, oMf_ob1, oMf_ob2, norm=np.linalg.norm):
        """

        :param oMf_ob1:
        :param oMf_ob2:
        :return:
        """

        return norm(pin.log(oMf_ob1.inverse()*oMf_ob2).vector)


    def nearest_vertex(self, cube_q_rand, min_dist = 10):
        '''returns the index of the Node of G with the configuration closest to cube_q_rand  '''

        idx = -1
        for (i, node) in enumerate(self.G):
            # the distance is computed using cube's C space (represented by an SE3 transform).
            # Actually, it's only based on the translation as the orientation is fixed.
            dist = self.distance_SE3(node[2], cube_q_rand)
            if dist < min_dist:
                min_dist = dist
                idx = i

        return idx


    def add_edge_and_vertex(self, parent_index, new_robot_q, oMf_cube_new):
        self.G += [(parent_index, new_robot_q, oMf_cube_new)]

    def lerp(self, q0, q1, t):
        return q0 * (1 - t) + q1 * t

    def new_conf(self, oMf_cube_near, oMf_cube_rand, discretisation_steps=1, delta_q=np.inf, render_attempts=False):

        """
        Return the closest oMf_cube_new (and the corresponding new_robot_q) s.t. the path
        oMf_cube_near => oMf_cube_new is the longest (using Pl¨ucker Coordinates) along the linear interpolation
        (oMf_cube_near, oMf_cube_rand) that is collision free and of length <  delta_q

        :param oMf_cube_near:
        :param oMf_cube_rand:
        :param discretisation_steps:
        :param delta_q:
        :param render_attempts:
        :return:
        """

        oMf_cube_end = oMf_cube_rand.copy()

        new_robot_q, success = computeqgrasppose(
            self.robot, q, self.cube, oMf_cube_end, viz=(self.viz if render_attempts else None))

        dist = self.distance_SE3(oMf_cube_near, oMf_cube_rand)

        if dist > delta_q:
            #compute the configuration that corresponds to a path of length delta_q in Pl¨ucker Coordinates
            #and then recompute the corresponding SE3 by exponentiation.
            oMf_cube_end = pin.exp(
                self.lerp(pin.log(oMf_cube_near).vector, pin.log(oMf_cube_rand).vector, delta_q / dist)
            )

        dt = 1.0 / discretisation_steps
        for i in range(1, discretisation_steps):
            oMf_cube_new = pin.exp(self.lerp(pin.log(oMf_cube_near).vector, pin.log(oMf_cube_end).vector, dt * i))

            # The cube position is updated too.
            new_robot_q, success = computeqgrasppose(
                self.robot, q, self.cube, oMf_cube_new, viz=(self.viz if render_attempts else None))

            collision_response = self.collision_check(new_robot_q)

            if not success or collision_response:

                oMf_cube_new = pin.exp(self.lerp(pin.log(oMf_cube_near), pin.log(oMf_cube_end), dt * (i - 1)))

                new_robot_q, success = computeqgrasppose(
                    self.robot, q, self.cube, oMf_cube_new, viz=(self.viz if render_attempts else None))

                assert success, "It should never happen as this was a valid solution previously!"

                return oMf_cube_new, new_robot_q

        return oMf_cube_end, new_robot_q


    def valid_edge(self, oMf_cube_new, oMf_cube_goal, discretisation_steps, eps=1e-3):

        oMf_cube_new, new_robot_q = self.new_conf(oMf_cube_new, oMf_cube_goal, discretisation_steps)

        return self.distance_SE3(oMf_cube_goal, oMf_cube_new) < eps

    def rrt(self, q_init, q_goal, cubeplacementq0, cubeplacementqgoal,
            k, delta_q, lowerTranslationOffsets, upperTranslationOffsets, discretisation_steps):

        self.G = [(None, q_init.copy(), cubeplacementq0.copy())]

        for _ in range(k):

            oMf_cube_rand, new_robot_q, success = self.next_random_cube_configuration(
                q=q_init,
                cube_init_position=cubeplacementq0,
                lowerTranslationOffsets=lowerTranslationOffsets,
                upperTranslationOffsets=upperTranslationOffsets,
                checkcollision=True
            )

            assert success, "This should never happen! The previous function should return only if a new configuration is identified!"

            # The distance is computed using cube's configuration.
            q_near_index = self.nearest_vertex(oMf_cube_rand)

            q_near = self.G[q_near_index][1]
            oMf_cube_near = self.G[q_near_index][2]

            oMf_cube_new, new_robot_q = self.new_conf(
                oMf_cube_near=oMf_cube_near,
                oMf_cube_rand=oMf_cube_rand,
                discretisation_steps=discretisation_steps,
                delta_q = delta_q
            )

            self.add_edge_and_vertex(
                parent_index=q_near_index,new_robot_q=new_robot_q, oMf_cube_new=oMf_cube_new)

            if self.valid_edge(
                    oMf_cube_new=oMf_cube_new,
                    oMf_cube_goal=cubeplacementqgoal,
                    discretisation_steps=discretisation_steps):
                print ("Path found!")
                self.add_edge_and_vertex(parent_index=len(self.G)-1, new_robot_q=q_goal, oMf_cube_new=cubeplacementqgoal)

                return self.G, True

        print("path not found")
        return self.G, False

    def getpath(self):
        path = []
        node = self.G[-1]
        while node[0] is not None:
            path = [node[1]] + path
            node = self.G[node[0]]
        path = [self.G[0][1]] + path
        return path

    def displaypath(self):

        path = self.getpath()

        for q in path:
            if self.viz:
                updatevisuals(viz,self.robot,self.cube,q)
                time.sleep(self.dt)


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal, viz=None):

    path_generator = GeneratedPath(robot=robot, cube=cube, viz=viz)

    # lowerTranslationOffsets = np.array([-0.2, -0.1, 0])
    # upperTranslationOffsets = np.array([0.3, 0.5, 0.3])

    lowerTranslationOffsets = np.array([-0.2, -0.1, -0.1])
    upperTranslationOffsets = np.array([0.3, 0.5, 0.3])


    _, success = path_generator.rrt(
        q_init=qinit,
        q_goal=qgoal,
        cubeplacementq0=cubeplacementq0,
        cubeplacementqgoal=cubeplacementqgoal,
        k=2000,
        delta_q=0.1,
        lowerTranslationOffsets=lowerTranslationOffsets,
        upperTranslationOffsets=upperTranslationOffsets,
        discretisation_steps=10)

    assert success, ":)  There is nothing to see here."

    pause(5)
    path_generator.displaypath()

    updatevisuals(viz, path_generator.robot, path_generator.cube, qgoal)

    # return [qinit, qgoal]
    return  path_generator.getpath()

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    q = robot.q0.copy()

    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    else:
        print ("Valid initial AND end configuration")
    
    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, viz=viz)
    
    # displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt