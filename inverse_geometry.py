#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from scipy.optimize import lsq_linear

from numpy.linalg import norm
from time import sleep

from tools import setcubeplacement

from setup_meshcat import updatevisuals

def oneStepIK_2arms_simultaneously(
        J_LH, J_RH,
        q, q_min, q_max,
        vq_min, vq_max,
        v_LH, v_RH,
        rho, mu, W, lambda_, q_posture, tol=1e-10
):
    #not essential, just keeping the notation clean for future use
    rho = np.sqrt(rho)

    # A_LH = np.vstack([rho * J_LH, mu * W, lambda_ * np.eye(q.size)])
    # b_LH = np.concatenate([rho * v_LH, mu * W @ (q_posture - q), np.zeros(q.size)])
    #
    # A_RH = np.vstack([rho * J_RH, mu * W, lambda_ * np.eye(q.size)])
    # b_RH = np.concatenate([rho * v_RH, mu * W @ (q_posture - q), np.zeros(q.size)])
    #
    # A = np.vstack([A_LH, A_RH])
    # b = np.concatenate([b_LH, b_RH])

    A = np.vstack([rho * J_LH, rho * J_RH, mu * W, lambda_ * np.eye(q.size)])
    b = np.concatenate([rho * v_LH, rho * v_RH, mu * W @ (q_posture - q), np.zeros(q.size)])

    # I take into consideration both the velocity limits as well as the configuration limits.
    dq_min = np.maximum(q_min - q, vq_min)
    dq_max = np.minimum(q_max - q, vq_max)

    res = lsq_linear(A, b, bounds=(dq_min, dq_max), method='trf', tol=tol)
    delta_q = res.x

    return delta_q

def IK_loop_2arms_simultaneously(
        robot, cube,
        q, q_min, q_max, vq_min, vq_max,
        frameID_current_LH, oMf_target_LH, frameID_current_RH, oMf_target_RH,
        rho, mu, W, lambda_, q_posture, error_stop=1e-3, max_steps=2000, alpha=0.3,
        viz = None, sleep_time=0.0
):

    success = False

    while(1):

        pin.forwardKinematics(model=robot.model, data=robot.data, q=q)
        pin.updateFramePlacements(model=robot.model, data=robot.data)
        pin.computeJointJacobians(robot.model, robot.data, q)

        oMf_current_LH = robot.data.oMf[frameID_current_LH]
        oMf_current_RH = robot.data.oMf[frameID_current_RH]

        tool_error_LH = pin.log(oMf_current_LH.inverse() * oMf_target_LH).vector
        tool_error_RH = pin.log(oMf_current_RH.inverse() * oMf_target_RH).vector

        # I could also consider: λ*∥W(q−q_posture)∥^2 as part of the stopping criteria...
        if norm(tool_error_LH) + norm(tool_error_RH) <= error_stop:
            success = True
            break

        tool_Jtool_LH = pin.computeFrameJacobian(robot.model, robot.data, q, frameID_current_LH)
        tool_Jtool_RH = pin.computeFrameJacobian(robot.model, robot.data, q, frameID_current_RH)

        delta_q = oneStepIK_2arms_simultaneously(
            J_LH=tool_Jtool_LH, J_RH=tool_Jtool_RH,
            q=robot.q0, q_min=q_min, q_max=q_max,
            vq_min=vq_min, vq_max=vq_max,
            v_LH=tool_error_LH, v_RH=tool_error_RH,
            rho=rho,
            mu=mu, W=W, lambda_=lambda_, q_posture=q_posture
        )

        # q += alpha * delta_q
        # not really a necessity for this homework; however it's more general.
        q = pin.integrate(robot.model, q, alpha*delta_q)

        sleep(sleep_time)
        if viz:
            updatevisuals(viz, robot, cube, q)

        max_steps-=1
        if max_steps <=0:
            break

    return q, success

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    frameID_tool_LH = robot.model.getFrameId(LEFT_HAND)
    frameID_tool_RH = robot.model.getFrameId(RIGHT_HAND)

    oMcube_LH_hook = getcubeplacement(cube, LEFT_HOOK)
    oMcube_RH_hook = getcubeplacement(cube, RIGHT_HOOK)

    size = qcurrent.size

    # The selected parameters are not optimized with a particular objective in mind other than reaching the end goal.
    # However, the numerical dumping term and the posture bias are fully functional.
    # I also added the configuration bonds just to make sure we don't end up with a final solution that is not plausible.
    # I also added the velocity bounds... as an exercises for the next tasks.

    # In this homework, I'm using IK with the same meaning as one can find in the literature.

    #todo add the equation for clarity!

    q_target, success = IK_loop_2arms_simultaneously(
        robot=robot,
        cube=cube,
        q=qcurrent,
        q_min=robot.model.lowerPositionLimit,
        q_max=robot.model.upperPositionLimit,
        vq_min=-robot.model.velocityLimit,
        vq_max= robot.model.velocityLimit,
        frameID_current_LH=frameID_tool_LH,
        oMf_target_LH=oMcube_LH_hook,
        frameID_current_RH = frameID_tool_RH,
        oMf_target_RH = oMcube_RH_hook,
        rho=4,
        mu=0.01,
        W=np.identity(size),
        lambda_=0.3,
        q_posture=qcurrent,
        alpha=1,
        viz=viz,
        sleep_time=(1/1000.0 if viz else 0)
    )

    return q_target, success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat

    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    updatevisuals(viz, robot, cube, q0)

    assert successinit == True

    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    updatevisuals(viz, robot, cube, qe)

    assert successend == True