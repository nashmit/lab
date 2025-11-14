#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON


import numpy as np
import pinocchio as pin

from bezier import Bezier

from scipy.optimize import lsq_linear


rng = np.random.default_rng(1)  # optional seed



def build_trajectory(path_in_joint_space, T):

    q_of_t = Bezier(
        [path_in_joint_space[0]] +
        [path_in_joint_space[0]] +
        path_in_joint_space +
        [path_in_joint_space[-1]] +
        [path_in_joint_space[-1]],
        t_max=T)

    dq_of_t = q_of_t.derivative(1)
    ddq_of_t = dq_of_t.derivative(1)

    return q_of_t, dq_of_t, ddq_of_t

def adjusting_time_trajectory(
        robot,
        path,
        initial_guess_trajs,
        initial_guess_total_time,
        ok_consecutive_samples_to_stop=10
):
    """
    Since this controller can't look into the future ( like an MPC would ) I can't really optimize/plan for time.
    However, I can do this small hack to increase the chance for a feasible path if the optimal one is not accessible.

    :param robot:
    :param initial_guess_trajs:
    :param initial_guess_total_time:
    :param samples_number:
    :return:
    """

    T = initial_guess_total_time

    q_min = robot.model.lowerPositionLimit
    q_max = robot.model.upperPositionLimit
    dq_min = -robot.model.velocityLimit
    dq_max =  robot.model.velocityLimit

    number_ok_consecutive_samples=0

    while(1):

        time_idx = np.random.uniform(0, T,1)

        current_q = initial_guess_trajs[0](time_idx)
        current_dq= initial_guess_trajs[1](time_idx)

        # in theory, this test is not necessary... it is only here for debugging reason.
        ok_position = np.all(
            (current_q >= q_min) &
            (current_q <= q_max)
        )

        # the scaling affects the velocity only...
        ok_velocity = np.all(
            (current_dq >= dq_min) &
            (current_dq <= dq_max)
        )

        if not ok_position or not ok_velocity:
            T *=1.2
            initial_guess_trajs = build_trajectory(path, T)
            number_ok_consecutive_samples=0
        else:
            number_ok_consecutive_samples+=1

        if number_ok_consecutive_samples > ok_consecutive_samples_to_stop:
            break

    return initial_guess_trajs, T



Kp = 70300.              # proportional gain (P of PD)
Kd = 3 * np.sqrt(Kp)   # derivative gain   (D of PD)

##LS weights

#for tracking the ddq
w_track = 1e+1
#keep contact forces small
w_f     = 1e-1
#regularise torques
w_tau   = 1e-5
#enforce dynamics strongly
w_dyn   = 5e+6
#enforce friction constraints
w_fric  = 1e+5
#discourage slack
w_s     = 1e+1
#friction coefficient
mu = 0.7

N_max =140  # max normal force (in Newton)


# maybe I need to switch the sign here...
tangent = np.array([0, 0, -1])  # np.array shape (3,), unit tangent in contact frame
normal = np.array([0, 1,  0])  # np.array shape (3,), unit normal in contact frame


def controllaw_with_friction(sim, robot, trajs, tcurrent, cube, DT):

    # ===============================================================
    # Inverse dynamics with contact forces as a least-squares problem
    # ===============================================================
    #
    # Goal:
    # -----
    # At each control step, we want to find:
    #
    #   - joint accelerations:  ddq  (shape: n)
    #   - joint torques:        tau  (shape: n)
    #   - contact forces:       T_L, N_L, T_R, N_R  (tangent & normal at left/right contacts)
    #
    # such that:
    #
    #   - we track a desired joint acceleration ddq_d (feedforward + PD),
    #   - we respect the joint-space dynamics (including contact forces),
    #   - we enforce friction constraints to maintain sticking contact,
    #   - we stay within joint/ joint velocity limits.
    #
    #
    # 1) Joint-space dynamics with contact forces
    # -------------------------------------------
    # The manipulator dynamics with two contacts (L, R) are:
    #
    #   tau = M(q) * ddq + h(q, dq) + J_L(q)^T * F_cL + J_R(q)^T * F_cR
    #
    # where:
    #   - M(q)   : n x n joint-space inertia matrix
    #   - h      : n-vector (Coriolis/centrifugal + gravity terms)
    #   - J_L, J_R: contact Jacobians (projected to 2D: tangent & normal) with shape (2 x n)
    #   - F_cL   : 2D contact force at left contact  [T_L, N_L]^T
    #   - F_cR   : 2D contact force at right contact [T_R, N_R]^T
    #   - mu     : the friction coefficient is considered known
    #
    # To make the Jacobians 2D, we:
    #
    #   - choose tangent (t) and normal (n) directions at each contact in the LOCAL contact frame,
    #   - extract the linear part of the full 6D Jacobian,
    #   - project it with S = [t^T; n^T] to get J_2D = S * J_linear.
    #
    #
    # 2) Desired joint acceleration (feedforward + feedback)
    # ------------------------------------------------------
    # We have a reference joint trajectory (q_ref, dq_ref, ddq_ref) from interpolation.
    # At each step, we define the desired acceleration with PD feedback:
    #
    #   e_q = q - q_ref
    #   e_v = dq - dq_ref
    #
    #   ddq_d = ddq_ref - Kp * e_q - Kd * e_v
    #
    # where Kp and Kd are diagonal or full PD gain matrices.
    #
    # We want the actual ddq to be close to ddq_d.
    #
    #
    # 3) Decision vector x
    # --------------------
    # We stack all unknowns into a single vector x:
    #
    #   x = [ ddq,
    #         tau,
    #         T_L, N_L, T_R, N_R,
    #         s_L_plus, s_L_minus, s_R_plus, s_R_minus ]^T
    #
    # Dimensions:
    #   - ddq: n
    #   - tau: n
    #   - contact forces: 4 scalars  (T_L, N_L, T_R, N_R)
    #   - slacks: 4 scalars  (s_L+, s_L-, s_R+, s_R-)
    #
    # So:
    #   x has dimension 2n + 8.
    #
    #
    # 4) Turning equality and inequality constraints into residuals
    # -------------------------------------------------------------
    #
    # We want to write the problem as:
    #
    #   minimize  ||A x - b||^2
    #   subject to l <= x <= u
    #
    # This is pure least-squares with simple bounds. ( IT TURNS OUT, I SHOULD HAVE KEPT SOME OF THE CONSTRAINTS AS HARD CONSTRAINTS
    # AND USE A QP... BUT YEA...
    #
    # The idea is:
    #   - Every "soft" equality is turned into a residual r = A_row * x - b_row,
    #     added to the cost with a weight.
    #   - Every inequality that we want to keep as inequality becomes either:
    #       - a bound on x (for simple variable bounds), or
    #       - an equality with slack + penalty (our friction constraints).
    #
    #
    # 4.1 Tracking residual (ddq ~ ddq_d)
    # -----------------------------------
    # We want to minimise:
    #
    #   w_track * ||ddq - ddq_d||^2
    #
    # This is a residual of size n:
    #
    #   r_track = sqrt(w_track) * (ddq - ddq_d)
    #
    # In matrix form:
    #   r_track = A_track * x - b_track
    #
    # where:
    #   A_track = sqrt(w_track) * [ I_n, 0, 0_4, 0_4 ]
    #   b_track = sqrt(w_track) * ddq_d
    #
    # Here the blocks are:
    #   - I_n  : n x n on ddq
    #   - 0    : n x n on tau
    #   - 0_4  : n x 4 on [T_L, N_L, T_R, N_R]
    #   - 0_4  : n x 4 on [s_L+, s_L-, s_R+, s_R-]
    #
    #
    # 4.2 Dynamics residual (enforce tau ~ M ddq + h + J^T F)
    # -------------------------------------------------------
    # Original dynamics:
    #
    #   tau = M ddq + h + J_L^T [T_L, N_L]^T + J_R^T [T_R, N_R]^T
    #
    # Rearranged as:
    #
    #   tau - M ddq - J_L^T F_cL - J_R^T F_cR = h
    #
    # Define:
    #
    #   r_dyn = sqrt(w_dyn) * (tau - M ddq - J_L^T F_cL - J_R^T F_cR - h)
    #
    # This is a residual of size n. In matrix form:
    #
    #   r_dyn = A_dyn * x - b_dyn
    #
    # where:
    #   A_dyn = sqrt(w_dyn) * [ -M, I_n, -J_L^T, -J_R^T, 0_(n x 4) ]
    #   b_dyn = sqrt(w_dyn) * h
    #
    # Here:
    #   - -M      multiplies ddq
    #   - I_n     multiplies tau
    #   - -J_L^T  multiplies [T_L, N_L]
    #   - -J_R^T  multiplies [T_R, N_R]
    #   - 0       ignores slacks
    #
    #
    # 4.3 Contact force regularization (keep forces small)
    # ----------------------------------------------------
    # We want to discourage very large contact forces by penalising:
    #
    #   w_f * (T_L^2 + N_L^2 + T_R^2 + N_R^2)
    #
    # Define a 4D residual:
    #
    #   r_f = sqrt(w_f) * [T_L, N_L, T_R, N_R]^T
    #
    # In matrix form:
    #
    #   A_f = sqrt(w_f) * [ 0_(4 x n), 0_(4 x n), I_4, 0_(4 x 4) ]
    #   b_f = 0_4
    #
    # where I_4 acts on [T_L, N_L, T_R, N_R]^T.
    #
    #
    # 4.4 Torque regularisation
    # -------------------------
    # Similar idea for torques:
    #
    #   w_tau * ||tau||^2
    #
    # Residual:
    #
    #   r_tau = sqrt(w_tau) * tau
    #
    # In matrix form:
    #
    #   A_tau = sqrt(w_tau) * [ 0_(n x n), I_n, 0_(n x 4), 0_(n x 4) ]
    #   b_tau = 0_n
    #
    #
    # 4.5 Friction constraints as equalities with slacks
    # --------------------------------------------------
    # Coulomb friction (2D model) at each contact i in its local frame:
    #
    #   N_i >= 0        (unilaterality)
    #   |T_i| <= mu * N_i   (friction cone)
    #
    # We enforce N_i >= 0 as a bound on x directly (see bounds below).
    #
    # For the two friction inequalities we introduce slacks s_i >= 0:
    #
    #   T_L - mu N_L + s_L+ = 0
    #  -T_L - mu N_L + s_L- = 0
    #
    #   T_R - mu N_R + s_R+ = 0
    #  -T_R - mu N_R + s_R- = 0
    #
    # Then we penalise these equalities:
    #
    #   w_fric * (r_{L+}^2 + r_{L-}^2 + r_{R+}^2 + r_{R-}^2)
    #
    # where they are:
    #
    #   r_{L+} = T_L - mu N_L + s_L+
    #   r_{L-} = -T_L
    ################################################################

    # Dimensions
    n = robot.model.nv  # total joints ( dof )
    n_x = 2 * n + 8  # ddq, tau, TL,NL,TR,NR, four slacks

    # x = [ddq (n), tau (n), T_L, N_L, T_R, N_R, s_L+, s_L-, s_R+, s_R-]

    #curren state
    q, vq = sim.getpybulletstate()

    #trajectory reference
    traj_q  = trajs[0](tcurrent)
    traj_dq = trajs[1](tcurrent)
    traj_ddq= trajs[2](tcurrent)

    e_q = q  - traj_q
    e_v = vq - traj_dq

    #accelertion desired
    ddq_d = traj_ddq - Kp * e_q - Kd

    pin.forwardKinematics(robot.model, robot.data, q, vq)
    M = pin.crba(robot.model, robot.data, q)
    h = pin.rnea(robot.model, robot.data, q, vq, np.zeros_like(vq))
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    frame_id_L = robot.model.getFrameId(LEFT_HAND)
    frame_id_R = robot.model.getFrameId(RIGHT_HAND)

    J_full_L = pin.getFrameJacobian(robot.model, robot.data, frame_id_L, pin.ReferenceFrame.LOCAL)
    J_full_R = pin.getFrameJacobian(robot.model, robot.data, frame_id_R, pin.ReferenceFrame.LOCAL)

    J_lin_L = J_full_L[3:6, :]
    J_lin_R = J_full_R[3:6, :]

    S = np.vstack([tangent, normal])
    J_L_2d = S @ J_lin_L  # shape (2, n)
    J_R_2d = S @ J_lin_R  # shape (2, n)

    sf_track = np.sqrt(w_track)
    # sf_f = np.sqrt(w_f)
    sf_tau = np.sqrt(w_tau)
    sf_dyn = np.sqrt(w_dyn)
    sf_fric = np.sqrt(w_fric)
    sf_s = np.sqrt(w_s)

    #get friction coefficient
    # I need to query the simulation for the friction coefficient of the cube.
    # Meanwhile, I am using a constant as a placeholder. (above)

    #build the "A" and "b" for LLS
    A_rows = []
    b_rows = []

    I_n = np.eye(n)
    Z_n_n = np.zeros((n, n))
    Z_n_4 = np.zeros((n, 4))

    A_track = sf_track * np.hstack([I_n, Z_n_n, Z_n_4, Z_n_4])
    b_track = ddq_d.copy()

    A_rows.append(A_track)
    b_rows.append(b_track)

    ###########

    A_tau = np.hstack([
        Z_n_n,  # ddq
        sf_tau * I_n,  # tau
        Z_n_4,  # contact forces
        Z_n_4  # slacks
    ])
    b_tau = np.zeros(n)

    A_rows.append(A_tau)
    b_rows.append(b_tau)

    ################

    JL_T = J_L_2d.T  # shape (n,2)
    JR_T = J_R_2d.T  # shape (n,2)

    A_fc_dyn = np.hstack([-JL_T, -JR_T])  # (n,4)

    A_dyn = sf_dyn * np.hstack([
        -M,  # ddq
        I_n,  # tau
        A_fc_dyn,  # [T_L,N_L,T_R,N_R]
        Z_n_4  # slacks
    ])
    b_dyn = sf_dyn * h

    A_rows.append(A_dyn)
    b_rows.append(b_dyn)

    #######################

    A_fc_fric = np.array([
        [1.0, -mu, 0.0, 0.0],  # r_{L,+}
        [-1.0, -mu, 0.0, 0.0],  # r_{L,-}
        [0.0, 0.0, 1.0, -mu],  # r_{R,+}
        [0.0, 0.0, -1.0, -mu],  # r_{R,-}
    ])

    A_slack_fric = np.array([
        [1.0, 0.0, 0.0, 0.0],  # s_L+
        [0.0, 1.0, 0.0, 0.0],  # s_L-
        [0.0, 0.0, 1.0, 0.0],  # s_R+
        [0.0, 0.0, 0.0, 1.0],  # s_R-
    ])

    Z_4_n = np.zeros((4, n))

    A_fric = sf_fric * np.hstack([
        Z_4_n,  # ddq
        Z_4_n,  # tau
        A_fc_fric,  # [T_L,N_L,T_R,N_R]
        A_slack_fric  # [s_L+,s_L-,s_R+,s_R-]
    ])
    b_fric = np.zeros(4)

    A_rows.append(A_fric)
    b_rows.append(b_fric)

    #################

    Z_4_n = np.zeros((4, n))
    Z_4_4 = np.zeros((4, 4))

    A_slack_reg = sf_s * np.hstack([
        Z_4_n,  # ddq
        Z_4_n,  # tau
        Z_4_4,  # contact forces
        np.eye(4)  # slacks
    ])
    b_slack_reg = np.zeros(4)

    A_rows.append(A_slack_reg)
    b_rows.append(b_slack_reg)

    A = np.vstack(A_rows)  # shape (m, 2n+8)
    b = np.concatenate(b_rows)  # shape (m,)

    #####################

    q_min = robot.model.lowerPositionLimit
    q_max = robot.model.upperPositionLimit
    v_min = -robot.model.velocityLimit
    v_max =  robot.model.velocityLimit
    tau_min = -robot.model.effortLimit
    tau_max =  robot.model.effortLimit

    ####################

    idx_ddq = slice(0, n)  # ddq
    idx_tau = slice(n, 2 * n)  # tau

    idx_TL = 2 * n + 0
    idx_NL = 2 * n + 1
    idx_TR = 2 * n + 2
    idx_NR = 2 * n + 3

    idx_sLp = 2 * n + 4
    idx_sLm = 2 * n + 5
    idx_sRp = 2 * n + 6
    idx_sRm = 2 * n + 7

    l = -np.inf * np.ones(n_x)
    u = np.inf * np.ones(n_x)

    ddq_min_vel = (v_min - vq) / DT
    ddq_max_vel = (v_max - vq) / DT

    ddq_min_pos = 2.0 * (q_min - q - DT * vq) / (DT ** 2)
    ddq_max_pos = 2.0 * (q_max - q - DT * vq) / (DT ** 2)

    ddq_min = np.maximum(ddq_min_vel, ddq_min_pos)
    ddq_max = np.minimum(ddq_max_vel, ddq_max_pos)

    l[idx_ddq] = ddq_min
    u[idx_ddq] = ddq_max

    l[idx_tau] = tau_min
    u[idx_tau] = tau_max

    l[idx_NL] = 0.0
    u[idx_NL] = N_max

    l[idx_NR] = 0.0
    u[idx_NR] = N_max

    # not required, but helped clipping huge outliers in infeasible cases :)
    T_max = 2 * mu * N_max

    l[idx_TL] = -T_max
    u[idx_TL] = T_max

    l[idx_TR] = -T_max
    u[idx_TR] = T_max

    l[idx_sLp] = 0.0
    l[idx_sLm] = 0.0
    l[idx_sRp] = 0.0
    l[idx_sRm] = 0.0
    # u[...] can stay +inf; the cost term w_s * s^2 keeps them small.

    #########################

    res = lsq_linear(A, b, bounds=(l, u), method='trf')
    X = res.x
    torques = X[slice(n, 2 * n)]

    sim.step(torques)


def controllaw(sim, robot, trajs, tcurrent, cube, DT):

    """
    The simpler solution....
    """

    # Dimensions
    n = robot.model.nv  # total joints ( dof )
    n_x = 2 * n # ddq, tau

    # x = [ddq (n), tau (n)]

    #curren state
    q, vq = sim.getpybulletstate()

    #trajectory reference
    traj_q  = trajs[0](tcurrent)
    traj_dq = trajs[1](tcurrent)
    traj_ddq= trajs[2](tcurrent)

    e_q = q  - traj_q
    e_v = vq - traj_dq

    #accelertion desired
    ddq_d = traj_ddq - Kp * e_q - Kd

    pin.forwardKinematics(robot.model, robot.data, q, vq)
    M = pin.crba(robot.model, robot.data, q)
    h = pin.rnea(robot.model, robot.data, q, vq, np.zeros_like(vq))
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    sf_track = np.sqrt(w_track)
    # sf_f = np.sqrt(w_f)
    sf_tau = np.sqrt(w_tau)
    sf_dyn = np.sqrt(w_dyn)
    sf_fric = np.sqrt(w_fric)
    sf_s = np.sqrt(w_s)

    #build the "A" and "b" for LLS
    A_rows = []
    b_rows = []

    I_n = np.eye(n)
    Z_n_n = np.zeros((n, n))
    Z_n_4 = np.zeros((n, 4))

    A_track = sf_track * np.hstack([
        I_n,
        Z_n_n,
        # Z_n_4,
        # Z_n_4
    ])
    b_track = ddq_d.copy()

    A_rows.append(A_track)
    b_rows.append(b_track)

    ###########

    A_tau = np.hstack([
        Z_n_n,  # ddq
        sf_tau * I_n,  # tau
        # Z_n_4,  # contact forces
        # Z_n_4  # slacks
    ])
    b_tau = np.zeros(n)

    A_rows.append(A_tau)
    b_rows.append(b_tau)

    ################

    # JL_T = J_L_2d.T  # shape (n,2)
    # JR_T = J_R_2d.T  # shape (n,2)
    #
    # A_fc_dyn = np.hstack([-JL_T, -JR_T])  # (n,4)

    A_dyn = sf_dyn * np.hstack([
        -M,  # ddq
        I_n,  # tau
        # A_fc_dyn,  # [T_L,N_L,T_R,N_R]
        # Z_n_4,
        # Z_n_4  # slacks
    ])
    b_dyn = sf_dyn * h

    A_rows.append(A_dyn)
    b_rows.append(b_dyn)

    #######################

    # A_fc_fric = np.array([
    #     [1.0, -mu, 0.0, 0.0],  # r_{L,+}
    #     [-1.0, -mu, 0.0, 0.0],  # r_{L,-}
    #     [0.0, 0.0, 1.0, -mu],  # r_{R,+}
    #     [0.0, 0.0, -1.0, -mu],  # r_{R,-}
    # ])
    #
    # A_slack_fric = np.array([
    #     [1.0, 0.0, 0.0, 0.0],  # s_L+
    #     [0.0, 1.0, 0.0, 0.0],  # s_L-
    #     [0.0, 0.0, 1.0, 0.0],  # s_R+
    #     [0.0, 0.0, 0.0, 1.0],  # s_R-
    # ])
    #
    # Z_4_n = np.zeros((4, n))
    #
    # A_fric = sf_fric * np.hstack([
    #     Z_4_n,  # ddq
    #     Z_4_n,  # tau
    #     A_fc_fric,  # [T_L,N_L,T_R,N_R]
    #     A_slack_fric  # [s_L+,s_L-,s_R+,s_R-]
    # ])
    # b_fric = np.zeros(4)
    #
    # A_rows.append(A_fric)
    # b_rows.append(b_fric)

    #################

    # Z_4_n = np.zeros((4, n))
    # Z_4_4 = np.zeros((4, 4))
    #
    # A_slack_reg = sf_s * np.hstack([
    #     Z_4_n,  # ddq
    #     Z_4_n,  # tau
    #     Z_4_4,  # contact forces
    #     np.eye(4)  # slacks
    # ])
    # b_slack_reg = np.zeros(4)
    #
    # A_rows.append(A_slack_reg)
    # b_rows.append(b_slack_reg)

    A = np.vstack(A_rows)  # shape (m, 2n+8)
    b = np.concatenate(b_rows)  # shape (m,)

    #####################

    q_min = robot.model.lowerPositionLimit
    q_max = robot.model.upperPositionLimit
    v_min = -robot.model.velocityLimit
    v_max =  robot.model.velocityLimit
    tau_min = -robot.model.effortLimit
    tau_max =  robot.model.effortLimit

    ####################

    idx_ddq = slice(0, n)  # ddq
    idx_tau = slice(n, 2 * n)  # tau

    # idx_TL = 2 * n + 0
    # idx_NL = 2 * n + 1
    # idx_TR = 2 * n + 2
    # idx_NR = 2 * n + 3
    #
    # idx_sLp = 2 * n + 4
    # idx_sLm = 2 * n + 5
    # idx_sRp = 2 * n + 6
    # idx_sRm = 2 * n + 7

    l = -np.inf * np.ones(n_x)
    u = np.inf * np.ones(n_x)

    ddq_min_vel = (v_min - vq) / DT
    ddq_max_vel = (v_max - vq) / DT

    ddq_min_pos = 2.0 * (q_min - q - DT * vq) / (DT ** 2)
    ddq_max_pos = 2.0 * (q_max - q - DT * vq) / (DT ** 2)

    ddq_min = np.maximum(ddq_min_vel, ddq_min_pos)
    ddq_max = np.minimum(ddq_max_vel, ddq_max_pos)

    l[idx_ddq] = ddq_min
    u[idx_ddq] = ddq_max

    l[idx_tau] = tau_min
    u[idx_tau] = tau_max

    # l[idx_NL] = 0.0
    # u[idx_NL] = N_max
    #
    # l[idx_NR] = 0.0
    # u[idx_NR] = N_max

    # not required, but helped clipping huge outliers in infeasible cases :)
    # T_max = 2 * mu * N_max
    #
    # l[idx_TL] = -T_max
    # u[idx_TL] = T_max
    #
    # l[idx_TR] = -T_max
    # u[idx_TR] = T_max
    #
    # l[idx_sLp] = 0.0
    # l[idx_sLm] = 0.0
    # l[idx_sRp] = 0.0
    # l[idx_sRm] = 0.0
    # u[...] can stay +inf; the cost term w_s * s^2 keeps them small.

    #########################

    res = lsq_linear(A, b, bounds=(l, u), method='trf')
    X = res.x
    torques = X[slice(n, 2 * n)]

    sim.step(torques)



if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    
    #setting initial configuration
    sim.setqsim(q0)

    initial_guess_total_time = 15

    initial_guess_trajs = build_trajectory(
        path_in_joint_space=path, T=initial_guess_total_time
    )

    trajs, total_time = adjusting_time_trajectory(
        robot=robot,
        path=path,
        initial_guess_trajs=initial_guess_trajs,
        initial_guess_total_time=initial_guess_total_time,
        ok_consecutive_samples_to_stop=1000
    )

    tcur = 0.
    while tcur < total_time:
        # I added "DT" again because I need it to enforce constraints over "qqd" based on "q" and "qd".
        # I could have modified the function, but I didn't want to mess up with the automatic testing,
        # you might have in place!
        rununtil(controllaw_with_friction, DT, sim, robot, trajs, tcur, cube, DT)
        tcur += DT