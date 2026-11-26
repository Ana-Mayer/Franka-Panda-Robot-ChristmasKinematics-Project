"""
Straight-line gripper moves made with our own inverse kinematics (IK), and
putting the spoon back in its holder with them.

A move goes from where the gripper is now to a target POSE (position + orientation),
not to taught joint angles:
  1. Interpolate the pose: the position along a straight line, the orientation
     turning about one fixed axis (SE3.interp).
  2. Time it with the same quintic as the shelf turn (gentle start and stop).
  3. Solve IK at every time step, starting from the previous step's joint angles.
  4. Check the result against the robot server's limits before it runs.
"""

import numpy as np
from spatialmath import SE3

from shelf_turn import DT, check_trajectory, smooth_fraction, solve_ik

LINE_SPEED = 0.05        # top speed of the gripper along a line [m/s]
TURN_SPEED = 0.5         # top speed of the gripper's rotation [rad/s]

HOLDER_CLEARANCE = 0.12  # how far above its holder [m] the spoon is lowered from
SPOON_INSERT_SPEED = 0.03  # slower while the spoon goes into the holder [m/s]
BACK_OFF = 0.05          # how far the gripper backs away after letting go [m]


def rotation_angle(R):
    """The angle [rad] of a rotation matrix: how far it turns, about whatever axis."""
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def plan_line(panda, q_start, T_goal, speed=LINE_SPEED, turn_speed=TURN_SPEED, dt=DT):
    """
    Joint trajectory that moves the gripper in a straight line from FK(q_start) to T_goal.
    Returns (q_traj, dt).
    """
    q = np.asarray(q_start, dtype=float)
    T_start = panda.fkine(q)
    distance = np.linalg.norm(T_goal.t - T_start.t)
    angle = rotation_angle(T_start.R.T @ T_goal.R)
    # long enough for both the distance and the turn; the quintic peaks at 1.875x the average speed
    duration = max(1.0, 1.875 * distance / speed, 1.875 * angle / turn_speed)
    n = int(np.ceil(duration / dt))

    q_traj = [q]
    for k in range(1, n + 1):
        T = T_start.interp(T_goal, smooth_fraction(k / n))   # pose part of the way along the line
        q = solve_ik(panda, T, q)                             # seed = previous step: no jumps
        q_traj.append(q)
    return np.array(q_traj), dt


def move_line(robot, panda, T_goal, name, speed=LINE_SPEED, turn_speed=TURN_SPEED):
    """Plan a straight-line IK move from where the robot is now, check it, and run it."""
    q_now = robot.get_current_joint_positions()
    q_traj, dt = plan_line(panda, q_now, T_goal, speed, turn_speed)
    check_trajectory(panda, q_traj, dt, name)
    robot.run_joint_trajectory(q_traj, dt)


def return_spoon(robot, panda, gripper, q_spoon_grab, q_spoon_lift):
    """
    Put the spoon back in its holder and let go.

    q_spoon_grab: the taught joint angles where we grasped the spoon in its holder.
    q_spoon_lift: the taught joint angles after we lifted it out.
    Neither is used as a move target: FK turns them into the holder's pose and the
    direction the spoon came out, and IK plans new poses from those.
    """
    T_holder = panda.fkine(np.asarray(q_spoon_grab, dtype=float))   # gripper pose with the spoon in the holder
    out = panda.fkine(np.asarray(q_spoon_lift, dtype=float)).t - T_holder.t
    out = out / np.linalg.norm(out)                                  # direction the spoon came out

    T_above = SE3.Trans(HOLDER_CLEARANCE * out) * T_holder   # above the holder, same orientation (world-frame shift)
    T_back = T_holder * SE3.Tz(-BACK_OFF)                    # back along the fingers (hand-frame shift)

    print("Returning the spoon to its holder")
    move_line(robot, panda, T_above, "above the holder")
    move_line(robot, panda, T_holder, "into the holder", speed=SPOON_INSERT_SPEED)
    gripper.open(0.1)
    move_line(robot, panda, T_back, "away from the spoon")
