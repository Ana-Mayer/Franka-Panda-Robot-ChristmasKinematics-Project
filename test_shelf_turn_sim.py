"""
Test shelf_turn.py against the SIMULATED robot server (no real robot).

    PYTHONPATH=/tmp/simpkgs python -m csc376_franky_server --sim --bind 127.0.0.1   # other terminal
    CSC376_ROBOT_HOST=127.0.0.1 python test_shelf_turn_sim.py

Swift (the visualizer) opens in a browser tab:
  solid Panda           = the plan, played before each move
  translucent blue Panda = where the (simulated) robot really is
  red disc              = the circle the gripper follows around the shelf centre
Add NO_VIZ=1 in front of the command to run without it.
"""
import os

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
from spatialmath import SE3
from csc376_franky import FrankaJointTrajectoryController
from csc376_franky.motion_generator import RuckigMotionGenerator
from csc376_franky.visualizer import RtbVisualizer

import shelf_turn as st

np.set_printoptions(precision=4, suppress=True)
panda = rtb.models.Panda()
robot = FrankaJointTrajectoryController()
motion_generator = RuckigMotionGenerator()

robot.move_home(0.3)

visualizer = None
if not os.environ.get("NO_VIZ"):
    visualizer = RtbVisualizer(panda, robot.get_current_joint_positions(), real_robot=robot)
    # Mark the circle around the shelf centre (fitted to the taught points)
    center = st.fit_center(panda, list(st.TAUGHT_TURN_POINTS) + [st.Q_LEG_END])
    T_grasp = panda.fkine(np.asarray(st.TAUGHT_TURN_POINTS[0]))
    radius = np.hypot(T_grasp.t[0] - center[0], T_grasp.t[1] - center[1])
    visualizer.env.add(sg.Cylinder(radius=radius, length=0.002, color=(1.0, 0.2, 0.2, 0.5),
                                   pose=SE3(center[0], center[1], T_grasp.t[2] - 0.02)))

# Go to the leg grasp the same way csc376_a4.py does (Ruckig, straight line)
q_now = robot.get_current_joint_positions()
traj, dt = motion_generator.calculate_cartesian_pose_trajectory(
    panda.fkine(q_now), panda.fkine(st.TAUGHT_TURN_POINTS[0]), 0.06, 0.02, 0.05)
q_traj = motion_generator.cartesian_pose_to_joint_trajectory(panda, q_now, traj)
if visualizer is not None:
    visualizer.run_joint_trajectory(q_traj, dt)
robot.run_joint_trajectory(q_traj, dt)
print("at grasp:", np.round(robot.get_current_joint_positions(), 4))

plans = st.run_shelf_turn(robot, panda, visualizer=visualizer)

# Where did the gripper end up, compared with the taught end position?
q_end = np.asarray(robot.get_current_joint_positions())
T_end = panda.fkine(q_end)
T_goal = panda.fkine(st.Q_LEG_END)
print("end joints:", np.round(q_end, 4))
print(f"gripper is {1000 * np.linalg.norm(T_end.t[:2] - T_goal.t[:2]):.2f} mm (horizontally) "
      f"from the taught end position, height {T_end.t[2]:.4f} m")

if visualizer is not None:
    input("Press Enter to close the visualizer...")
    visualizer.stop()
robot.close()
