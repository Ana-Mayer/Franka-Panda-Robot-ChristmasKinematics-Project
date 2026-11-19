import numpy as np
import roboticstoolbox as rtb
from csc376_franky.vizualizer import RtbVisualizer
from csc376_franky.motion_generator import RuckigMotionGenerator
from csc376_franky import Gripper
from spatialmath import SE3

from csc376_franky import FrankaJointTrajectoryController

def main():
    np.set_printoptions(precision=4, suppress=True,)

    # I. Speed factor settings (causing issues, exceeding acceleration threshold)
    relative_vel_factor = 0.06
    relative_acc_factor = 0.01
    relative_jerk_factor = 0.05

    # II. RTB, Ruckig, csc376_franky, and Visualizer setup
    panda_rtb_model = rtb.models.Panda()
    motion_generator = RuckigMotionGenerator()

    csc376_franky_robot = FrankaJointTrajectoryController() # The robot server on this station's miniPC
    csc376_gripper = Gripper() # the robot gripper object that we will utilize to open/close to gripper
    print(csc376_franky_robot.get_current_joint_positions())
    print(csc376_gripper.state)
    csc376_gripper.open(0.05)
    #csc376_gripper.grasp(0.017, 0.07995638996362686, 0.5)
    # Almays start code at home position and with the gripper all the way open
    # csc376_franky_robot.move_home(0.01) # Move home
    # csc376_gripper.open(0.05) 


    q_start = csc376_franky_robot.get_current_joint_positions()
    # The translucent blue Panda shows where the real robot is
    visualizer = RtbVisualizer(panda_rtb_model, q_start, real_robot=csc376_franky_robot)

    # III. Calculate your goal
    se3_start = panda_rtb_model.fkine(q_start)
    se3_target = panda_rtb_model.fkine([0.0714, 0.7686, -0.0928, -2.3038, 0.0203, 1.497, 0.7049])
    # Grab Cup 1: [0.0714, 0.7686, -0.0928, -2.3038, 0.0203, 1.497, 0.7049]
    # Turntable 1: [0.1152, 0.8946, -0.7418, -2.3414, -1.7717, 2.0637, 0.8603]
    # Turntable 1 Placement: [0.0377,  0.9485, -0.6766, -2.2881, -1.7716, 2.0664, 0.8738]
    # Reverse from turntable: [0.4314, 0.9245, -0.8674, -2.3994, -1.5953, 2.246,  0.7938]
    # Turntable Column Grab: [0.233, 1.0381, -0.9726, -2.1166, -1.5948, 2.2983, 0.5578]
    # Close gripper after this step^.
    # After turntable rotation: [0.1413, 1.1646, -1.0913, -2.2967, -1.8882, 2.4854, 0.987 ]
    # Open gripper after this step^
    # Move away from turntable 1: [0.01, 0.7922, -0.2304, -2.0334, -0.9806, 1.5827, 0.4381]
    # Grab Cup 2: [0.7782, 1.2754, -0.9127, -2.2612, 0.3216, 1.5339, -0.0672]

    # IV. Visualize the trajectory in simulation
    cartesian_traj, dt = motion_generator.calculate_cartesian_pose_trajectory(se3_start, se3_target,
                                                                              relative_vel_factor, relative_acc_factor, relative_jerk_factor) # Interpolation and trajectory parameterization
    q_traj = motion_generator.cartesian_pose_to_joint_trajectory(panda_rtb_model, q_start, cartesian_traj)

    # input("Press enter, to run in visualizer\n")
    # visualizer.run_joint_trajectory(q_traj, dt)
    # csc376_franky_robot.run_joint_trajectory(q_traj, dt)

    return visualizer

if __name__ == "__main__":
    visualizer = main()
    visualizer.stop() # Makes sure render thread ends