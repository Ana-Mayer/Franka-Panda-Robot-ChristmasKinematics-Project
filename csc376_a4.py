import numpy as np
import roboticstoolbox as rtb
from csc376_franky.vizualizer import RtbVisualizer
from csc376_franky.motion_generator import RuckigMotionGenerator
from csc376_franky import Gripper
from spatialmath import SE3

from csc376_franky import FrankaJointTrajectoryController

import shelf_turn  # smooth shelf turn through our taught points (FK + IK), see shelf_turn.py


def move_arm(motion_generator, csc376_franky_robot, se3_target, relative_vel_factor, relative_acc_factor, relative_jerk_factor):
    panda_rtb_model = rtb.models.Panda()
    q_start = csc376_franky_robot.get_current_joint_positions()
    se3_start = panda_rtb_model.fkine(q_start)

    cartesian_traj, dt = motion_generator.calculate_cartesian_pose_trajectory(se3_start, se3_target,
                                                                              relative_vel_factor, relative_acc_factor, relative_jerk_factor) # Interpolation and trajectory parameterization
    q_traj = motion_generator.cartesian_pose_to_joint_trajectory(panda_rtb_model, q_start, cartesian_traj)

    # input("Press enter, to run in visualizer\n")
    # visualizer.run_joint_trajectory(q_traj, dt)
    csc376_franky_robot.run_joint_trajectory(q_traj, dt)

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

    # Almays start code at home position and with the gripper all the way open
    csc376_franky_robot.move_home(0.1) # Move home
    csc376_gripper.open(0.05) 


    q_start = csc376_franky_robot.get_current_joint_positions()
    # The translucent blue Panda shows where the real robot is
    visualizer = RtbVisualizer(panda_rtb_model, q_start, real_robot=csc376_franky_robot)
    cup_values = [[ 0.2974 , 0.9507, -0.5666, -2.214 , -0.0652 , 1.5174 , 0.3292], [0.0714, 0.7686, -0.0928, -2.3038, 0.0203, 1.497, 0.7049], [ 0.237 ,  0.765  , 0.0621 ,-2.2566  ,0.3139 , 1.4961  ,0.8341]]

    # Position cups on the turntable:
    # for i in range(len(cup_values)):
    #     # Moves to starting position in front of cups
    #     se3_target = panda_rtb_model.fkine([ 0.1034  ,0.9171 ,-0.0998, -1.8636, 0.0514 , 1.19   , 0.7142])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.08, 0.02, 0.05)

    #     # Moves to grab the cup (depending on which loop iteration it is on)
    #     se3_target = panda_rtb_model.fkine(cup_values[i])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)

    #     # Grabs the cup
    #     csc376_gripper.grasp(0.017, 0.07995638996362686, 0.5)

    #     # Moves cup out of the way of obstacles before continuing
    #     se3_target = SE3.Tz(0.03) * SE3.Tx(0.1) * se3_target
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)

    #     # Moves cup over to the bottom turntable shelf
    #     se3_target = panda_rtb_model.fkine([0.1152, 0.8946, -0.7418, -2.3414, -1.7717, 2.0637, 0.8603])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.08, 0.02, 0.05)

    #     # Places the cup on the turntable 
    #     se3_target = panda_rtb_model.fkine([0.0377,  0.9485, -0.6766, -2.2881, -1.7716, 2.0664, 0.8738])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)

    #     # Releases the cup
    #     csc376_gripper.open(0.05)

    #     # Moves the robot arm out from underneath the upper turntable shelf
    #     se3_target = panda_rtb_model.fkine([0.4314, 0.9245, -0.8674, -2.3994, -1.5953, 2.246,  0.7938])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)

    #     # if the last cup has just been placed, do not rotate turntable, and return to the starting position
    #     if (i == len(cup_values) - 1):
    #         se3_target = panda_rtb_model.fkine([ 0.1034  ,0.9171 ,-0.0998, -1.8636, 0.0514 , 1.19   , 0.7142])
    #         move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    #         break;

    #     # Move the robot arm to position itself in front of the turntable cylinder
    #     se3_target = panda_rtb_model.fkine([ 0.1046 , 1.103  ,-0.6042 ,-1.7349, -0.9421 , 1.8784,  0.1902])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)

    #     # Move the end effector such that the cylinder is between its grippers
    #     se3_target = panda_rtb_model.fkine([-0.0335 , 1.2175 ,-0.6945, -1.6928 ,-0.9427,  1.8866 , 0.1465])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)

    #     # Grasp the cylinder
    #     csc376_gripper.grasp(0.079, 0.07995638996362686, 0.5)

    #     # Turn the shelf: one smooth TURN through our taught points, then a PUSH to the
    #     # leg's end position (taught points and settings are in shelf_turn.py).
    #     # Previews each move in the visualizer, then asks for YES before the robot moves.
    #     shelf_turn.run_shelf_turn(csc376_franky_robot, panda_rtb_model, ask=False)

    #     # Open the grippers to release the cylinder
    #     csc376_gripper.open(0.05)

    #     # Move back to the starting position
    #     se3_target = panda_rtb_model.fkine([-0.0296 , 0.8752 ,-0.6361 ,-2.6688 ,-2.3438 , 1.8078 , 1.3451])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    
    se3_target = panda_rtb_model.fkine([ 0.1034  ,0.9171 ,-0.0998, -1.8636, 0.0514 , 1.19   , 0.7142])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.08, 0.02, 0.05)
    # #[ 0.1542  0.4514  0.874  -2.1559  0.8927  3.0592  2.3997]
    # # Move towards the spoon
    se3_target = panda_rtb_model.fkine([ 0.1542 , 0.4514 , 0.874 , -2.1559  ,0.8927 , 3.0592 , 2.3997])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([ 0.2014 , 0.6341 , 0.8781 ,-2.0568 , 0.8929 , 3.1484 , 2.2932])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    # Grab Spoon
    csc376_gripper.grasp(0.025739362463355064, 0.07995638996362686, 0.5)
    se3_target = panda_rtb_model.fkine([ 0.141  , 0.1975 , 0.6516 ,-2.1597 , 1.8019 , 2.3135 , 1.6888])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([ 0.8643  ,0.4895 ,-0.3838 ,-2.3638 , 1.8645  ,2.0388 , 2.6958])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([ 0.8634  ,0.5367 ,-0.3864 ,-2.3182 , 1.8678 , 2.0766  ,1.6159])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([ 0.7639 , 0.5912, -0.3391 ,-2.2979,  1.8691 , 1.954  , 1.309 ])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([ 0.6895 , 0.5909, -0.4108 ,-2.3183 , 1.8777 , 1.4471 , 1.1206])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([ 0.0887 , 0.559 , -0.7574, -2.0717 ,-1.2117 , 1.7462 , 0.0422])

    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([ 0.0889 , 0.5587 ,-0.7598, -2.0729 ,-1.2117 , 1.7461,-1.7529])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)

    # csc376_gripper.open(0.05)
    return visualizer

if __name__ == "__main__":
    visualizer = main()
    visualizer.stop() # Makes sure render thread ends


# General Saved Values
    # III. Calculate your goal
    # Grab Cup 1: [ 0.2974  0.9507 -0.5666 -2.214  -0.0652  1.5174  0.3292]
    # Turntable 1: [0.1152, 0.8946, -0.7418, -2.3414, -1.7717, 2.0637, 0.8603]
    # Turntable 1 Placement: [0.0377,  0.9485, -0.6766, -2.2881, -1.7716, 2.0664, 0.8738]
    # Reverse from turntable: [0.4314, 0.9245, -0.8674, -2.3994, -1.5953, 2.246,  0.7938]
    # Postition gripper in front of cylinder [ 0.1046  1.103  -0.6042 -1.7349 -0.9421  1.8784  0.1902]
    # Turntable Column Grab: [0.233, 1.0381, -0.9726, -2.1166, -1.5948, 2.2983, 0.5578]
    # Close gripper after this step^.
    # After turntable rotation: [0.1413, 1.1646, -1.0913, -2.2967, -1.8882, 2.4854, 0.987 ]
    # Open gripper after this step^
    # Move away from turntable 1: [0.01, 0.7922, -0.2304, -2.0334, -0.9806, 1.5827, 0.4381]
    # Grab Cup 2: [0.0714, 0.7686, -0.0928, -2.3038, 0.0203, 1.497, 0.7049]
    # Grab Cup 3: [ 0.237 ,  0.765  , 0.0621 ,-2.2566  ,0.3139 , 1.4961  ,0.8341]
    # Backup gripper after rotation: [-0.0296  0.8752 -0.6361 -2.6688 -2.3438  1.8078  1.3451]
    # Move cup to starting position: [ 0.1034  0.9171 -0.0998 -1.8636  0.0514  1.19    0.7142]

# Rotation Saved Values:
    # This replaces the five move_arm calls through these taught points:
    #[-0.2731  0.9406 -0.4004 -1.897  -1.3958  1.8664  0.4168]
    #[-0.2656  0.7917 -0.4253 -2.1973 -1.8771  1.9651  0.6798]
    #[-0.2875  0.7395 -0.4812 -2.4418 -2.4816  1.8374  1.0573]
    #[-0.2555  0.9424 -0.6522 -2.5317 -2.8819  1.4759  1.3648]
    #[-0.0258  1.4237 -0.9528 -2.4458 -2.8822  1.4895  1.8457]
    # End position (refused by the robot as a full pose, so only its position is used):
    #[-0.0246  1.66   -1.0082 -2.4151 -2.8761  1.2171  1.977 ]

# Grab Spoon Saved Values
    #[ 0.0887  0.559  -0.7574 -2.0717 -1.2117  1.7462  0.0422]
    # [ 0.0889  0.5587 -0.7598 -2.0729 -1.2117  1.7461 -1.7529]
    # [ 0.8557  0.2714  0.0631 -2.0374  1.734   2.5698  1.7079]

    # [ 0.7333 -0.0204 -0.0891 -2.4195  1.7277  2.2916  1.7022]
    # [ 0.8643  0.4895 -0.3838 -2.3638  1.8645  2.0388  2.6958]
    # [ 0.8634  0.5367 -0.3864 -2.3182  1.8678  2.0766  1.6159]
    # [ 0.7639  0.5912 -0.3391 -2.2979  1.8691  1.954   1.309 ]
    # [ 0.6895  0.5909 -0.4108 -2.3183  1.8777  1.4471  1.1206]