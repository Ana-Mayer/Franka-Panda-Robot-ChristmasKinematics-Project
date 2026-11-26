import numpy as np
import roboticstoolbox as rtb
from csc376_franky.vizualizer import RtbVisualizer
from csc376_franky.motion_generator import RuckigMotionGenerator
from csc376_franky import Gripper
from spatialmath import SE3

from csc376_franky import FrankaJointTrajectoryController

import shelf_turn  # smooth shelf turn through our taught points (FK + IK), see shelf_turn.py
import ik_moves    # straight-line moves with our own IK, and putting the spoon back, see ik_moves.py

# Speed presets for move_arm: (velocity, acceleration, jerk) 
FAST = (0.15, 0.05, 0.10)     # free moves: nothing in the gripper, or an empty spoon
CAREFUL = (0.10, 0.03, 0.10)  # close to objects, or carrying a cup
SLOW = (0.06, 0.01, 0.05)     # our original speed
CARRY = tuple(1.6 * f for f in SLOW)  # the long moves carrying the spoon: 1.6x our original speed

# Taught spoon poses (also used by ik_moves.return_spoon to find the holder)
Q_SPOON_GRAB = [0.2014, 0.6341, 0.8781, -2.0568, 0.8929, 3.1484, 2.2932]
Q_SPOON_LIFT = [0.141, 0.1975, 0.6516, -2.1597, 1.8019, 2.3135, 1.6888]
Q_SPOON_AT_CUP = [0.0887, 0.559, -0.7574, -2.0717, -1.2117, 1.7462, 0.0422]
Q_SPOON_BY_SUGAR = [0.6895, 0.5909, -0.4108, -2.3183, 1.8777, 1.4471, 1.1206]


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

    # I. Speed factor settings: see FAST, CAREFUL and SLOW at the top of the file

    # II. RTB, Ruckig, csc376_franky, and Visualizer setup
    panda_rtb_model = rtb.models.Panda()
    motion_generator = RuckigMotionGenerator()

    csc376_franky_robot = FrankaJointTrajectoryController() # The robot server on this station's miniPC
    csc376_gripper = Gripper() # the robot gripper object that we will utilize to open/close to gripper
    print(csc376_franky_robot.get_current_joint_positions())
    print(csc376_gripper.state)

    # Almays start code at home position and with the gripper all the way open
    csc376_franky_robot.move_home(0.2) # Move home
    csc376_gripper.open(0.1)


    q_start = csc376_franky_robot.get_current_joint_positions()
    # The translucent blue Panda shows where the real robot is
    visualizer = RtbVisualizer(panda_rtb_model, q_start, real_robot=csc376_franky_robot)
    cup_values = [[ 0.2974 , 0.9507, -0.5666, -2.214 , -0.0652 , 1.5174 , 0.3292], [0.0714, 0.7686, -0.0928, -2.3038, 0.0203, 1.497, 0.7049], [ 0.237 ,  0.765  , 0.0621 ,-2.2566  ,0.3139 , 1.4961  ,0.8341]]

    # Position cups on the turntable:
    # for i in range(len(cup_values)):
    #     # Moves to starting position in front of cups
    #     se3_target = panda_rtb_model.fkine([ 0.1034  ,0.9171 ,-0.0998, -1.8636, 0.0514 , 1.19   , 0.7142])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)

    #     # Moves to grab the cup (depending on which loop iteration it is on)
    #     se3_target = panda_rtb_model.fkine(cup_values[i])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)

    #     # Grabs the cup
    #     csc376_gripper.grasp(0.017, 0.07995638996362686, 0.5)

    #     # Moves cup out of the way of obstacles before continuing
    #     se3_target = SE3.Tz(0.03) * SE3.Tx(0.1) * se3_target
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)

    #     # Moves cup over to the bottom turntable shelf
    #     se3_target = panda_rtb_model.fkine([0.1152, 0.8946, -0.7418, -2.3414, -1.7717, 2.0637, 0.8603])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)

    #     # Places the cup on the turntable
    #     se3_target = panda_rtb_model.fkine([0.0377,  0.9485, -0.6766, -2.2881, -1.7716, 2.0664, 0.8738])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)

    #     # Releases the cup
    #     csc376_gripper.open(0.1)

    #     # Moves the robot arm out from underneath the upper turntable shelf
    #     se3_target = panda_rtb_model.fkine([0.4314, 0.9245, -0.8674, -2.3994, -1.5953, 2.246,  0.7938])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)

    #     # if the last cup has just been placed, do not rotate turntable, and return to the starting position
    #     if (i == len(cup_values) - 1):
    #         se3_target = panda_rtb_model.fkine([ 0.1034  ,0.9171 ,-0.0998, -1.8636, 0.0514 , 1.19   , 0.7142])
    #         move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    #         break;

    #     # Move the robot arm to position itself in front of the turntable cylinder
    #     se3_target = panda_rtb_model.fkine([ 0.1046 , 1.103  ,-0.6042 ,-1.7349, -0.9421 , 1.8784,  0.1902])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)

    #     # Move the end effector such that the cylinder is between its grippers
    #     se3_target = panda_rtb_model.fkine([-0.0335 , 1.2175 ,-0.6945, -1.6928 ,-0.9427,  1.8866 , 0.1465])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)

    #     # Grasp the cylinder
    #     csc376_gripper.grasp(0.079, 0.07995638996362686, 0.5)

    #     # Turn the shelf: one smooth TURN through our taught points, then a PUSH to the
    #     # leg's end position (taught points and settings are in shelf_turn.py).
    #     shelf_turn.run_shelf_turn(csc376_franky_robot, panda_rtb_model)

    #     # Open the grippers to release the cylinder
    #     csc376_gripper.open(0.1)

    #     # Move back to the starting position
    #     se3_target = panda_rtb_model.fkine([-0.0296 , 0.8752 ,-0.6361 ,-2.6688 ,-2.3438 , 1.8078 , 1.3451])
    #     move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)

    # [ 0.2651,  1.2926, -0.8798, -1.8775, -2.5912,  2.2531,  1.4068]
    se3_target = panda_rtb_model.fkine([ 0.2651,  1.2926, -0.8798, -1.8775, -2.5912,  2.2531,  1.4068])
    move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    csc376_gripper.grasp(0.079, 0.07995638996362686, 10)
    se3_target = panda_rtb_model.fkine([ 0.801 ,  1.5235, -1.5574, -2.1236, -2.8819,  2.8281,  2.0613])
    move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # se3_target = panda_rtb_model.fkine([ 0.6297,  1.5283, -1.5859, -2.3901, -2.884 ,  2.8271,  2.0449])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # se3_target = panda_rtb_model.fkine([ 0.4552,  1.0255, -1.4108, -2.3187, -2.8835,  2.8263,  1.6973])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # # [ 0.801 ,  1.5235, -1.5574, -2.1236, -2.8819,  2.8281,  2.0613]
    # # [ 0.6297,  1.5283, -1.5859, -2.3901, -2.884 ,  2.8271,  2.0449]
    # # [ 0.4552,  1.0255, -1.4108, -2.3187, -2.8835,  2.8263,  1.6973]
    # csc376_gripper.open(0.1)
    # se3_target = panda_rtb_model.fkine([ 1.0181,  1.3209, -1.5455, -2.3676, -2.8467,  3.4939,  2.0268])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # # [ 1.0181,  1.3209, -1.5455, -2.3676, -2.8467,  3.4939,  2.0268]

    # se3_target = panda_rtb_model.fkine([ 0.1034  ,0.9171 ,-0.0998, -1.8636, 0.0514 , 1.19   , 0.7142])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    # # Move towards the spoon
    # se3_target = panda_rtb_model.fkine([ 0.1542 , 0.4514 , 0.874 , -2.1559  ,0.8927 , 3.0592 , 2.3997])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    # se3_target = panda_rtb_model.fkine(Q_SPOON_GRAB)
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # # Grab Spoon
    # csc376_gripper.grasp(0.025739362463355064, 0.07995638996362686, 0.5)
    # # Move spoon up to avoid collision with the spoon stand
    # se3_target = panda_rtb_model.fkine(Q_SPOON_LIFT)
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # # Move spoon down and rotate the end effector to prepare for scooping the sugar
    # se3_target = panda_rtb_model.fkine([ 0.8643  ,0.4895 ,-0.3838 ,-2.3638 , 1.8645  ,2.0388 , 2.6958])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # # Scoop the sugar (short moves of the spoon: FAST)
    # se3_target = panda_rtb_model.fkine([ 0.8634  ,0.5367 ,-0.3864 ,-2.3182 , 1.8678 , 2.0766  ,1.6159])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    # # Lift the spoon up to avoid collision with the sugar container
    # se3_target = panda_rtb_model.fkine([ 0.7639 , 0.5912, -0.3391 ,-2.2979,  1.8691 , 1.954  , 1.309 ])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    # # Move spoon to the cup on rotating shelf and rotate the spoon such that the sugar is in the cup
    # se3_target = panda_rtb_model.fkine(Q_SPOON_BY_SUGAR)
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    # # The long move to the cup carrying the full spoon: CARRY (1.6x our original speed)
    # se3_target = panda_rtb_model.fkine(Q_SPOON_AT_CUP)
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    # # Tip the spoon to pour the sugar into the cup
    # se3_target = panda_rtb_model.fkine([ 0.0889 , 0.5587 ,-0.7598, -2.0729 ,-1.2117 , 1.7461,-1.7529])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)

    # # Put the spoon back. First retrace our path (known to be clear): turn the
    # # empty spoon back up, then the long move back beside the sugar (CARRY).
    # se3_target = panda_rtb_model.fkine(Q_SPOON_AT_CUP)
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    # se3_target = panda_rtb_model.fkine(Q_SPOON_BY_SUGAR)
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *FAST)
    # # Then inverse kinematics: above the holder, straight down into it, let go, back off
    # ik_moves.return_spoon(csc376_franky_robot, panda_rtb_model, csc376_gripper,
    #                       Q_SPOON_GRAB, Q_SPOON_LIFT)
    # se3_target = panda_rtb_model.fkine([ 0.4478,  0.6346,  0.366 , -2.4139,  2.2961,  1.8726,  0.6787])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *SLOW)
    # se3_target = panda_rtb_model.fkine([ 0.5211,  0.7579,  0.3502, -2.3116,  2.3611,  1.7776,  0.6515])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # csc376_gripper.grasp(0.079, 0.07995638996362686, 0.7)
    # se3_target = panda_rtb_model.fkine([ 0.5735,  1.4443, -1.5997, -2.1595, -0.5895,  2.8805, -0.3586])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # se3_target = panda_rtb_model.fkine([-0.3675, -0.0557, -0.6446, -2.2735, -1.6615,  2.6693,  1.645 ])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # se3_target = panda_rtb_model.fkine([ 0.5735,  1.4443, -1.5997, -2.1595, -0.5895,  2.8805, -0.3586])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # se3_target = panda_rtb_model.fkine([ 0.5211,  0.7579,  0.3502, -2.3116,  2.3611,  1.7776,  0.6515])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *CAREFUL)
    # csc376_gripper.open(0.1)
    # se3_target = panda_rtb_model.fkine([ 0.4478,  0.6346,  0.366 , -2.4139,  2.2961,  1.8726,  0.6787])
    # move_arm(motion_generator, csc376_franky_robot, se3_target, *SLOW)
    # csc376_franky_robot.move_home(0.2) # Move home
    # [ 0.4478,  0.6346,  0.366 , -2.4139,  2.2961,  1.8726,  0.6787]
    # [ 0.3982,  0.8047,  0.4627, -2.3149,  2.3465,  1.8158,  0.6143]
    # [ 0.2865,  0.7408, -1.7828, -2.561 ,  2.8227,  2.0167,  1.6625]
    # [ 0.4197,  0.7415, -1.6946, -2.6228,  2.8127,  2.0171, -0.1666]
    # [ 0.4473,  0.7415, -1.6924, -2.6175,  2.8129,  2.0168, -0.4685]
    # [ 0.1011,  0.415 , -0.2393, -2.4167,  0.8606,  1.3049,  0.9121]

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