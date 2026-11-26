import numpy as np
import roboticstoolbox as rtb
from csc376_franky.vizualizer import RtbVisualizer
from csc376_franky.motion_generator import RuckigMotionGenerator
from csc376_franky import Gripper
from spatialmath import SE3

from csc376_franky import FrankaJointTrajectoryController

import shelf_turn  # smooth shelf turn through our taught points (FK + IK), see shelf_turn.py
import ik_moves    # straight-line moves with our own IK, and putting the spoon back, see ik_moves.py

def main():
    np.set_printoptions(precision=4, suppress=True,)

    # I. Speed factor settings: see FAST, CAREFUL and SLOW at the top of the file

    # II. RTB, Ruckig, csc376_franky, and Visualizer setup
    panda_rtb_model = rtb.models.Panda()
    motion_generator = RuckigMotionGenerator()

    csc376_franky_robot = FrankaJointTrajectoryController() # The robot server on this station's miniPC
    csc376_gripper = Gripper() # the robot gripper object that we will utilize to open/close to gripper
    print(csc376_gripper.state)
    print(np.array2string(csc376_franky_robot.get_current_joint_positions(), separator=', '))

if __name__ == "__main__":
    visualizer = main()
    visualizer.stop() # Makes sure render thread ends