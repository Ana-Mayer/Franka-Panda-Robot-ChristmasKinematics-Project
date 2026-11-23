import numpy as np
import roboticstoolbox as rtb
from csc376_franky.vizualizer import RtbVisualizer
from csc376_franky.motion_generator import RuckigMotionGenerator
from csc376_franky import Gripper
from spatialmath import SE3

from csc376_franky import FrankaJointTrajectoryController


def rotate_shelf(
    motion_generator,
    robot,
    panda,
    angle_deg=25,                 # + = counter-clockwise seen from above
    radius=0.11,
    center=(0.2586, -0.5892),
    execute=False
):
    """
    Plan one counterclockwise shelf rotation.

    Robot must already be holding the shelf leg, gripper pointing at the centre.
    The hand turns with the shelf so it keeps pointing at the centre.
    Does not execute unless execute=True.
    """

    q_start = np.asarray(
        robot.get_current_joint_positions(),
        dtype=float
    )

    T_start = panda.fkine(q_start)
    p = T_start.t

    center = np.asarray(center)

    # Starting angle around shelf centre
    theta_start = np.arctan2(
        p[1] - center[1],
        p[0] - center[0]
    )

    # Check radius against measured geometry
    measured_radius = np.linalg.norm(
        p[:2] - center
    )

    if abs(measured_radius - radius) > 0.01:
        raise RuntimeError(
            f"Gripper is not at the expected shelf radius: "
            f"measured {measured_radius:.4f} m, expected {radius:.4f} m."
        )

    # Divide the rotation into 12 short segments
    segments = 12
    total_angle = np.deg2rad(angle_deg)

    planned = []
    q_current = q_start.copy()
    T_current = T_start

    for i in range(1, segments + 1):

        angle = total_angle * i / segments
        theta = theta_start + angle

        # Calculate next point on the circle
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = p[2]

        # FIX: turn the hand with the shelf (about the world vertical axis),
        # so the gripper keeps pointing at the shelf centre.
        # (Last version used R = T_start.R, which kept the hand fixed.)
        R = (SE3.Rz(angle) * T_start).R
        T_next = SE3.Rt(R, [x, y, z])

        # Coarse Ruckig planner
        cartesian_traj, dt = (
            motion_generator.calculate_cartesian_pose_trajectory(
                T_current,
                T_next,
                0.02,
                0.01,
                0.05
            )
        )

        try:
            q_traj = np.asarray(
                motion_generator.cartesian_pose_to_joint_trajectory(
                    panda,
                    q_current,
                    cartesian_traj
                )
            )
        except RuntimeError:
            print(
                f"IK failed on segment {i}/{segments} "
                f"(arc angle {np.degrees(angle):.1f} deg), "
                f"target {np.round(T_next.t, 4)}"
            )
            raise

        if not np.all(np.isfinite(q_traj)):
            raise RuntimeError("Invalid joint trajectory")

        if np.max(np.abs(np.diff(q_traj, axis=0))) > 0.10:
            raise RuntimeError("Large joint step detected")

        if np.any(q_traj < panda.qlim[0]) or \
           np.any(q_traj > panda.qlim[1]):
            raise RuntimeError("Joint limit exceeded")

        planned.append((q_traj, dt))
        q_current = q_traj[-1].copy()
        T_current = panda.fkine(q_current)

    print("Planned", segments, "arc segments.")

    if not execute:
        print("Planning only: no robot movement.")
        return planned

    # Execute only after an explicit confirmation.
    confirmation = input(
        "Execute shelf rotation on connected robot? "
        "Type YES: "
    )

    if confirmation != "YES":
        print("Rotation cancelled.")
        return planned

    for i, (q_traj, dt) in enumerate(planned):
        actual_q = np.asarray(
            robot.get_current_joint_positions()
        )

        if not np.allclose(actual_q, q_traj[0], atol=0.01):
            raise RuntimeError(
                f"Robot position mismatch at segment {i+1}"
            )

        robot.run_joint_trajectory(q_traj, dt)

    print("Shelf rotation complete.")
    return planned


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
    # csc376_gripper.open(0.05)
    #csc376_gripper.grasp(0.017, 0.07995638996362686, 0.5)
    # Almays start code at home position and with the gripper all the way open
    csc376_franky_robot.move_home(0.08) # Move home
    csc376_gripper.open(0.05) 


    q_start = csc376_franky_robot.get_current_joint_positions()
    # The translucent blue Panda shows where the real robot is
    visualizer = RtbVisualizer(panda_rtb_model, q_start, real_robot=csc376_franky_robot)

    # III. Calculate your goal
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
    # Grab Cup 3: [1.5221, 1.4311, -1.0157, -2.1599, 0.862, 0.8393, 0.0766]
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    csc376_gripper.grasp(0.017, 0.07995638996362686, 0.5)
    se3_target = SE3.Tz(0.03) * SE3.Tx(0.1) * se3_target
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([0.1152, 0.8946, -0.7418, -2.3414, -1.7717, 2.0637, 0.8603])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.08, 0.02, 0.05)
    se3_target = panda_rtb_model.fkine([0.0377,  0.9485, -0.6766, -2.2881, -1.7716, 2.0664, 0.8738])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    csc376_gripper.open(0.05)
    se3_target = panda_rtb_model.fkine([0.4314, 0.9245, -0.8674, -2.3994, -1.5953, 2.246,  0.7938])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([-0.2865,  1.0618, -0.4065, -1.6739, -1.1614,  1.7426,  0.2524])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    csc376_gripper.grasp(0.079, 0.07995638996362686, 0.5)
    se3_target = panda_rtb_model.fkine([-0.2731  ,0.9406, -0.4004 ,-1.897  ,-1.3958  ,1.8664  ,0.4168])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([-0.2656 , 0.7917 ,-0.4253, -2.1973 ,-1.8771 , 1.9651 , 0.6798])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([-0.2875  ,0.7395 ,-0.4812, -2.4418, -2.4816 , 1.8374 , 1.0573])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([-0.2555 , 0.9424 ,-0.6522 ,-2.5317 ,-2.8819 , 1.4759  ,1.3648])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    se3_target = panda_rtb_model.fkine([-0.0258 , 1.4237 ,-0.9528,-2.4458 ,-2.8822 , 1.4895 , 1.8457])
    move_arm(motion_generator, csc376_franky_robot, se3_target, 0.06, 0.01, 0.05)
    #[-0.2731  0.9406 -0.4004 -1.897  -1.3958  1.8664  0.4168]
    #[-0.2656  0.7917 -0.4253 -2.1973 -1.8771  1.9651  0.6798]
    #[-0.2875  0.7395 -0.4812 -2.4418 -2.4816  1.8374  1.0573]
    #[-0.2555  0.9424 -0.6522 -2.5317 -2.8819  1.4759  1.3648]
    #[-0.0258  1.4237 -0.9528 -2.4458 -2.8822  1.4895  1.8457]


    #[-0.0246  1.66   -1.0082 -2.4151 -2.8761  1.2171  1.977 ]



    csc376_gripper.open(0.05)
    return visualizer

if __name__ == "__main__":
    visualizer = main()
    visualizer.stop() # Makes sure render thread ends