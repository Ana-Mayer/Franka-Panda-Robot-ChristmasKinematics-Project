
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg

from spatialmath import SE3
from csc376_franky.visualizer import RtbVisualizer
from csc376_franky.motion_generator import RuckigMotionGenerator
from csc376_franky import FrankaJointTrajectoryController


# ==========================================
# CONFIGURATION
# ==========================================

# Simulated Panda server
ROBOT_HOST = "127.0.0.1"

# Cartesian movement in metres (WORLD frame)
MOVE_X = 0.0
MOVE_Y = 0.0
MOVE_Z = -0.40  # Move end effector 40 cm down

# Motion constraints
RELATIVE_VEL = 0.02
RELATIVE_ACC = 0.01
RELATIVE_JERK = 0.05

# Gripper opening width in metres
GRIPPER_WIDTH = 0.07
GRIPPER_SPEED = 0.10

# Table dimensions
TABLE_LENGTH = 1.79
TABLE_WIDTH = 0.90
TABLE_THICKNESS = 0.04

# Table position relative to Panda base
TABLE_X = 0.20
TABLE_Y = 0.0
TABLE_TOP_Z = 0.0

BLUE = [0.12, 0.30, 0.55, 1.0]


# ==========================================
# TABLE VISUALIZATION
# ==========================================

def add_table(env):
    """Add rectangular tabletop along Y-axis."""

    tabletop = sg.Cuboid(
        [
            TABLE_WIDTH,
            TABLE_LENGTH,
            TABLE_THICKNESS
        ],
        pose=SE3(
            TABLE_X,
            TABLE_Y,
            TABLE_TOP_Z - TABLE_THICKNESS / 2
        ),
        color=BLUE
    )

    env.add_shape(tabletop)


# ==========================================
# MAIN
# ==========================================

def main():
    np.set_printoptions(precision=4, suppress=True)

    visualizer = None

    # Connect ONLY to local simulated server.
    with FrankaJointTrajectoryController(
        host=ROBOT_HOST
    ) as robot:

        # 1. Initialize robot model and motion generator
        panda = rtb.models.Panda()
        motion_generator = RuckigMotionGenerator()

        # Obtain actual simulated joint positions
        q_start = robot.get_current_joint_positions()

        print("Starting joints:", q_start)

        try:
            # ==========================================
            # 2. START SWIFT
            # ==========================================

            visualizer = RtbVisualizer(
                panda,
                q_start,
                real_robot=robot
            )

            add_table(visualizer.env)

            print("Table added.")

            # Visualize gripper opening
            visualizer.move_gripper(
                GRIPPER_WIDTH,
                GRIPPER_SPEED
            )

            
            # ==========================================
            # 3. CALCULATE TARGET POSE
            #    TRANSLATION + GRIPPER ROTATION
            # ==========================================

            T_start = panda.fkine(q_start)

            # Match the working CSC376 lab movement:
            # Translation and -90 degree Y-axis rotation
            T_requested = (
                T_start.inv()
                * SE3(0.6, 0, 0.442)
                * SE3.Ry(-np.pi / 2)
            )

            print("\nStarting EE pose:", T_start)
            print("Requested EE pose:", T_requested)

            # Solve IK as in the working lab example
            ik_solution = panda.ikine_LM(
                T_requested,
                q0=q_start
            )

            if not ik_solution.success:
                raise RuntimeError(
                    "IK failed to find the target pose."
                )

            q_goal = ik_solution.q

            # Reconstruct achieved pose using FK
            T_goal = panda.fkine(q_goal)

            print("\nTarget joints:", q_goal)
            print("Final EE pose:", T_goal)
            print("Final EE position:", T_goal.t)

            # if T_goal.t[2] <= TABLE_TOP_Z:
            #     raise ValueError(
            #         "Target end effector is at or "
            #         "below the tabletop."
            #     )

            # ==========================================
            # 4. VALIDATE TARGET WITH IK
            # ==========================================

            ik_solution = panda.ikine_LM(
                T_goal,
                q0=q_start,
                joint_limits=True,
                ilimit=100,
                method="sugihara",
                k=1e-4
            )

            if not ik_solution.success:
                raise RuntimeError(
                    "IK could not find a valid "
                    "configuration for the target."
                )

            q_goal = ik_solution.q

            # Verify final pose
            T_verified = panda.fkine(q_goal)

            position_error = np.linalg.norm(
                T_verified.t - T_goal.t
            )

            print("\nIK successful.")
            print("Target joints:", q_goal)
            print("Position error:", position_error)

            if position_error > 0.005:
                raise RuntimeError(
                    "IK position error exceeds 5 mm."
                )

            # ==========================================
            # 5. GENERATE RUCKIG TRAJECTORY
            # ==========================================

            cartesian_traj, dt = (
                motion_generator
                .calculate_cartesian_pose_trajectory(
                    T_start,
                    T_goal,
                    RELATIVE_VEL,
                    RELATIVE_ACC,
                    RELATIVE_JERK
                )
            )

            q_traj = (
                motion_generator
                .cartesian_pose_to_joint_trajectory(
                    panda,
                    q_start,
                    cartesian_traj
                )
            )

            if not q_traj:
                raise RuntimeError("Empty trajectory.")

            print("\nTrajectory generated.")
            print("Points:", len(q_traj))
            print("dt:", dt)
            print("Duration:", (len(q_traj) - 1) * dt)

            # Check EE height along the planned path
            min_ee_z = min(
                panda.fkine(q).t[2] for q in q_traj
            )

            # if min_ee_z <= TABLE_TOP_Z:
            #     raise RuntimeError(
            #         "Planned end-effector path "
            #         "reaches the tabletop."
            #     )

            # ==========================================
            # 6. PREVIEW IN SWIFT
            # ==========================================

            input(
                "\nPress Enter to preview in Swift..."
            )

            visualizer.run_joint_trajectory(
                q_traj,
                dt
            )

            print("Swift preview complete.")

            # ==========================================
            # 7. EXECUTE ON SIMULATED PANDA
            # ==========================================

            confirmation = input(
                "\nType 'yes' to execute on "
                "the SIMULATED Panda: "
            )

            if confirmation != "yes":
                print("Execution cancelled.")
                return

            # Recheck the robot's starting position
            q_actual = (
                robot.get_current_joint_positions()
            )

            if not np.allclose(
                q_actual,
                q_start,
                atol=0.01
            ):
                raise RuntimeError(
                    "Simulated robot has moved. "
                    "Replan from current position."
                )

            # Execute on simulated robot server
            robot.run_joint_trajectory(
                q_traj,
                dt
            )

            print(
                "\nSimulated Panda movement complete."
            )

            print(
                "Final simulated joints:",
                robot.get_current_joint_positions()
            )

            input(
                "\nPress Enter to close Swift..."
            )

        finally:
            if visualizer is not None:
                visualizer.stop()


if __name__ == "__main__":
    main()
