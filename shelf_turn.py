"""
Turn the round shelf with ONE smooth gripper motion through our taught points.

We taught the robot a few gripper poses by hand (TAUGHT_TURN_POINTS) and one
final position for the leg (Q_LEG_END). Moving between them with move_arm goes in
straight lines and stops at every point. Instead, this file:

1. Reads each taught point with forward kinematics (FK): its angle around the
   shelf centre, its distance from the centre, and how far the hand has turned.
2. Draws one smooth curve through those points, at one fixed height (the lowest
   of the points).
3. Walks along the curve in small time steps, with a gentle start and stop,
   and uses inverse kinematics (IK) to get the joint angles at each step.

The motion has two parts:
  TURN: through the taught points; the hand turns as it did when we taught it.
  PUSH: from the last taught point to the leg's final position; the hand stops
        turning (turning further runs joint 5 into its limit) and just pushes.
"""

import numpy as np
import roboticstoolbox as rtb
from scipy.interpolate import PchipInterpolator
from spatialmath import SE3

# ---------------------------------------------------------------- settings ---
# x, y of the shelf's centre [m], robot base frame. None = fit it to the taught points
# (fit_center below). The old hand-measured (0.2586, -0.5892) was about 4 cm off.
SHELF_CENTER = None

# Taught by hand, in order. The first is where we grasp the leg.
TAUGHT_TURN_POINTS = [
    [-0.2865, 1.0618, -0.4065, -1.6739, -1.1614, 1.7426, 0.2524],   # grasp
    [-0.2731, 0.9406, -0.4004, -1.8970, -1.3958, 1.8664, 0.4168],
    [-0.2656, 0.7917, -0.4253, -2.1973, -1.8771, 1.9651, 0.6798],
    [-0.2875, 0.7395, -0.4812, -2.4418, -2.4816, 1.8374, 1.0573],
]
# Where the leg should end up. Only its POSITION is used: its hand orientation
# is too close to joint 5's limit, which is why the robot refused it.
Q_LEG_END = [-0.0246, 1.6600, -1.0082, -2.4151, -2.8761, 1.2171, 1.9770]

GRIPPER_SPEED = 0.02     # top speed of the gripper along the curve [m/s]
DT = 0.05                # time between trajectory points [s]; the robot fills in 1 kHz between them

# The robot server's checks (see robot-interface/README.md, "What the miniPC checks")
SERVER_LIMIT_MARGIN = 0.01                                      # rad from each joint limit
SERVER_MAX_STEP = 0.025                                         # rad between two points
SERVER_MAX_VEL = 0.3 * np.array([2.175] * 4 + [2.61] * 3)       # rad/s
SERVER_MAX_ACC = np.array([15, 7.5, 10, 12.5, 15, 20, 20.0])    # rad/s^2


# ------------------------------------------------- step 1: read the points ---
def fit_center(panda, q_points):
    """
    The centre of the circle that best fits the gripper positions (FK) of the points.

    A point (x, y) is on a circle with centre (cx, cy) and radius r when
        x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
    which is LINEAR in the unknowns cx, cy and k = r^2 - cx^2 - cy^2.
    So least squares on one row per point finds them.
    """
    xy = np.array([panda.fkine(np.asarray(q, dtype=float)).t[:2] for q in q_points])
    A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
    b = xy[:, 0] ** 2 + xy[:, 1] ** 2
    cx, cy, k = np.linalg.lstsq(A, b, rcond=None)[0]
    radius = np.sqrt(k + cx**2 + cy**2)
    off = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - radius
    print(f"Fitted shelf centre ({cx:.4f}, {cy:.4f}), radius {radius:.4f} m; "
          f"the points are within {1000 * np.abs(off).max():.1f} mm of that circle")
    return (cx, cy)


def describe_point(panda, q, center, R_grasp):
    """
    FK of a taught joint configuration, described around the shelf:
    (angle around the centre [rad], distance from the centre [m], height [m],
     how far the hand has turned about the vertical since the grasp [rad]).
    """
    T = panda.fkine(np.asarray(q, dtype=float))
    dx, dy = T.t[0] - center[0], T.t[1] - center[1]
    angle = np.arctan2(dy, dx)
    distance = np.hypot(dx, dy)
    R_rel = T.R @ R_grasp.T                        # rotation from the grasp orientation to this one
    hand_turn = np.arctan2(R_rel[1, 0], R_rel[0, 0])  # its part about the vertical (z) axis
    return angle, distance, T.t[2], hand_turn


# ------------------------------------------------- step 2: the smooth curve ---
class ShelfCurve:
    """
    A smooth curve through the taught points, described by the angle around the
    shelf centre. For any angle it gives the gripper's distance from the centre and
    how far the hand is turned. PCHIP passes exactly through every point and never
    overshoots between them (a plain cubic spline can bulge outward).
    """

    def __init__(self, panda, q_turn_points, q_end, center):
        self.center = np.asarray(center, dtype=float)
        self.R_grasp = panda.fkine(np.asarray(q_turn_points[0], dtype=float)).R

        turn = np.array([describe_point(panda, q, center, self.R_grasp) for q in q_turn_points])
        end = describe_point(panda, q_end, center, self.R_grasp)

        angles = np.unwrap(np.r_[turn[:, 0], end[0]])
        if not np.all(np.diff(angles) > 0):
            raise ValueError("The points must go around the shelf in one direction (counter-clockwise)")

        self.start_angle = angles[0]
        self.push_angle = angles[-2]     # the last taught point: PUSH starts here
        self.end_angle = angles[-1]
        self.height = min(np.r_[turn[:, 2], end[2]])   # one height: the lowest point's

        # distance from the centre: through every point, the end included
        self.distance = PchipInterpolator(angles, np.r_[turn[:, 1], end[1]])
        # hand turn: through the taught points only; it stays constant during PUSH
        self.hand_turn = PchipInterpolator(angles[:-1], np.unwrap(turn[:, 3]))

        print("Taught points (angle around centre, distance, height, hand turn):")
        for a, d, z, h in turn:
            print(f"  {np.degrees(a):6.1f} deg  {d:.4f} m  {z:.4f} m  {np.degrees(h):6.1f} deg")
        print(f"  end {np.degrees(end[0]):6.1f} deg  {end[1]:.4f} m  (position only)")
        print(f"Curve height {self.height:.4f} m")

    def pose(self, angle):
        """The gripper pose at `angle` [rad] around the shelf centre."""
        d = float(self.distance(angle))
        x = self.center[0] + d * np.cos(angle)
        y = self.center[1] + d * np.sin(angle)
        h = float(self.hand_turn(min(angle, self.push_angle)))  # frozen after the last taught point
        R = SE3.Rz(h).R @ self.R_grasp                          # turn the hand about the vertical
        return SE3.Rt(R, [x, y, self.height])


# --------------------------------------- step 3: time steps + inverse kinematics ---
def joint_margin(panda, q):
    """Distance [rad] from the closest joint to its limit."""
    lower, upper = panda.qlim
    return float(np.min(np.minimum(q - lower, upper - q)))


def solve_ik(panda, T, q_seed):
    """Joint angles for pose T, starting the search from q_seed (the previous point)."""
    sol = panda.ikine_LM(T, q0=q_seed, joint_limits=True)
    if not sol.success:
        raise RuntimeError(f"IK found no solution for the gripper at {np.round(T.t, 4)}")
    return np.asarray(sol.q, dtype=float)


def smooth_fraction(tau):
    """0 -> 1 as tau goes 0 -> 1, with zero speed and zero acceleration at both ends."""
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def plan_along_curve(panda, curve, q_start, a_from, a_to, speed=GRIPPER_SPEED, dt=DT):
    """
    One smooth move along the curve, from angle a_from to a_to [rad].
    Returns (q_traj, dt): one row of 7 joint angles per time step.
    """
    # length of the curve, to choose a duration that respects `speed`
    a = np.linspace(a_from, a_to, 200)
    xy = np.array([curve.pose(ai).t[:2] for ai in a])
    length = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    duration = max(2.0, 1.875 * length / speed)   # smooth_fraction peaks at 1.875x the average speed
    n = int(np.ceil(duration / dt))

    q = np.asarray(q_start, dtype=float)
    q_traj = [q]
    for k in range(1, n + 1):
        angle = a_from + smooth_fraction(k / n) * (a_to - a_from)
        q = solve_ik(panda, curve.pose(angle), q)   # seed = previous point: no jumps
        q_traj.append(q)
    return np.array(q_traj), dt


def check_trajectory(panda, q_traj, dt, name):
    """The same checks the robot server makes, so a bad plan fails here, not on the robot."""
    step = np.diff(q_traj, axis=0)                    # how far each joint moves per point
    acc = np.abs(np.diff(step, axis=0)) / dt**2       # how fast that changes
    worst_margin = min(joint_margin(panda, q) for q in q_traj)
    worst_step = np.abs(step).max()
    worst_vel = (np.abs(step) / dt / SERVER_MAX_VEL).max()   # 1.0 = at the server's limit
    worst_acc = (acc / SERVER_MAX_ACC).max()
    print(f"  {name}: {len(q_traj)} points, {len(q_traj) * dt:.1f} s, "
          f"closest to a joint limit {worst_margin:.3f} rad, largest step {worst_step:.4f} rad, "
          f"top speed {worst_vel:.0%} and top acceleration {worst_acc:.0%} of the server limits")
    if worst_margin < SERVER_LIMIT_MARGIN:
        raise RuntimeError(f"{name}: a joint comes within {worst_margin:.3f} rad of its limit")
    if worst_step > SERVER_MAX_STEP:
        raise RuntimeError(f"{name}: a joint jumps {worst_step:.3f} rad between two points")
    if worst_vel > 1.0:
        raise RuntimeError(f"{name}: too fast; lower GRIPPER_SPEED")
    if worst_acc > 1.0:
        raise RuntimeError(f"{name}: a joint's speed changes too abruptly; lower GRIPPER_SPEED")


def plan_shelf_turn(panda, q_now, turn_points=TAUGHT_TURN_POINTS, q_end=Q_LEG_END,
                    center=SHELF_CENTER):
    """Plan TURN then PUSH, starting from q_now (the robot holding the leg at the grasp)."""
    q_now = np.asarray(q_now, dtype=float)
    # The robot is never exactly at the taught grasp: start the curve where it really is.
    points = [q_now] + [np.asarray(q, dtype=float) for q in turn_points[1:]]
    if center is None:
        center = fit_center(panda, list(turn_points) + [q_end])
    curve = ShelfCurve(panda, points, q_end, center)
    print(f"TURN {np.degrees(curve.push_angle - curve.start_angle):.1f} deg, "
          f"PUSH {np.degrees(curve.end_angle - curve.push_angle):.1f} deg")

    plans = []
    q_traj, dt = plan_along_curve(panda, curve, q_now, curve.start_angle, curve.push_angle)
    plans.append(("TURN", q_traj, dt))
    q_traj, dt = plan_along_curve(panda, curve, q_traj[-1], curve.push_angle, curve.end_angle)
    plans.append(("PUSH", q_traj, dt))

    for name, q_traj, dt in plans:
        check_trajectory(panda, q_traj, dt, name)
    return plans


def run_shelf_turn(robot, panda, visualizer=None, ask=True):
    """
    Plan from where the robot is now (it must already hold the leg), then run the plan.
    With a visualizer, each move is shown first. With ask=True, nothing moves until
    you type YES.
    """
    q_now = np.asarray(robot.get_current_joint_positions(), dtype=float)
    plans = plan_shelf_turn(panda, q_now)

    if visualizer is not None:
        for name, q_traj, dt in plans:
            print(f"Showing {name} in the visualizer")
            visualizer.run_joint_trajectory(q_traj, dt)

    if ask and input("Run the shelf turn on the robot? Type YES: ") != "YES":
        print("Cancelled; the robot did not move.")
        return plans

    for name, q_traj, dt in plans:
        print(f"Running {name}")
        robot.run_joint_trajectory(q_traj, dt)
    print("Shelf turn done.")
    return plans


if __name__ == "__main__":
    # Planning only: no robot needed.
    np.set_printoptions(precision=4, suppress=True)
    plan_shelf_turn(rtb.models.Panda(), TAUGHT_TURN_POINTS[0])
