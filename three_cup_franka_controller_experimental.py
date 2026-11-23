# NOTE: This is swift only and unstable (haven't added collision detection)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import math
import argparse
from scipy.interpolate import PchipInterpolator
from csc376_franky import FrankaJointTrajectoryController
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
from spatialmath import SE3
from csc376_franky.visualizer import RtbVisualizer

# ----------------------- PARAMETERS ----------------------------------
SHELF_RADIUS = 0.11
SHELF_HEIGHT = 0.20
SHELF_CLEARANCE = 0.015   # inflated cylinder radius; approximate
SHELF_ROTATION_DEG = 90.0  # Preview only; real indexing requires calibrated 120 deg
SHELF_ARC_STEPS = 100
TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS = 1.79, 0.90, 0.04
TABLE_X, TABLE_Y = 0.20, 0.0
PREVIEW_DT = 0.035       # Swift preview only
ROBOT_DT = 0.01          # Robot controller sample interval
MAX_JOINT_STEP = 0.020  # Course miniPC accepts at most 0.025 rad
TRANSFER_STEPS = 70
LIFT_HEIGHT = 0.07
OPEN_WIDTH, CLOSED_WIDTH = 0.07, 0.025

# Limit risk of IK branch jumps while contacting the shelf.
MAX_IK_JUMP = 0.25
MAX_ELBOW_DRIFT = 1.0

# RRT parameters (configuration space, 7 joint angles, radians)
RRT_SEED = 18
RRT_ITERATIONS = 2400
RRT_STEP = 0.24
EDGE_RESOLUTION = 0.075
GOAL_BIAS = 0.18
JOINT_LIMIT_MARGIN = np.deg2rad(2.0)
RADIAL_RELEASE_DISTANCE = 0.15  # 15cm gradual TCP release before RRT
RADIAL_RELEASE_STEPS = 36
RELEASE_INCREMENTS = 0.015

Q = {
    "cup1": np.array([0.0714, 0.7686, -0.0928, -2.3038, 0.0203, 1.497, 0.7049]),
    "approach": np.array([0.1152, 0.8946, -0.7418, -2.3414, -1.7717, 2.0637, 0.8603]),
    "place": np.array([0.0377, 0.9485, -0.6766, -2.2881, -1.7716, 2.0664, 0.8738]),
    "retreat": np.array([0.4314, 0.9245, -0.8674, -2.3994, -1.5953, 2.246, 0.7938]),
    "leg": np.array([0.233, 1.0381, -0.9726, -2.1166, -1.5948, 2.2983, 0.5578]),
    "rotated_measured": np.array([0.1413, 1.1646, -1.0913, -2.2967, -1.8882, 2.4854, 0.987]),
    "exit": np.array([0.01, 0.7922, -0.2304, -2.0334, -0.9806, 1.5827, 0.4381]),
    "cup2": np.array([0.7782, 1.2754, -0.9127, -2.2612, 0.3216, 1.5339, -0.0672]),
    "cup3": np.array([1.5221, 1.4311, -1.0157, -2.1599, 0.862, 0.8393, 0.0766]),
}


def candidate_centers(p0, p1, radius):
    a, b = p0[:2], p1[:2]
    chord_vec = b - a
    length = np.linalg.norm(chord_vec)
    if not (0 < length <= 2 * radius):
        raise ValueError("Recorded endpoints incompatible with the measured shelf radius")
    midpoint = 0.5 * (a + b)
    normal = np.array([-chord_vec[1], chord_vec[0]]) / length
    offset = math.sqrt(max(0, radius**2 - length**2 / 4))
    return midpoint + offset * normal, midpoint - offset * normal


def ccw_angle(p0, p1, center):
    a = math.atan2(p0[1] - center[1], p0[0] - center[0])
    b = math.atan2(p1[1] - center[1], p1[0] - center[0])
    return (b - a) % (2 * math.pi)


def joint_interpolation(q0, q1, n=TRANSFER_STEPS):
    return np.asarray(rtb.jtraj(q0, q1, n).q)


def ik_pose(panda, T, seed):
    sol = panda.ikine_LM(T, q0=seed, joint_limits=True, ilimit=130)
    if not sol.success:
        raise RuntimeError("Inverse kinematics failed")
    q = np.asarray(sol.q, dtype=float)
    if np.max(np.abs(q - seed)) > MAX_IK_JUMP:
        raise RuntimeError("IK branch jump detected")
    return q


def lift_path(panda, q0):
    start_pose = panda.fkine(q0)
    qs = [np.asarray(q0, dtype=float)]
    try:
        for dz in np.linspace(0, LIFT_HEIGHT, 18)[1:]:
            qs.append(ik_pose(panda, SE3.Tz(float(dz)) * start_pose, qs[-1]))
        return np.asarray(qs)
    except RuntimeError as exc:
        print(f"  WARNING: optional cup lift skipped: {exc}")
        return None


def build_shelf_arc(panda, center):
    start = panda.fkine(Q["leg"])
    p = start.t
    theta0 = math.atan2(p[1] - center[1], p[0] - center[0])
    joints = [Q["leg"].copy()]
    for u in np.linspace(0.0, 1.0, SHELF_ARC_STEPS)[1:]:
        eased = 3 * u**2 - 2 * u**3
        angle = math.radians(SHELF_ROTATION_DEG) * eased
        theta = theta0 + angle
        x = center[0] + SHELF_RADIUS * math.cos(theta)
        y = center[1] + SHELF_RADIUS * math.sin(theta)
        orientation = (SE3.Rz(angle) * start).R
        desired = SE3.Rt(orientation, [x, y, p[2]])
        q_next = ik_pose(panda, desired, joints[-1])
        if np.max(np.abs(q_next[2:4] - Q["leg"][2:4])) > MAX_ELBOW_DRIFT:
            raise RuntimeError(f"Shelf-arc elbow drift too large near {math.degrees(angle):.1f} degrees")
        joints.append(q_next)
    return np.asarray(joints)


class CollisionEnvironment:
    """Conservative collision checks for moving robot links and table/shelf.

    Assumes model collision meshes exist and Coal works. No silently skipping checks.
    The robot base is intentionally on the tabletop, so nonmoving base links are
    excluded from TABLE contacts; the moving arm + gripper links are still checked.
    """

    def __init__(self, panda, center):
        self.robot = panda
        self.table = sg.Cuboid(
            [TABLE_WIDTH, TABLE_LENGTH, TABLE_THICKNESS],
            pose=SE3(TABLE_X, TABLE_Y, -TABLE_THICKNESS / 2),
        )
        self.shelf = sg.Cylinder(
            radius=SHELF_RADIUS + SHELF_CLEARANCE,
            length=SHELF_HEIGHT,
            pose=SE3(center[0], center[1], SHELF_HEIGHT / 2),
        )
        self.lower = np.asarray(panda.qlim[0], dtype=float) + JOINT_LIMIT_MARGIN
        self.upper = np.asarray(panda.qlim[1], dtype=float) - JOINT_LIMIT_MARGIN
        self.moving_links = [link for link in panda.links if link.isjoint]
        for gripper in panda.grippers:
            self.moving_links.extend(gripper.links)
        self.moving_links = [link for link in self.moving_links if len(link.collision) > 0]
        if len(self.moving_links) < 5:
            raise RuntimeError(
                "Robot model lacks enough moving-link collision geometry. "
                "Cannot claim collision-aware planning; check model and Coal installation."
            )
        # Validate installed collision backend and model geometry. The base can
        # legitimately touch the tabletop, so check moving links individually.
        try:
            panda.iscollided(Q["exit"], self.shelf)
        except Exception as exc:
            raise RuntimeError(
                f"External collision API/model failed: {type(exc).__name__}: {exc}"
            ) from exc
        # In the installed CSC376 RTB 1.4.2 build, `isselfcollided` may
        # not exist. Never represent its absence as passing a safety test.
        self.self_collision_available = False
        try:
            panda.iscollided(Q["exit"], panda)
            self.self_collision_available = True
        except (AttributeError, TypeError, ValueError, NotImplementedError) as exc:
            print("WARNING: self-collision check unavailable:", exc)
        print("Self-collision checking available:", self.self_collision_available)

    def collisions(self, q, check_shelf=True):
        q = np.asarray(q, dtype=float)
        reasons = []
        if np.any(q < self.lower) or np.any(q > self.upper):
            return ["joint limit / 2-degree margin"]
        # An external collision query updates link collision transforms.
        self.robot.iscollided(q, self.table)
        if self.self_collision_available:
            if self.robot.iscollided(q, self.robot):
                reasons.append("self collision")
        # The fixed mounting base may intentionally contact the table.
        # Check the modeled moving links against the table only.
        for link in self.moving_links:
            if link.iscollided(self.table):
                reasons.append("table (moving link)")
                break
        if check_shelf and self.robot.iscollided(q, self.shelf):
            reasons.append("shelf")
        return reasons

    def edge_clear(self, q_a, q_b, check_shelf=True):
        q_a = np.asarray(q_a)
        q_b = np.asarray(q_b)
        n = max(2, int(math.ceil(np.max(np.abs(q_b - q_a)) / EDGE_RESOLUTION)) + 1)
        for u in np.linspace(0, 1, n):
            if self.collisions(q_a + u * (q_b - q_a), check_shelf):
                return False
        return True


def release_from_shelf(panda, env, q_rotated, center):
    """Retract the grasp TCP radially while allowing deliberate shelf contact.

    This is NOT collision-checked against the shelf during release, because the
    gripper is physically touching it. Other moving links are checked against
    the table and against one another. At the end, full shelf clearance is required.
    """
    T = panda.fkine(q_rotated)
    radial = T.t[:2] - center
    radial = radial / np.linalg.norm(radial)
    qs = [np.asarray(q_rotated, dtype=float)]
    for dist in np.linspace(0.0, RADIAL_RELEASE_DISTANCE, RADIAL_RELEASE_STEPS)[1:]:
        point = T.t.copy()
        point[:2] += radial * float(dist)
        goal = SE3.Rt(T.R, point)
        q_next = ik_pose(panda, goal, qs[-1])
        if not env.edge_clear(qs[-1], q_next, check_shelf=False):
            raise RuntimeError(
                f"Release path hits table/self or joint margin near {dist:.3f}m. "
                "Need a measured safe release pose; do not increase limits."
            )
        qs.append(q_next)
    if env.collisions(qs[-1], check_shelf=True):
        raise RuntimeError(
            "After 15cm release, the full arm is still in shelf collision "
            "(or another unsafe configuration). Need a measured shelf-clear "
            "pose or a more accurate shelf model."
        )
    return np.asarray(qs)


class Tree:
    def __init__(self, q):
        self.nodes = [np.asarray(q, dtype=float).copy()]
        self.parent = [-1]

    def nearest(self, q):
        distances = np.linalg.norm(np.asarray(self.nodes) - q, axis=1)
        return int(np.argmin(distances))

    def add(self, q, parent):
        self.nodes.append(np.asarray(q, dtype=float).copy())
        self.parent.append(parent)
        return len(self.nodes) - 1

    def trace(self, index):
        chain = []
        while index >= 0:
            chain.append(self.nodes[index])
            index = self.parent[index]
        return chain[::-1]


def grow_toward(tree, target, env, until_reached=False):
    current_idx = tree.nearest(target)
    while True:
        current = tree.nodes[current_idx]
        diff = target - current
        distance = np.linalg.norm(diff)
        if distance < 1e-9:
            return current_idx, True
        nxt = target if distance <= RRT_STEP else current + diff / distance * RRT_STEP
        if not env.edge_clear(current, nxt):
            return current_idx, False
        current_idx = tree.add(nxt, current_idx)
        reached = distance <= RRT_STEP
        if reached or not until_reached:
            return current_idx, reached


def rrt_connect(start, goal, env, rng):
    """Bidirectional RRT-Connect: returns joint waypoints, or raises.

    Every tree edge is discretely collision-tested. Path is smoothed only when
    the shortcut is also collision-tested. No controller timing is generated.
    """
    for label, q in [("Start", start), ("Goal", goal)]:
        problems = env.collisions(q)
        if problems:
            raise RuntimeError(f"{label} is unsafe: {problems}")
    if env.edge_clear(start, goal):
        return np.stack([start, goal])

    a, b = Tree(start), Tree(goal)
    a_is_start = True
    for step in range(RRT_ITERATIONS):
        sample = b.nodes[0] if rng.random() < GOAL_BIAS else rng.uniform(env.lower, env.upper)
        idx_a, _ = grow_toward(a, sample, env)
        # Skip connecting if no extension was made.
        idx_b, reached = grow_toward(b, a.nodes[idx_a], env, until_reached=True)
        if reached:
            branch_a, branch_b = a.trace(idx_a), b.trace(idx_b)
            if a_is_start:
                path = branch_a + branch_b[::-1][1:]
            else:
                path = branch_b + branch_a[::-1][1:]
            print(f"RRT-Connect found {len(path)} joint waypoints after {step+1} iterations")
            return np.asarray(path)
        a, b = b, a
        a_is_start = not a_is_start
    raise RuntimeError(
        f"RRT-Connect found no shelf-clear exit in {RRT_ITERATIONS} iterations. "
        "Do not use a direct interpolated fallback. A measured safe exit pose "
        "or improved collision model is needed."
    )


def shortcut_path(path, env, rng, attempts=120):
    path = [q.copy() for q in path]
    for _ in range(attempts):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j > i+1 and env.edge_clear(path[i], path[j]):
            path = path[:i+1] + path[j:]
    return np.asarray(path)


def densify_path(path, env):
    out = [path[0].copy()]
    for a, b in zip(path[:-1], path[1:]):
        n = max(2, int(math.ceil(np.max(np.abs(b-a)) / 0.045)) + 1)
        for u in np.linspace(0, 1, n)[1:]:
            q = a + u * (b - a)
            if env.collisions(q):
                raise RuntimeError("Postprocessing introduced a collision; rejecting preview")
            out.append(q)
    return np.asarray(out)


# ============================================================================
# COURSE CONTROLLER ADAPTER
# ============================================================================
# This code does NOT use the fixed preview dt for physical execution.
# PCHIP preserves joint waypoints, and time scaling enforces conservative
# estimates of the course server's joint velocity / acceleration / step tests.
# It is NOT a substitute for verified geometry, continuous collision checking,
# FCI monitoring, and supervision by the lab staff.

JOINT_V_MAX = 0.20 * np.array([2.175]*4 + [2.61]*3)
JOINT_A_MAX = 0.50 * np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])


def time_parameterize(path, dt=ROBOT_DT):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
        raise ValueError("Expected at least two 7-joint waypoints")
    # Remove consecutive duplicate points; PCHIP requires strictly increasing s.
    keep = [0]
    for i in range(1, len(path)):
        if np.linalg.norm(path[i]-path[keep[-1]]) > 1e-9:
            keep.append(i)
    path = path[keep]
    if len(path) < 2:
        return np.vstack([path[0], path[0]]), dt
    intervals = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0., np.cumsum(intervals)]
    spline = PchipInterpolator(s/s[-1], path, axis=0)
    # Start/stop at rest using quintic time law. Adapt duration to satisfy
    # finite-difference limits of the COURSE server, not just Cartesian speed.
    duration = max(4.0, s[-1]/0.10)
    for _ in range(12):
        count = int(math.ceil(duration/dt)) + 1
        if count > 120000 or count * dt > 600:
            raise RuntimeError("Trajectory exceeds controller point/duration limits")
        u = np.linspace(0., 1., count)
        smooth = 10*u**3 - 15*u**4 + 6*u**5
        result = np.asarray(spline(smooth), dtype=float)
        # Match endpoints exactly.
        result[0], result[-1] = path[0], path[-1]
        step = np.max(np.abs(np.diff(result,axis=0)), axis=0)
        vel = np.max(np.abs(np.diff(result,axis=0)/dt), axis=0)
        acc = np.max(np.abs(np.diff(result,n=2,axis=0)/(dt*dt)),axis=0)
        if (np.all(step < MAX_JOINT_STEP) and
            np.all(vel < JOINT_V_MAX) and
            np.all(acc < JOINT_A_MAX)):
            return result, dt
        duration *= 1.45
    raise RuntimeError("Failed to time-parameterize path conservatively")


def preflight(panda, path, env, check_shelf=True, allow_contact=False):
    # Every link collision and joint-limit check is discrete, not continuous.
    if allow_contact and check_shelf:
        raise ValueError("Contradictory shelf-contact settings")
    for i, q in enumerate(path):
        hits = env.collisions(q, check_shelf=check_shelf)
        if hits:
            raise RuntimeError(f"Rejected waypoint {i}/{len(path)}: {hits}")
    for a, b in zip(path[:-1],path[1:]):
        if not env.edge_clear(a,b,check_shelf=check_shelf):
            raise RuntimeError("Rejected edge between two sampled waypoints")


def require_exact_start(controller, expected):
    observed = np.asarray(controller.get_current_joint_positions(), dtype=float)
    err = np.max(np.abs(observed-expected))
    if err > 0.01:
        raise RuntimeError(f"Robot moved since planning (max error {err:.4f} rad). "
                           "Replan from current state; do not force a connection.")


def main():
    parser = argparse.ArgumentParser(description="CSC376 Panda three-cup experiment")
    parser.add_argument("--mode", choices=["preview", "sim", "real"], default="preview")
    parser.add_argument("--cycles", type=int, choices=[1,2,3], default=1)
    parser.add_argument("--allow-unverified-real", action="store_true",
                        help="Lab-supervised only. Acknowledges missing calibration/geometry validation.")
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)
    panda = rtb.models.Panda()
    first = panda.fkine(Q["leg"]).t
    last = panda.fkine(Q["rotated_measured"]).t
    candidates = candidate_centers(first,last,SHELF_RADIUS)
    center = min(candidates, key=lambda c: ccw_angle(first,last,c))
    print("Shelf pivot ESTIMATE:", center)
    print(f"Shelf rotation: {SHELF_ROTATION_DEG:g} deg; actual indexing requires ~120 deg")

    # Real hardware constraints: the shelf and TCP are not calibrated, the
    # cup transfer paths have not been checked fully and shelf rotation 90 != 120.
    # Never run a repeated uncalibrated shelf grasp on hardware.
    if args.mode == "real":
        if args.cycles != 1:
            raise RuntimeError("Real mode is restricted to ONE cycle until shelf index geometry is measured")
        if not args.allow_unverified_real:
            raise RuntimeError(
                "Physical robot execution is BLOCKED by default. Current shelf/TCP/"
                "cup geometry is approximate. Review all paths with lab staff; "
                "then use --allow-unverified-real only under direct supervision.")
        print("WARNING: USER-ACKNOWLEDGED EXPERIMENTAL REAL ROBOT MODE")

    arc = build_shelf_arc(panda,center)
    print("Shelf arc IK solved:",len(arc))
    env=CollisionEnvironment(panda,center)
    print("Moving links with collision geometry:",len(env.moving_links))
    if args.mode == "real" and not env.self_collision_available:
        raise RuntimeError("Physical execution blocked: the installed Panda model "
                           "cannot perform a self-collision check. Use --mode sim "
                           "and update the collision model before real execution.")
    print("Planning release + RRT exit...")
    release=release_from_shelf(panda,env,arc[-1],center)
    rng=np.random.default_rng(RRT_SEED)
    raw_exit=rrt_connect(release[-1],Q["exit"],env,rng)
    exit_path=densify_path(shortcut_path(raw_exit,env,rng),env)
    print("Exit path planned:",len(exit_path),"samples")

    controller=None
    viewer=None
    try:
        if args.mode != "preview":
            if args.mode == "sim":
                controller=FrankaJointTrajectoryController(host="127.0.0.1")
            else:
                # Default host is the station's miniPC, per course API.
                controller=FrankaJointTrajectoryController()
            print("Connected controller current q:",controller.get_current_joint_positions())

        # Start Swift at current physical/simulated pose where applicable.
        q_initial=(np.asarray(controller.get_current_joint_positions(),dtype=float)
                   if controller is not None else Q["cup1"].copy())
        viewer=RtbVisualizer(panda,q_initial,real_robot=controller)
        current=q_initial.copy()

        def run(path,label,check_shelf=True,contact=False):
            nonlocal current
            path=np.asarray(path,dtype=float)
            if not np.allclose(path[0],current,atol=1e-6):
                raise RuntimeError(f"Path discontinuity in {label}")
            # Keep drawing possible but reject unsafe physical moves.
            print("\nMOVE:",label)
            if args.mode == "preview":
                viewer.run_joint_trajectory(path,PREVIEW_DT)
            else:
                # For the shelf leg, the coarse cylinder overlaps the gripper.
                # Do not interpret the contact exception as collision approval.
                preflight(panda,path,env,check_shelf=check_shelf,allow_contact=contact)
                timed,dt=time_parameterize(path)
                # Smoothing can change the path: CHECK AGAIN after timing.
                resampled_for_check = np.vstack([timed[::4], timed[-1]])
                preflight(panda,resampled_for_check,env,
                          check_shelf=check_shelf,allow_contact=contact)
                viewer.run_joint_trajectory(timed,dt)
                answer=input(f"Execute '{label}' on {args.mode.upper()} robot? "
                             "Type EXACTLY EXECUTE: ")
                if answer != "EXECUTE":
                    raise RuntimeError("User cancelled. No command sent for this segment")
                require_exact_start(controller,timed[0])
                controller.run_joint_trajectory(timed,dt)
                require_exact_start(controller,timed[-1])
            current=path[-1].copy()

        def joint_move(target,label,check_shelf=True):
            run(joint_interpolation(current,np.asarray(target,dtype=float)),
                label,check_shelf=check_shelf)

        def gripper(width,label):
            print("GRIPPER:",label)
            viewer.move_gripper(width,0.1)
            if controller is not None:
                response=input(f"Command gripper width {width:.3f}m "
                               f"on {args.mode.upper()}? Type GRIPPER: ")
                if response != "GRIPPER":
                    raise RuntimeError("Gripper command cancelled")
                # Course API supports robot.move_gripper(width).
                # Width is not automatically a secure grasp of the real object.
                controller.move_gripper(width)

        gripper(OPEN_WIDTH,"open")
        input("Press Enter to start preview / individually confirmed movements...")
        for cup in range(1,args.cycles+1):
            print(f"\n========== CUP {cup} ==========")
            joint_move(Q[f"cup{cup}"],f"Cup {cup} pickup pose [NOT CALIBRATED]")
            gripper(CLOSED_WIDTH,"grasp cup (width is an estimate)")
            lift=lift_path(panda,current)
            if lift is None:
                raise RuntimeError("Cup lift IK failed; stopping (no skip on controller)")
            run(lift,"Lift cup 7cm")
            joint_move(Q["approach"],"Move toward shelf")
            joint_move(Q["place"],"Place cup")
            gripper(OPEN_WIDTH,"release cup")
            joint_move(Q["retreat"],"Retreat from cup placement",check_shelf=False)
            joint_move(Q["leg"],"Approach shelf leg",check_shelf=False)
            gripper(CLOSED_WIDTH,"grasp shelf leg (not calibrated)")
            run(arc,"Rotate shelf along CCW arc",check_shelf=False,contact=True)
            gripper(OPEN_WIDTH,"release shelf leg")
            run(release,"Retract shelf leg radially",check_shelf=False,contact=True)
            run(exit_path,"RRT collision-checked exit")
        print("All requested cycles completed.")
        input("Press Enter to close Swift...")
    finally:
        if viewer is not None:
            viewer.stop()
        if controller is not None:
            # close() disconnects; DO NOT call stop() unless actively stopping a trajectory.
            controller.close()


if __name__ == "__main__":
    main()
