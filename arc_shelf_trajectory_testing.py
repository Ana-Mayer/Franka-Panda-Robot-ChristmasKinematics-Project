# NOTE: This is swift only and unstable (haven't added collision detection)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import math
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
from spatialmath import SE3
from csc376_franky.visualizer import RtbVisualizer

# ----------------------- PARAMETERS ----------------------------------
SHELF_RADIUS = 0.11
SHELF_HEIGHT = 0.20
SHELF_CLEARANCE = 0.015   # inflated cylinder radius; approximate
SHELF_ROTATION_DEG = 90.0  # 90-degree diagnostic; actual target ~120 degrees
SHELF_ARC_STEPS = 100
TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS = 1.79, 0.90, 0.04
TABLE_X, TABLE_Y = 0.20, 0.0
DT = 0.035               # PREVIEW ONLY; NOT safe robot timing
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
        # Force immediate backend validation. Do not swallow missing-Coal errors.
        try:
            panda.iscollided(Q["exit"], self.shelf)
            self.collisions(Q["exit"], check_shelf=False)
        except Exception as exc:
            raise RuntimeError(
                "Collision engine unavailable or incompatible. Install Coal in the "
                "Dev Container (pip install 'roboticstoolbox-python[collision]')."
            ) from exc

    def collisions(self, q, check_shelf=True):
        q = np.asarray(q, dtype=float)
        reasons = []
        if np.any(q < self.lower) or np.any(q > self.upper):
            reasons.append("joint limit / 2-degree margin")
            return reasons
        # Updates world transforms used by each link collision object.
        if self.robot.isselfcollided(q):
            reasons.append("self collision")
        # Refresh world transforms for per-link external collision queries.
        # (isselfcollided(q) has updated robot link positions.)
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


def main():
    np.set_printoptions(precision=4, suppress=True)
    panda = rtb.models.Panda()
    first = panda.fkine(Q["leg"]).t
    last = panda.fkine(Q["rotated_measured"]).t
    candidate_a, candidate_b = candidate_centers(first, last, SHELF_RADIUS)
    # Always choose the green CCW candidate, regardless of array naming.
    center = min([candidate_a, candidate_b], key=lambda c: ccw_angle(first, last, c))
    print("Estimated shelf centre XY:", center)
    print(f"CCW rotation: {SHELF_ROTATION_DEG:g} degrees, r={SHELF_RADIUS}m")
    print("SWIFT-ONLY. NO ROBOT SERVER / PHYSICAL ROBOT CONNECTION.")
    print("WARNING: approximate geometry, and motion is NOT time-parameterized for hardware.")
    arc = build_shelf_arc(panda, center)
    print("Arc IK:", len(arc), "waypoints")

    env_collision = CollisionEnvironment(panda, center)
    print("Moving links with collision shapes:", len(env_collision.moving_links))
    print("Planning radial release and collision-aware exit...")
    release = release_from_shelf(panda, env_collision, arc[-1], center)
    rng = np.random.default_rng(RRT_SEED)
    exit_waypoints = rrt_connect(release[-1], Q["exit"], env_collision, rng)
    exit_waypoints = shortcut_path(exit_waypoints, env_collision, rng)
    exit_trajectory = densify_path(exit_waypoints, env_collision)
    print("Checked exit trajectory:", len(exit_trajectory), "samples")
    print("NOTE: discrete geometry checking, not continuous collision certification.")

    current = Q["cup1"].copy()
    viewer = None
    try:
        viewer = RtbVisualizer(panda, current)

        def play(path, message):
            nonlocal current
            path = np.asarray(path, dtype=float)
            if not np.allclose(path[0], current, atol=1e-4):
                raise ValueError(f"Discontinuous preview at {message}")
            print(message)
            viewer.run_joint_trajectory(path, DT)
            current = path[-1].copy()

        def move(target, message):
            play(joint_interpolation(current, target), message)

        viewer.move_gripper(OPEN_WIDTH, 0.1)
        input("Press Enter to preview all 3 cup cycles in Swift...")
        for cup in range(1, 4):
            print(f"\n===== CUP {cup} =====")
            move(Q[f"cup{cup}"], f"Cup {cup}: approach recorded pickup [UNVALIDATED]")
            viewer.move_gripper(CLOSED_WIDTH, 0.1)
            print("Close gripper (visual only)")
            lift = lift_path(panda, current)
            if lift is not None:
                play(lift, "Lift cup 7cm [UNVALIDATED]")
            move(Q["approach"], "Transfer toward shelf [UNVALIDATED]")
            move(Q["place"], "Place cup [UNVALIDATED]")
            viewer.move_gripper(OPEN_WIDTH, 0.1)
            move(Q["retreat"], "Retreat after placement [UNVALIDATED]")
            move(Q["leg"], "Approach shelf leg [UNVALIDATED]")
            viewer.move_gripper(CLOSED_WIDTH, 0.1)
            play(arc, f"Rotate shelf {SHELF_ROTATION_DEG:g} degrees CCW [contact preview]")
            viewer.move_gripper(OPEN_WIDTH, 0.1)
            play(release, "Release leg and retract radially [shelf contact exception]")
            play(exit_trajectory, "Collision-checked joint-space exit [preview only]")

        print("\nDone. No hardware commands were sent.")
        input("Press Enter to close Swift...")
    finally:
        if viewer is not None:
            viewer.stop()


if __name__ == "__main__":
    main()
