"""
Идентификация маркеров дронами и ровером
"""

import math
import time

import numpy as np
import sverk_interfaces
import sverk_rover
from simple_offboard_py import Frame, SimpleOffboard

SCOUT_ID, STRIKER_ID, ROVER_ID = "drone_1", "drone_2", "rover_1"
SCOUT_START_IN_ROVER = (0.0, -1.0)

CAMERA = dict(width=640, height=480, fov=65, frame_rate=12, enable=True)
BALLOON_RADIUS = 0.15
CAM_UP = math.radians(10.0)
CAM_FORWARD, CAM_HEIGHT = 0.0527, 0.0278
NEEDLE_FORWARD, NEEDLE_HEIGHT = 0.084, 0.00324649

SCAN_HEIGHT = 1.15
FINAL_SAFETY_MARGIN = 0.06
ROVER_TIMEOUT_MARGIN = 65.0
TRANSIT_HEIGHT = 1.9
VIEW_DISTANCE = 1.52
VIEW_CLEARANCE = 0.8
SITE_LINK = 1.0
SITE_RADIUS = 1.0
# Merge panoramic fragments of the same dense site BEFORE flying there.
SITE_SEED_MERGE = 2.20
FLIGHT_LIMIT = 270

ROVER_STANDOFF = 1.0
VANTAGE_RADIUS = 1.6
ROVER_SPEED = 0.10
ROVER_ACCEL = 0.05
ROVER_TURN_RATE = 0.35


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def color_masks(frame):
    """RGB пороги, дополненные нормированной насыщенностью.

    Не требует cv2: зависимости исходного симулятора сохранены.
    """
    rgb = frame[..., :3].astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    saturation = (maximum - minimum) / np.maximum(maximum, 1)
    colorful = saturation > 0.22
    return {
        "red": colorful & (r > 65) & (r > g * 1.43) & (r > b * 1.35),
        "yellow": colorful & (r > 85) & (g > 75) & (b < np.minimum(r, g) * 0.67)
                  & (np.abs(r - g) < np.minimum(r, g) * 0.39),
        "green": colorful & (g > 68) & (g > r * 1.29) & (g > b * 1.22),
    }


def blobs(mask, step, min_pixels=8):
    small = mask[::step, ::step]
    ys, xs = np.nonzero(small)
    remaining = set(zip(ys.tolist(), xs.tolist()))
    h, w = small.shape
    found = []

    while remaining:
        seed = remaining.pop()
        stack, component = [seed], [seed]

        while stack:
            y, x = stack.pop()

            for n in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if n in remaining:
                    remaining.remove(n)
                    stack.append(n)
                    component.append(n)

        if len(component) < min_pixels:
            continue

        rows = {}

        for y, x in component:
            lo, hi = rows.get(y, (x, x))
            rows[y] = (min(lo, x), max(hi, x))

        width = max(hi - lo + 1 for lo, hi in rows.values())
        band = [(y, (lo + hi) / 2) for y, (lo, hi) in rows.items() if hi - lo + 1 >= 0.95 * width]
        cy = sum(y for y, _ in band) / len(band)
        cx = sum(x for _, x in band) / len(band)
        ys_c = [y for y, _ in component]
        xs_c = [x for _, x in component]
        edge = min(xs_c) == 0 or min(ys_c) == 0 or max(xs_c) == w - 1 or max(ys_c) == h - 1

        found.append({"cx": (cx + 0.5) * step, "cy": (cy + 0.5) * step, "width": width * step,
                      "left": min(xs_c) * step, "right": (max(xs_c) + 1) * step,
                      "top": min(ys_c) * step, "bottom": (max(ys_c) + 1) * step, "edge": edge,
                      "round": len(component) / (math.pi * width * width / 4)})

    return found


def detect_balloons(frame):
    h, w = frame.shape[:2]
    fx = h / (2 * math.tan(math.radians(CAMERA["fov"]) / 2))
    step = 2 if w >= 480 else 1
    result = []

    for color, mask in color_masks(frame).items():
        for blob in blobs(mask, step, min_pixels=7):
            if blob["width"] < 5 or blob["round"] < 0.30 or blob["round"] > 1.85:
                continue
            depth = math.sqrt((2 * fx * BALLOON_RADIUS / blob["width"]) ** 2 + BALLOON_RADIUS ** 2)
            u = (blob["cx"] - w / 2) / fx
            v = (blob["cy"] - h / 2) / fx

            result.append({
                "color": color, **blob,
                "forward": CAM_FORWARD + depth * (math.cos(CAM_UP) + v * math.sin(CAM_UP)),
                "left": -u * depth,
                "up": CAM_HEIGHT + depth * (math.sin(CAM_UP) - v * math.cos(CAM_UP)),
            })

    return result


class Microdrone:
    def __init__(self, drone_id):
        self.id = drone_id
        self.fc = SimpleOffboard()
        self.fc._drone_id = drone_id
        self.camera = sverk_interfaces.init(drone_id=drone_id).camera
        self.origin = (0.0, 0.0, 0.0)
        self.started = None

    def _raw(self):
        north, east, down, _, _, yaw = self.fc.get_position()

        return -north, east, -down, wrap(math.pi - yaw)

    def set_origin(self):
        x, y, z, _ = self._raw()
        self.origin = (x, y, z)

    def pose(self):
        x, y, z, yaw = self._raw()

        return x - self.origin[0], y - self.origin[1], z - self.origin[2], yaw

    def time_left(self):
        return FLIGHT_LIMIT - (time.monotonic() - self.started) if self.started else FLIGHT_LIMIT

    def takeoff(self, height):
        self.camera.configure(**CAMERA)

        if not self.fc.takeoff(height):
            raise RuntimeError(f"{self.id}: взлёт не завершён")

        self.started = time.monotonic()

    def goto(self, x, y, z, yaw, time_limit=40.0):
        ox, oy, oz = self.origin
        ok = self.fc.navigate(-(x + ox), y + oy, -(z + oz), Frame.LOCAL_NED,
                              yaw=wrap(math.pi - yaw), tolerance=0.08, time_limit=time_limit)
        deadline = time.monotonic() + 10

        while abs(wrap(self.pose()[3] - yaw)) > math.radians(6) and time.monotonic() < deadline:
            time.sleep(0.1)

        return ok

    def turn(self, yaw):
        x, y, z, _ = self.pose()

        return self.goto(x, y, z, yaw)

    def velocity(self, forward=0.0, left=0.0, up=0.0, yaw=None):
        heading = self.pose()[3] if yaw is None else yaw
        vx = forward * math.cos(heading) - left * math.sin(heading)
        vy = forward * math.sin(heading) + left * math.cos(heading)
        self.fc.set_velocity(-vx, vy, -up, Frame.LOCAL_NED, yaw=wrap(math.pi - heading), yaw_rate=None)

    def look(self, settle=0.4):
        time.sleep(settle)
        frame = self.camera.read_numpy(encoding="rgb8", timeout=1.5)

        if frame is None:
            return

        x, y, z, yaw = self.pose()

        for b in detect_balloons(frame):
            b["x"] = x + b["forward"] * math.cos(yaw) - b["left"] * math.sin(yaw)
            b["y"] = y + b["forward"] * math.sin(yaw) + b["left"] * math.cos(yaw)
            b["z"] = z + b["up"]

            yield b

    def land(self):
        try:
            self.velocity()
            self.fc.land()

            print(f"{self.id}: посадка выполнена", flush=True)
        except Exception as error:
            print(f"{self.id}: посадка не подтверждена: {error}", flush=True)


class Channel:
    def __init__(self):
        self.messages = []

    def send(self, sender, kind, **data):
        self.messages.append((sender, kind, data))

        print(f"[канал] {sender} -> {kind}: {data}", flush=True)

        return data


def cluster(points, link):
    groups = []

    for p in points:
        joined = [g for g in groups if any(math.dist(p[:2], q[:2]) < link for q in g)]
        merged = [p] + [q for g in joined for q in g]
        groups = [g for g in groups if g not in joined] + [merged]

    return groups


def scout_mission(scout):
    scout.set_origin()
    scout.takeoff(SCAN_HEIGHT)

    print("Разведчик: панорамный осмотр", flush=True)

    seen = []

    for k in range(12):
        scout.turn(math.radians(30 * k))
        seen += [(b["x"], b["y"], b["color"]) for b in scout.look() if not b["edge"]]

    groups = [g for g in cluster(seen, SITE_LINK) if len(g) >= 2]
    raw_queue = [(float(np.median([p[0] for p in g])),
                  float(np.median([p[1] for p in g]))) for g in groups]
    # Nearby groups may be fragments of ONE site, especially when several
    # balloons occlude each other during the initial panorama.
    seed_groups = cluster(raw_queue, SITE_SEED_MERGE)
    queue = [(float(np.mean([p[0] for p in g])),
              float(np.mean([p[1] for p in g]))) for g in seed_groups]
    print('Разведчик: панорамные фрагменты:',
          [(round(a,2),round(b,2)) for a,b in raw_queue], flush=True)
    print('Разведчик: объединены фрагменты:',
          [[(round(a,2),round(b,2)) for a,b in g] for g in seed_groups],flush=True)
    known = [(p[0], p[1]) for p in seen]

    print("Разведчик: кандидаты по панораме:", [(round(x, 2), round(y, 2)) for x, y in queue], flush=True)

    report = []

    while queue:
        if scout.time_left() < 60:
            print("Разведчик: заряд на исходе, осмотр прекращён", flush=True)
            break

        x, y, _, _ = scout.pose()
        queue.sort(key=lambda q: math.hypot(q[0] - x, q[1] - y))   # ближайший — следующим
        site = queue.pop(0)

        if any(math.hypot(site[0] - r["x"], site[1] - r["y"]) < SITE_SEED_MERGE for r in report):
            continue

        result, others = inspect_site(scout, site, len(report), known)
        twin = [r for r in report if math.hypot(result["x"] - r["x"], result["y"] - r["y"]) < SITE_SEED_MERGE]

        if twin:
            twin[0].update({c: max(twin[0][c], result[c]) for c in ("red", "yellow", "green")})
        elif result["red"] + result["yellow"] + result["green"]:
            report.append(result)

            print(f"Разведчик: участок {result['label']} ({result['x']:.2f}, {result['y']:.2f}): "
                  f"красных {result['red']}, жёлтых {result['yellow']}, зелёных {result['green']}", flush=True)

        for g in cluster(others, SITE_RADIUS):
            if len(g) < 2:
                continue

            c = (float(np.mean([p[0] for p in g])), float(np.mean([p[1] for p in g])))

            if all(math.hypot(c[0] - r["x"], c[1] - r["y"]) > SITE_SEED_MERGE for r in report) \
                    and all(math.dist(c, q) > SITE_SEED_MERGE for q in queue):
                queue.append(c)

    climb_and_go(scout, 0.0, 0.0, 0.0)
    scout.goto(0.0, 0.0, 0.5, 0.0)
    scout.land()

    return report


def climb_and_go(drone, x, y, yaw):
    px, py, _, current = drone.pose()
    drone.goto(px, py, TRANSIT_HEIGHT, current)
    drone.goto(x, y, TRANSIT_HEIGHT, yaw)


def segment_clear(a, b, known, margin):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy or 1e-9

    for k in known:
        t = max(0.0, min(1.0, ((k[0] - a[0]) * dx + (k[1] - a[1]) * dy) / length2))

        if math.hypot(k[0] - a[0] - t * dx, k[1] - a[1] - t * dy) < margin:
            return False

    return True


def fuse_observations(frames, center, radius=SITE_RADIUS):
    """Учитываем все правдоподобные шары в ОБЛАСТИ участка.

    v6 ошибочно отбрасывала перекрытые силуэты и контуры с round вне
    0.43..1.60, а потом переносила центр участка к оставшемуся одному шару.
    Одна физическая цель не может быть двумя треками в одном кадре.
    """
    tracks = []
    peak = {c: 0 for c in ('red', 'yellow', 'green')}
    stats = []
    for view, objects in enumerate(frames):
        # Сохраняем исходный центр панорамы: оценка расстояния по ширине
        # частично закрытого шара может оказаться ошибочной.
        region = [b for b in objects if not b['edge'] and
                  math.hypot(b['x'] - center[0], b['y'] - center[1]) < radius + 0.35]
        accepted = [b for b in region if 0.30 <= b['round'] <= 1.85 and b['width'] >= 5]
        # Отбрасывать перекрытые шары по bbox нельзя: для густого участка
        # B это уничтожало практически все наблюдения.
        stats.append((len(objects), len(region), len(accepted)))
        for c in peak:
            peak[c] = max(peak[c], sum(b['color'] == c for b in accepted))
        possible = []
        for i, obj in enumerate(accepted):
            for j, tr in enumerate(tracks):
                if tr['color'] != obj['color'] or view in tr['views']:
                    continue
                pos = np.median(np.asarray(tr['points']), axis=0)
                dxy = math.hypot(float(pos[0]-obj['x']),float(pos[1]-obj['y']))
                dz = abs(float(pos[2]-obj['z']))
                if dxy < 0.31 and dz < 0.36:
                    possible.append((dxy + 0.25*dz, i, j))
        used_obj, used_track = set(), set()
        for _, i, j in sorted(possible):
            if i in used_obj or j in used_track:
                continue
            obj = accepted[i]
            tracks[j]['points'].append((obj['x'], obj['y'], obj['z']))
            tracks[j]['views'].add(view)
            used_obj.add(i)
            used_track.add(j)
        for i, obj in enumerate(accepted):
            if i not in used_obj:
                tracks.append({'color':obj['color'],
                               'points':[(obj['x'],obj['y'],obj['z'])],
                               'views':{view}})
    # Повторные 3D-оценки одной цели с разных ракурсов могут сильно расходиться.
    # Сливаем только очень близкие треки, НЕ наблюдавшиеся одновременно.
    while True:
        pairs=[]
        for i,a in enumerate(tracks):
            for j in range(i+1,len(tracks)):
                b=tracks[j]
                if a['color'] != b['color'] or a['views'] & b['views']:
                    continue
                pa=np.median(np.asarray(a['points']),axis=0)
                pb=np.median(np.asarray(b['points']),axis=0)
                d=math.dist(pa,pb)
                if d < 0.285:
                    pairs.append((d,i,j))
        if not pairs:
            break
        _,i,j=min(pairs)
        tracks[i]['points'].extend(tracks[j]['points'])
        tracks[i]['views'].update(tracks[j]['views'])
        del tracks[j]
    unique=[]
    for t in tracks:
        pos=np.median(np.asarray(t['points']),axis=0)
        if math.hypot(pos[0]-center[0],pos[1]-center[1]) < radius+0.35:
            unique.append((t['color'],tuple(float(x) for x in pos),len(t['views'])))
    print(f'Разведчик: фильтрация кадров (всего/в зоне/принято)={stats}',flush=True)
    return unique,peak


def _orbit_step(scout, goal, center, known, speed=0.19, timeout=9.0):
    """Плавный перелёт к короткой точке окружности, с постоянным взглядом к центру."""
    deadline = time.monotonic() + timeout
    slow_frames = 0
    while time.monotonic() < deadline:
        x, y, z, yaw = scout.pose()
        dx, dy = goal[0]-x, goal[1]-y
        dist = math.hypot(dx,dy)
        face = math.atan2(center[1]-y, center[0]-x)
        if dist < 0.12:
            scout.velocity(yaw=face)
            return True
        # 2D potential collision envelope. Stop if a known marker is close
        # to the next part of the path; do not continue at scan altitude.
        ahead = (x + min(dist, 0.35)*dx/max(dist,1e-6),
                 y + min(dist, 0.35)*dy/max(dist,1e-6))
        if not segment_clear((x,y),ahead,known,0.48):
            scout.velocity(yaw=face)
            return False
        # Navigation velocity is expressed in heading-aligned coordinates.
        vx = dx/max(dist,1e-6)*min(speed,0.75*dist)
        vy = dy/max(dist,1e-6)*min(speed,0.75*dist)
        forward = vx*math.cos(face) + vy*math.sin(face)
        left = -vx*math.sin(face) + vy*math.cos(face)
        scout.velocity(forward=forward,left=left,
                       up=max(-0.10,min(0.10,1.0*(SCAN_HEIGHT-z))),yaw=face)
        time.sleep(0.09)
    scout.velocity()
    return False


def inspect_site(scout, site, index, known):
    """Ближний кольцевой осмотр: малый шаг, плавный ход, камера к центру.

    Большие скачки с одного диаметрально противоположного ракурса на другой
    в v7 вызывали ненужные резкие манёвры и слишком большие удаления.
    """
    sx, sy = site
    px, py, _, _ = scout.pose()
    base = math.atan2(py-sy,px-sx)
    frames, everything = [], []
    orbit_radius = 1.52
    nsteps = 12
    # Keep clear of nearby balloons even if the panoramic center is imperfect.
    # A radius increment is capped; going far away to force a viewpoint is forbidden.
    checkpoints = []
    for k in range(nsteps):
        angle=base + 2*math.pi*k/nsteps
        chosen=None
        for radius in (orbit_radius, 1.67, 1.82):
            point=(sx+radius*math.cos(angle),sy+radius*math.sin(angle))
            if all(math.dist(point,q[:2]) >= 0.57 for q in known):
                chosen=point
                break
        checkpoints.append(chosen)
    # First point is reached using high transit altitude; later points are
    # reached on the ring without repeated 1.9m climbs whenever path is clear.
    started=False
    for k,point in enumerate(checkpoints):
        if scout.time_left()<65:
            print('Разведчик: оставляю время на возвращение',flush=True)
            break
        if point is None:
            print(f'Разведчик: сектор {k+1}/{nsteps} занят шарами, пропускаю',flush=True)
            continue
        angle=base+2*math.pi*k/nsteps
        face=math.atan2(sy-point[1],sx-point[0])
        x,y,z,_=scout.pose()
        # A blocked arc is crossed only at safe transit altitude.
        clear=segment_clear((x,y),point,known,0.52)
        try:
            if not started or not clear or math.hypot(point[0]-x,point[1]-y)>0.95:
                print(f'Разведчик: сектор {k+1}: переход сверху',flush=True)
                climb_and_go(scout,point[0],point[1],face)
                ok=scout.goto(point[0],point[1],SCAN_HEIGHT,face,time_limit=18)
            else:
                ok=_orbit_step(scout,point,(sx,sy),known,speed=0.19,timeout=11)
                if not ok:
                    # unsafe short segment, do not blindly push forward
                    climb_and_go(scout,point[0],point[1],face)
                    ok=scout.goto(point[0],point[1],SCAN_HEIGHT,face,time_limit=18)
            if not ok:
                print(f'Разведчик: сектор {k+1}: позиция не достигнута',flush=True)
                continue
            started=True
            frame=list(scout.look(settle=0.16))
            frames.append(frame)
            everything.extend((b['x'],b['y'],b['color']) for b in frame if not b['edge'])
            known.extend((b['x'],b['y']) for b in frame if not b['edge'])
            count={c:sum(b['color']==c and not b['edge'] for b in frame)
                   for c in ('red','yellow','green')}
            print(f"Разведчик: участок {index+1}, сектор {k+1}/{nsteps}, "
                  f"R={count['red']}, Y={count['yellow']}, G={count['green']}, "
                  f"расстояние от центра={math.hypot(point[0]-sx,point[1]-sy):.2f} м",flush=True)
        except Exception as exc:
            scout.velocity()
            print(f'Разведчик: сектор {k+1}: ошибка навигации {exc}',flush=True)
            continue
    scout.velocity()
    unique, peak = fuse_observations(frames,(sx,sy),radius=SITE_RADIUS+0.20)
    reliable=[v for v in unique if v[2]>=2]
    if reliable:
        sx=float(np.median([v[1][0] for v in reliable]))
        sy=float(np.median([v[1][1] for v in reliable]))
    # Strong tracks only: a stationary physical balloon should be visible from
    # multiple nearby sectors. Isolated detections (one or two sectors) are
    # often duplicate or misplaced triangulations from overlapping silhouettes.
    confirmed={c:sum(v[0]==c and v[2]>=3 for v in unique)
               for c in ('red','yellow','green')}
    tentative={c:sum(v[0]==c and v[2]>=2 for v in unique)
               for c in ('red','yellow','green')}
    # Do not discard simultaneous detections merely because a short orbit
    # leaves an object visible in only one/two sectors.
    # 'peak' is after geometrical filtering to THIS site; it is a lower-bound
    # candidate, not a claim that every raw image blob is real.
    counts={c:max(confirmed[c], tentative[c], peak[c])
            for c in ('red','yellow','green')}
    print(f'Разведчик: итог по трекам И одновременному кадру={counts}',flush=True)
    print(f'Разведчик: проверенные треки (3+ сектора)={confirmed}; '
          f'предварительные (2+ сектора)={tentative}',flush=True)
    print(f'Разведчик: участок {index+1}, треки={counts}, '
          f'подтверждено={confirmed}, максимум одного кадра={peak}',flush=True)
    print(f'Разведчик: уникальные шары участка {index+1}: '
          +str([(c,round(p[0],2),round(p[1],2),round(p[2],2),n)
                for c,p,n in unique]),flush=True)
    extent=max((math.hypot(v[1][0]-sx,v[1][1]-sy) for v in unique),default=0.5)
    label='ABCDEFGH'[index] if index<8 else str(index+1)
    others=[q for q in everything if math.hypot(q[0]-sx,q[1]-sy)>SITE_RADIUS+0.4]
    return {'label':label,'x':sx,'y':sy,'extent':round(extent,2),**counts},others


class Carrier:
    def __init__(self, rover):
        self.rover = rover
        pose = rover.get_robot_pose()
        self.origin = (pose["x"], pose["y"])

    def pose(self):
        p = self.rover.get_robot_pose()

        return p["x"] - self.origin[0], p["y"] - self.origin[1], math.radians(p["yaw_deg"])

    def turn_to(self, yaw, timeout=30.0):
        rate, last = 0.0, time.monotonic()
        deadline = last + timeout

        while time.monotonic() < deadline:
            error = wrap(yaw - self.pose()[2])

            if abs(error) < math.radians(2) and abs(rate) < 0.05:
                break

            now = time.monotonic()
            dt, last = max(now - last, 0.02), now
            wanted = max(-ROVER_TURN_RATE, min(ROVER_TURN_RATE, 1.5 * error))
            rate += max(-0.3 * dt, min(0.3 * dt, wanted - rate))
            self.rover.set_velocity(wz=rate)
            time.sleep(0.05)

        self.rover.stop_motion()

    def drive_to(self, x, y, timeout=None):
        """Медленный контроллер точки: ограничение скорости 0.10 м/с."""
        px, py, _ = self.pose()
        start_range = math.hypot(x - px, y - py)
        if timeout is None:
            timeout = max(180.0, 2.5 * start_range / ROVER_SPEED + ROVER_TIMEOUT_MARGIN)
        self.turn_to(math.atan2(y - py, x - px), timeout=45.0)
        speed, last = 0.0, time.monotonic()
        deadline = last + timeout
        last_progress, prior_distance = last, start_range
        reached = False
        try:
            while time.monotonic() < deadline:
                px, py, yaw = self.pose()
                distance = math.hypot(x - px, y - py)
                if distance < 0.065:
                    reached = True
                    break
                if prior_distance - distance > 0.04:
                    prior_distance, last_progress = distance, time.monotonic()
                if time.monotonic() - last_progress > 30:
                    raise RuntimeError(f"Ровер не продвигается к точке: остаток {distance:.2f} м")
                now = time.monotonic()
                dt, last = min(max(now - last, 0.02), 0.2), now
                heading_error = wrap(math.atan2(y - py, x - px) - yaw)
                # Сначала довернуть, чтобы не срезать путь через соседние маркеры.
                wanted = min(ROVER_SPEED, math.sqrt(max(0.0, 2 * ROVER_ACCEL * distance)))
                if abs(heading_error) > math.radians(32):
                    wanted = 0.0
                else:
                    wanted *= max(0.35, math.cos(heading_error))
                speed += max(-ROVER_ACCEL * dt, min(ROVER_ACCEL * dt, wanted - speed))
                speed = max(0.0, min(ROVER_SPEED, speed))
                steer = max(-ROVER_TURN_RATE, min(ROVER_TURN_RATE, 1.45 * heading_error))
                self.rover.set_velocity(vx=speed, wz=steer)
                time.sleep(0.05)
        finally:
            self.rover.stop_motion()
        if not reached:
            raise TimeoutError(f"Ровер не достиг точки ({x:.2f}, {y:.2f})")
        px, py, _ = self.pose()
        print(f"Ровер: точка достигнута, ошибка {math.hypot(x-px,y-py):.3f} м", flush=True)


def deliver(carrier, site, extent, striker):
    px, py, _ = carrier.pose()
    distance = math.hypot(site[0] - px, site[1] - py)
    standoff = max(ROVER_STANDOFF, extent + 0.6)
    k = max(0.0, (distance - standoff) / max(distance, 1e-9))
    stop = (px + (site[0] - px) * k, py + (site[1] - py) * k)

    print(f"Ровер: везу исполнителя к ({stop[0]:.2f}, {stop[1]:.2f})", flush=True)

    carrier.rover.set_led_preset("navigation")
    carrier.drive_to(*stop)
    carrier.turn_to(math.atan2(site[1] - stop[1], site[0] - stop[0]))
    carrier.rover.set_led_preset("success")
    rover = carrier.rover.get_robot_pose()
    sx, sy, sz, _ = striker._raw()
    offset = math.hypot(sx - rover["x"], sy - rover["y"])

    print(f"Ровер: прибыл. Исполнитель на палубе: {'да' if offset < 0.07 and sz > 0.12 else 'НЕТ'} "
          f"(смещение {offset:.3f} м)", flush=True)

    return carrier.pose()[:2]


def path_clear(drone, red, others):
    x, y, _, _ = drone.pose()
    dx, dy = red["x"] - x, red["y"] - y
    length = math.hypot(dx, dy) or 1.0

    for b in others:
        if b is red:
            continue

        t = ((b["x"] - x) * dx + (b["y"] - y) * dy) / length

        if t <= 0 or t >= length:
            continue

        side = abs((b["x"] - x) * dy - (b["y"] - y) * dx) / length

        if side < 2 * BALLOON_RADIUS + 0.08 and abs(b["z"] - red["z"]) < 2 * BALLOON_RADIUS + 0.1:
            return False

    return True


def occluded(blob, seen):
    for b in seen:
        if b is not blob and b["width"] > blob["width"] \
                and b["left"] < blob["right"] + 2 and b["right"] > blob["left"] - 2 \
                and b["top"] < blob["bottom"] + 2 and b["bottom"] > blob["top"] - 2:
            return True

    return False


def choose_red(drone, target, skip=(), allow_occluded=False, radius=SITE_RADIUS):
    seen = list(drone.look(settle=0.3))
    reds = sorted((b for b in seen if b["color"] == "red" and not b["edge"]
                   and (allow_occluded or not occluded(b, seen))
                   and math.hypot(b["x"] - target[0], b["y"] - target[1]) < radius
                   and all(math.dist((b["x"], b["y"]), q) > 0.25 for q in skip)),
                  key=lambda b: b["forward"])

    # The wide 0.38-m global path clearance falsely rejects red targets in
    # densely spaced plots. We will check non-red occlusion immediately before
    # final alignment, using the camera instead.
    for red in reds:
        nonred = [b for b in seen if b['color'] != 'red' and b is not red
                  and b['forward'] < red['forward']-0.17
                  and abs(b['left']-red['left']) < 0.21
                  and abs(b['up']-red['up']) < 0.22]
        if not nonred:
            return red, len(reds)
    return None, len(reds)


def pop_balloon(drone, red):
    """Camera servo, cautious recovery from frame loss, then real needle stroke.

    All geometry is in the drone's own local map. In particular, do NOT
    abort just because a sphere leaves the FOV at close range. Never ram
    without a previous visual alignment or a reasonably verified path.
    """
    def clip(x, bound):
        return max(-bound, min(bound, x))

    lock = np.array([red['x'], red['y'], red['z']], dtype=float)
    px, py, pz, _ = drone.pose()
    heading = math.atan2(lock[1] - py, lock[0] - px)
    print(f'{drone.id}: цель R {tuple(round(v,2) for v in lock)}', flush=True)
    drone.turn(heading)

    last_valid = None
    aligned = 0
    lost = 0
    previous_range = 10.0
    deadline = time.monotonic() + 50.0

    while time.monotonic() < deadline:
        seen = list(drone.look(settle=0.0))
        candidates = [b for b in seen
                      if b['color'] == 'red' and not b['edge']
                      and math.dist((b['x'], b['y'], b['z']), lock) < 0.95]
        # Prefer alignment in the camera rather than noisy 3-D triangulation.
        if candidates:
            b = min(candidates, key=lambda v: (
                abs(v['left']) + 0.65 * abs(v['up'] - NEEDLE_HEIGHT)
                + 0.3 * math.dist((v['x'], v['y'], v['z']), lock)))
            # Protect non-red markers: only actual *nearer* silhouettes
            # are blockers, and require tight overlap with the needle lane.
            blocking = [q for q in seen if q is not b and q['color'] != 'red'
                        and 0.05 < q['forward'] < b['forward'] - 0.17
                        and abs(q['left'] - b['left']) < 0.21
                        and abs(q['up'] - b['up']) < 0.22]
            if blocking:
                drone.velocity()
                print(f'{drone.id}: некрасный шар перед целью, другую позицию',flush=True)
                return None
            forward, left, up = b['forward'], b['left'], b['up'] - NEEDLE_HEIGHT
            previous_range = forward
            last_valid = (forward, left, up)
            lock = 0.88 * lock + 0.12 * np.asarray((b['x'], b['y'], b['z']))
            lost = 0
            advance = forward > 0.61
            drone.velocity(forward=clip(max(0., (forward-0.56)*0.42), 0.18) if advance else 0.,
                           left=clip(left * 0.8, 0.11),
                           up=clip(up * 0.85, 0.14), yaw=heading)
            if not advance and abs(left) < 0.035 and abs(up) < 0.035:
                aligned += 1
                if aligned >= 3:
                    break
            else:
                aligned = 0
        else:
            lost += 1
            drone.velocity(yaw=heading)
            # The target often vanishes as the drone closes in. If visually
            # aligned at short range, proceed using the last known observation.
            if last_valid and previous_range < 0.83 and \
                    abs(last_valid[1]) < 0.055 and abs(last_valid[2]) < 0.055:
                print(f'{drone.id}: шар вышел из кадра вблизи, '
                      f'продолжаю по последнему наведению',flush=True)
                break
            if lost == 4 and last_valid is None:
                # A small heading correction, not an abrupt multi-point tour.
                x,y,z,_=drone.pose()
                h=math.atan2(lock[1]-y, lock[0]-x)
                drone.turn(h)
                heading=h
            if lost >= 12:
                print(f'{drone.id}: шар не найден с безопасной дистанции',flush=True)
                return False
        time.sleep(0.08)
    else:
        drone.velocity()
        print(f'{drone.id}: истекло время наведения',flush=True)
        return False

    drone.velocity(yaw=heading)
    if last_valid is None:
        return False
    forward,left,up=last_valid
    # Freeze the camera-certified lane. Drive *past* the sphere's center;
    # the needle, NOT the drone body, must intersect the collision geometry.
    # Limit the stroke: we must not strike an unseen green/yellow sphere.
    stroke = min(0.95, max(0.18, forward + 0.01))
    x0,y0,z0,heading=drone.pose()
    target_z=z0+up
    start=time.monotonic()
    deadline=start+max(15.0,stroke/0.10+5)
    best=0.0
    stalled=time.monotonic()
    print(f'{drone.id}: КОНТАКТНЫЙ ПРОКОЛ: '
          f'камера до центра {forward:.2f} м, заданный ход {stroke:.2f} м',flush=True)
    reached=False
    while time.monotonic()<deadline:
        x,y,z,_=drone.pose()
        dx,dy=x-x0,y-y0
        along=dx*math.cos(heading)+dy*math.sin(heading)
        lateral=-dx*math.sin(heading)+dy*math.cos(heading)
        if along >= stroke-0.025:
            reached=True
            break
        if along > best + 0.006:
            best=along
            stalled=time.monotonic()
        if time.monotonic()-stalled>5:
            print(f'{drone.id}: остановка по отсутствию продвижения',flush=True)
            break
        drone.velocity(forward=0.105,
                       left=clip(1.3*(left-lateral),0.065),
                       up=clip(1.45*(target_z-z),0.10),yaw=heading)
        time.sleep(0.075)
    drone.velocity(yaw=heading)
    print(f'{drone.id}: фактический контактный ход {best:.2f}/{stroke:.2f} м, '
          f'полный={reached}',flush=True)
    drone.velocity(forward=-0.11,yaw=heading)
    time.sleep(1.6)
    drone.velocity(yaw=heading)
    # A vanished target alone isn't enough proof; require completed stroke
    # and subsequent absence at the old position.
    after=list(drone.look(settle=0.35))
    still=[b for b in after if b['color']=='red'
           and math.dist((b['x'],b['y'],b['z']),lock)<0.38]
    if still:
        print(f'{drone.id}: после контакта красный шар всё ещё виден',flush=True)
        return False
    return reached

def strike_mission(striker, target, expected, extent):
    striker.set_origin()
    striker.takeoff(0.8)
    height = striker.pose()[2]
    base = math.atan2(-target[1], -target[0]) if math.hypot(*target) > 0.1 else 0.0
    vantages = [base + math.radians(a) for a in (0, 60, -60, 120, -120, 180)]

    radius = max(VANTAGE_RADIUS, extent + 0.7)

    def vantage(angle):
        point = (target[0] + radius * math.cos(angle), target[1] + radius * math.sin(angle))

        return point, math.atan2(target[1] - point[1], target[0] - point[0])

    current = vantages[0]
    point, face = vantage(current)
    striker.goto(point[0], point[1], height, face)
    popped, misses = 0, 0
    blocked = {}

    while misses < 3 and striker.time_left() > 25:
        found = None

        for allow_occluded in (False, True):
            tour = [current] if popped >= expected else [current] + [v for v in vantages if v != current]

            for angle in tour:
                if striker.time_left() < 25:
                    break

                point, face = vantage(angle)

                if angle != current:
                    climb_and_go(striker, point[0], point[1], face)    # смена ракурса — выше шаров
                    current = angle

                striker.goto(point[0], point[1], height, face)

                for delta in (0, 30, -30):
                    striker.turn(face + math.radians(delta))
                    found, visible = choose_red(striker, target, blocked.get(angle, []), allow_occluded)

                    if found:
                        break

                if found:
                    break

            if found:
                break

        if not found:
            print(f"{striker.id}: доступных красных шаров на участке больше нет", flush=True)
            break

        result = pop_balloon(striker, found)

        if result:
            popped += 1
            misses = 0

            print(f"{striker.id}: красный шар обработан ({popped})", flush=True)
        else:
            blocked.setdefault(current, []).append((found["x"], found["y"]))

            if result is False:
                misses += 1

            print(f"{striker.id}: прокол не удался, повторяю", flush=True)

        point, face = vantage(current)
        striker.goto(point[0], point[1], height, face)

    striker.land()

    return popped


def main():
    channel = Channel()
    scout = Microdrone(SCOUT_ID)
    striker = Microdrone(STRIKER_ID)
    rover = sverk_rover.init(ROVER_ID)
    started = time.monotonic()

    try:
        carrier = Carrier(rover)
        rover.set_led_preset("idle")

        report = scout_mission(scout)

        if not report:
            raise RuntimeError("Участки не найдены")

        best = max(report, key=lambda s: (s["red"], -math.hypot(s["x"], s["y"])))

        if best["red"] == 0:
            raise RuntimeError("Красных шаров не найдено")

        site = (best["x"] + SCOUT_START_IN_ROVER[0], best["y"] + SCOUT_START_IN_ROVER[1])
        message = channel.send("разведчик", "участок", label=best["label"],
                               x=round(site[0], 2), y=round(site[1], 2), red=best["red"],
                               extent=best["extent"])

        stop = deliver(carrier, (message["x"], message["y"]), message["extent"], striker)
        order = channel.send("ровер", "взлёт", x=round(message["x"] - stop[0], 2),
                             y=round(message["y"] - stop[1], 2), red=message["red"],
                             extent=message["extent"])
        popped = strike_mission(striker, (order["x"], order["y"]), order["red"], order["extent"])

        print("\n===== ИТОГ =====", flush=True)

        for s in report:
            print(f"Участок {s['label']}: красных {s['red']}, жёлтых {s['yellow']}, зелёных {s['green']}", flush=True)

        print(f"Выбран участок {best['label']}, координаты переданы роверу: ({message['x']}, {message['y']})", flush=True)
        print(f"Ровер доставил исполнителя, обработано красных шаров: {popped}", flush=True)
        print(f"Время попытки: {time.monotonic() - started:.0f} с", flush=True)
    finally:
        for drone in (scout, striker):
            try:
                if drone.started and drone.pose()[2] > 0.15:
                    drone.land()
            except Exception as error:
                print(f"{drone.id}: ошибка при посадке: {error}", flush=True)
        try:
            rover.stop_motion()
        except Exception:
            pass

        rover.close()


if __name__ == "__main__":
    main()