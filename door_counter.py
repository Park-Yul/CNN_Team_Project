# door_total.py  (USB 웹캠 + yolo26n NCNN, 라즈베리파이 5)
# 세로선 1개 + 가로선 1개
# 같은 사람이 세로선과 가로선을 "둘 다 같은 방향(IN 또는 OUT)으로" 지났을 때만 카운트 (AND 조건)
# 실행: source ~/yolo-env/bin/activate && cd ~/door_counter && python door_total.py
# 키(SHOW=True일 때): q 종료, r 학생 수 0으로 리셋 / SHOW=False일 때: Ctrl+C 종료

import time
from pathlib import Path
import cv2
from ultralytics import YOLO

# ---------------- 설정 ----------------
BASE = Path(__file__).resolve().parent
MODEL = str(BASE / "models" / "yolo26n_ncnn_model")
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
IMGSZ = 320

# 세로선 (좌우 이동 판정): 화면 '폭' 대비 비율, 0.0 = 왼쪽 끝, 1.0 = 오른쪽 끝
USE_X_LINE = True
LINE_X_RATIO = 0.80
X_IN_DIRECTION = "R2L"       # 오른쪽->왼쪽이 IN 이면 "R2L", 왼쪽->오른쪽이 IN 이면 "L2R"

# 가로선 (위아래 이동 판정): 화면 '높이' 대비 비율, 0.0 = 위쪽 끝, 1.0 = 아래쪽 끝
USE_Y_LINE = True
LINE_Y_RATIO = 0.70
Y_IN_DIRECTION = "T2B"       # 위->아래가 IN 이면 "T2B", 아래->위가 IN 이면 "B2T"

BAND = 10                    # 선 양옆 불감대(px). 선 위에서 박스가 흔들려도 중복 카운트 안 되게. 0이면 끔
STALE_FRAMES = 60            # 이 프레임 수 동안 안 보인 ID는 상태 삭제
SHOW = True                  # FPS 측정할 땐 False
DEBUG = True                 # 새 ID / 카운트 이벤트를 터미널에 출력

LINE_X = int(FRAME_W * LINE_X_RATIO)
LINE_Y = int(FRAME_H * LINE_Y_RATIO)

# ---------------- 초기화 ----------------
model = YOLO(MODEL, task="detect")
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
assert cap.isOpened(), "웹캠을 열 수 없음: ls /dev/video* 로 확인"
assert USE_X_LINE or USE_Y_LINE, "세로선/가로선 중 하나는 켜야 함"

x_side = {}       # track_id -> 'L' / 'R'  (세로선 기준 상태)
y_side = {}       # track_id -> 'T' / 'B'  (가로선 기준 상태)
pend_x = {}       # track_id -> 세로선을 마지막으로 지난 방향 'IN'/'OUT' (래치)
pend_y = {}       # track_id -> 가로선을 마지막으로 지난 방향 'IN'/'OUT' (래치)
last_seen = {}    # track_id -> 마지막으로 보인 프레임 번호
total = 10
in_cnt = out_cnt = 0


def classify(v, prev, line, low_name, high_name):
    """선 하나 기준 상태 판정. 불감대(line±BAND) 안에서는 이전 상태 유지 (슈미트 트리거)."""
    if v < line - BAND:
        return low_name
    if v > line + BAND:
        return high_name
    if prev is None:                      # 불감대 안에서 처음 잡힘 -> 선의 어느 쪽인지로
        return low_name if v < line else high_name
    return prev


def apply(event):
    """학생 수 반영"""
    global total, in_cnt, out_cnt
    if event == "IN":
        in_cnt += 1
        total += 1
    else:
        out_cnt += 1
        if total > 0:
            total -= 1
        else:
            print("경고: 학생 수 0인데 OUT 발생 (이전에 IN을 놓쳤을 가능성)")


def crossed(tid, axis, event, prev, new, pos):
    """선 하나를 지났을 때 호출. 두 선의 래치가 같은 방향으로 모이면 카운트."""
    (pend_x if axis == "세로선" else pend_y)[tid] = event      # 래치 set (최근 방향으로 덮어씀)
    if DEBUG:
        print(f"[통과]  id{tid} {axis} {prev}->{new} {event}  위치={pos:.0f}"
              f"  (세로:{pend_x.get(tid, '-')} 가로:{pend_y.get(tid, '-')})")

    # 한 축만 켠 경우엔 그 축만으로 판정
    need_x, need_y = USE_X_LINE, USE_Y_LINE
    ex = pend_x.get(tid) if need_x else event
    ey = pend_y.get(tid) if need_y else event
    if ex == ey == event:                                       # AND 조건 성립
        apply(event)
        pend_x.pop(tid, None)                                   # 래치 clear
        pend_y.pop(tid, None)
        if DEBUG:
            print(f"[{event:3}]  id{tid}  두 선 모두 통과 -> 학생 수={total}")


frame_no, fps_avg, prev_t = 0, 0.0, time.time()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 읽기 실패")
            break

        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        classes=[0], imgsz=IMGSZ, verbose=False)[0]

        if r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            for tid, (x1, y1, x2, y2) in zip(ids, boxes):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                is_new = tid not in last_seen
                last_seen[tid] = frame_no

                # ---- 세로선: 좌우 이동 ----
                if USE_X_LINE:
                    px = x_side.get(tid)
                    nx = classify(cx, px, LINE_X, "L", "R")
                    if px and nx != px:
                        ev = "IN" if f"{px}2{nx}" == X_IN_DIRECTION else "OUT"
                        crossed(tid, "세로선", ev, px, nx, cx)
                    x_side[tid] = nx

                # ---- 가로선: 위아래 이동 ----
                if USE_Y_LINE:
                    py = y_side.get(tid)
                    ny = classify(cy, py, LINE_Y, "T", "B")
                    if py and ny != py:
                        ev = "IN" if f"{py}2{ny}" == Y_IN_DIRECTION else "OUT"
                        crossed(tid, "가로선", ev, py, ny, cy)
                    y_side[tid] = ny

                if DEBUG and is_new:
                    print(f"[새 ID] f{frame_no} id{tid}  cx={cx:.0f} cy={cy:.0f}"
                          f"  초기 {x_side.get(tid, '-')}{y_side.get(tid, '-')}")

                if SHOW:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"id{tid} {x_side.get(tid, '')}{y_side.get(tid, '')}"
                                       f" x:{pend_x.get(tid, '-')} y:{pend_y.get(tid, '-')}",
                                (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 오래 안 보인 ID 정리 (메모리 누수 방지)
        for tid in [t for t, f in last_seen.items() if frame_no - f > STALE_FRAMES]:
            for d in (x_side, y_side, pend_x, pend_y, last_seen):
                d.pop(tid, None)

        # FPS (이동평균)
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        fps_avg = fps if frame_no == 0 else 0.9 * fps_avg + 0.1 * fps
        frame_no += 1

        if frame_no % 15 == 0:
            print(f"FPS={fps_avg:5.1f}  학생 수={total}  (IN={in_cnt} OUT={out_cnt})")

        if SHOW:
            if USE_X_LINE:
                cv2.line(frame, (LINE_X, 0), (LINE_X, FRAME_H), (0, 0, 255), 2)       # 세로선: 빨강
            if USE_Y_LINE:
                cv2.line(frame, (0, LINE_Y), (FRAME_W, LINE_Y), (0, 255, 255), 2)     # 가로선: 노랑
            cv2.putText(frame, f"Total: {total}  (IN {in_cnt} / OUT {out_cnt})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("door_total", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                total = 0
                print("학생 수 리셋 -> 0")
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    if SHOW:
        cv2.destroyAllWindows()
    print(f"최종  학생 수={total}  IN={in_cnt}  OUT={out_cnt}  평균FPS={fps_avg:.1f}")
