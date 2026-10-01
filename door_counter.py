# door_ab.py  (USB 웹캠 + yolo26n NCNN, 라즈베리파이 5)
# 문 주변을 둥근 모서리 사각형으로 감싸서 A(문 쪽), 나머지 전부를 B(교실 쪽)로 나눔
#   A 사각형: 왼쪽/오른쪽/아래 경계는 비율로 설정, 위쪽은 화면 끝까지 열려 있음(기본)
#   IN  : A -> B 로 경계를 넘는 순간 바로 카운트 (학생 수 +1)
#   OUT : B 에서 시작한 사람이 A 에서 사라질 때 확정 (학생 수 -1)
#   IN 된 사람은 그때부터 'B(안쪽) 출신'으로 바뀜 -> 되돌아 나가면 OUT 으로 상쇄
#   같은 사람이 경계를 여러 번 오가도 IN 은 한 번만
# 실행: source ~/yolo-env/bin/activate && cd ~/door_counter && python door_ab.py
# 키(SHOW=True일 때): q 종료, r 학생 수 0으로 리셋 / SHOW=False일 때: Ctrl+C 종료

import time
from pathlib import Path
import cv2
from ultralytics import YOLO

# ---------------- 설정 ----------------
BASE = Path(__file__).resolve().parent
MODEL = str(BASE / "models" / "yolo26n_ncnn_model")
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480      # 카메라에 요청하는 해상도 (회전 전)
IMGSZ = 320

ROTATE = None                # 영상 회전: None(안 돌림), "CW"(시계 90°), "CCW"(반시계 90°), "180"

# ---- A(문 쪽) 구역: 둥근 모서리 사각형 (회전 후 화면 기준 비율) ----
A_LEFT_RATIO = 0.55          # 왼쪽 경계   (화면 폭 대비, 0.0 왼쪽 ~ 1.0 오른쪽)
A_RIGHT_RATIO = 0.91         # 오른쪽 경계 (화면 폭 대비)
A_BOTTOM_RATIO = 0.65        # 아래 경계   (화면 높이 대비, 0.0 위 ~ 1.0 아래)
A_TOP_RATIO = None           # 위 경계. None 이면 화면 위쪽 끝까지 A
R_BL_RATIO = 0.17            # 왼쪽 아래 모서리 곡선 반지름 (화면 높이 대비, 0이면 직각)
R_BR_RATIO = 0.06            # 오른쪽 아래 모서리 곡선 반지름 (화면 높이 대비, 0이면 직각)
USE_FOOT_Y = False           # 위/아래 경계 판정에 발 위치(박스 아래쪽 y) 사용. False면 박스 중심 y

BAND = 15                    # 경계 양옆 불감대(px). 경계 위에서 흔들려도 구역이 안 바뀜
LOST_FRAMES = 15             # 이 프레임 수 동안 안 보이면 "사라짐"으로 보고 카운트 확정
                             #   작을수록 빨리 확정 / 너무 작으면 잠깐 가려진 걸 사라진 걸로 오판
MERGE_DIST = 120             # 새 ID가 방금 사라진 ID와 이 거리(px) 안에서 나타나면 같은 사람으로 보고 이어붙임
                             #   ID 스위치로 한 사람이 A->A, B->B 두 개로 쪼개지는 것을 막음
SHOW = True                  # FPS 측정할 땐 False
DEBUG = True                 # 새 ID / 구역 이동 / 확정 결과를 터미널에 출력

ROT_CODE = {None: None,
            "CW": cv2.ROTATE_90_CLOCKWISE,
            "CCW": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "180": cv2.ROTATE_180}
assert ROTATE in ROT_CODE, 'ROTATE 는 None(따옴표 없이), "CW", "CCW", "180" 중 하나'

if ROTATE in ("CW", "CCW"):
    FRAME_W, FRAME_H = CAM_H, CAM_W
else:
    FRAME_W, FRAME_H = CAM_W, CAM_H

AL = int(FRAME_W * A_LEFT_RATIO)
AR = int(FRAME_W * A_RIGHT_RATIO)
AB = int(FRAME_H * A_BOTTOM_RATIO)
AT = int(FRAME_H * A_TOP_RATIO) if A_TOP_RATIO is not None else None
R_BL = int(FRAME_H * R_BL_RATIO)
R_BR = int(FRAME_H * R_BR_RATIO)
assert AL < AR, "A_LEFT_RATIO < A_RIGHT_RATIO 이어야 함"
assert AT is None or AT < AB, "A_TOP_RATIO < A_BOTTOM_RATIO 이어야 함"


def a_dist(x, y):
    """A 구역 경계까지의 부호 있는 거리(px). + 면 A 안쪽, - 면 B 쪽.
    A = 왼쪽/오른쪽/아래(와 선택적으로 위) 경계로 둘러싼 사각형, 아래 두 모서리는 둥글게."""
    dl = x - AL                          # 왼쪽 경계에서 안쪽으로 +
    dr = AR - x                          # 오른쪽 경계에서 안쪽으로 +
    db = AB - y                          # 아래 경계에서 위쪽(안쪽)으로 +
    dt = (y - AT) if AT is not None else float("inf")
    if R_BL > 0 and dl < R_BL and db < R_BL:          # 왼쪽 아래 둥근 모서리
        return R_BL - ((R_BL - dl) ** 2 + (R_BL - db) ** 2) ** 0.5
    if R_BR > 0 and dr < R_BR and db < R_BR:          # 오른쪽 아래 둥근 모서리
        return R_BR - ((R_BR - dr) ** 2 + (R_BR - db) ** 2) ** 0.5
    return min(dl, dr, db, dt)                        # 그 외: 가장 가까운 직선까지


def get_zone(x, y, prev):
    """구역 판정 (A/B). 경계 불감대 안이면 이전 구역 유지 (슈미트 트리거)."""
    d = a_dist(x, y)
    if d > BAND:
        return "A"
    if d < -BAND:
        return "B"
    if prev is not None:
        return prev
    return "A" if d > 0 else "B"


# ---------------- 초기화 ----------------
model = YOLO(MODEL, task="detect")
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
assert cap.isOpened(), "웹캠을 열 수 없음: ls /dev/video* 로 확인"

origin_of = {}    # track_id -> 처음 보인 구역 'A'/'B'
zone_of = {}      # track_id -> 현재(마지막) 구역 'A'/'B'
last_seen = {}    # track_id -> 마지막으로 보인 프레임 번호
last_pos = {}     # track_id -> 마지막으로 보인 위치 (cx, cy)
total = 10
in_cnt = out_cnt = 0


def finalize(tid, reason):
    """ID가 사라졌을 때 호출: 처음 구역 -> 마지막 구역으로 IN/OUT 확정 후 상태 삭제."""
    global total, in_cnt, out_cnt
    o, z = origin_of.pop(tid, None), zone_of.pop(tid, None)
    last_seen.pop(tid, None)
    last_pos.pop(tid, None)
    if o == "B" and z == "A":
        out_cnt += 1
        if total > 0:
            total -= 1
        else:
            print("경고: 학생 수 0인데 OUT 발생 (이전에 IN을 놓쳤을 가능성)")
        result = "OUT"
    else:
        result = None
    if DEBUG:
        tag = f"[확정 {result}]" if result else "[무효]  "
        print(f"{tag} id{tid}  {o}->{z}  ({reason})  학생 수={total}")


if DEBUG:
    print(f"회전={ROTATE}  화면={FRAME_W}x{FRAME_H}  A구역 x={AL}~{AR}, y={AT if AT is not None else 0}~{AB}"
          f"  모서리 R(왼아래)={R_BL} R(오른아래)={R_BR}  사라짐 판정={LOST_FRAMES}프레임")

frame_no, fps_avg, prev_t = 0, 0.0, time.time()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 읽기 실패")
            break

        if ROT_CODE[ROTATE] is not None:
            frame = cv2.rotate(frame, ROT_CODE[ROTATE])

        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        classes=[0], imgsz=IMGSZ, verbose=False)[0]

        # ---- 보이는 사람: 처음 구역 기록 + 현재 구역 갱신 ----
        if r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            visible = set(ids)                     # 이번 프레임에 보이는 ID 전체 (연결 후보에서 제외용)
            for tid, (x1, y1, x2, y2) in zip(ids, boxes):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                py = y2 if USE_FOOT_Y else cy          # 판정용 y (발 또는 중심)
                prev = zone_of.get(tid)
                new = get_zone(cx, py, prev)
                last_seen[tid] = frame_no

                if prev is None:
                    # 방금 사라진(이번 프레임에 안 보인) ID 중 가장 가까운 것을 찾음
                    best, best_d = None, MERGE_DIST
                    for old, (ox, oy) in last_pos.items():
                        if old in visible:             # 지금 화면에 보이는 사람은 후보 아님
                            continue
                        d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                        if d < best_d:
                            best, best_d = old, d
                    if best is not None:
                        # 같은 사람으로 판단 -> 시작 구역을 물려받고 옛 ID는 조용히 삭제 (확정 안 함)
                        origin_of[tid] = origin_of.pop(best)
                        prev = zone_of.pop(best)
                        new = get_zone(cx, py, prev)
                        last_seen.pop(best, None)
                        last_pos.pop(best, None)
                        if DEBUG:
                            print(f"[연결]  f{frame_no} id{best} -> id{tid}  거리={best_d:.0f}px"
                                  f"  시작 구역={origin_of[tid]} 유지")
                    else:
                        origin_of[tid] = new
                        if DEBUG:
                            print(f"[새 ID] f{frame_no} id{tid}  ({cx:.0f},{py:.0f})  시작 구역={new}")

                # 구역이 바뀜 (새 ID는 prev가 None이라 제외, 연결된 ID는 포함)
                if prev is not None and new != prev:
                    if prev == "A" and new == "B" and origin_of[tid] == "A":
                        # IN: 경계를 넘는 순간 바로 카운트, 이후 이 사람은 '안쪽 출신'으로 취급
                        in_cnt += 1
                        total += 1
                        origin_of[tid] = "B"
                        if DEBUG:
                            print(f"[IN ]   f{frame_no} id{tid}  A->B  ({cx:.0f},{py:.0f})  학생 수={total}")
                    elif DEBUG:
                        print(f"[이동]  f{frame_no} id{tid}  {prev}->{new}  ({cx:.0f},{py:.0f})")
                zone_of[tid] = new
                last_pos[tid] = (cx, cy)

                if SHOW:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(py)), 4, (0, 255, 0), -1)   # 판정 기준점
                    cv2.putText(frame, f"id{tid} {origin_of[tid]}>{new}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---- 사라진 사람: LOST_FRAMES 동안 안 보이면 카운트 확정 ----
        for tid in [t for t, f in last_seen.items() if frame_no - f > LOST_FRAMES]:
            finalize(tid, "사라짐")

        # FPS (이동평균)
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        fps_avg = fps if frame_no == 0 else 0.9 * fps_avg + 0.1 * fps
        frame_no += 1

        if frame_no % 15 == 0:
            print(f"FPS={fps_avg:5.1f}  학생 수={total}  (IN={in_cnt} OUT={out_cnt})  추적 중={len(zone_of)}명")

        if SHOW:
            # ---- A 구역 경계: 왼쪽선 + 왼아래 곡선 + 아래선 + 오른아래 곡선 + 오른쪽선 ----
            col = (0, 0, 255)
            top = AT if AT is not None else 0
            cv2.line(frame, (AL, top), (AL, AB - R_BL), col, 2)                    # 왼쪽
            cv2.line(frame, (AR, top), (AR, AB - R_BR), col, 2)                    # 오른쪽
            cv2.line(frame, (AL + R_BL, AB), (AR - R_BR, AB), col, 2)              # 아래
            if AT is not None:
                cv2.line(frame, (AL, AT), (AR, AT), col, 2)                        # 위
            if R_BL > 0:
                cv2.ellipse(frame, (AL + R_BL, AB - R_BL), (R_BL, R_BL), 0, 90, 180, col, 2)
            if R_BR > 0:
                cv2.ellipse(frame, (AR - R_BR, AB - R_BR), (R_BR, R_BR), 0, 0, 90, col, 2)
            cv2.putText(frame, "A", ((AL + AR) // 2 - 10, (top + AB) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            # B 글자: A 왼쪽에 공간이 있으면 왼쪽, 없으면 A 아래
            bx, by = (AL // 2, FRAME_H // 2) if AL > 80 else ((AL + AR) // 2, (AB + FRAME_H) // 2)
            cv2.putText(frame, "B", (bx - 10, by), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Total: {total}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"IN {in_cnt} / OUT {out_cnt}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("door_ab", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                total = 0
                print("학생 수 리셋 -> 0")
except KeyboardInterrupt:
    pass
finally:
    # 종료 시점에 아직 화면에 남아 있던 사람도 판정 (마지막 구역 기준)
    for tid in list(last_seen):
        finalize(tid, "프로그램 종료")
    cap.release()
    if SHOW:
        cv2.destroyAllWindows()
    print(f"최종  학생 수={total}  IN={in_cnt}  OUT={out_cnt}  평균FPS={fps_avg:.1f}")