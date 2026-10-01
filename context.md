# 프로젝트 컨텍스트: 라즈베리파이 5 문 출입 학생 수 카운터

> 이 문서는 지금까지의 설계 논의를 요약한 컨텍스트입니다. 새 대화에 붙여넣으면 이어서 작업할 수 있습니다.

## 1. 목표

- 교실 문 **옆 벽**에 카메라 1대를 달고, **현재 교실 안에 있는 학생 수**를 실시간으로 센다.
- 통과 횟수가 아니라 **재실 인원**이므로 오차가 누적된다 → 판정이 엄격해야 한다.
- 이후 학생 수를 TCP 소켓으로 다른 라즈베리파이 5에 전송할 예정.

## 2. 환경

| 항목 | 내용 |
|---|---|
| 보드 | Raspberry Pi 5, Raspberry Pi OS 64-bit (Bookworm) |
| 카메라 | **USB 웹캠** (Pi Camera 아님), **YUYV만 지원**, 640×480 @ 30fps |
| 가상환경 | `~/yolo-env` (`--system-site-packages`), `source ~/yolo-env/bin/activate` |
| 패키지 | `ultralytics` 8.4.x (`[export]`로 설치해서 TensorFlow·CUDA가 불필요하게 깔려 있음) |
| 모델 | **YOLO26n**, NCNN 변환 (`imgsz=320`) |
| 프로젝트 폴더 | `~/door_counter/` (`door_ab.py`, `models/yolo26n_ncnn_model/`, `recordings/`, `results/`) |
| 원격 접속 | Windows + MobaXterm (SSH + X11). X11 인증 문제는 MobaXterm 설정에서 해결 |
| 측정 FPS | 약 15.8 (창 표시 포함) |

## 3. 아키텍처

```
USB 웹캠 → (선택) 90° 회전 → YOLO26n 검출 (person만) → ByteTrack 추적 (ID 부여)
        → ID별 A/B 구역 상태 머신 (직접 구현) → 학생 수
```

- **방향은 모델이 아는 게 아니다.** 검출 + 추적으로 생긴 궤적을 **직접 만든 규칙**으로 해석한다.
- Ultralytics `ObjectCounter`는 **ID당 1회만** 세서(들어왔다 나가면 +1이 남음) 버리고 판정을 직접 구현했다.

## 4. 구역 정의 (현재)

- **A (문 쪽)**: 문 주변을 감싸는 **둥근 모서리 사각형**. 위쪽은 화면 끝까지 열려 있음.
- **B (교실 쪽)**: A를 제외한 화면 전체.
- 판정 기준점: **박스 중심** `(cx, cy)` (발 기준은 시도 후 중심으로 결정).
- 경계 판정: A 경계까지의 **부호 있는 거리(SDF)** 계산 → `+BAND` 초과면 A, `-BAND` 미만이면 B, 그 사이는 이전 구역 유지 (**슈미트 트리거 / 불감대**).

현재 설정값 (640×480, 회전 없음):

| 설정 | 값 | 의미 |
|---|---|---|
| `A_LEFT_RATIO` | 0.49 | A 왼쪽 경계 |
| `A_RIGHT_RATIO` | 0.91 | A 오른쪽 경계 |
| `A_BOTTOM_RATIO` | 0.65 | A 아래 경계 |
| `A_TOP_RATIO` | None | 화면 위 끝까지 A |
| `R_BL_RATIO` | 0.17 | 왼쪽 아래 모서리 곡선 반지름 (높이 대비) |
| `R_BR_RATIO` | 0.06 | 오른쪽 아래 모서리 곡선 반지름 |
| `BAND` | 15px | 경계 불감대 |
| `LOST_FRAMES` | 15 | 16프레임 안 보이면 "사라짐" (15FPS에서 약 1초) |
| `MERGE_DIST` | 120px | ID 스위치 보정 거리 |
| `ROTATE` | None | `"CW"`, `"CCW"`, `"180"` 가능 (따옴표 없는 `None` 주의) |
| `total` | 10 | 시작 학생 수 (코드에 직접 지정) |

## 5. 카운트 규칙 (확정)

| 경로 | 결과 | 시점 |
|---|---|---|
| A → B | **IN (+1)** | 경계를 넘는 **즉시** (사라질 필요 없음) |
| B → A → 사라짐 | **OUT (−1)** | ID가 **사라진 뒤** (`LOST_FRAMES`) |
| B → A → (안 사라지고) B | 없음 | **B → B로 간주** |
| A → A | 없음 | 문 앞만 지나감 |
| B → B | 없음 | 교실 안에서만 움직임 |
| A → B → A → 사라짐 | IN 후 OUT, 합계 0 | IN 순간 그 사람의 시작 구역을 B로 전환 |
| A → B → A → B | IN 한 번만 | 시작 구역이 A일 때만 IN |

구현 핵심:
- `origin_of[tid]`: 처음 보인 구역 (IN 시 `"B"`로 바뀜)
- `zone_of[tid]`: 현재 구역
- IN: `prev == "A" and new == "B" and origin_of[tid] == "A"` → 즉시 +1, `origin_of[tid] = "B"`
- OUT: `finalize()`에서 `origin == "B" and last_zone == "A"` → −1 (0 미만 방지)

## 6. ID 스위치 보정

- 문제: 한 사람이 경계 근처에서 ID가 바뀌면 `A->A` + `B->B` 두 개로 쪼개져 둘 다 무효가 됨.
- 해결: 새 ID가 나타나면, **이번 프레임에 안 보이는 ID** 중 `MERGE_DIST` 안에 있던 가장 가까운 ID의 시작 구역을 물려받음 (`[연결]` 로그).
- 버그 수정 이력: 같은 프레임 안의 처리 순서 때문에 보이는 사람과 잘못 연결되던 문제 → 이번 프레임에 보이는 ID를 `visible` 집합으로 먼저 모아서 제외.
- 한계: 두 사람이 붙어서 지나가면 잘못 연결될 수 있음.

## 7. 기술 선택과 근거

| 선택 | 이유 | 근거 / 주의 |
|---|---|---|
| Ultralytics | 검출·추적·변환·학습이 한 패키지 | AGPL-3.0 |
| YOLO26n | 엣지 기기용 설계, 2.4M params | 공식 RPi 가이드. **"CPU 43% 빠름"은 Intel Xeon + ONNX 기준**이라 RPi 근거 아님 → 직접 벤치마크 필요 |
| NCNN | ARM NEON 최적화, RPi에서 가장 빠른 형식 | 공식 RPi 가이드 |
| imgsz 320 | 640 대비 연산량 약 1/4 | 문 앞 사람은 화면에 크게 잡힘 |
| ByteTrack | 저신뢰 박스도 2차 매칭 → 가림에 강함 | ByteTrack 논문 (ECCV 2022), 기본 `track_buffer` 30프레임 |
| 소프트웨어 회전 | YOLO는 똑바로 선 사람으로 학습됨 → 추론 전에 회전 | COCO 학습 데이터 |
| 경계 = SDF | 경계 모양이 바뀌어도 판정 로직(비교기 2개)은 그대로 | |

**정정 사항:** YOLO26은 **기본적으로 NMS를 사용**한다. NMS-free는 `nms=False`를 줘야 적용된다. 현재 코드는 기본값(NMS 사용). NCNN + `track()`에서 `nms=False` 호환은 미확인.

## 8. 설계 변천 (버린 방식과 이유)

1. ObjectCounter → ID당 1회 제한
2. 선 1개 + 불감대 → 조정 불편
3. 세로선 A/B 2개 (10%/90%) → 박스가 크면 중심이 선에 못 닿음
4. 세로선 + 가로선 AND → 너무 엄격해서 놓침
5. X자 4구역 (각도·중심 조절) → 실제 문과 구역이 안 맞음
6. 세로선 1개 A/B + 회전 → 문 아래까지 A로 잡힘
7. + 문 쪽 가로선 + 곡선 모서리 → 카메라 각도 변경
8. **둥근 사각형 A (현재)**

## 9. 해결한 환경 문제

- `pip install "ultralytics[export]"`의 filelock 경고는 무시 가능 (다음엔 `pip install ultralytics`만).
- 웹캠이 MJPG 미지원 → MJPG 설정 삭제.
- X11 `No authorisation provided` → MobaXterm 원격 X11 인증 설정 해제.
- `python`만 입력하면 대화형 모드 → `exit()` 또는 `Ctrl+D`. 실행은 `python door_ab.py`.
- `ROTATE = "None"`(문자열) 에러 → `None`(따옴표 없이).

## 10. 남은 과제

- [ ] 새 카메라 각도에서 A 사각형 비율 확정
- [ ] 1단계 측정: 1명 / 2명 줄지어 / 2명 나란히 / 들어왔다 돌아가기, FPS, CPU 온도
- [ ] yolo11n vs yolo26n, `nms=False` 효과 **직접 벤치마크** (`yolo benchmark ... imgsz=320 format=ncnn`)
- [ ] `LOST_FRAMES`를 초 단위로 바꿀지 결정 (FPS에 따라 대기 시간이 변함), ByteTrack `track_buffer`(30)와의 관계 고려
- [ ] 필요 시 2단계(머리 검출, CrowdHuman) / 3단계(직접 촬영 데이터로 파인튜닝)
- [ ] 학생 수를 TCP 소켓으로 다른 라즈베리파이에 전송

## 11. 하드웨어 관점 연결 (면접용)

- ID별 A/B 상태 레지스터 + 전이 시 출력 → **Mealy FSM**
- 경계 불감대 → **슈미트 트리거 / 히스테리시스**
- 시작 구역 래치 + ID 소멸 시 OUT 확정 → **트랜잭션 완료 시점 출력**
- 90° 회전 → 계산이 아니라 **메모리 주소 재배치**
- imgsz 절반 → **MAC 연산 약 1/4**
- NCNN → ARM **NEON SIMD**, pack4 데이터 레이아웃
- X11 전송으로 FPS 저하 → **I/O 병목**

한 줄 설명: **"검출은 YOLO26n, 추적은 ByteTrack, 출입 판정은 불감대를 둔 ID별 상태 머신으로 직접 설계했고, ID 스위치는 거리 기반 재연결로 보정했다."**
