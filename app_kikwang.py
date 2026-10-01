#!/usr/bin/env python3
"""
라즈베리파이5 door_counter 웹 컨트롤러
- 로그인(SSH 인증) -> 메인(초기 인원수 설정 / 실행)
- 이 파일은 라즈베리파이(10.10.17.124) 위에서 실행해야 합니다.

설치:  pip install flask paramiko
실행:  python3 app.py        ->  http://10.10.17.124:8000
"""
import atexit
import os
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

import paramiko
from flask import (Flask, Response, redirect, render_template_string,
                   request, session, url_for)

PI_HOST = os.environ.get("PI_HOST", "10.10.17.124")
PI_PORT = int(os.environ.get("PI_PORT", "22"))
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))
VENV_ACTIVATE = "yolo-env/bin/activate"
SCRIPT_NAME = "door_counter.py"      # 홈디렉터리 아래에서 자동 검색
# 'total = 숫자' 형태의 줄 (== 비교, += 대입 등은 제외)
TOTAL_RE = re.compile(r"^(\s*total\s*=(?!=)\s*)(-?\d+)(?!\d)")
FRAME_PATH = "/tmp/door_frame.jpg"
RUNNER_PATH = "/tmp/stream_runner.py"
LOG_PATH = "/tmp/door_counter.log"
PID_PATH = "/tmp/door_counter.pid"

# 웹페이지 닫힘 감지(heartbeat) 설정
LEAVE_GRACE = 10     # 탭 닫힘 신호(/leaving) 후, 이 시간(초) 동안 heartbeat가 없으면 종료
HB_TIMEOUT = 90      # 신호 없이 heartbeat가 이 시간(초) 끊기면 종료 (브라우저 강제종료 등)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
SESSIONS = {}  # token -> {"client": SSHClient, "user": str, "venv": str}

# ---------------------------------------------------------------------------
# door_counter.py 를 그대로 실행하되, cv2.imshow 화면을 JPEG 파일로 내보내는 래퍼
# (SSH/서버에는 모니터가 없으므로 창 대신 웹으로 화면을 전달)
# ---------------------------------------------------------------------------
RUNNER_CODE = r'''
import os, sys, time, runpy
import cv2

FRAME = "/tmp/door_frame.jpg"

def _imshow(name, frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with open(FRAME + ".tmp", "wb") as f:
            f.write(buf.tobytes())
        os.replace(FRAME + ".tmp", FRAME)

def _wait(delay=1):
    time.sleep(0.001)
    return -1

def _noop(*a, **k):
    return None

cv2.imshow = _imshow
cv2.waitKey = _wait
for _n in ("namedWindow", "destroyAllWindows", "destroyWindow",
           "resizeWindow", "moveWindow", "setWindowProperty"):
    setattr(cv2, _n, _noop)

script = os.path.abspath(sys.argv[1])
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
'''

# ---------------------------------------------------------------------------
# 템플릿
# ---------------------------------------------------------------------------
BASE = """
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Door Counter Controller</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--fg:#e2e8f0;--acc:#38bdf8;--acc2:#22c55e;--err:#f87171}
@media (prefers-color-scheme: light){:root{--bg:#f1f5f9;--card:#fff;--fg:#0f172a}}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;background:var(--bg);color:var(--fg);min-height:100vh}
.wrap{max-width:720px;margin:0 auto;padding:32px 20px}
h1{font-size:1.5rem;margin:0 0 6px} .sub{opacity:.7;margin-bottom:24px;font-size:.9rem}
.card{background:var(--card);border-radius:14px;padding:24px;box-shadow:0 4px 18px rgba(0,0,0,.15)}
label{display:block;margin:14px 0 6px;font-size:.9rem}
input{width:100%;padding:12px;border-radius:8px;border:1px solid #64748b55;background:transparent;color:inherit;font-size:1rem}
button,.btn{display:inline-block;margin-top:18px;padding:13px 20px;border:0;border-radius:10px;background:var(--acc);color:#04263a;font-weight:700;font-size:1rem;cursor:pointer;text-decoration:none}
.btn.g{background:var(--acc2);color:#052e16} .btn.r{background:var(--err);color:#450a0a} .btn.s{background:#64748b55;color:var(--fg)}
.banners{display:grid;gap:16px;grid-template-columns:1fr 1fr}
@media(max-width:560px){.banners{grid-template-columns:1fr}}
.banner{display:block;padding:40px 20px;border-radius:14px;text-align:center;font-size:1.3rem;font-weight:800;text-decoration:none;color:#04263a;background:var(--acc)}
.banner.g{background:var(--acc2);color:#052e16}
.msg{margin-top:14px;padding:10px 12px;border-radius:8px;background:#64748b33;font-size:.9rem;white-space:pre-wrap}
.msg.err{background:#f8717133;color:var(--err)} .msg.ok{background:#22c55e33}
img.cam{width:100%;border-radius:10px;background:#000;min-height:240px}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}
.top a{color:var(--acc);font-size:.9rem}
.lnk{margin:0;padding:0;background:none;color:var(--acc);font-weight:400;font-size:.9rem;border:0;cursor:pointer}
.st{font-size:.9rem;margin:12px 0}
</style></head><body><div class="wrap">
{% if user %}<div class="top"><span>👤 {{ user }}@{{ host }}</span>
<form method="post" action="{{ url_for('logout') }}" style="margin:0"><button class="lnk" type="submit">로그아웃</button></form></div>{% endif %}
{{ body|safe }}
</div>
{% if hb %}<script>
(function(){
  function ping(){ fetch('/ping',{method:'POST',cache:'no-store',keepalive:true}).catch(function(){}); }
  ping(); setInterval(ping,3000);
  window.addEventListener('pagehide',function(){ try{navigator.sendBeacon('/leaving');}catch(e){} });
})();
</script>{% endif %}
</body></html>
"""

LOGIN = """
<h1>Raspberry Pi 5 접속</h1><div class="sub">{{ host }} SSH 계정으로 로그인하세요</div>
<form class="card" method="post">
  <label>Username</label><input name="username" autocomplete="username" required autofocus>
  <label>Password</label><input name="password" type="password" autocomplete="current-password" required>
  {% if error %}<div class="msg err">{{ error }}</div>{% endif %}
  <button type="submit">접속</button>
</form>
"""

MAIN = """
<h1>Door Counter</h1><div class="sub">{{ venv }}<br>{{ script }}</div>
{% if msg %}<div class="msg {{ cls }}" style="margin:0 0 16px">{{ msg }}</div>{% endif %}
<div class="st">{{ status }}</div>
<div class="banners">
  <a class="banner" href="{{ url_for('setcount') }}">초기 인원수 설정</a>
  <a class="banner g" href="{{ url_for('run') }}">실행</a>
</div>
"""

SETCOUNT = """
<h1>초기 인원수 설정</h1>
<div class="sub">{{ path }}</div>
<form class="card" method="post">
  <div class="msg">현재 {{ line }}행: <code>{{ current }}</code></div>
  <label>인원수를 설정해주세요</label>
  <input name="total" type="number" min="0" step="1" required autofocus>
  {% if msg %}<div class="msg {{ cls }}">{{ msg }}</div>{% endif %}
  <button type="submit">저장</button>
  <a class="btn s" href="{{ url_for('main') }}">돌아가기</a>
</form>
"""

RUN = """
<h1>실행 화면</h1>
<div class="card">
  <div class="st" style="margin-top:0">{{ status }}</div>
  {% if msg %}<div class="msg {{ cls }}">{{ msg }}</div>{% endif %}
  <img class="cam" src="{{ url_for('video') }}" alt="웹캠 화면 (시작 후 몇 초 걸릴 수 있습니다)">
  <form method="post" action="{{ url_for('stop') }}" style="display:inline">
    <button class="r" type="submit">중지</button>
  </form>
  <a class="btn s" href="{{ url_for('main') }}">돌아가기</a>
  <a class="btn s" href="{{ url_for('log') }}" target="_blank">로그</a>
</div>
"""

BYE = """
<h1>로그아웃 완료</h1>
<div class="card">
  <div class="msg ok">door_counter.py 프로세스를 종료했고, 웹 서버를 종료해 포트 {{ port }}를 반환합니다.</div>
  <div class="sub" style="margin-top:14px">다시 사용하려면 라즈베리파이에서 <code>python3 app.py</code> 를 다시 실행하세요.</div>
</div>
"""


def page(body_tpl, hb=True, **ctx):
    body = render_template_string(body_tpl, host=PI_HOST, **ctx)
    return render_template_string(BASE, body=body, host=PI_HOST,
                                  user=current_user(), hb=hb)


# ---------------------------------------------------------------------------
# SSH 헬퍼
# ---------------------------------------------------------------------------
def current():
    tok = session.get("tok")
    return SESSIONS.get(tok) if tok else None


def current_user():
    c = current()
    return c["user"] if c else None


def get_client():
    c = current()
    if not c:
        return None
    tr = c["client"].get_transport()
    if not tr or not tr.is_active():
        SESSIONS.pop(session.get("tok"), None)
        return None
    return c["client"]


def ssh_exec(client, cmd, timeout=20):
    """홈디렉터리에서 bash로 실행"""
    full = "cd ~ && " + cmd
    _, out, err = client.exec_command("bash -lc " + shlex.quote(full), timeout=timeout)
    o, e = out.read().decode(errors="replace"), err.read().decode(errors="replace")
    return out.channel.recv_exit_status(), o, e


# 'stream_runner.py' 를 '[s]tream_runner.py' 로 쓰는 이유: pkill -f 가 자기 자신(bash -lc ...)의
# 명령줄과 매칭되어 셸이 같이 죽는 문제를 피하기 위함.
STOP_CMD = r"""
PIDF=__PIDF__
if [ -f "$PIDF" ] && grep -aq 'stream_[r]unner' "/proc/$(cat "$PIDF")/cmdline" 2>/dev/null; then
  PID=$(cat "$PIDF")
  kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
  for i in 1 2 3 4 5 6; do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
  kill -KILL -- "-$PID" 2>/dev/null
  kill -KILL "$PID" 2>/dev/null
fi
rm -f "$PIDF"
pkill -TERM -u "$(id -un)" -f '[s]tream_runner.py' 2>/dev/null
sleep 0.5
pkill -KILL -u "$(id -un)" -f '[s]tream_runner.py' 2>/dev/null
rm -f __FRAME__ __FRAME__.tmp
if pgrep -u "$(id -un)" -f '[s]tream_runner.py' >/dev/null; then echo RUNNING; else echo STOPPED; fi
""".replace("__PIDF__", PID_PATH).replace("__FRAME__", FRAME_PATH)

STATUS_CMD = ("if pgrep -u \"$(id -un)\" -f '[s]tream_runner.py' >/dev/null; "
              "then echo RUNNING; else echo STOPPED; fi")


def stop_runner(client):
    """door_counter(래퍼 포함) 프로세스 그룹을 종료. 종료되었으면 True"""
    try:
        if client is not None:
            tr = client.get_transport()
            if tr and tr.is_active():
                _, out, _ = ssh_exec(client, STOP_CMD, timeout=30)
                return "STOPPED" in out
    except Exception:
        pass
    # SSH 연결이 끊긴 경우: 같은 계정으로 실행 중이라면 로컬에서라도 종료 시도
    try:
        subprocess.run(["pkill", "-TERM", "-f", "[s]tream_runner.py"], timeout=5)
        time.sleep(0.5)
        subprocess.run(["pkill", "-KILL", "-f", "[s]tream_runner.py"], timeout=5)
        for p in (FRAME_PATH, PID_PATH):
            try:
                os.remove(p)
            except OSError:
                pass
    except Exception:
        pass
    return False


def runner_status(client):
    try:
        _, out, _ = ssh_exec(client, STATUS_CMD, timeout=10)
        return "● 실행 중" if "RUNNING" in out else "■ 정지됨"
    except Exception:
        return "(상태 확인 실패)"


def login_required(fn):
    from functools import wraps

    @wraps(fn)
    def w(*a, **k):
        if not get_client():
            return redirect(url_for("login"))
        return fn(*a, **k)
    return w


def find_script(client):
    """홈디렉터리 아래에서 door_counter.py 를 찾아 절대경로를 반환 (없으면 None).
    여러 개면 CNN_Team_Project 경로 우선, 그다음 가장 최근 수정된 것."""
    cmd = (f'find "$HOME" -type f -name {SCRIPT_NAME} '
           '-not -path "*/.cache/*" -not -path "*/site-packages/*" '
           '-not -path "*/.local/*" -printf "%T@ %p\\n" 2>/dev/null | sort -rn')
    _, out, _ = ssh_exec(client, cmd, timeout=60)
    paths = [ln.split(" ", 1)[1] for ln in out.splitlines() if " " in ln]
    if not paths:
        return None
    preferred = [p for p in paths if "CNN_Team_Project" in p]
    return (preferred or paths)[0]


def get_script(client):
    c = current()
    if not c.get("script"):
        c["script"] = find_script(client)
    return c["script"]


def find_total_line(lines):
    """'total = 숫자' 줄을 자동 검색. 여러 개면 들여쓰기가 가장 얕은(전역) 줄, 그다음 첫 줄.
    반환: (0-based 인덱스, 후보 개수) 또는 (None, 0)"""
    cands = []
    for i, ln in enumerate(lines):
        m = TOTAL_RE.match(ln)
        if m:
            indent = len(ln) - len(ln.lstrip())
            cands.append((indent, i))
    if not cands:
        return None, 0
    cands.sort()
    return cands[0][1], len(cands)


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return page(LOGIN, error=None)

    user = request.form.get("username", "").strip()
    pw = request.form.get("password", "")
    cl = paramiko.SSHClient()
    cl.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cl.connect(PI_HOST, port=PI_PORT, username=user, password=pw,
                   timeout=10, allow_agent=False, look_for_keys=False)
    except paramiko.AuthenticationException:
        return page(LOGIN, error="username 또는 비밀번호가 올바르지 않습니다.")
    except Exception as e:
        return page(LOGIN, error=f"SSH 연결 실패: {e}")

    # 접속 즉시 홈에서 가상환경 activate (성공 여부 확인)
    code, out, err = ssh_exec(cl, f"source {VENV_ACTIVATE} && python -V")
    venv = (f"✅ yolo-env 활성화됨 ({(out or err).strip()})" if code == 0
            else f"⚠️ yolo-env 활성화 실패: {err.strip()}")

    script = find_script(cl)   # door_counter.py 자동 검색
    tok = secrets.token_hex(16)
    SESSIONS[tok] = {"client": cl, "user": user, "venv": venv, "script": script}
    session["tok"] = tok
    return redirect(url_for("main"))


@app.route("/main")
@login_required
def main():
    cl = get_client()
    script = get_script(cl)
    flash = session.pop("flash", None) or (None, None)
    return page(MAIN, venv=current()["venv"],
                script=(f"📄 {script}" if script else f"⚠️ {SCRIPT_NAME} 를 찾지 못했습니다"),
                status=f"door_counter.py: {runner_status(cl)}", msg=flash[0], cls=flash[1])


@app.route("/setcount", methods=["GET", "POST"])
@login_required
def setcount():
    cl = get_client()
    msg = cls = None
    path = get_script(cl)
    if not path:
        return page(SETCOUNT, path="", line="-", current="",
                    msg=f"홈디렉터리에서 {SCRIPT_NAME} 를 찾지 못했습니다.", cls="err")
    try:
        sftp = cl.open_sftp()
        with sftp.open(path, "r") as f:
            text = f.read().decode("utf-8")
    except Exception as e:
        return page(SETCOUNT, path=path, line="-", current="(읽기 실패)",
                    msg=f"파일을 읽을 수 없습니다: {e}", cls="err")

    lines = text.split("\n")
    idx, n = find_total_line(lines)
    if idx is None:
        sftp.close()
        return page(SETCOUNT, path=path, line="-", current="",
                    msg="파일에서 'total = 숫자' 형태의 줄을 찾지 못했습니다.", cls="err")
    cur = lines[idx]

    if request.method == "POST":
        raw = request.form.get("total", "").strip()
        if not re.fullmatch(r"\d+", raw):
            msg, cls = "0 이상의 정수를 입력해주세요.", "err"
        else:
            new = TOTAL_RE.sub(lambda m: m.group(1) + raw, cur, count=1)
            try:
                with sftp.open(path + ".bak", "w") as f:   # 백업
                    f.write(text.encode("utf-8"))
                lines[idx] = new
                with sftp.open(path, "w") as f:            # 저장
                    f.write("\n".join(lines).encode("utf-8"))
                cur = new
                msg, cls = f"저장 완료: {idx + 1}행 total = {raw}  (백업: {SCRIPT_NAME}.bak)", "ok"
            except Exception as e:
                msg, cls = f"저장 실패: {e}", "err"
    if n > 1 and not msg:
        msg, cls = f"'total = 숫자' 줄이 {n}개 발견되어 가장 바깥쪽(들여쓰기 얕은) 줄을 선택했습니다.", ""
    sftp.close()
    return page(SETCOUNT, path=path, line=idx + 1, current=cur.strip(), msg=msg, cls=cls)


@app.route("/run")
@login_required
def run():
    cl = get_client()
    script = get_script(cl)
    if not script:
        return page(RUN, status="", msg=f"홈디렉터리에서 {SCRIPT_NAME} 를 찾지 못했습니다.", cls="err")
    try:
        stop_runner(cl)                     # 이전 실행이 남아 있으면 먼저 종료
        sftp = cl.open_sftp()
        with sftp.open(RUNNER_PATH, "w") as f:
            f.write(RUNNER_CODE.encode())
        sftp.close()
        # 홈에서 venv activate 후 python ./.../door_counter.py 실행(래퍼 경유).
        # setsid: 별도 프로세스 그룹으로 띄워 나중에 그룹 전체를 한 번에 종료할 수 있게 함.
        cmd = (f"source {VENV_ACTIVATE} && "
               f"(setsid nohup python {RUNNER_PATH} {shlex.quote(script)} "
               f"> {LOG_PATH} 2>&1 & echo $! > {PID_PATH}); "
               f"sleep 1.5; "
               f"if kill -0 $(cat {PID_PATH}) 2>/dev/null; then echo OK; else echo DEAD; fi")
        _, out, err = ssh_exec(cl, cmd, timeout=30)
        if "OK" in out:
            msg, cls = f"실행 중: {script}", "ok"
        else:
            _, tail, _ = ssh_exec(cl, f"tail -n 15 {LOG_PATH} 2>/dev/null")
            msg, cls = f"실행 직후 종료되었습니다.\n{tail or err}", "err"
    except Exception as e:
        msg, cls = f"실행 실패: {e}", "err"
    return page(RUN, status=f"door_counter.py: {runner_status(cl)}", msg=msg, cls=cls)


@app.route("/stop", methods=["POST"])
@login_required
def stop():
    ok = stop_runner(get_client())
    session["flash"] = (("⏹ door_counter.py 를 중지했습니다.", "ok") if ok
                        else ("⚠️ 중지를 확인하지 못했습니다. 로그를 확인하세요.", "err"))
    return redirect(url_for("main"))


@app.route("/log")
@login_required
def log():
    _, out, _ = ssh_exec(get_client(), f"tail -n 60 {LOG_PATH} 2>/dev/null")
    return Response(out or "(로그 없음)", mimetype="text/plain; charset=utf-8")


@app.route("/video")
@login_required
def video():
    def gen():
        last, data, sent_at = None, None, 0
        while not STOPPING.is_set():
            try:
                m = os.path.getmtime(FRAME_PATH)
                if m != last:
                    last = m
                    with open(FRAME_PATH, "rb") as f:
                        data = f.read()
                    sent_at = 0
            except FileNotFoundError:
                pass
            now = time.time()
            # 새 프레임이 있거나 2초 이상 정체되면 전송 -> 브라우저가 닫혔을 때 쓰기 오류로 스레드 종료
            if data and (sent_at == 0 or now - sent_at > 2):
                sent_at = now
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# 종료 / 포트 반환
# ---------------------------------------------------------------------------
STOPPING = threading.Event()
HB = {"last": None, "leaving": None}


def cleanup_all():
    """모든 세션의 door_counter 종료 + SSH 연결 닫기 (여러 번 호출해도 안전)"""
    items = list(SESSIONS.items())
    SESSIONS.clear()
    if not items:
        stop_runner(None)       # 세션이 없어도 남은 프로세스가 있으면 정리 시도
    for _, c in items:
        try:
            stop_runner(c["client"])
        except Exception:
            pass
        try:
            c["client"].close()
        except Exception:
            pass


def shutdown_server(reason=""):
    if STOPPING.is_set():
        return
    STOPPING.set()
    print(f"[종료] {reason} -> door_counter 종료, 웹 서버 종료(포트 {WEB_PORT} 반환)", flush=True)
    cleanup_all()
    t = threading.Timer(5, lambda: os._exit(0))   # 정상 종료가 안 될 때의 안전장치
    t.daemon = True
    t.start()
    try:
        os.kill(os.getpid(), signal.SIGINT)        # app.run() 정상 종료 -> 소켓 close
    except Exception:
        os._exit(0)


def watchdog():
    while not STOPPING.is_set():
        time.sleep(2)
        last, leaving = HB["last"], HB["leaving"]
        if last is None:
            continue                                # 아직 아무도 접속 안 함
        now = time.time()
        if leaving and now - leaving > LEAVE_GRACE:
            shutdown_server("웹페이지 닫힘 감지")
        elif now - last > HB_TIMEOUT:
            shutdown_server(f"{HB_TIMEOUT}초 동안 브라우저 응답 없음")


@app.route("/ping", methods=["POST"])
def ping():
    HB["last"] = time.time()
    HB["leaving"] = None
    return ("", 204)


@app.route("/leaving", methods=["POST"])
def leaving():
    HB["leaving"] = time.time()    # 새로고침/페이지 이동이면 곧바로 /ping 이 와서 취소됨
    return ("", 204)


@app.route("/logout", methods=["POST"])
def logout():
    c = SESSIONS.pop(session.pop("tok", None), None)
    if c:
        stop_runner(c["client"])   # 로그아웃 시 door_counter 종료
        try:
            c["client"].close()
        except Exception:
            pass
    html = page(BYE, hb=False, port=WEB_PORT)
    threading.Timer(1.5, shutdown_server, args=("로그아웃",)).start()
    return html


def _on_signal(signum, frame):
    cleanup_all()
    sys.exit(0)


def port_in_use(port):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


if __name__ == "__main__":
    if port_in_use(WEB_PORT):
        print(f"[오류] 포트 {WEB_PORT} 를 이미 다른 프로세스가 사용 중입니다.\n"
              f"  이전에 실행한 app.py 가 남아 있다면:  fuser -k {WEB_PORT}/tcp\n"
              f"  또는 다른 포트로 실행:  WEB_PORT=8001 python3 app.py")
        sys.exit(1)
    for sig in (signal.SIGTERM, signal.SIGHUP):    # kill / 터미널 닫힘
        signal.signal(sig, _on_signal)
    signal.signal(signal.SIGINT, signal.default_int_handler)  # nohup/& 로 실행해도 종료 신호가 먹히도록
    atexit.register(cleanup_all)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        STOPPING.set()
        cleanup_all()
        print(f"서버 종료 완료 - 포트 {WEB_PORT} 반환됨")