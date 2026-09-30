#!/usr/bin/env python3
"""
라즈베리파이5 door_counter 웹 컨트롤러
- 로그인(SSH 인증) -> 메인(초기 인원수 설정 / 실행)
- 이 파일은 라즈베리파이(10.10.17.105) 위에서 실행해야 합니다.

설치:  pip install flask paramiko
실행:  python3 app.py        ->  http://10.10.17.117:8000
"""
import os
import re
import secrets
import shlex
import time

import paramiko
from flask import (Flask, Response, redirect, render_template_string,
                   request, session, url_for)

PI_HOST = os.environ.get("PI_HOST", "10.10.17.105")
PI_PORT = int(os.environ.get("PI_PORT", "22"))
VENV_ACTIVATE = "work/yolo-env/bin/activate"
SCRIPT_REL = "work/team_project/door_counter.py"
TOTAL_LINE = 97                      # total 값이 있는 줄 번호(1부터)
FRAME_PATH = "/tmp/door_frame.jpg"
RUNNER_PATH = "/tmp/stream_runner.py"
LOG_PATH = "/tmp/door_counter.log"

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

script = os.path.expanduser("~/work/team_project/door_counter.py")
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
</style></head><body><div class="wrap">
{% if user %}<div class="top"><span>👤 {{ user }}@{{ host }}</span><a href="{{ url_for('logout') }}">로그아웃</a></div>{% endif %}
{{ body|safe }}
</div></body></html>
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
<h1>Door Counter</h1><div class="sub">{{ venv }}</div>
<div class="banners">
  <a class="banner" href="{{ url_for('setcount') }}">초기 인원수 설정</a>
  <a class="banner g" href="{{ url_for('run') }}">실행</a>
</div>
"""

SETCOUNT = """
<h1>초기 인원수 설정</h1>
<div class="sub">door_counter.py {{ line }}번째 줄</div>
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
  {% if msg %}<div class="msg {{ cls }}">{{ msg }}</div>{% endif %}
  <img class="cam" src="{{ url_for('video') }}" alt="웹캠 화면 (시작 후 몇 초 걸릴 수 있습니다)">
  <form method="post" action="{{ url_for('stop') }}" style="display:inline">
    <button class="r" type="submit">중지</button>
  </form>
  <a class="btn s" href="{{ url_for('main') }}">돌아가기</a>
  <a class="btn s" href="{{ url_for('log') }}" target="_blank">로그</a>
</div>
"""


def page(body_tpl, **ctx):
    body = render_template_string(body_tpl, host=PI_HOST, **ctx)
    return render_template_string(BASE, body=body, host=PI_HOST, user=current_user())


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


def login_required(fn):
    from functools import wraps

    @wraps(fn)
    def w(*a, **k):
        if not get_client():
            return redirect(url_for("login"))
        return fn(*a, **k)
    return w


def read_script_lines(client):
    sftp = client.open_sftp()
    try:
        path = sftp.normalize(SCRIPT_REL)
        with sftp.open(path, "r") as f:
            text = f.read().decode("utf-8")
        return sftp, path, text
    except Exception:
        sftp.close()
        raise


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

    tok = secrets.token_hex(16)
    SESSIONS[tok] = {"client": cl, "user": user, "venv": venv}
    session["tok"] = tok
    return redirect(url_for("main"))


@app.route("/main")
@login_required
def main():
    return page(MAIN, venv=current()["venv"])


@app.route("/setcount", methods=["GET", "POST"])
@login_required
def setcount():
    cl = get_client()
    msg = cls = None
    try:
        sftp, path, text = read_script_lines(cl)
    except Exception as e:
        return page(SETCOUNT, line=TOTAL_LINE, current="(읽기 실패)",
                    msg=f"파일을 읽을 수 없습니다: {e}", cls="err")

    lines = text.split("\n")
    cur = lines[TOTAL_LINE - 1] if len(lines) >= TOTAL_LINE else ""

    if request.method == "POST":
        raw = request.form.get("total", "").strip()
        if not re.fullmatch(r"\d+", raw):
            msg, cls = "0 이상의 정수를 입력해주세요.", "err"
        elif not re.match(r"^\s*total\s*=", cur):
            msg, cls = (f"{TOTAL_LINE}행이 'total = ...' 형태가 아니어서 수정하지 않았습니다.\n"
                        f"현재 내용: {cur}"), "err"
        else:
            new = re.sub(r"^(\s*total\s*=\s*)-?\d+", lambda m: m.group(1) + raw, cur, count=1)
            if new == cur and not re.match(r"^\s*total\s*=\s*-?\d+", cur):
                msg, cls = f"숫자 값을 찾지 못했습니다: {cur}", "err"
            else:
                try:
                    with sftp.open(path + ".bak", "w") as f:   # 백업
                        f.write(text.encode("utf-8"))
                    lines[TOTAL_LINE - 1] = new
                    with sftp.open(path, "w") as f:            # 저장
                        f.write("\n".join(lines).encode("utf-8"))
                    cur = new
                    msg, cls = f"저장 완료: total = {raw}  (백업: door_counter.py.bak)", "ok"
                except Exception as e:
                    msg, cls = f"저장 실패: {e}", "err"
    sftp.close()
    return page(SETCOUNT, line=TOTAL_LINE, current=cur.strip(), msg=msg, cls=cls)


@app.route("/run")
@login_required
def run():
    cl = get_client()
    try:
        ssh_exec(cl, f"pkill -u $USER -f {RUNNER_PATH} || true")
        sftp = cl.open_sftp()
        with sftp.open(RUNNER_PATH, "w") as f:
            f.write(RUNNER_CODE.encode())
        sftp.close()
        try:
            os.remove(FRAME_PATH)
        except FileNotFoundError:
            pass
        # python ./work/CNN_Team_Project/door_counter.py 와 동일하게 홈에서 실행(래퍼 경유)
        cmd = (f"cd ~ && source {VENV_ACTIVATE} && "
               f"nohup python {RUNNER_PATH} > {LOG_PATH} 2>&1 &")
        cl.exec_command("bash -lc " + shlex.quote(cmd))
        msg, cls = "door_counter.py 실행 중…", "ok"
    except Exception as e:
        msg, cls = f"실행 실패: {e}", "err"
    return page(RUN, msg=msg, cls=cls)


@app.route("/stop", methods=["POST"])
@login_required
def stop():
    ssh_exec(get_client(), f"pkill -u $USER -f {RUNNER_PATH} || true")
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
        last = None
        while True:
            try:
                m = os.path.getmtime(FRAME_PATH)
                if m != last:
                    last = m
                    with open(FRAME_PATH, "rb") as f:
                        data = f.read()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            except FileNotFoundError:
                pass
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logout")
def logout():
    c = SESSIONS.pop(session.pop("tok", None), None)
    if c:
        c["client"].close()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)

