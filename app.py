"""Car damage detection: Flask + YOLOv8.

Run:  python app.py            (then open http://127.0.0.1:5000)
Env:  APP_USER / APP_PASSWORD  login credentials (default admin / admin)
      SECRET_KEY               session signing key (random per start if unset)
      CAMERA_INDEX             webcam number (default 0)
"""
import base64
import hmac
import io
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from functools import wraps

# best.pt is this project's own trusted file. Newer PyTorch refuses to unpickle it
# unless this is set (the cause of the "Weights only load failed" error).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2
from flask import (Flask, Response, redirect, render_template, request,
                   session, url_for)
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
for _d in (UPLOAD_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
VIDEO_EXTENSIONS = {"mp4"}
DEFAULT_CONF = 0.25
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
RESULT_MAX_AGE = 24 * 3600  # delete processed videos after a day

# Severity is an estimate from how much of the picture the damage boxes cover.
SEVERITY_LEVELS = [(0.05, "minor"), (0.15, "moderate"), (1.01, "severe")]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24).hex()

ADMIN_USER = os.environ.get("APP_USER", "admin")
ADMIN_PASSWORD = os.environ.get("APP_PASSWORD", "admin")

model = YOLO(MODEL_PATH)
model_lock = threading.Lock()      # one prediction at a time; the model is not thread-safe
camera_stop = threading.Event()    # set by /stop, ends the live stream


# ---------------------------------------------------------------- helpers
def extension(filename):
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def read_conf():
    """Confidence threshold from the form (percent), clamped to 5-95."""
    try:
        pct = float(request.form.get("conf", DEFAULT_CONF * 100))
    except ValueError:
        pct = DEFAULT_CONF * 100
    return max(5.0, min(95.0, pct)) / 100


def zone_key(label):
    """Map a class name to the color key used in the CSS."""
    text = label.lower().replace("_", " ")
    if "wind" in text and "shield" in text:
        return "windshield"
    for key in ("window", "door", "headlight", "mirror", "dent", "hood", "bumper"):
        if key in text:
            return key
    return "other"


def summarize(result):
    """Turn one YOLO result into counts, coverage and a severity estimate."""
    height, width = result.orig_shape[:2]
    image_area = float(height * width) or 1.0
    best = {}
    counts = Counter()
    coverage = 0.0
    if result.boxes is not None and len(result.boxes):
        for cls, conf, xyxy in zip(result.boxes.cls.tolist(),
                                   result.boxes.conf.tolist(),
                                   result.boxes.xyxy.tolist()):
            label = result.names[int(cls)]
            x1, y1, x2, y2 = xyxy
            coverage += max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
            counts[label] += 1
            best[label] = max(best.get(label, 0.0), conf)
    coverage = min(coverage, 1.0)
    if not counts:
        severity = "none"
    else:
        severity = next(name for limit, name in SEVERITY_LEVELS if coverage < limit)
    rows = [{"label": label, "zone": zone_key(label), "count": n,
             "confidence": round(best[label] * 100)}
            for label, n in counts.most_common()]
    return {"rows": rows, "total": sum(counts.values()),
            "coverage": round(coverage * 100, 1), "severity": severity}


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def safe_next(target):
    """Only allow redirects to paths inside this site."""
    return target if target and target.startswith("/") and not target.startswith("//") else None


def prune_results():
    cutoff = time.time() - RESULT_MAX_AGE
    for name in os.listdir(RESULT_DIR):
        path = os.path.join(RESULT_DIR, name)
        if name.endswith(".mp4") and os.path.getmtime(path) < cutoff:
            os.remove(path)


def make_browser_playable(raw_path, final_path):
    """OpenCV writes mp4v, which most browsers will not play. Re-encode to H.264 if ffmpeg exists."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path, "-vcodec", "libx264",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", final_path]
        if subprocess.run(cmd).returncode == 0 and os.path.exists(final_path):
            os.remove(raw_path)
            return
    os.replace(raw_path, final_path)


def process_video(source, conf):
    """Run the model on every frame and save an annotated copy. Returns (filename, stats)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("That video could not be opened.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 120 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    token = uuid.uuid4().hex[:10]
    raw_path = os.path.join(RESULT_DIR, f"raw_{token}.mp4")
    name = f"result_{token}.mp4"
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames = damaged_frames = 0
    seen_in_frames = Counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with model_lock:
                result = model.predict(source=frame, conf=conf, verbose=False)[0]
            writer.write(result.plot())
            frames += 1
            labels = {result.names[int(c)] for c in result.boxes.cls.tolist()} if len(result.boxes) else set()
            if labels:
                damaged_frames += 1
                seen_in_frames.update(labels)
    finally:
        cap.release()
        writer.release()

    if frames == 0:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        raise ValueError("No frames could be read from that video.")

    make_browser_playable(raw_path, os.path.join(RESULT_DIR, name))
    rows = [{"label": label, "zone": zone_key(label), "count": n,
             "confidence": round(100 * n / frames)}
            for label, n in seen_in_frames.most_common()]
    stats = {"rows": rows, "frames": frames, "damaged_frames": damaged_frames,
             "seconds": round(frames / fps, 1)}
    return name, stats


# ---------------------------------------------------------------- pages
@app.route("/")
@app.route("/first")
def first():
    return render_template("first.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    target = safe_next(request.values.get("next"))
    error = None
    if request.method == "POST":
        user_ok = hmac.compare_digest(request.form.get("uname", ""), ADMIN_USER)
        pass_ok = hmac.compare_digest(request.form.get("pwd", ""), ADMIN_PASSWORD)
        if user_ok and pass_ok:
            session["user"] = ADMIN_USER
            return redirect(target or url_for("image"))
        error = "Username or password is incorrect. Check both and try again."
    return render_template("login.html", error=error, next=target), (401 if error else 200)


@app.route("/logout")
def logout():
    session.clear()
    camera_stop.set()
    return redirect(url_for("first"))


@app.route("/chart")
def chart():
    return render_template("chart.html")


@app.route("/performance")
def performance():
    return render_template("performance.html")


# ---------------------------------------------------------------- photo
@app.route("/image")
@login_required
def image():
    return render_template("image.html")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        return redirect(url_for("image"))

    def fail(message):
        return render_template("image.html", error=message), 400

    file = request.files.get("file")
    if file is None or file.filename == "":
        return fail("Choose a photo before you start.")
    if extension(file.filename) not in IMAGE_EXTENSIONS:
        return fail("That file is not a JPG or PNG. Choose a different photo.")
    try:
        picture = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return fail("That image could not be read. Try saving it again as a JPG or PNG.")

    conf = read_conf()
    with model_lock:
        result = model.predict(source=picture, conf=conf, verbose=False)[0]

    annotated = Image.fromarray(result.plot()[..., ::-1])  # plot() is BGR, PIL wants RGB
    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return render_template("image.html", detection_results=encoded,
                           summary=summarize(result), conf=round(conf * 100))


# ---------------------------------------------------------------- video
@app.route("/video")
@login_required
def video():
    return render_template("video.html")


@app.route("/predict_img", methods=["GET", "POST"])
@login_required
def predict_img():
    if request.method == "GET":
        return redirect(url_for("video"))

    def fail(message):
        return render_template("video.html", error=message), 400

    file = request.files.get("file")
    if file is None or file.filename == "":
        return fail("Choose a video before you start.")
    if extension(file.filename) not in VIDEO_EXTENSIONS:
        return fail("Only MP4 videos work here. Choose a different file.")

    prune_results()
    upload_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}")
    file.save(upload_path)
    conf = read_conf()
    try:
        name, stats = process_video(upload_path, conf)
    except ValueError as exc:
        return fail(str(exc))
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)
    return render_template("video.html", video_url=url_for("static", filename=f"results/{name}"),
                           stats=stats, conf=round(conf * 100))


# ---------------------------------------------------------------- live camera
def stream_frames(cap):
    try:
        while not camera_stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            with model_lock:
                result = model.predict(source=frame, conf=DEFAULT_CONF, verbose=False)[0]
            ok, jpeg = cv2.imencode(".jpg", result.plot())
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.route("/webcam")
@login_required
def webcam():
    camera_stop.clear()
    return render_template("webcam.html")


@app.route("/video_feed")
@login_required
def video_feed():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        return Response("No camera found", status=503, mimetype="text/plain")
    return Response(stream_frames(cap), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop", methods=["POST"])
@login_required
def stop():
    camera_stop.set()
    return redirect(url_for("first"))


@app.errorhandler(413)
def too_large(_):
    return render_template("video.html", error="That file is over the 500 MB limit."), 413


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Car damage detection web app")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True)
