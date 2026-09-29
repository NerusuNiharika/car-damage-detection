# Car damage detection (YOLOv8 + Flask)

Finds and labels vehicle damage in photos, videos and a live webcam feed.
Classes: damaged door, window, headlight, mirror, dent, hood, bumper, windshield. Reported accuracy: 91%.

## Run
1. Install Python 3.10 - 3.12. On Windows also install the Visual C++ Redistributable (x64).
2. Double-click `run.bat` (Windows) or run `./run.sh` (Mac/Linux). Or by hand:
   `pip install -r requirements.txt` then `python app.py`
3. Open http://127.0.0.1:5000 and sign in (default `admin` / `admin`).

Options: `python app.py --port 8000`, `--host 0.0.0.0` (lets other devices on your network connect; change the password first).

## Settings (environment variables)
| Variable | Purpose | Default |
|---|---|---|
| APP_USER, APP_PASSWORD | Login | admin / admin |
| SECRET_KEY | Signs login sessions | random each start |
| CAMERA_INDEX | Which webcam | 0 |

## Pages
- Photo check: upload JPG/PNG, adjust sensitivity, get boxes, a damage list and a severity estimate.
- Video check: upload MP4, get an annotated video you can play and download, plus frame statistics.
- Live camera: real-time detection from the webcam, with a Stop button.
- Model accuracy and Class breakdown: public pages showing the model's results.

## Layout
```
app.py              server and detection logic
best.pt             trained YOLOv8 weights
templates/          pages (base.html is the shared layout)
static/css/theme.css, static/js/app.js, static/confusion_matrix.png
static/results/     processed videos (deleted after 24 h)
samples/            photos and a video to try
model/              training notebook
```
Severity (minor / moderate / severe) is a simple estimate from the share of the photo covered by damage boxes (under 5%, 5-15%, over 15%). Change `SEVERITY_LEVELS` in `app.py` to tune it.
