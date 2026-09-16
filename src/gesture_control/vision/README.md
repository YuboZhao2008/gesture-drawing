# Vision integration

Install `python -m pip install -e ".[dev,vision]"` on Python 3.12+. The optional
extra uses MediaPipe Tasks and **opencv-contrib-python**, which supplies `cv2`
and satisfies MediaPipe's own dependency. Do not install opencv-python alongside
it: both distributions write the same module. Geometry and public package imports
work without these extras; camera, detector and drawing imports are lazy.

Download the official version-1 model explicitly (no network activity is initiated
by our adapter):

```powershell
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile models/hand_landmarker.task
```

The model is ignored by Git. Backend API and model documentation:
[MediaPipe Hand Landmarker for Python](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

```python
from gesture_control.vision import OpenCVCamera, MediaPipeHandDetector, PalmNormalizer

normalizer = PalmNormalizer()
with MediaPipeHandDetector("models/hand_landmarker.task") as detector:
    with OpenCVCamera(index=0, width=640, height=480, fps=30) as camera:
        captured = camera.read()
        frame = detector.detect(captured.image, captured.context)
        normalized = [normalizer.normalize(hand, frame.context) for hand in frame.hands]
        # Production: pass captured.image/context to app.process_frame instead.
        # Inject this same detector into the app and the normalizer into recognition.
```

`CameraConfig.index` and `target_fps` map to camera `index` and `fps`;
`min_detection_confidence` maps to the detector's same-named argument. Requested
resolution is a vision constructor option, without expanding shared configuration.
Actual image dimensions are recorded in each `FrameContext`. Camera images are
uint8 HxWx3 BGR arrays, unmirrored. Frame timestamps are elapsed monotonic seconds,
measured after capture. `open()` is idempotent; `read()` requires an open camera.
Startup/read failures raise and release capture. They are not hand absence.
Both components support `release()`, `close()` and context managers.

The detector implements `HandDetector.detect(image, context) -> HandFrame`. It
uses one MediaPipe Tasks HandLandmarker in synchronous VIDEO mode with one hand.
It preserves the original context and all x/y/z values, including offscreen points.
Blank scenes return `HandFrame(context, ())` and must reach downstream processing.
Malformed images/results and backend failures raise; they are not fabricated absence.
VIDEO milliseconds derive only from context time. Ties after rounding are advanced
by one millisecond for the backend while the core context remains unchanged.
Frame IDs must increase and source timestamps must not decrease.

Tasks handedness labels are retained for ordinary unmirrored input. If the supplied
image is mirrored, set `input_is_mirrored=True` to swap anatomical labels; coordinates
still describe that supplied image. Prefer unmirrored capture and mirror only the
preview. `HandLandmarks.confidence` holds the handedness category score (zero if
unavailable), **not a detection probability**. Tasks exposes thresholds for detection,
presence and tracking but no per-result scores for them. No core fields are added.

Track IDs have a detector-instance prefix and increasing counter. A track continues
only across consecutive visible observations with compatible handedness, unchanged
resolution, a time gap <= 0.25 seconds, and wrist displacement <= 0.25 image-width
units (defaults configurable). Any empty frame ends detector continuity; the next
observation receives a new ID. Temporal owns all grace/dwell/reacquisition policy.
This intentionally conservative single-hand heuristic cannot identify a replacement
hand appearing in the same place without a visible gap. Fast motion or handedness
classification flips can split a track. There are no persistent biometric IDs.
Create a new detector and reset temporal processing for each new camera stream.

`PalmNormalizer.normalize(hand, context) -> NormalizedHand | None` follows the shared
contract exactly: subtract wrist, multiply y differences by height/width, divide all
axes by 3D wrist-to-middle-MCP distance. `scale` is in image-width units. Rotation,
handedness and relative z geometry remain intact. No handedness canonicalization is
performed. Scale <= 1e-9 returns None; recognition must retain the ID and emit UNKNOWN.
Normalization is deterministic, translation invariant and uniformly scale invariant.

Helpers exported by `gesture_control.vision`:

- `landmark(hand, LandmarkIndex)` accepts raw or normalized hands.
- `index_fingertip(hand)` returns the source `Landmark3D`.
- `pointer_coordinates(hand)` returns source x/y fractions, with no clipping/mirroring.
- `fingertip_pixels(hand, context)` returns x*width, y*height camera coordinates.
- `hand_bounding_box(hand)` returns unclipped left/top/right/bottom fractions.
- `palm_center(hand)` averages wrist and four MCPs in the input coordinate space.
- `draw_hand_frame(image, frame)` draws joints, connections, box and tip in place.
  Copy explicitly when an unmodified frame is also needed. Rendering coordinates
  are clipped for OpenCV; observations are never changed.

Optional smoke tests (no display or saved frames):

```powershell
python -m gesture_control.vision.smoke --blank --frames 5
python -m gesture_control.vision.smoke --camera 0 --frames 30
```

The webcam command fails explicitly when no camera is available. Neither smoke test
runs under ordinary pytest. Hardware-independent vision tests use mocked capture
and backend results; dependency-specific tests skip when the vision extra is absent.
Run all checks with the extra installed to validate them.
