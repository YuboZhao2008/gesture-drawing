"""Benchmark-only selection of the production waitKey fallback; no GUI worker."""

import sys

import cv2

from gesture_control.runner import main

cv2.pollKey = None
raise SystemExit(main(sys.argv[1:]))
