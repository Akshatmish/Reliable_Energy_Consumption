#!/bin/bash
gunicorn -w 1 -k sync --bind 0.0.0.0:$PORT --timeout 120 app:app
