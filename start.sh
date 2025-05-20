#!/bin/bash
gunicorn -w 2 -k sync app:app