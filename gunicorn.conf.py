import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 2  # Reduced to 2 for Render free tier
threads = 2
timeout = 300  # 5 minutes
loglevel = "info"
accesslog = "-"
errorlog = "-"
worker_class = "gthread"  # Changed to WSGI worker
max_requests = 1000
max_requests_jitter = 50
keepalive = 5
