import psutil
import os

def track():
    
    memory_threshold = 90 # threshold in percentage
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > memory_threshold:
        os._exit(1)
