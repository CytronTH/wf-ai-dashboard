#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time
import datetime

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
# Use pull_up_down=GPIO.PUD_DOWN to ensure stable reading if floating
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

print("Monitoring GPIO 23...")
print("Press Ctrl+C to exit")
print("-" * 25)

try:
    while True:
        state = GPIO.input(23)
        if state:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] GPIO 23 is HIGH (NG Triggered!)")
            # Wait for 1 second to avoid spamming the same pulse
            time.sleep(1)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nExiting...")
    GPIO.cleanup()
