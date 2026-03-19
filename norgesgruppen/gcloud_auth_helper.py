"""Helper to extract gcloud auth URL and feed back verification code."""
import subprocess
import sys
import re
import os

os.environ["PATH"] = os.path.expanduser("~/google-cloud-sdk/bin") + ":" + os.environ["PATH"]

proc = subprocess.Popen(
    ["gcloud", "auth", "login", "--no-launch-browser"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

output = ""
while True:
    char = proc.stdout.read(1)
    if not char:
        break
    output += char
    sys.stdout.write(char)
    sys.stdout.flush()
    if "enter the verification code" in output.lower():
        break

# Wait for user to provide the code
code = input("\n>>> PASTE CODE HERE: ")
proc.stdin.write(code + "\n")
proc.stdin.flush()

# Read remaining output
rest = proc.communicate()[0]
print(rest)
