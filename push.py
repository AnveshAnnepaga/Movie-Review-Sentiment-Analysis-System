import subprocess
import sys

commands = [
    "git add .",
    'git commit -m "Fix Streamlit deployment, update UI, and add README"',
    "git push"
]

for cmd in commands:
    print(f"Executing: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if process.stdout:
        print(process.stdout)
    if process.stderr:
        print(process.stderr)
    if process.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)
        
print("Successfully pushed to GitHub!")
