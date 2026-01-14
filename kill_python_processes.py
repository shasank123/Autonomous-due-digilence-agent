# kill_python_processes.py - Find and stop Python processes
import subprocess
import sys

print("Finding Python processes...")
result = subprocess.run(['tasklist'], capture_output=True, text=True)

python_processes = []
for line in result.stdout.split('\n'):
    if 'python' in line.lower():
        parts = line.split()
        if len(parts) >= 2:
            pid = parts[1]
            python_processes.append(pid)
            print(f"Found: PID {pid} - {parts[0]}")

if not python_processes:
    print("\nNo Python processes found running.")
else:
    print(f"\n{len(python_processes)} Python process(es) found.")
    print("\nTo kill them, run:")
    for pid in python_processes:
        print(f"  taskkill /F /PID {pid}")
    
    print("\nOr run this script with 'kill' argument:")
    print("  python kill_python_processes.py kill")

if len(sys.argv) > 1 and sys.argv[1] == 'kill':
    print("\n=== Killing processes ===")
    for pid in python_processes:
        try:
            subprocess.run(['taskkill', '/F', '/PID', pid], check=True)
            print(f"[OK] Killed PID {pid}")
        except:
            print(f"[FAILED] Could not kill PID {pid}")
