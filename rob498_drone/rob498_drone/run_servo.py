import os
import subprocess

# Set paths
arduino_cli_path = "arduino-cli"  # Change if necessary
project_dir = "ArduinoProjects/Servo"
board_type = "arduino:avr:uno"  # Replace with your board type
port = "/dev/ttyUSB0"  # Replace with your board's port (e.g., COM3 for Windows)

# Compile the sketch
compile_cmd = f"{arduino_cli_path} compile --fqbn {board_type} {project_dir}"
upload_cmd = f"{arduino_cli_path} upload -p {port} --fqbn {board_type} {project_dir}"

try:
    print("Compiling...")
    subprocess.run(compile_cmd, shell=True, check=True)
    print("Uploading...")
    subprocess.run(upload_cmd, shell=True, check=True)
    print("Upload complete! The Arduino is now running the sketch.")
except subprocess.CalledProcessError as e:
    print("Error:", e)
