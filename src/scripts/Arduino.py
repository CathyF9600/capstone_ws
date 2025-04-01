import serial
import threading
import time

class ArduinoInterface:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        print('hello')
        self.ser = None  # Initialize as None
        try:
            print('a')
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            print('b')
            if not self.ser.is_open:
                raise serial.SerialException("Port exists but is not open.")
            print("Arduino connected successfully on port {}".format(port))
        except (serial.SerialException, OSError) as e:
            print("Error: Could not connect to Arduino on port {}. {}".format(port, e))
            self.ser = None  # Mark as not connected

        self.lock = threading.Lock()
        self.waiting_for_confirm = False
        self.last_command = None

        if self.ser:  # Only start thread if connection was successful
            self.confirm_thread = threading.Thread(target=self._listen_for_confirmations)
            self.confirm_thread.daemon = True
            self.confirm_thread.start()

    def _listen_for_confirmations(self):
        """Background thread to read serial and check for confirmations."""
        while self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode().strip()
                    if line == "{}_CONFIRM".format(self.last_command):
                        with self.lock:
                            self.waiting_for_confirm = False
                            self.last_command = None
            except serial.SerialException:
                print("Error: Lost connection to Arduino.")
                self.ser = None  # Mark as disconnected
                break
            time.sleep(0.05)

    def send_command(self, command):
        """Send a command only if no other command is pending confirmation."""
        if not self.is_connected():
            print("Error: Arduino is not connected.")
            return False

        with self.lock:
            if self.waiting_for_confirm:
                print("Waiting for confirmation before sending another command.")
                return False
            self.last_command = command
            self.waiting_for_confirm = True

        full_command = "{}\n".format(command)
        try:
            self.ser.write(full_command.encode())
            print("[Jetson to Arduino] Sent command: {}".format(command))
            return True
        except serial.SerialException:
            print("Error: Failed to send command.")
            self.ser = None  # Mark as disconnected
            return False

    def is_connected(self):
        """Check if the Arduino is connected."""
        return self.ser is not None and self.ser.is_open

    def close(self):
        if self.ser:
            self.ser.close()
            print("Serial connection closed.")
            self.ser = None

a = ArduinoInterface()
while True:
    c = input('Enter your command (ON/OFF): ')
    a.send_command(c)