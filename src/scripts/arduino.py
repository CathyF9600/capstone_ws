import serial
import threading
import time

class ArduinoInterface:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        self.ser = None  # Initialize as None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
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
        last_check_time = time.time()
        TIMEOUT = 3.0  # seconds

        while self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    print(f"[Arduino → Jetson] Received: '{line}'")

                    expected = f"{self.last_command}_CONFIRM" if self.last_command else None
                    if expected and line.upper() == expected.upper():
                        with self.lock:
                            self.waiting_for_confirm = False
                            self.last_command = None
                            last_check_time = time.time()  # Reset timer
                    else:
                        print(f"Unexpected response: '{line}' (expected: '{expected}')")
                else:
                    # Check timeout
                    if self.waiting_for_confirm and (time.time() - last_check_time > TIMEOUT):
                        with self.lock:
                            print("Timeout waiting for confirmation. Resetting state.")
                            self.waiting_for_confirm = False
                            self.last_command = None
                            last_check_time = time.time()
            except serial.SerialException:
                print("Error: Lost connection to Arduino.")
                self.ser = None
                break
            time.sleep(0.05)


    def send_command(self, command):
        if not self.is_connected():
            print("Error: Arduino is not connected.")
            return False

        should_wait_for_confirm = command.upper() in ["ON", "OFF"]

        with self.lock:
            if should_wait_for_confirm and self.waiting_for_confirm:
                print("Waiting for confirmation before sending another command.")
                return False
            self.last_command = command if should_wait_for_confirm else None
            self.waiting_for_confirm = should_wait_for_confirm

        full_command = "{}\n".format(command)
        try:
            self.ser.write(full_command.encode())
            print("[Jetson to Arduino] Sent command: {}".format(command))
            return True
        except serial.SerialException:
            print("Error: Failed to send command.")
            self.ser = None
            return False


    def is_connected(self):
        """Check if the Arduino is connected."""
        return self.ser is not None and self.ser.is_open

    def close(self):
        if self.ser:
            self.ser.close()
            print("Serial connection closed.")
            self.ser = None

a = ArduinoInterface(baudrate=9600)
while True:
    c = input('Enter your command (o (open)/ c (close)): ')
    if c == 'o': c = 'ON'
    elif c == 'c': c = 'OFF'
    a.send_command(c)
