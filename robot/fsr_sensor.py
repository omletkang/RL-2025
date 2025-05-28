import serial

class FSRSensor:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def read(self):
        """Reads one line from serial and returns A0, A1 as integers."""
        while True:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    val0, val1 = map(int, line.split(","))
                    return val0, val1
            except (ValueError, UnicodeDecodeError):
                # Skip malformed lines
                continue

    def close(self):
        self.ser.close()

    def __del__(self):
        # Ensure the port closes if object is deleted
        if self.ser and self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":

    sensor = FSRSensor(port="/dev/ttyUSB0")

    try:
        for _ in range(100):
            a0, a1 = sensor.read()
            print(f"{a0}, {a1}")
            
    finally:
        sensor.close()