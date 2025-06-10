import serial

class FSRSensor:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    # def read(self):
    #     """Reads one line from serial and returns A0, A1 as integers."""
    #     while True:
    #         try:
    #             line = self.ser.readline().decode('utf-8').strip()
    #             if line:
    #                 val0, val1 = map(int, line.split(","))
    #                 now_ms = int(time.time() * 1000)  # 현재 시각 (ms)
    #                 print("c")
    #                 print("va10: %d" % val0)
    #                 print("va11: %d" % val1)
    #                 print("time (ms):", now_ms)
    #                 return val0, val1
    #         except (ValueError, UnicodeDecodeError):
    #             # Skip malformed lines
    #             continue

    # def read(self):
    #     # 방법 1: 버퍼에 남은 모든 줄을 소비하고 마지막 값만 리턴
    #     lines = []
    #     while self.ser.in_waiting:
    #         line = self.ser.readline().decode('utf-8', errors='ignore').strip()
    #         if line:
    #             lines.append(line)
    #     if lines:
    #         val0, val1 = map(int, lines[-1].split(","))
    #         return val0, val1
    #     else:
    #         # 버퍼가 비었으면 새로 한 줄 받기
    #         while True:
    #             try:
    #                 line = self.ser.readline().decode('utf-8', errors='ignore').strip()
    #                 if line:
    #                     val0, val1 = map(int, line.split(","))
    #                     return val0, val1
    #             except (ValueError, UnicodeDecodeError):
    #                 continue
    def read(self):
        """시리얼 버퍼에 쌓인 줄 모두 소비하고, 가장 마지막 값만 반환"""
        latest_line = None
        # 버퍼에 데이터가 있으면 계속 읽어서 마지막 값만 남김
        while self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                print(f"Warning: Unexpected sensor value '{line}'")
                continue
            latest_line = parts
        # 버퍼가 비었으면 새로 한 줄 읽기
        if latest_line is not None:
            val0, val1 = map(int, latest_line)
            # print(val0, val1)
            return val0, val1
        else:
            while True:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) != 2:
                        print(f"Warning: Unexpected sensor value '{line}'")
                        continue
                    val0, val1 = map(int, parts)
                    # print(val0, val1)
                    return val0, val1
                except (ValueError, UnicodeDecodeError) as e:
                    print(f"Exception: {e}, line: {repr(line)}")
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