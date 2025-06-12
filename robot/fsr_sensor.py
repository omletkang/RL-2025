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
    # def read(self):
    #     """
    #     시리얼 포트에서 한 줄을 읽어 정수 값 두 개를 반환합니다.
    #     timeout을 사용하여 프로그램이 멈추는 것을 방지합니다.
    #     """
    #     try:
    #         # readline()은 timeout 시간 동안 기다렸다가 데이터가 없으면 빈 바이트(b'')를 반환합니다.
    #         line_bytes = self.ser.readline()

    #         # 타임아웃으로 인해 빈 데이터가 수신된 경우
    #         if not line_bytes:
    #             # print("Warning: No data received within timeout period.")
    #             return None, None # 또는 이전 값을 유지하거나 다른 적절한 처리를 합니다.

    #         # 수신된 데이터를 디코딩
    #         line = line_bytes.decode('utf-8', errors='ignore').strip()

    #         if not line:
    #             return None, None

    #         # 데이터 파싱
    #         parts = line.split(",")
    #         if len(parts) != 2:
    #             print(f"Warning: Unexpected sensor value format '{line}'")
    #             return None, None
            
    #         val0, val1 = map(int, parts)
    #         return val0, val1

    #     except (ValueError, UnicodeDecodeError) as e:
    #         # 데이터 포맷이 잘못되었거나(e.g. "123,abc"), 디코딩 오류가 발생한 경우
    #         print(f"Exception during parsing: {e}, on line: {repr(line_bytes)}")
    #         return None, None
    #     except Exception as e:
    #         # 그 외 시리얼 통신 관련 예외 처리
    #         print(f"An unexpected error occurred: {e}")
    #         return None, None
    
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