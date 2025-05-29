import rtde_control
import rtde_receive
import time

def main():
    ROBOT_IP = "192.168.0.1"  # Change to your robot's IP

    # Connect to RTDE interfaces
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    # Get current pose
    current_pose = rtde_r.getActualTCPPose()  # [x, y, z, Rx, Ry, Rz]
    print("Current TCP Pose:", current_pose)

    # Desired joint positions in radians
    q = [0.0, -1.57, -1.57, 1.57, -1.57, 0.0]

    # Send servoJ command (realtime joint-level control)
    # print("Sending servoJ command...")
    # rtde_c.servoJ(q, 0.5, 0.5, 0.01, 0.2, 300)  # target_q, velocity, acceleration, dt, lookahead_time, gain
    print("Sending moveL command...")
    p = [0.1946, -0.2212, 0.04040, 1.10417, 1.12385, -0.027471]
    rtde_c.moveL(p, 0.03, 0.2) # velocity, accelration
    time.sleep(2.0)

    rtde_c.servoStop()
    rtde_c.disconnect()

if __name__ == '__main__':
    print('hi')
    main()
