import rtde_control
import rtde_receive
import time


class URRobot:
    def __init__(self, ip="192.168.0.1"):
        self.ip = ip
        self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)

    def get_tcp_pose(self):
        """Returns current TCP pose [x, y, z, Rx, Ry, Rz]."""
        return self.rtde_r.getActualTCPPose()

    def move_tcp(self, pose, velocity=0.05, acceleration=0.2):
        """
        Moves robot in task space using moveL.
        :param pose: List of 6 values [x, y, z, Rx, Ry, Rz]
        """
        self.rtde_c.moveL(pose, velocity, acceleration) # MoveL instead of ServoL

    def move_joint(self, q, velocity=1.0, acceleration=1.0):
        """
        Moves robot in joint space using moveJ.
        :param q: List of 6 joint angles (radians)
        """
        self.rtde_c.moveJ(q, velocity, acceleration)
    
    def set_tcp_z(self, z_val):
        pose = self.get_tcp_pose()
        new_pose = pose.copy()
        new_pose[2] = z_val
        self.move_tcp(new_pose)  # or moveP
        

    def move_to_initial_pose(self):
        pose = self.get_tcp_pose()
        pose[2] = 0.0404  # or the saved `initial_z`
        # self.moveL(pose)
        self.rtde_c.moveL(pose, 0.06, 0.2)


    def stop(self):
        self.rtde_c.servoStop()

    def close(self):
        self.rtde_c.disconnect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass



if __name__ == "__main__":

    robot = URRobot(ip="192.168.0.1")

    try:
        print("Current TCP Pose:", robot.get_tcp_pose())

        pose = [0.1946, -0.2212, 0.1640, 1.10417, 1.12385, -0.027471]
        robot.move_tcp(pose, velocity=0.06, acceleration=0.2)
        time.sleep(2.0)

    finally:
        robot.stop()
        robot.close()