"""机械臂连接与连续帧 chunk 透传（Reception 文件夹内自包含版本）。

只保留 action chunk 闭环执行所需的能力：
    - 连接机械臂
    - 读取当前关节角
    - rm_movej 平滑运动到某一关节角（用于运动到 chunk 起点 / 回初始位）
    - 按采样周期把一段连续帧 chunk 通过 CANFD 透传执行
"""

import sys
import time
from pathlib import Path

GMM_DIR = Path(__file__).resolve().parents[2] / "GMM"
if str(GMM_DIR) not in sys.path:
    sys.path.append(str(GMM_DIR))

from Robotic_Arm.rm_robot_interface import (  # pyright: ignore[reportMissingImports]
    RoboticArm,
    rm_thread_mode_e,
)


ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080

SAMPLE_PERIOD = 0.02  # 透传采样周期，约 50Hz
MOVE_SPEED = 20
CANFD_FOLLOW = False  # 低跟随对 Python 定时更稳


class RobotArm:
    def __init__(self, robot_ip=ROBOT_IP, robot_port=ROBOT_PORT):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(robot_ip, robot_port)
        print("机械臂连接 ID:", self.handle.id)

    def close(self):
        self.arm.rm_delete_robot_arm()

    def get_joints(self):
        ret_code, joints = self.arm.rm_get_joint_degree()
        if ret_code != 0:
            raise RuntimeError(f"读取关节角失败，状态码: {ret_code}")
        return [float(joint) for joint in joints]

    def move_joint(self, joints, speed=MOVE_SPEED):
        ret = self.arm.rm_movej(list(joints), speed, 0, 0, 1)
        if ret != 0:
            raise RuntimeError(f"运动到目标关节角失败，状态码: {ret}")

    def stream_chunk(self, frames, period=SAMPLE_PERIOD, follow=CANFD_FOLLOW):
        """按时间顺序连续透传一段 chunk（每帧 6 维关节角）。"""
        if not frames:
            return 0

        failed_count = 0
        stream_start = time.monotonic()
        for index, frame in enumerate(frames):
            self._sleep_until(stream_start + index * period)
            ret = self.arm.rm_movej_canfd(list(frame), follow)
            if ret != 0:
                failed_count += 1
                if failed_count <= 5:
                    print(f"chunk 透传第 {index} 帧失败，状态码: {ret}")
        return failed_count

    @staticmethod
    def _sleep_until(target_time):
        while True:
            remaining = target_time - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.002))
