"""连续帧示教数据采集（action chunk 训练数据，Reception 自包含版本）。

采集流程：在「拖动示教」模式下手动拖动机械臂走一遍技能动作，脚本按固定采样
周期同步采集 (相机图像 + 当前 6 个关节角)，保存成 action_chunk_model.py 训练
器要求的连续帧格式：

    data_chunk/<技能>/<示教序号>/000000_j1_j2_j3_j4_j5_j6.jpg
    data_chunk/<技能>/<示教序号>/000001_j1_j2_j3_j4_j5_j6.jpg
    ...

每条示教是一个按帧顺序排列的图片目录：
    - 文件名前缀（000000、000001 ...）用于排序，表示时间顺序；
    - 后面 6 个数字是该帧机械臂的 6 个关节角，单位度。

采集完成后用 `python action_chunk_model.py <技能>` 即可训练对应技能模型。
"""

import threading
import time
from pathlib import Path

import cv2

from robot_arm import RobotArm, SAMPLE_PERIOD


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_chunk"

JOINT_DIM = 6
CAMERA_INDEX = 1
CAMERA_WARMUP_FRAMES = 5  # 打开相机后丢掉几帧旧缓存

# 技能选项（目录名需与 reception_LLM.py 的 SKILL_CHECKPOINTS 输出文件名一致，
# 训练后 skill/<目录名>.pth 才能被主程序直接加载）。
SKILL_OPTIONS = {
    "1": ("抓卡", "grasp_card"),
    "2": ("还卡", "return_card"),
    "3": ("抓水", "grasp_water"),
    "4": ("抓笔", "grasp_pen"),
    "5": ("抓纸", "grasp_paper"),
}


def joints_to_filename(frame_index, joints):
    """组合成 `帧序号_j1_..._j6` 的文件名（不含扩展名）。"""
    joint_part = "_".join(f"{joint:.3f}" for joint in joints)
    return f"{frame_index:06d}_{joint_part}"


def next_demo_dir(skill_dir):
    """在技能目录下创建下一个递增编号的示教子目录。"""
    skill_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in skill_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    demo_index = (max(existing) + 1) if existing else 0
    demo_dir = skill_dir / f"{demo_index:03d}"
    demo_dir.mkdir(parents=True, exist_ok=True)
    return demo_dir


def open_camera(camera_index=CAMERA_INDEX):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {camera_index}")
    for _ in range(CAMERA_WARMUP_FRAMES):
        cap.read()
    return cap


def _wait_enter_to_stop(stop_event):
    input()
    stop_event.set()


class DemoCollector:
    def __init__(self, camera_index=CAMERA_INDEX):
        self.robot_arm = RobotArm()
        self.cap = open_camera(camera_index)

    def close(self):
        self.cap.release()
        self.robot_arm.close()
        cv2.destroyAllWindows()

    def record_demo(self, skill_name, sample_period=SAMPLE_PERIOD):
        """录制一条连续帧示教，返回保存的帧数。"""
        skill_dir = DATA_DIR / skill_name
        demo_dir = next_demo_dir(skill_dir)
        print(f"\n本条示教将保存到: {demo_dir}")

        input("按 Enter 进入拖动示教并开始录制，开始后请手动拖动机械臂演示动作...")

        ret = self.robot_arm.arm.rm_start_drag_teach(1)
        if ret != 0:
            raise RuntimeError(f"开始拖动示教失败，状态码: {ret}")

        stop_event = threading.Event()
        stopper = threading.Thread(target=_wait_enter_to_stop, args=(stop_event,), daemon=True)
        stopper.start()
        print("正在录制连续帧（图像 + 关节角），再按 Enter 停止录制。")

        frame_index = 0
        loop_start = time.monotonic()
        try:
            while not stop_event.is_set():
                self._sleep_until(loop_start + frame_index * sample_period)

                ok, frame = self.cap.read()
                if not ok:
                    print("读取摄像头数据失败，跳过该帧。")
                    continue

                try:
                    joints = self.robot_arm.get_joints()
                except RuntimeError as exc:
                    print(exc)
                    continue
                if len(joints) < JOINT_DIM:
                    print(f"关节角维度不足，跳过该帧: {joints}")
                    continue

                file_name = joints_to_filename(frame_index, joints[:JOINT_DIM]) + ".jpg"
                if cv2.imwrite(str(demo_dir / file_name), frame):
                    frame_index += 1
                else:
                    print(f"图像保存失败: {file_name}")

                cv2.imshow("Demo Collect (press Enter in console to stop)", frame)
                cv2.waitKey(1)
        finally:
            stop_ret = self.robot_arm.arm.rm_stop_drag_teach()
            if stop_ret != 0:
                print(f"停止拖动示教失败，状态码: {stop_ret}")

        if frame_index < 2:
            print("采集帧数不足（<2 帧），该条示教无效。")
        else:
            print(f"本条示教录制完成，共 {frame_index} 帧，保存在 {demo_dir}")
        return frame_index


def choose_skill():
    print("\n请选择要采集的技能:")
    for key, (cn_name, dir_name) in SKILL_OPTIONS.items():
        print(f"{key}. {cn_name} ({dir_name})")
    print("或直接输入自定义技能目录名。")
    choice = input("输入选项或目录名: ").strip()

    if choice in SKILL_OPTIONS:
        return SKILL_OPTIONS[choice][1]
    return choice or None


def main():
    collector = DemoCollector()
    try:
        while True:
            skill_name = choose_skill()
            if not skill_name:
                print("未选择技能。")
                continue

            collector.record_demo(skill_name)

            again = input("\n继续采集下一条示教？(y 继续 / 其它键退出): ").strip().lower()
            if again != "y":
                break
    except KeyboardInterrupt:
        print("\n已中断采集。")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
