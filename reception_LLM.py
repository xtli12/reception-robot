import json
import os
import queue
import sys
import threading
from pathlib import Path

import cv2
import sounddevice as sd
import torch
import vosk
import win32com.client
from openai import OpenAI

import LLM_API
from action_chunk_model import (
    ActionChunkCNN,
    CHUNK_SIZE,
    JOINT_DIM,
    predict_chunk,
)
from hand_control import close_dexterous_hand, open_dexterous_hand
from robot_arm import RobotArm, SAMPLE_PERIOD


SCRIPT_DIR = Path(__file__).resolve().parent
VOSK_MODEL_NAME = "vosk-model-small-cn-0.22"


def resolve_vosk_model_path():
    env_path = os.getenv("VOSK_MODEL_PATH")
    if env_path:
        return Path(env_path)

    direct_path = SCRIPT_DIR / VOSK_MODEL_NAME
    if direct_path.exists():
        return direct_path

    search_roots = [
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        SCRIPT_DIR.parents[1],
        Path.cwd(),
    ]
    checked_roots = set()
    for root in search_roots:
        root = root.resolve()
        if root in checked_roots or not root.exists():
            continue
        checked_roots.add(root)
        for path in root.rglob(VOSK_MODEL_NAME):
            if path.is_dir():
                return path

    return direct_path


VOSK_MODEL_PATH = resolve_vosk_model_path()

ROBOT_HOME_JOINTS = [
    -88.2760009765625,
    3.490000009536743,
    90.88700103759766,
    -6.796000003814697,
    -6.206999778747559,
    156.9600067138672,
]
HOME_SPEED = 20

CHECKPOINT_DIR = SCRIPT_DIR / "skill"
SKILL_CHECKPOINTS = {
    "抓卡": CHECKPOINT_DIR / "grasp_card.pth",
    "还卡": CHECKPOINT_DIR / "return_card.pth",
    "抓水1": CHECKPOINT_DIR / "grasp_water.pth",
    "抓笔": CHECKPOINT_DIR / "grasp_pen.pth",
    "抓纸": CHECKPOINT_DIR / "grasp_paper.pth",
}

# 连续帧 action chunk 闭环执行参数。
CAMERA_INDEX = 1
CHUNK_MOVE_SPEED = 20  # 运动到首段 chunk 起点的速度
MAX_CHUNKS = 60  # 单个技能闭环最多生成多少段 chunk（上限保护）
CONVERGENCE_THRESHOLD = 0.5  # 一段 chunk 内关节角变化小于该值（度）即认为动作收敛/结束
# 抓取时刻不再用「MAX_CHUNKS 的固定比例段序号」，改为基于机械臂实际运动收敛信号：
# 机械臂先运动起来（某段 chunk 帧间最大变化超过 ACTIVE 阈值），随后某段 chunk
# 运动速度回落到 SETTLE 阈值以下，即认为到达抓取/接触点，闭合灵巧手。这样无论
# 实际动作用多少段都能贴合真实进度。
HAND_CLOSE_ACTIVE_SPEED = 1.0  # 度，判定机械臂已开始运动的最小帧间变化
HAND_CLOSE_SETTLE_SPEED = 0.4  # 度，运动起来后帧间变化回落到该值以下视为到达抓取点
HAND_CLOSE_MIN_CHUNKS = 1  # 至少执行多少段后才允许触发闭手（避免起步误触发）

STAGE_WAIT_APPOINTMENT = "wait_appointment"
STAGE_WAIT_ID_CARD = "wait_id_card"
STAGE_WAIT_PEN_REQUEST = "wait_pen_request"
STAGE_WAIT_THANKS_AFTER_PEN = "wait_thanks_after_pen"
STAGE_WAIT_THANKS_AFTER_WATER = "wait_thanks_after_water"
STAGE_DONE = "done"
EARLY_RECOGNITION_KEYWORDS = [
    "你好",
    "预约",
    "约了",
    "教授",
    "带了",
    "给",
    "身份证",
    "笔",
    "比",
    "一支",
    "有没有",
    "谢谢",
    "感谢",
    "填完",
    "写完",
    "录完",
    "还给我",
    "还卡",
    "身份证还",
]


def contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def is_return_card_request(text):
    return contains_any(
        text,
        [
            "谢谢",
            "感谢",
            "填完",
            "写完",
            "录完",
            "录入完",
            "办完",
            "还给我",
            "还卡",
            "卡还",
            "身份证还",
            "把卡",
            "我的身份证",
        ],
    )


class VoskSpeechRecognizer:
    def __init__(self, model_path=VOSK_MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"没有找到Vosk中文语音模型: {model_path}\n"
                "请下载中文模型并放到该路径，或设置环境变量 VOSK_MODEL_PATH。"
            )
        self.model = vosk.Model(str(model_path))
        self.audio_queue = queue.Queue()

    def listen_once(self):
        recognizer = vosk.KaldiRecognizer(self.model, 16000)

        def callback(indata, _frames, _time_info, status):
            if status:
                print("录音状态:", status, file=sys.stderr)
            self.audio_queue.put(bytes(indata))

        print("\n请说话...")
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                data = self.audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = self.correct_text(result.get("text", ""))
                    if text:
                        print("识别结果:", text)
                        return text

                partial = self.correct_text(json.loads(recognizer.PartialResult()).get("partial", ""))
                if len(partial) >= 2 and contains_any(partial, EARLY_RECOGNITION_KEYWORDS):
                    print("识别结果:", partial)
                    return partial

    @staticmethod
    def correct_text(text):
        text = text.replace(" ", "")
        corrections = {
            "叫兽": "教授",
            "身份证吗": "身份证",
            "神分证": "身份证",
            "带啦": "带了",
            "代了": "带了",
            "笔吗": "笔",
            "带比": "带笔",
            "没带比": "没带笔",
            "有没有比": "有没有笔",
            "给我一支比": "给我一支笔",
            "喝水平": "喝瓶水",
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        return text


class StreamingSpeaker:
    def __init__(self):
        self._lock = threading.Lock()

    def speak_text(self, text):
        text = text.strip()
        if not text:
            return

        with self._lock:
            pythoncom_module = None
            try:
                import pythoncom

                pythoncom_module = pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pythoncom_module = None

            try:
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak(text)
            finally:
                if pythoncom_module is not None:
                    pythoncom_module.CoUninitialize()

    def speak_stream(self, text_stream, fallback_text):
        full_text = ""
        sentence_buffer = ""

        for chunk in text_stream:
            if not chunk:
                continue
            print(chunk, end="", flush=True)
            full_text += chunk
            sentence_buffer += chunk

            if any(mark in sentence_buffer for mark in "，。！？；\n"):
                self.speak_text(sentence_buffer)
                sentence_buffer = ""

        if sentence_buffer.strip():
            self.speak_text(sentence_buffer)

        print()
        return full_text.strip() or fallback_text


class LLMResponder:
    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY") or getattr(LLM_API.dashscope, "api_key", None)
        if not api_key:
            self.client = None
            print("未配置DASHSCOPE_API_KEY，将使用本地固定回复。")
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    def decide(self, user_text, current_stage, history):
        if self.client is None:
            return {
                "steps": [{"say": "当前没有连接大模型，无法完成智能决策。", "skill": "none"}],
                "next_stage": current_stage,
            }

        messages = [
            {
                "role": "system",
                "content": self._decision_system_prompt(),
            },
            {
                "role": "user",
                "content": (
                    f"当前阶段: {current_stage}\n"
                    f"最近对话历史: {json.dumps(history[-6:], ensure_ascii=False)}\n"
                    f"用户最新语音识别文本: {user_text}\n\n"
                    "请只输出一个JSON对象，不要输出Markdown。"
                ),
            },
        ]

        try:
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                temperature=0.1,
            )
            content = completion.choices[0].message.content
            return self._parse_decision(content, current_stage)
        except Exception as exc:
            print(f"大模型决策失败: {exc}")
            return {
                "steps": [{"say": "我刚才没有听清，请您再说一遍。", "skill": "none"}],
                "next_stage": current_stage,
            }

    def stream_reply(self, user_text, fallback_text, scene_instruction):
        if self.client is None:
            yield fallback_text
            return

        messages = [
            {
                "role": "system",
                "content": (
                    "你是前台接待机器人。请用第一人称、自然、礼貌、简短的中文回复。"
                    "不要输出动作名称、代码、JSON或括号说明。"
                    "回复必须适合直接语音播报，尽量不超过30个汉字。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户说：{user_text}\n"
                    f"当前接待流程要求：{scene_instruction}\n"
                    f"推荐回复：{fallback_text}\n"
                    "请基于推荐回复做一句自然口语化表达。"
                ),
            },
        ]

        try:
            stream = self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                stream=True,
            )
            has_output = False
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    has_output = True
                    yield delta
            if not has_output:
                yield fallback_text
        except Exception as exc:
            print(f"大模型调用失败，使用固定回复: {exc}")
            yield fallback_text

    @staticmethod
    def _decision_system_prompt():
        return """
你是前台接待机器人，需要根据访客语音、当前阶段、历史对话决定机器人要说什么以及调用什么技能。

可用技能只能从下面选择：
- none：不调用动作
- 抓卡：用CNN识别当前相机图片抓取访客递来的身份证/卡片，然后接录制轨迹后半段
- 抓纸：用CNN抓取表格/纸张，然后接录制轨迹后半段递给访客
- 抓笔：用CNN抓取笔，然后接录制轨迹后半段递给访客
- 抓水1：用CNN抓取水瓶，然后接录制轨迹后半段递给访客
- 还卡：用CNN抓取已录入完成的卡片，然后接录制轨迹后半段还给访客

阶段只能从下面选择：
- wait_appointment：等待访客说明来意
- wait_id_card：等待访客递身份证/卡片
- wait_pen_request：访客填表，等待是否需要笔
- wait_thanks_after_pen：已经递笔，等待访客感谢或继续填表
- wait_thanks_after_water：已经递水，等待录入完成/访客索要身份证/卡片
- done：流程完成

典型接待流程：
1. 访客说预约教授/老师/来访，回复“好的，您带身份证了吗，需要录入一下信息。”，不调用技能，进入wait_id_card。
2. 访客表示带了、给你、递身份证，先说“好的。”并调用抓卡；再说“录入需要等一会，您先填个表。”并调用抓纸；进入wait_pen_request。
3. 访客说忘记带笔、有笔吗、给我一支笔，回复“有的，我现在帮您拿一只。”并调用抓笔；进入wait_thanks_after_pen。
4. 访客感谢递笔，先回复“不用谢。”，再说“先生您稍等，您先喝瓶水。”并调用抓水1；进入wait_thanks_after_water。
5. 访客说填完了、写完了、录完了吗、把卡还给我、把身份证还给我、谢谢等，回复“信息录入完毕，这是您的卡片。”并调用还卡；进入done。
6. 如果信息不足，礼貌追问，不调用技能，阶段保持不变。

输出JSON格式必须严格如下：
{
  "steps": [
    {"say": "要播报的话", "skill": "none或技能名"}
  ],
  "next_stage": "下一阶段"
}
不要输出其它文字。
""".strip()

    @staticmethod
    def _parse_decision(content, current_stage):
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                content = content.replace("json\n", "", 1).replace("JSON\n", "", 1)
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start : end + 1]
            decision = json.loads(content)
        except json.JSONDecodeError:
            print(f"大模型决策JSON解析失败: {content}")
            return {
                "steps": [{"say": "我刚才没有听清，请您再说一遍。", "skill": "none"}],
                "next_stage": current_stage,
            }

        valid_skills = {"none", *SKILL_CHECKPOINTS.keys()}
        valid_stages = {
            STAGE_WAIT_APPOINTMENT,
            STAGE_WAIT_ID_CARD,
            STAGE_WAIT_PEN_REQUEST,
            STAGE_WAIT_THANKS_AFTER_PEN,
            STAGE_WAIT_THANKS_AFTER_WATER,
            STAGE_DONE,
        }

        steps = []
        for step in decision.get("steps", []):
            say_text = str(step.get("say", "")).strip()
            skill = str(step.get("skill", "none")).strip()
            if skill not in valid_skills:
                skill = "none"
            if say_text or skill != "none":
                steps.append({"say": say_text, "skill": skill})

        if not steps:
            steps = [{"say": "我刚才没有听清，请您再说一遍。", "skill": "none"}]

        next_stage = str(decision.get("next_stage", current_stage)).strip()
        if next_stage not in valid_stages:
            next_stage = current_stage

        return {"steps": steps, "next_stage": next_stage}


class ActionChunkPolicy:
    """连续帧 action chunk 动作生成与闭环执行。

    动作生成模式：每次根据当前相机图像生成一段连续帧 chunk (CHUNK_SIZE 帧)，
    机械臂按采样周期透传执行整段 chunk，再用新图像生成下一段，循环往复，
    直到动作收敛（机械臂停止运动）或达到最大段数。整段动作完全由模型生成，
    不再回放预录制轨迹。
    """

    def __init__(self, camera_index=CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def execute(self, skill_key, robot_arm):
        checkpoint_path = self._resolve_checkpoint(skill_key)
        model = self._load_model(checkpoint_path)
        print(f"{skill_key} action chunk 模型: {checkpoint_path}")

        print(f"{skill_key} 开始连续帧 action chunk 闭环执行...")
        open_dexterous_hand()  # 抓取前先张开灵巧手
        hand_closed = False
        arm_active = False  # 机械臂是否已经真正运动起来（用于过滤起步阶段）

        for chunk_index in range(MAX_CHUNKS):
            frame = self._capture_frame()
            chunk = predict_chunk(model, self.device, frame)

            if chunk_index == 0:
                # 首段先平滑运动到 chunk 起点，避免第一帧透传跳变过大。
                robot_arm.move_joint(chunk[0], speed=CHUNK_MOVE_SPEED)

            robot_arm.stream_chunk(chunk, period=SAMPLE_PERIOD)

            # 基于实际运动收敛信号触发闭手：先动起来，再回落到抓取点。
            chunk_speed = self._chunk_speed(chunk)
            if chunk_speed >= HAND_CLOSE_ACTIVE_SPEED:
                arm_active = True

            reached_grasp = (
                not hand_closed
                and arm_active
                and (chunk_index + 1) >= HAND_CLOSE_MIN_CHUNKS
                and chunk_speed <= HAND_CLOSE_SETTLE_SPEED
            )
            if reached_grasp:
                print(f"{skill_key} 运动收敛到抓取点（第 {chunk_index + 1} 段），闭合灵巧手...")
                close_dexterous_hand()
                hand_closed = True

            if self._chunk_converged(chunk):
                print(f"{skill_key} 动作已收敛，结束连续帧生成（共 {chunk_index + 1} 段）。")
                break
        else:
            print(f"{skill_key} 达到最大 chunk 段数 {MAX_CHUNKS}，结束执行。")

        if not hand_closed:
            print(f"{skill_key} 动作结束前补充闭合灵巧手...")
            close_dexterous_hand()

    @staticmethod
    def _chunk_speed(chunk):
        """一段 chunk 的运动速度，用帧间最大关节角变化（度）表示。"""
        if len(chunk) < 2:
            return 0.0
        max_delta = 0.0
        for prev_frame, curr_frame in zip(chunk, chunk[1:]):
            for prev, curr in zip(prev_frame, curr_frame):
                max_delta = max(max_delta, abs(curr - prev))
        return max_delta

    @staticmethod
    def _chunk_converged(chunk):
        if len(chunk) < 2:
            return True
        first = chunk[0]
        max_delta = 0.0
        for frame in chunk[1:]:
            for current, start in zip(frame, first):
                max_delta = max(max_delta, abs(current - start))
        return max_delta < CONVERGENCE_THRESHOLD

    def _resolve_checkpoint(self, skill_key):
        configured_path = SKILL_CHECKPOINTS[skill_key]
        if configured_path.exists():
            return configured_path

        keyword_map = {
            "抓卡": ["card", "ka", "抓卡"],
            "还卡": ["return", "huanka", "还卡"],
            "抓水1": ["water", "shui", "抓水"],
            "抓笔": ["pen", "bi", "抓笔"],
            "抓纸": ["paper", "zhi", "抓纸"],
        }
        candidates = []
        for checkpoint in CHECKPOINT_DIR.glob("*.pth"):
            lower_name = checkpoint.name.lower()
            if any(keyword.lower() in lower_name for keyword in keyword_map.get(skill_key, [])):
                candidates.append(checkpoint)

        if candidates:
            return sorted(candidates)[0]

        raise FileNotFoundError(
            f"没有找到 {skill_key} 的 action chunk 模型。\n"
            f"请放到配置路径: {configured_path}\n"
            f"或放到 {CHECKPOINT_DIR}，文件名包含技能关键词。"
        )

    def _load_model(self, checkpoint_path):
        cache_key = str(checkpoint_path)
        if cache_key in self.models:
            return self.models[cache_key]

        model = ActionChunkCNN(chunk_size=CHUNK_SIZE, joint_dim=JOINT_DIM).to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        self.models[cache_key] = model
        return model

    def _capture_frame(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头: {self.camera_index}")

        # 丢掉几帧旧缓存，尽量使用实时画面。
        frame = None
        for _ in range(3):
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("无法读取摄像头数据")

        cv2.imshow("Action Chunk Camera", frame)
        cv2.waitKey(1)
        return frame


class ReceptionRobot:
    def __init__(self):
        self.recognizer = VoskSpeechRecognizer()
        self.speaker = StreamingSpeaker()
        self.llm = LLMResponder()
        self.robot_arm = RobotArm()
        self.policy = ActionChunkPolicy()
        self.stage = STAGE_WAIT_APPOINTMENT
        self.history = []

    def close(self):
        self.policy.close()
        self.robot_arm.close()

    def move_home(self):
        print("回到前台初始位置...")
        self.robot_arm.move_joint(ROBOT_HOME_JOINTS, speed=HOME_SPEED)

    def say(self, user_text, fallback_text, scene_instruction):
        print("机器人: ", end="", flush=True)
        stream = self.llm.stream_reply(user_text, fallback_text, scene_instruction)
        return self.speaker.speak_stream(stream, fallback_text)

    def say_async(self, user_text, fallback_text, scene_instruction):
        thread = threading.Thread(
            target=self.say,
            args=(user_text, fallback_text, scene_instruction),
            daemon=True,
        )
        thread.start()
        return thread

    def run_skill(self, skill_key):
        print(f"执行动作技能（连续帧 action chunk）: {skill_key}")
        self.policy.execute(skill_key, self.robot_arm)
        self.move_home()

    def say_then_skill(self, user_text, fallback_text, scene_instruction, skill_key):
        self.say_while_skill(user_text, fallback_text, scene_instruction, skill_key)

    def say_while_skill(self, user_text, fallback_text, scene_instruction, skill_key):
        speak_thread = self.say_async(user_text, fallback_text, scene_instruction)
        self.run_skill(skill_key)
        speak_thread.join()

    def handle_user_text(self, text):
        decision = self.llm.decide(text, self.stage, self.history)
        print("LLM决策:", json.dumps(decision, ensure_ascii=False))

        for step in decision["steps"]:
            say_text = step["say"]
            skill = step["skill"]

            if skill == "none":
                if say_text:
                    self.say(text, say_text, "请将这句话自然口语化后播报。")
                continue

            if say_text:
                self.say_while_skill(text, say_text, "请将这句话自然口语化后播报。", skill)
            else:
                self.run_skill(skill)

        self.history.append(
            {
                "user": text,
                "decision": decision,
                "stage_before": self.stage,
                "stage_after": decision["next_stage"],
            }
        )
        self.stage = decision["next_stage"]

    def run(self):
        self.move_home()
        print("前台接待机器人已启动。按 Ctrl+C 退出。")
        while True:
            text = self.recognizer.listen_once()
            self.handle_user_text(text)


def main():
    robot = ReceptionRobot()
    try:
        robot.run()
    except KeyboardInterrupt:
        print("\n已退出前台接待流程。")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
