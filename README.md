# Reception Robot 前台接待机器人

基于「语音识别 + 大模型决策 + 视觉动作生成」的前台接待机器人。机器人通过麦克风识别访客语音，调用通义千问（Qwen）大模型进行对话与流程决策，再根据决策调用对应技能，由 action chunk 视觉模型实时生成机械臂动作完成抓取/递送等接待任务。

入口脚本：

```powershell
python reception_LLM.py
```

---


## 系统架构

整个系统由四个模块协同工作，形成「听 → 想 → 说 + 做」的闭环：

```
访客语音
   │  ① 语音识别 (Vosk 离线中文模型)
   ▼
识别文本 ──② 决策 (Qwen 大模型)──> { steps: [{say, skill}], next_stage }
   │                                      │
   │ ③ 语音播报                            │ ④ 动作执行
   ▼ (SAPI 流式 TTS)                       ▼ (action chunk 视觉闭环)
  扬声器                              机械臂 + 灵巧手
```

| 模块 | 实现 | 说明 |
| --- | --- | --- |
| 语音识别 | `VoskSpeechRecognizer` | Vosk 离线中文模型，支持关键词提前触发与常见错别字纠正 |
| 对话决策 | `LLMResponder` | 调用 Qwen，根据阶段/历史/语音输出结构化 JSON 决策 |
| 语音播报 | `StreamingSpeaker` | Windows SAPI 流式分句 TTS，边生成边播报 |
| 动作执行 | `ActionChunkPolicy` | 视觉 action chunk 模型 + 机械臂 CANFD 透传 + 灵巧手开合 |

主控类 `ReceptionRobot` 串联上述模块，并维护当前接待阶段（`stage`）和对话历史（`history`）。

---

## 典型接待流程

机器人按阶段（stage）推进一次完整接待：

1. **wait_appointment**：访客说明来意（如“预约了教授”）→ 机器人询问是否带身份证。
2. **wait_id_card**：访客递身份证 → 调用 `抓卡` 接卡，再调用 `抓纸` 递表格。
3. **wait_pen_request**：访客填表，若需要笔 → 调用 `抓笔` 递笔。
4. **wait_thanks_after_pen**：访客致谢后 → 调用 `抓水1` 递水。
5. **wait_thanks_after_water**：信息录入完成、访客索要证件 → 调用 `还卡` 还卡。
6. **done**：流程完成。

> 阶段流转由大模型根据上下文判断，机器人具备容错追问能力（信息不足时不调用技能、阶段保持不变）。

---

## 技能与动作执行

可用技能（均对应 `skill/` 下一个 action chunk 模型）：

| 技能 | 模型文件 | 含义 |
| --- | --- | --- |
| `抓卡` | `skill/grasp_card.pth` | 抓取访客递来的身份证/卡片 |
| `还卡` | `skill/return_card.pth` | 把录入完成的卡片还给访客 |
| `抓纸` | `skill/grasp_paper.pth` | 抓取并递出表格/纸张 |
| `抓笔` | `skill/grasp_pen.pth` | 抓取并递出笔 |
| `抓水1` | `skill/grasp_water.pth` | 抓取并递出水瓶 |

### 连续帧 action chunk 闭环执行

每个技能的动作完全由视觉模型实时生成，不回放预录制轨迹。执行流程：

1. 对当前相机图像推理一次，生成未来 `CHUNK_SIZE` 帧（默认 16 帧）的连续关节角动作 chunk。
2. 首段先平滑运动到 chunk 起点（避免首帧跳变），随后机械臂按采样周期（约 50Hz）通过 CANFD 连续透传整段 chunk。
3. 再采集新图像生成下一段 chunk，循环往复（**看图 → 生成 chunk → 执行 → 再看图**）。
4. 基于实际运动收敛信号闭合灵巧手：机械臂先运动起来（帧间变化超过 `HAND_CLOSE_ACTIVE_SPEED`），随后某段 chunk 速度回落到 `HAND_CLOSE_SETTLE_SPEED` 以下（到达抓取/接触点）时闭手。无论实际动作用多少段都能贴合真实进度。
5. 当某段 chunk 内关节角变化小于 `CONVERGENCE_THRESHOLD`（动作收敛）或达到 `MAX_CHUNKS` 时结束，并回到前台初始位置。

---

## 大模型决策

`reception_LLM.py` 通过 DashScope 的 OpenAI 兼容接口调用 Qwen：

```python
model="qwen-plus"
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

大模型承担两个任务：

- **决策**（`decide`）：依据当前阶段、最近对话历史与用户语音文本，输出结构化的下一步动作。
- **回复润色**（`stream_reply`）：将推荐回复改写为自然口语，并流式输出供语音播报。

决策必须返回严格的 JSON：

```json
{
  "steps": [
    {"say": "要播报的话", "skill": "none或技能名"}
  ],
  "next_stage": "下一阶段"
}
```

约束（非法值会被回退处理）：

- 允许的 `skill`：`none`、`抓卡`、`抓纸`、`抓笔`、`抓水1`、`还卡`
- 允许的 `next_stage`：`wait_appointment`、`wait_id_card`、`wait_pen_request`、`wait_thanks_after_pen`、`wait_thanks_after_water`、`done`

> 若未配置 API Key，程序仍能启动，但会使用本地固定回复，无法进行智能决策。

---

## 项目文件结构

```
Reception/
├── reception_LLM.py        # 主入口：语音识别 + 大模型决策 + 播报 + 技能调度
├── action_chunk_model.py   # action chunk 模型结构、数据集与训练脚本
├── robot_arm.py            # 机械臂连接与连续帧 chunk 的 CANFD 透传
├── hand_control.py         # 灵巧手 Modbus 开合控制
├── LLM_API.py              # DashScope/Qwen 接口示例（含 api_key 兜底来源）
├── skill/                  # 各技能训练好的 .pth 模型
│   ├── grasp_card.pth
│   ├── return_card.pth
│   ├── grasp_paper.pth
│   ├── grasp_pen.pth
│   └── grasp_water.pth
└── vosk-model-small-cn-0.22/  # Vosk 离线中文语音模型
```

> `robot_arm.py` 会自动把上层的 `GMM/` 目录加入 `sys.path`，以加载睿尔曼机械臂 SDK（`Robotic_Arm.rm_robot_interface`）。

---

## 关键参数

| 参数 | 位置 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `CHUNK_SIZE` | `action_chunk_model.py` | `16` | 每段生成的连续帧数 K |
| `JOINT_DIM` | `action_chunk_model.py` | `6` | 机械臂关节维度 |
| `MAX_CHUNKS` | `reception_LLM.py` | `60` | 单技能闭环最多生成的 chunk 段数（上限保护） |
| `CONVERGENCE_THRESHOLD` | `reception_LLM.py` | `0.5` | 段内关节角变化（度）小于该值判定动作结束 |
| `HAND_CLOSE_ACTIVE_SPEED` | `reception_LLM.py` | `1.0` | 判定机械臂已开始运动的最小帧间变化（度） |
| `HAND_CLOSE_SETTLE_SPEED` | `reception_LLM.py` | `0.4` | 运动后帧间变化回落到该值以下视为到达抓取点（度） |
| `HAND_CLOSE_MIN_CHUNKS` | `reception_LLM.py` | `1` | 至少执行多少段后才允许触发闭手（避免起步误触发） |
| `CAMERA_INDEX` | `reception_LLM.py` | `1` | 相机设备索引 |
| `CHUNK_MOVE_SPEED` | `reception_LLM.py` | `20` | 运动到首段 chunk 起点的速度 |
| `ROBOT_HOME_JOINTS` | `reception_LLM.py` | — | 前台初始位置的 6 个关节角 |
| `ROBOT_IP` / `ROBOT_PORT` | `robot_arm.py` | `192.168.1.18` / `8080` | 机械臂网络地址 |
| `SAMPLE_PERIOD` | `robot_arm.py` | `0.02` | 透传采样周期（约 50Hz） |
| `HAND_PORT` | `hand_control.py` | `COM12` | 灵巧手 Modbus 串口 |
| `HAND_BAUDRATE` | `hand_control.py` | `115200` | 灵巧手串口波特率 |

---

## 环境与依赖

- **操作系统**：Windows（依赖 Windows SAPI 语音合成与 `pywin32`）
- **Python**：3.8+
- **硬件**：睿尔曼机械臂、灵巧手（Modbus RTU 串口）、USB 相机、麦克风、扬声器

安装主要 Python 依赖：

```powershell
pip install openai dashscope torch opencv-python sounddevice vosk pywin32 pymodbus
```

> 此外需保证睿尔曼机械臂 Python SDK 及其 C API 动态库可正常加载（位于上层 `GMM/` 目录）。

---

## 运行前配置

### 1. 配置 DashScope API Key

推荐使用环境变量（优先级最高）：

```powershell
$env:DASHSCOPE_API_KEY="你的DashScope API Key"
```

若未设置环境变量，程序会回退使用 `LLM_API.py` 中的 `dashscope.api_key`。

### 2. 准备 Vosk 中文语音模型

下载 `vosk-model-small-cn-0.22` 并放到本目录，或通过环境变量指定路径：

```powershell
$env:VOSK_MODEL_PATH="模型所在目录"
```

> 程序会自动在当前目录及上层目录递归查找该模型目录。

### 3. 准备技能模型

将训练好的 `.pth` 模型放入 `skill/`，文件名按 [技能与动作执行](#技能与动作执行) 表格对应；也可放入 `skill/` 且文件名包含技能关键词（如 `card`、`pen`、`water`、`paper`、`return`），程序会按关键词自动匹配。

### 4. 配置硬件参数

按实际硬件修改：

- 机械臂 IP/端口：`robot_arm.py` 的 `ROBOT_IP` / `ROBOT_PORT`
- 灵巧手串口：`hand_control.py` 的 `HAND_PORT`
- 相机索引：`reception_LLM.py` 的 `CAMERA_INDEX`
- 前台初始位姿：`reception_LLM.py` 的 `ROBOT_HOME_JOINTS`

---

## 运行

在 `Reception` 目录下执行：

```powershell
python reception_LLM.py
```

启动后程序会：

1. 加载 Vosk 语音模型。
2. 连接机械臂并移动到前台初始位置。
3. 持续监听访客语音。
4. 调用 Qwen 返回结构化决策。
5. 根据决策流式播报回复，并执行对应技能。

按 `Ctrl+C` 退出。

---
