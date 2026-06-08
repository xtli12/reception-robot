"""灵巧手 Modbus 控制（Reception 文件夹内自包含版本）。

从 CNN_inference_control.py 抽取灵巧手开合相关逻辑，去掉了与 CNN 模型相关的
依赖，使 Reception 目录可独立运行。
"""

from pymodbus.client.sync import ModbusSerialClient as ModbusClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder


HAND_PORT = "COM12"
HAND_BAUDRATE = 115200
HAND_SLAVE_ID = 2
HAND_START_ADDRESS = 1135
HAND_CLOSE_VALUES = [40000, 40000, 0, 0, 10000, 65535]
HAND_OPEN_VALUES = [0, 0, 0, 0, 10000, 65535]


def send_hand_values(values, action_name):
    client = ModbusClient(method="rtu", port=HAND_PORT, baudrate=HAND_BAUDRATE, timeout=1)
    if not client.connect():
        print(f"灵巧手连接失败: {HAND_PORT}")
        return False

    try:
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)
        for value in values:
            builder.add_16bit_uint(value)

        payload = builder.to_registers()
        result = client.write_registers(HAND_START_ADDRESS, payload, unit=HAND_SLAVE_ID)
        if result.isError():
            print(f"灵巧手{action_name}失败: {result}")
            return False

        print(f"灵巧手已发送{action_name}指令: {values}")
        return True
    finally:
        client.close()


def close_dexterous_hand():
    return send_hand_values(HAND_CLOSE_VALUES, "闭合")


def open_dexterous_hand():
    return send_hand_values(HAND_OPEN_VALUES, "张开")
