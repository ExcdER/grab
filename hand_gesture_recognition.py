# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import time
import os
import pyautogui  # 用于全屏截图
from PIL import ImageGrab, Image  # 用于处理截图和剪贴板

# 初始化 MediaPipe 手部模块
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# 关键点索引
FINGER_TIPS = [8, 12, 16, 20]  # 四根手指的指尖
FINGER_PIP = [6, 10, 14, 18]   # 指尖下面一节关节

# 记录状态
previous_was_palm = False  
palm_persist_frames = 0  # 记录手掌出现的帧数
screenshot_counter = 0  # 截图计数，防止重复截图

def is_fist(landmarks):
    """ 判断是否为拳头（不分左右手、不分方向） """
    for tip, pip in zip(FINGER_TIPS, FINGER_PIP):
        if landmarks[tip].y < landmarks[pip].y:  # 指尖高于PIP关节，说明手指伸直
            return False
    return True

def is_open_palm(landmarks):
    """ 判断是否为张开的手掌（不分方向） """
    for tip, pip in zip(FINGER_TIPS, FINGER_PIP):
        if landmarks[tip].y > landmarks[pip].y:  # 指尖低于PIP关节，说明手指弯曲
            return False
    return True

def take_full_screenshot():
    """ 截取全屏截图并复制到剪贴板 """
    global screenshot_counter
    time.sleep(3)  # 休息3秒，避免重复触发

    timestamp = time.strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    save_dir = "screenshots"  # 目标文件夹
    os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在，则创建

    filename = os.path.join(save_dir, f"screenshot_{timestamp}_{screenshot_counter}.png")
    
    # 截取全屏
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"Screenshot saved: {filename}")

    # 使用 Pillow 将图片读取并复制到剪贴板
    img = Image.open(filename)
    img.show()  # 显示截图
    img.save("temp_screenshot.png")  # 保存为临时文件，防止无法复制
    img_clipboard = Image.open("temp_screenshot.png")

    # 将图片复制到剪贴板
    img_clipboard.save("temp_screenshot.png", format='PNG')
    ImageGrab.grabclipboard()  # 处理剪贴板操作
    print("Screenshot copied to clipboard!")

    screenshot_counter += 1  # 计数增加

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 翻转图像（镜像效果），转换 BGR 到 RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 进行手部检测
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制手部关键点
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 获取关键点坐标
            landmarks = hand_landmarks.landmark
            
            # **检测手掌**
            if is_open_palm(landmarks):
                palm_persist_frames += 1
                if palm_persist_frames >= 3:  # 需要至少 3 帧确认手掌
                    previous_was_palm = True  
                    cv2.putText(frame, "Open Palm", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                palm_persist_frames = 0  # 不是手掌就重置

            # **检测拳头**
            if is_fist(landmarks) and previous_was_palm:
                cv2.putText(frame, "Fist Detected! Taking Screenshot!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                take_full_screenshot()  # **全屏截图 + 复制到剪贴板**
                
                previous_was_palm = False  # 触发后重置，防止连续触发

    # 显示视频
    cv2.imshow("Hand Gesture Recognition", frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()