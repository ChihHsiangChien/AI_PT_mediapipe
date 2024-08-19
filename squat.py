import cv2
import mediapipe as mp
import numpy as np
import math
import winsound  # For Windows beep sound, use other alternatives for different platforms

# 初始化 MediaPipe Pose 模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 计算两点之间的角度
def calculate_angle(a, b, c):
    a = np.array(a)  # 第一个点
    b = np.array(b)  # 第二个点 (作为角的顶点)
    c = np.array(c)  # 第三个点

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# 打开视频流
cap = cv2.VideoCapture(0)
horizon_angle_threshold = 90  # Set the horizon angle threshold

while cap.isOpened():
    ret, frame = cap.read()
    
    # 水平翻转图像
    frame = cv2.flip(frame, 1)

    # 将图像转为RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # 检测关键点
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 获取髋部、膝盖和肩膀的坐标
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        # 计算髋部的角度
        hip_angle = calculate_angle(shoulder, hip, knee)

        # 在图像左上角显示角度（使用较大的字体）
        cv2.putText(frame, f'Hip Angle: {int(hip_angle)}',
                    (20, 50),  # 位置在左上角
                    cv2.FONT_HERSHEY_SIMPLEX, 2,  # 字体大小为2
                    (255, 255,255), 3, cv2.LINE_AA  # 绿色字体，线宽为3
                    )

        # 检测髋部是否低于膝盖
        if hip[1] > knee[1]:
            circle_color = (255, 255, 255)  # 白色
            #winsound.Beep(1000, 200)  # 1000 Hz for 200 ms (adjust as needed)

        else:
            circle_color = (0, 0, 255)  # 红色

        # 绘制大圆圈在右上角
        cv2.circle(frame, (frame.shape[1] - 50, 50), 40, circle_color, -1)

        # 绘制关键点和骨架
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 触发蜂鸣声，当髋部角度达到或超过设定的水平角度阈值时
        #if hip_angle <= horizon_angle_threshold:
        #    winsound.Beep(1000, 200)  # 1000 Hz for 200 ms (adjust as needed)

    cv2.imshow('Mediapipe Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
