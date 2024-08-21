import cv2
import mediapipe as mp
import numpy as np

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

# 初始化变量
counter = 0
stage = None  # 当前状态 ('A'或'B')

# 打开视频流
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # 将图像转为RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # 检测关键点
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 获取髋部和膝盖的坐标
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        # 计算髋部的角度
        hip_angle = calculate_angle(shoulder, hip, knee)

        # 检查当前的状态 (standing 或 squat)
        if hip[1] < knee[1]:  # 髋部高于膝盖 (站立状态)
            if stage == 'standing':
                stage = 'squat'
                counter += 1  # 完成一次深蹲
        elif hip[1] > knee[1]:  # 髋部低于膝盖 (深蹲状态)
            stage = 'standing'
        # 获取肩膀和鼻子的关键点
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

        # 计算肩膀的宽度，并确定头部的宽度和高度
        shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        head_width = shoulder_width *.6
        head_height = head_width * 1.8

        # 将头部宽度和高度转换为像素
        head_width_px = int(head_width * frame.shape[1])
        head_height_px = int(head_height * frame.shape[0])

        # 计算头部矩形的边界
        x_center = int(nose[0] * frame.shape[1])
        y_center = int(nose[1] * frame.shape[0])

        x_min = x_center - head_width_px // 2
        y_min = y_center - head_height_px // 2
        x_max = x_center + head_width_px // 2
        y_max = y_center + head_height_px // 2

        # 绘制灰色矩形
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (128, 128, 128), -1)
        # 显示计数和髋部角度
        cv2.putText(frame, f'Squat Count: {counter}',
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # 在图像左上角显示角度（使用较大的字体）
        cv2.putText(frame, f'Hip Angle: {int(hip_angle)}',
                    (10, 100),  # 位置在左上角
                    cv2.FONT_HERSHEY_SIMPLEX, 2,  # 字体大小为2
                    (0, 255, 0), 2, cv2.LINE_AA  # 绿色字体，线宽为3
                    )

        # 检测髋部是否低于膝盖
        if hip[1] > knee[1]:
            circle_color = (255, 255, 255)  # 白色
        else:
            circle_color = (0, 0, 255)  # 红色

        # 绘制大圆圈在右上角
        cv2.circle(frame, (frame.shape[1] - 50, 50), 40, circle_color, -1)


        # 绘制关键点和骨架
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Mediapipe Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
