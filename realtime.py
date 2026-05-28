import sys
sys.path.insert(0, r'D:\squat_libs')

import cv2
import mediapipe as mp
import numpy as np
import joblib
import csv
import time

# =============================================
# 모델 및 설정 로드
# =============================================
MODEL_PATH = r'C:\Users\tlswn\OneDrive\바탕 화면\squat'

xgb_model = joblib.load(f'{MODEL_PATH}\\xgboost.pkl')
xgb_scaler = joblib.load(f'{MODEL_PATH}\\xgboost_scaler.pkl')

LABEL_MAP = {
    0: 'Good Squat',
    1: 'Shallow Squat',
    2: 'Forward Lean',
    3: 'Knee Cave',
    4: 'Heel Rise',
    5: 'Asymmetric'
}

FEATURE_COLS = [
    'knee_angle_left', 'knee_angle_right',
    'hip_angle_left', 'hip_angle_right',
    'ankle_angle_left', 'ankle_angle_right',
    'knee_angle_avg', 'hip_angle_avg',
    'ankle_angle_avg', 'knee_asymmetry', 'heel_diff',
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# =============================================
# 각도 계산 함수
# =============================================
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def get_lm(landmarks, name, w=1, h=1):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return np.array([lm.x * w, lm.y * h])

def extract_angles(lm, w=1, h=1):
    try:
        vis_l = lm[mp_pose.PoseLandmark['LEFT_KNEE'].value].visibility
        vis_r = lm[mp_pose.PoseLandmark['RIGHT_KNEE'].value].visibility
        side  = 'LEFT' if vis_l >= vis_r else 'RIGHT'
        opp   = 'RIGHT' if side == 'LEFT' else 'LEFT'

        sh  = get_lm(lm, f'{side}_SHOULDER', w, h)
        hip = get_lm(lm, f'{side}_HIP',      w, h)
        kn  = get_lm(lm, f'{side}_KNEE',     w, h)
        an  = get_lm(lm, f'{side}_ANKLE',    w, h)
        ft  = get_lm(lm, f'{side}_FOOT_INDEX', w, h)
        he  = get_lm(lm, f'{side}_HEEL',     w, h)

        if kn[1] < hip[1]:
            kn, hip = hip, kn

        spine_top = (sh + hip) / 2
        k_ang = calc_angle(hip, kn, an)
        h_ang = calc_angle(spine_top, hip, kn)
        a_ang = calc_angle(kn, an, ft)

        try:
            kn2     = get_lm(lm, f'{opp}_KNEE',  w, h)
            hi2     = get_lm(lm, f'{opp}_HIP',   w, h)
            an2     = get_lm(lm, f'{opp}_ANKLE', w, h)
            k2      = calc_angle(hi2, kn2, an2)
            opp_vis = lm[mp_pose.PoseLandmark[f'{opp}_KNEE'].value].visibility
            asym    = abs(k_ang - k2) if opp_vis > 0.7 else 0.0
        except:
            k2   = k_ang
            asym = 0.0

        return {
            'knee_angle_left':   round(k_ang, 2),
            'knee_angle_right':  round(k2,    2),
            'hip_angle_left':    round(h_ang, 2),
            'hip_angle_right':   round(h_ang, 2),
            'ankle_angle_left':  round(a_ang, 2),
            'ankle_angle_right': round(a_ang, 2),
            'knee_angle_avg':    round(k_ang, 2),
            'hip_angle_avg':     round(h_ang, 2),
            'ankle_angle_avg':   round(a_ang, 2),
            'knee_asymmetry':    round(asym,  2),
            'heel_diff':         round(abs(he[1] - an[1]), 4),
        }
    except Exception as e:
        return None

# =============================================
# 실시간 분석
# =============================================
cap = cv2.VideoCapture(2)

cv2.namedWindow('Squat AI Coach', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Squat AI Coach', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'{MODEL_PATH}\\squat_result.mp4', fourcc, 20.0, (960, 720))

# CSV 저장 설정
csv_file = open(f'{MODEL_PATH}\\squat_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'label', 'knee', 'hip', 'heel', 'count'])

count = 0
position = 'up'
squat_was_good = False
smooth_labels = []

with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 720))
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        label = 'Detecting...'
        color = (128, 128, 128)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

            angles = extract_angles(results.pose_landmarks.landmark, w, h)

            if angles:
                X = np.array([[angles[c] for c in FEATURE_COLS]])
                X_sc = xgb_scaler.transform(X)
                pred = xgb_model.predict(X_sc)[0]

                smooth_labels.append(pred)
                if len(smooth_labels) > 3:
                    smooth_labels.pop(0)
                pred = max(set(smooth_labels), key=smooth_labels.count)

                label = LABEL_MAP.get(int(pred), 'Unknown')

                knee = angles['knee_angle_avg']
                if knee < 110 and position == 'up':
                    position = 'down'
                    squat_was_good = False
                elif int(pred) == 0 and position == 'down':
                    squat_was_good = True
                elif knee > 140 and position == 'down':
                    position = 'up'
                    if squat_was_good:
                        count += 1
                    squat_was_good = False

                color = (0, 255, 0) if int(pred) == 0 else (0, 0, 255)

                cv2.putText(frame, f'Knee: {angles["knee_angle_avg"]:.1f}', (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f'Hip: {angles["hip_angle_avg"]:.1f}', (10, 230),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f'Heel: {angles["heel_diff"]:.3f}', (10, 260),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # CSV 저장
                csv_writer.writerow([
                    time.strftime('%H:%M:%S'),
                    label,
                    round(angles['knee_angle_avg'], 1),
                    round(angles['hip_angle_avg'], 1),
                    round(angles['heel_diff'], 3),
                    count
                ])

        cv2.rectangle(frame, (0, 0), (500, 50), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.rectangle(frame, (0, 50), (200, 100), (0, 0, 0), -1)
        cv2.putText(frame, f'Count: {count}', (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # 영상 저장
        out.write(frame)

        cv2.imshow('Squat AI Coach', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
csv_file.close()
print(f'✅ 저장 완료!')
print(f'영상: {MODEL_PATH}\\squat_result.mp4')
print(f'CSV: {MODEL_PATH}\\squat_log.csv')
cv2.destroyAllWindows()