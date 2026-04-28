# 필수 환경 설정
print("필수 환경 설정 중...")
!pip install mediapipe==0.10.14 opencv-python-headless==4.10.0.84 pillow matplotlib > /dev/null 2>&1
!sudo apt-get install -y fonts-nanum > /dev/null 2>&1
!sudo fc-cache -fv > /dev/null 2>&1
!rm ~/.cache/matplotlib -rf > /dev/null 2>&1
print("설치 완료!")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os, io, joblib, warnings, subprocess
from collections import deque, Counter
from google.colab.patches import cv2_imshow
from google.colab import files
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import ImageFont, ImageDraw, Image
from IPython.display import HTML, display
from base64 import b64encode
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
plt.rcParams['axes.unicode_minus'] = False

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print("환경 설정 완료!")

# 필터링
from PIL import ImageFont, ImageDraw, Image

LABEL_MAP = {
    0: '올바른 스쿼트',    1: '얕은 스쿼트',
    2: '앞으로 기울어진 스쿼트', 3: '무릎 안쪽 꺾임',
    4: '발뒤꿈치 들림',    5: '비대칭 스쿼트',
}
FEEDBACK_MAP = {
    0: '완벽한 스쿼트입니다! 자세를 유지하세요.',
    1: '더 깊이 내려가세요. 무릎이 90도 이상 굽혀져야 합니다.',
    2: '상체를 세워주세요. 코어에 힘을 주고 가슴을 앞으로 내미세요.',
    3: '무릎이 안쪽으로 꺾이고 있습니다. 발끝 방향으로 무릎을 밀어주세요.',
    4: '발뒤꿈치가 들리고 있습니다. 발바닥 전체를 바닥에 붙이세요.',
    5: '좌우 균형이 맞지 않습니다. 양쪽 체중을 균등하게 분배하세요.',
}
COLOR_MAP = {
    0:(0,200,0), 1:(0,220,220), 2:(0,140,255),
    3:(0,0,255), 4:(200,0,200), 5:(255,100,0),
}
FEATURE_COLS = [
    'knee_angle_left','knee_angle_right',
    'hip_angle_left', 'hip_angle_right',
    'ankle_angle_left','ankle_angle_right',
    'knee_angle_avg', 'hip_angle_avg',
    'ankle_angle_avg','knee_asymmetry','heel_diff',
]

def put_korean_text(frame, text, pos, font_size=20, color=(255,255,255)):
    img_pil = Image.fromarray(frame)
    draw    = ImageDraw.Draw(img_pil)
    font    = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', font_size)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)

def calc_angle(a, b, c):
    a = np.array(a, dtype=float)[:2]
    b = np.array(b, dtype=float)[:2]
    c = np.array(c, dtype=float)[:2]
    ba = a - b
    bc = c - b
    cos_v = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_v, -1.0, 1.0))))

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
            print('  [보정] 무릎-엉덩이 y좌표 역전 감지 → 교환')
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

        print(f'  [디버그] side={side}, 무릎={k_ang:.1f}, 엉덩이={h_ang:.1f}, 발목={a_ang:.1f}')

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
        print(f'각도 추출 오류: {e}')
        return None

def extract_metrics(landmarks, w, h):
    def p(i): return np.array([landmarks[i].x*w, landmarks[i].y*h])
    vis_l = landmarks[25].visibility
    vis_r = landmarks[26].visibility
    if vis_l >= vis_r:
        sh, hip, knee, ankle, foot = p(11), p(23), p(25), p(27), p(31)
    else:
        sh, hip, knee, ankle, foot = p(12), p(24), p(26), p(28), p(32)

    if knee[1] < hip[1]:
        knee, hip = hip, knee

    spine_top = (sh + hip) / 2
    lk = calc_angle(hip, knee, ankle)
    sp = calc_angle(spine_top, hip, knee)
    la = calc_angle(knee, ankle, foot)
    ki = max(knee[0] - ankle[0], 0) / (abs(hip[0] - ankle[0]) + 1e-6)
    return {'lk':lk, 'rk':lk, 'sp':sp, 'la':la, 'ra':la, 'ki':ki, 'sym':0.0}

def rule_classify(angles):
    k, h  = angles['knee_angle_avg'], angles['hip_angle_avg']
    asym  = angles['knee_asymmetry']
    heel  = angles['heel_diff']

    if asym > 15:          return 5
    if heel > 0.05:        return 4
    if h < 60 or h > 120: return 2
    if asym > 10:          return 3
    if k < 50:             return 1
    if 50 <= k <= 80:      return 0
    if 80 < k <= 120:      return 1
    if k >= 150:           return -1
    return 1

def judge_squat_logic(m):
    fb, labels = [], []
    avg_k = (m['lk'] + m['rk']) / 2

    if avg_k < 50:     labels.append(1); fb.append(f'얕은 스쿼트({avg_k:.1f}도)')
    if avg_k > 80:     labels.append(1); fb.append(f'너무 깊음({avg_k:.1f}도)')
    if m['sp'] < 60:   labels.append(2); fb.append(f'상체 숙여짐({m["sp"]:.1f}도)')
    if m['sp'] > 120:  labels.append(2); fb.append(f'상체 뒤로 기울어짐({m["sp"]:.1f}도)')
    if m['ki'] > 0.15: labels.append(3); fb.append('무릎 꺾임')
    if m['la'] < 50 or m['ra'] < 50: labels.append(4); fb.append('뒤꿈치 들림')
    if m['sym'] > 15:  labels.append(5); fb.append(f'비대칭({m["sym"]:.1f}도)')

    if not labels and 50 <= avg_k <= 80:
        return 0, '완벽'
    return (labels[0] if labels else -1), ', '.join(fb)

def calc_class_probabilities(m):
    avg_k  = (m['lk'] + m['rk']) / 2
    scores = {i: 0.0 for i in range(6)}

    if avg_k < 50:     scores[1] = min((50 - avg_k)  / 25.0, 1.0) * 3.0
    if avg_k > 80:     scores[1] = min((avg_k - 80)  / 35.0, 1.0) * 2.0
    if m['sp'] < 60:   scores[2] = min((60 - m['sp'])/ 30.0, 1.0) * 3.0
    if m['sp'] > 120:  scores[2] = min((m['sp']-120) / 30.0, 1.0) * 3.0
    if m['ki'] > 0.15: scores[3] = min((m['ki']-0.15)/ 0.2,  1.0) * 4.0
    heel_err = max(50 - m['la'], 50 - m['ra'], 0)
    if heel_err > 0:   scores[4] = min(heel_err / 20.0, 1.0) * 3.0
    if m['sym'] > 15:  scores[5] = min((m['sym']-15) / 20.0, 1.0) * 3.0

    base = 2.0
    if 50 <= avg_k <= 80 and 60 <= m['sp'] <= 120 and m['ki'] <= 0.15:
        base += 2.0
    scores[0] = max(base - sum(scores[i] for i in range(1,6)), 0.0)

    vals  = np.array(list(scores.values()))
    probs = np.exp(vals - vals.max())
    probs /= probs.sum()
    return {LABEL_MAP[i]: round(float(probs[i])*100, 1) for i in range(6)}

class SquatClassifier:
    def __init__(self, model_type='random_forest'):
        self.model = None; self.scaler = None
        self.model_type = model_type; self._load()
    def _load(self):
        mp_path = f'models/{self.model_type}.pkl'
        sc_path = f'models/{self.model_type}_scaler.pkl'
        if os.path.exists(mp_path):
            self.model  = joblib.load(mp_path)
            self.scaler = joblib.load(sc_path)
            print(f'✅ 모델 로드: {mp_path}')
        else:
            print('ℹ️ 저장된 모델 없음 → 룰 기반 분류 사용')
    def predict(self, angles):
        if self.model is None:
            return rule_classify(angles), 0.80
        try:
            X   = np.array([[angles[c] for c in FEATURE_COLS]])
            if self.model_type == 'xgboost': X = self.scaler.transform(X)
            lbl = int(self.model.predict(X)[0])
            return lbl, float(self.model.predict_proba(X)[0][lbl])
        except:
            return rule_classify(angles), 0.75

def draw_skeleton_and_angles(frame, pose_landmarks, angles):
    h, w = frame.shape[:2]
    mp_drawing.draw_landmarks(
        frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,0,220), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0,220,0), thickness=2),
    )
    for lm_name, angle_key in [
        ('LEFT_KNEE',  'knee_angle_left'),
        ('RIGHT_KNEE', 'knee_angle_right'),
        ('LEFT_HIP',   'hip_angle_left'),
        ('RIGHT_HIP',  'hip_angle_right'),
    ]:
        lm_pt = pose_landmarks.landmark[mp_pose.PoseLandmark[lm_name].value]
        px, py = int(lm_pt.x * w), int(lm_pt.y * h)
        cv2.putText(frame, f'{angles[angle_key]:.0f}',
                    (px+5, py-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,0), 1, cv2.LINE_AA)

def draw_overlay(frame, angles, label, conf):
    color = COLOR_MAP[label]
    name  = LABEL_MAP[label]
    fb    = FEEDBACK_MAP[label]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (450,220), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    frame = put_korean_text(frame, f'자세: {name}',                (10,20),  25, color)
    frame = put_korean_text(frame, f'신뢰도: {conf*100:.1f}%',     (10,55),  20, color)
    frame = put_korean_text(frame, f'무릎: {angles["knee_angle_avg"]:.1f}°  엉덩이: {angles["hip_angle_avg"]:.1f}°', (10,88))
    frame = put_korean_text(frame, f'발목: {angles["ankle_angle_avg"]:.1f}°  비대칭: {angles["knee_asymmetry"]:.1f}°', (10,118))
    frame = put_korean_text(frame, f'피드백: {fb[:22]}',           (10,155), 18, (0,220,255))
    if len(fb) > 22:
        frame = put_korean_text(frame, fb[22:],                    (10,180), 18, (0,220,255))
    return frame

print('✅ 완료!')

# 이미지로 판정
print('📂 이미지 파일을 선택하세요')
uploaded = files.upload()

clf = SquatClassifier()

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, model_complexity=2) as pose:
    for filename, data in uploaded.items():
        print(f'\n[분석 중] {filename}')

        nparr = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w  = img.shape[:2]

        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print('  ❌ 포즈 감지 실패 — 전신이 보이는 사진을 사용하세요')
            continue

        lm = results.pose_landmarks.landmark

        # ★ 랜드마크 좌표 디버그 출력
        check = {
            'L_SHOULDER':11, 'R_SHOULDER':12,
            'L_HIP':23,      'R_HIP':24,
            'L_KNEE':25,     'R_KNEE':26,
            'L_ANKLE':27,    'R_ANKLE':28,
            'L_FOOT':31,     'R_FOOT':32,
        }
        print(f'  {"이름":<12} {"x(px)":>6} {"y(px)":>6} {"vis":>6}')
        print(f'  {"-"*34}')
        for name, idx in check.items():
            px  = int(lm[idx].x * w)
            py  = int(lm[idx].y * h)
            vis = lm[idx].visibility
            print(f'  {name:<12} {px:>6} {py:>6} {vis:>6.2f}')

        angles = extract_angles(lm, w, h)
        m      = extract_metrics(lm, w, h)

        if not angles:
            print('  ⚠️ 각도 계산 실패')
            continue

        label, conf   = clf.predict(angles)
        idx, final_fb = judge_squat_logic(m)
        final_probs   = calc_class_probabilities(m)

        if idx == -1:
            final_label = '서있는 상태 (스쿼트 아님)'
        else:
            final_label = LABEL_MAP.get(idx, LABEL_MAP.get(label, '알 수 없음'))

        # 스켈레톤 + 각도 숫자 그리기
        ann_img = img.copy()
        mp_drawing.draw_landmarks(
            ann_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,220), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0,220,0), thickness=2),
        )
        for lm_name, angle_key in [
            ('LEFT_KNEE',  'knee_angle_left'),
            ('RIGHT_KNEE', 'knee_angle_right'),
            ('LEFT_HIP',   'hip_angle_left'),
            ('RIGHT_HIP',  'hip_angle_right'),
        ]:
            lm_pt = results.pose_landmarks.landmark[mp_pose.PoseLandmark[lm_name].value]
            px, py = int(lm_pt.x * w), int(lm_pt.y * h)
            cv2.putText(ann_img, f'{angles[angle_key]:.0f}',
                        (px+5, py-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,0), 1, cv2.LINE_AA)

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                                 gridspec_kw={'width_ratios': [1.2, 1]})

        axes[0].imshow(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'판정: {final_label}', fontsize=22, fontweight='bold', pad=20)
        axes[0].axis('off')

        axes[1].axis('off')
        axes[1].text(0.05, 0.92,
                     f'[ 최종 결과: {final_label} ]\n피드백: {final_fb if final_fb else "완벽"}',
                     fontsize=16, linespacing=2.0, va='top', fontweight='bold')
        axes[1].text(0.05, 0.72, '[ 클래스별 확률 ]', fontsize=14)

        for i, (lbl_name, prob) in enumerate(final_probs.items()):
            y = 0.65 - i * 0.065
            axes[1].text(0.05, y, lbl_name, fontsize=13, va='center')
            axes[1].text(0.55, y, f'{prob:.1f}%', fontsize=13, va='center')

        axes[1].text(0.05, 0.22,
                     f'무릎각도 : {angles["knee_angle_avg"]:.1f}°\n'
                     f'엉덩이   : {angles["hip_angle_avg"]:.1f}°\n'
                     f'발목     : {angles["ankle_angle_avg"]:.1f}°\n'
                     f'비대칭도  : {angles["knee_asymmetry"]:.1f}°',
                     fontsize=12, va='top', linespacing=2.0, color='gray')

        plt.tight_layout()
        plt.show()

        print(f'  ✅ 자세    : {final_label} (신뢰도 {conf*100:.1f}%)')
        print(f'  📝 피드백  : {FEEDBACK_MAP.get(idx, FEEDBACK_MAP.get(label, "-"))}')
        print(f'  📐 무릎각도 : {angles["knee_angle_avg"]:.1f}°')
        print(f'  📐 엉덩이  : {angles["hip_angle_avg"]:.1f}°')
        print(f'  📐 발목    : {angles["ankle_angle_avg"]:.1f}°')
        print(f'  ↔  비대칭도 : {angles["knee_asymmetry"]:.1f}°')

# 영상으로 판정
print('📂 영상 파일을 선택하세요 (mp4, avi)')
uploaded_video = files.upload()

clf = SquatClassifier()

for filename, data in uploaded_video.items():
    with open(filename, 'wb') as f:
        f.write(data)

    print(f'\n[분석 중] {filename}')

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print('❌ 영상 열기 실패')
        continue

    fps        = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_path   = 'temp_output.mp4'
    final_path = 'final_output.mp4'

    writer = cv2.VideoWriter(raw_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (960, 720))

    label_history = []
    smooth_buf    = deque(maxlen=5)
    rep_count     = 0
    squat_state   = 'UP'
    frame_idx     = 0

    with mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f'  처리 중: {frame_idx}/{total} ({frame_idx/total*100:.1f}%)', end='\r')

            frame = cv2.resize(frame, (960, 720))
            h, w  = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                angles = extract_angles(results.pose_landmarks.landmark, w, h)

                if angles:
                    label, conf = clf.predict(angles)

                    # 스무딩
                    smooth_buf.append(label)
                    label_sm = Counter(smooth_buf).most_common(1)[0][0]
                    label_history.append(label_sm)

                    # 반복 카운트
                    k = angles['knee_angle_avg']
                    if squat_state == 'UP' and k < 62:
                        squat_state = 'DOWN'
                    elif squat_state == 'DOWN' and k > 72:
                        squat_state = 'UP'
                        rep_count  += 1

                    # 스켈레톤 + 각도 표시
                    draw_skeleton_and_angles(frame, results.pose_landmarks, angles)

                    # 오버레이 (label_sm 기준)
                    frame = draw_overlay(frame, angles, label_sm, conf)

                    # 반복 횟수 표시
                    cv2.putText(frame, f'Reps: {rep_count}',
                                (w-180, h-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0,255,180), 2, cv2.LINE_AA)

                    # 상태 표시
                    cv2.putText(frame, squat_state,
                                (w-180, h-50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (200,200,200), 1, cv2.LINE_AA)

            writer.write(frame)

    cap.release()
    writer.release()
    print(f'\n  ✅ 처리 완료 (총 {frame_idx}프레임)')

    # ffmpeg 변환
    print('🎬 ffmpeg 변환 중...')
    subprocess.run([
        'ffmpeg', '-y',
        '-i', raw_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        final_path
    ], capture_output=True)
    print('✅ 변환 완료:', final_path)

    # 요약
    if label_history:
        cnt = Counter(label_history)
        print(f'\n📊 영상 분석 요약 | 반복 횟수: {rep_count}회')
        print('─' * 35)
        for lbl, c in cnt.most_common():
            pct = c / len(label_history) * 100
            bar = '█' * int(pct // 5)
            print(f'  {LABEL_MAP[lbl]:<18} {bar} {pct:.1f}%')

        # 파이차트
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(list(cnt.values()),
               labels=[LABEL_MAP[k] for k in cnt],
               autopct='%1.1f%%', startangle=90)
        ax.set_title('자세 분포')
        plt.tight_layout()
        plt.show()

    # 영상 표시
    video    = open(final_path, 'rb').read()
    data_url = 'data:video/mp4;base64,' + b64encode(video).decode()
    display(HTML(f'''
    <video width=600 controls>
        <source src="{data_url}" type="video/mp4">
    </video>
    '''))
