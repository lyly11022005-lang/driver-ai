
import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# ==========================================
# 1. ĐƯỜNG DẪN VÀ CẤU HÌNH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "yawn_cnn.keras"
LANDMARKER_PATH = BASE_DIR / "face_landmarker.task"

IMG_SIZE = (64, 64)

EAR_THRESHOLD = 0.20
EYE_SECONDS = 1.5

MAR_THRESHOLD = 0.30
YAWN_SECONDS = 1.5
YAWN_PROB_THRESHOLD = 0.50
MAR_CNN_GATE = 0.15

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 78, 308, 82, 312, 87, 317]


# ==========================================
# 2. KIỂM TRA VÀ TẢI MÔ HÌNH CNN
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình ngáp: {MODEL_PATH}"
    )

if not LANDMARKER_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy Face Landmarker: {LANDMARKER_PATH}"
    )

print("DRIVER AI: Đang tải mô hình ngáp...")

yawn_model = tf.keras.models.load_model(str(MODEL_PATH))

print("DRIVER AI: Đã tải mô hình CNN.")


# ==========================================
# 3. HÀM TÍNH TOÁN
# ==========================================

def point_xy(landmarks, index, width, height):
    point = landmarks[index]
    return np.array(
        [point.x * width, point.y * height],
        dtype=np.float32
    )


def distance(a, b):
    return float(np.linalg.norm(a - b))


def calculate_ear(landmarks, indices, width, height):
    p1, p2, p3, p4, p5, p6 = [
        point_xy(landmarks, i, width, height)
        for i in indices
    ]

    horizontal = distance(p1, p4)

    if horizontal < 1:
        return 0.0

    return (
        distance(p2, p6) + distance(p3, p5)
    ) / (2.0 * horizontal)


def calculate_mar(landmarks, width, height):
    top = point_xy(landmarks, 13, width, height)
    bottom = point_xy(landmarks, 14, width, height)
    left = point_xy(landmarks, 61, width, height)
    right = point_xy(landmarks, 291, width, height)

    mouth_width = distance(left, right)

    if mouth_width < 1:
        return 0.0

    return distance(top, bottom) / mouth_width


def crop_mouth(frame, landmarks):
    height, width = frame.shape[:2]

    xs = [landmarks[i].x * width for i in MOUTH]
    ys = [landmarks[i].y * height for i in MOUTH]

    pad_x = max((max(xs) - min(xs)) * 0.8, 5)
    pad_y = max((max(ys) - min(ys)) * 0.8, 5)

    x1 = max(0, int(min(xs) - pad_x))
    x2 = min(width, int(max(xs) + pad_x))
    y1 = max(0, int(min(ys) - pad_y))
    y2 = min(height, int(max(ys) + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def predict_yawn(crop):
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32") / 255.0

    prediction = yawn_model(
        np.expand_dims(image, axis=0),
        training=False
    )

    return float(np.asarray(prediction).reshape(-1)[0])

# ==========================================
# HÀM LẤY TỌA ĐỘ KHUNG NHẬN DIỆN
# ==========================================

def get_landmark_box(landmarks, indices, width, height,
                     pad_x=0.35, pad_y=0.60):

    xs = [landmarks[i].x * width for i in indices]
    ys = [landmarks[i].y * height for i in indices]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    padding_x = (x_max - x_min) * pad_x
    padding_y = (y_max - y_min) * pad_y

    x1 = max(0, int(x_min - padding_x))
    y1 = max(0, int(y_min - padding_y))
    x2 = min(width, int(x_max + padding_x))
    y2 = min(height, int(y_max + padding_y))

    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1
    }

# ==========================================
# 4. BỘ XỬ LÝ AI CHO TỪNG NGƯỜI DÙNG
# ==========================================

class DriverAIEngine:

    def __init__(self):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(LANDMARKER_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1
        )

        # Mỗi kết nối có bộ nhận diện và bộ đếm riêng
        self.landmarker = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

        self.eye_start = None
        self.yawn_start = None
        self.frame_i = 0
        self.last_prob = 0.0
        self.was_alert = False
        self.alert_count = 0

    def process_frame(self, jpeg_bytes):

        # Giải mã ảnh JPEG nhận từ trình duyệt
        image_array = np.frombuffer(
            jpeg_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return {
                "error": "Không đọc được hình ảnh camera"
            }

        # Thu nhỏ ảnh để giảm tải xử lý
        height, width = frame.shape[:2]

        if width > 480:
            new_height = int(height * 480 / width)
            frame = cv2.resize(frame, (480, new_height))

        height, width = frame.shape[:2]

        now = time.monotonic()
        self.frame_i += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect(mp_image)

        eye_status = "KHÔNG THẤY MẮT"
        mouth_status = "KHÔNG THẤY MIỆNG"

        ear = 0.0
        mar = 0.0
        probability = 0.0

        eyes_closed = False
        eyes_long = False
        yawn_now = False
        yawn_long = False

        left_eye_box = None
        right_eye_box = None
        mouth_box = None

        face_found = bool(result.face_landmarks)

        if face_found:

            landmarks = result.face_landmarks[0]

            # LẤY TỌA ĐỘ MẮT VÀ MIỆNG
            left_eye_box = get_landmark_box(
                landmarks, LEFT_EYE, width, height
            )

            right_eye_box = get_landmark_box(
                landmarks, RIGHT_EYE, width, height
            )

            mouth_box = get_landmark_box(
                landmarks, MOUTH, width, height,
                pad_x=0.45, pad_y=0.80
            )

            # ----------------------------------
            # A. NHẬN DIỆN MẮT NHẮM
            # ----------------------------------

            left_ear = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )

            right_ear = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            ear = (left_ear + right_ear) / 2.0

            eyes_closed = (
                left_ear < EAR_THRESHOLD
                and right_ear < EAR_THRESHOLD
            )

            if eyes_closed:

                if self.eye_start is None:
                    self.eye_start = now

                elapsed = now - self.eye_start
                eyes_long = elapsed >= EYE_SECONDS

                eye_status = (
                    f"MẮT NHẮM {elapsed:.1f}/{EYE_SECONDS:.1f}s"
                )

            else:
                self.eye_start = None
                eye_status = "MẮT MỞ"

            # ----------------------------------
            # B. NHẬN DIỆN NGÁP
            # ----------------------------------

            mar = calculate_mar(
                landmarks, width, height
            )

            mouth_crop = crop_mouth(frame, landmarks)

            if mouth_crop is not None:

                if (
                    mar >= MAR_CNN_GATE
                    and self.frame_i % 2 == 0
                ):
                    self.last_prob = predict_yawn(mouth_crop)

                elif mar < MAR_CNN_GATE:
                    self.last_prob = 0.0

                probability = self.last_prob

                yawn_now = (
                    mar >= MAR_THRESHOLD
                    or probability >= YAWN_PROB_THRESHOLD
                )

                if yawn_now:

                    if self.yawn_start is None:
                        self.yawn_start = now

                    elapsed = now - self.yawn_start
                    yawn_long = elapsed >= YAWN_SECONDS

                    mouth_status = (
                        f"ĐANG NGÁP {elapsed:.1f}/{YAWN_SECONDS:.1f}s"
                    )

                else:
                    self.yawn_start = None
                    mouth_status = "KHÔNG NGÁP"

            else:
                self.yawn_start = None
                mouth_status = "KHÔNG ĐỌC ĐƯỢC MIỆNG"

        else:
            self.eye_start = None
            self.yawn_start = None
            self.last_prob = 0.0

            eye_status = "KHÔNG THẤY MẮT"
            mouth_status = "KHÔNG THẤY MIỆNG"

        # ----------------------------------
        # C. TỔNG HỢP CẢNH BÁO
        # ----------------------------------

        is_alert = face_found and (eyes_long or yawn_long)

        if not face_found:
            alert_message = "KHÔNG PHÁT HIỆN KHUÔN MẶT"
        elif is_alert:
            alert_message = "CẢNH BÁO: CÓ DẤU HIỆU BUỒN NGỦ"
        else:
            alert_message = "AN TOÀN - CHƯA PHÁT HIỆN"

        if is_alert and not self.was_alert:
            self.alert_count += 1

        self.was_alert = is_alert

        return {
            "face_found": face_found,

            # Kích thước ảnh AI xử lý
            "frame_width": width,
            "frame_height": height,

            # Tọa độ khung nhận diện
            "left_eye_box": left_eye_box,
            "right_eye_box": right_eye_box,
            "mouth_box": mouth_box,

            # Trạng thái nhận diện
            "eye_status": eye_status,
            "mouth_status": mouth_status,

            # Chỉ số AI
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "yawn_probability": round(probability, 3),

            # Dấu hiệu buồn ngủ
            "eyes_closed": eyes_closed,
            "yawn_detected": yawn_now,
            "is_alert": is_alert,
            "alert_message": alert_message,
            "alert_count": self.alert_count
        }
        
    def close(self):
        self.landmarker.close()
