import argparse
from datetime import datetime
import calendar
from functools import wraps
import os
from pathlib import Path
import re
import threading
import time
import uuid

import cv2
import numpy as np
import pymysql
from pymysql.cursors import DictCursor
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)
app.secret_key = os.environ.get("PLATE_APP_SECRET", "plate-detection-local-secret")
DB_CONFIG = {
    "host": os.environ.get("PLATE_DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("PLATE_DB_PORT", "3306")),
    "user": os.environ.get("PLATE_DB_USER", "root"),
    "password": os.environ.get("PLATE_DB_PASSWORD", ""),
    "database": os.environ.get("PLATE_DB_NAME", "plate_detection"),
    "charset": "utf8mb4",
    "cursorclass": DictCursor,
    "autocommit": True,
}
OUTPUT_DIR = Path("runtime_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
detector_state = None


def get_mysql_connection(database=True):
    config = DB_CONFIG.copy()
    if not database:
        config.pop("database", None)
    return pymysql.connect(**config)


def init_db():
    database_name = DB_CONFIG["database"]
    with get_mysql_connection(database=False) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )

    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(80) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at DATETIME NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plate_key VARCHAR(32) NOT NULL UNIQUE,
                    plate_number VARCHAR(32) NOT NULL,
                    owner_name VARCHAR(120) NOT NULL,
                    plate_date VARCHAR(16),
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    INDEX idx_vehicles_plate_key (plate_key)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plate_number VARCHAR(32) NOT NULL,
                    plate_date VARCHAR(16),
                    owner_name VARCHAR(120),
                    source VARCHAR(24) NOT NULL,
                    image_url VARCHAR(255) NOT NULL,
                    threshold DOUBLE NOT NULL,
                    created_at DATETIME NOT NULL,
                    INDEX idx_detections_created_at (created_at),
                    INDEX idx_detections_plate_number (plate_number)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            ensure_column(cursor, "detections", "owner_name", "VARCHAR(120)")
            cursor.execute("SELECT 1 FROM users WHERE username = %s", ("admin",))
            admin_exists = cursor.fetchone()
            if admin_exists is None:
                cursor.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (%s, %s, %s)",
                    ("admin", generate_password_hash("admin123"), datetime.now()),
                )


def ensure_column(cursor, table_name, column_name, definition):
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """,
        (DB_CONFIG["database"], table_name, column_name),
    )
    if cursor.fetchone() is None:
        cursor.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `{column_name}` {definition}")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "message": "Login diperlukan"}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def normalize_plate_key(plate_number):
    return re.sub(r'[^A-Z0-9]', '', (plate_number or "").upper())


def find_vehicle_by_plate(plate_number):
    plate_key = normalize_plate_key(plate_number)
    if not plate_key:
        return None

    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, plate_key, plate_number, owner_name, plate_date, created_at, updated_at
                FROM vehicles
                WHERE plate_key = %s
                LIMIT 1
                """,
                (plate_key,),
            )
            row = cursor.fetchone()

    if row is None:
        return None

    row["tax_status"] = get_tax_status(row.get("plate_date"))
    for key in ("created_at", "updated_at"):
        if isinstance(row.get(key), datetime):
            row[key] = row[key].isoformat(timespec="seconds")
    return row


def save_vehicle(plate_number, owner_name, plate_date):
    plate_key = normalize_plate_key(plate_number)
    if not plate_key:
        raise ValueError("Nomor plat tidak valid")

    owner_name = (owner_name or "").strip()
    if not owner_name:
        raise ValueError("Nama pemilik wajib diisi")

    normalized_date = plate_date if plate_date not in ("", "-") else None
    now = datetime.now()
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO vehicles (plate_key, plate_number, owner_name, plate_date, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    plate_number = VALUES(plate_number),
                    owner_name = VALUES(owner_name),
                    plate_date = VALUES(plate_date),
                    updated_at = VALUES(updated_at)
                """,
                (plate_key, plate_number, owner_name, normalized_date, now, now),
            )

    return find_vehicle_by_plate(plate_number)


def sync_vehicle_from_detection(plate_number, owner_name, plate_date):
    if not plate_number or not owner_name:
        return None
    return save_vehicle(plate_number, owner_name, plate_date)


def save_detection(plate_number, plate_date, source, image_url, threshold, owner_name=None):
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO detections (plate_number, plate_date, owner_name, source, image_url, threshold, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    plate_number,
                    plate_date if plate_date not in ("", "-") else None,
                    owner_name if owner_name not in ("", "-") else None,
                    source,
                    image_url,
                    float(threshold),
                    datetime.now(),
                ),
            )


def fetch_recent_detections(limit=25, filters=None):
    filters = filters or {}
    where = []
    params = []

    query = (filters.get("q") or "").strip()
    if query:
        where.append("(plate_number LIKE %s OR owner_name LIKE %s)")
        like = f"%{query}%"
        params.extend([like, like])

    source = (filters.get("source") or "").strip()
    if source:
        where.append("source = %s")
        params.append(source)

    date_from = (filters.get("date_from") or "").strip()
    if date_from:
        where.append("DATE(created_at) >= %s")
        params.append(date_from)

    date_to = (filters.get("date_to") or "").strip()
    if date_to:
        where.append("DATE(created_at) <= %s")
        params.append(date_to)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, plate_number, plate_date, owner_name, source, image_url, threshold, created_at
                FROM detections
                {where_sql}
                ORDER BY id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()
            for row in rows:
                if isinstance(row.get("created_at"), datetime):
                    row["created_at"] = row["created_at"].isoformat(timespec="seconds")
                row["tax_status"] = get_tax_status(row.get("plate_date"))
            status_filter = (filters.get("tax_status") or "").strip()
            if status_filter:
                rows = [row for row in rows if row["tax_status"]["class"] == status_filter]
            return rows


def fetch_detection_detail(detection_id):
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, plate_number, plate_date, owner_name, source, image_url, threshold, created_at
                FROM detections
                WHERE id = %s
                LIMIT 1
                """,
                (detection_id,),
            )
            row = cursor.fetchone()

    if row is None:
        return None

    if isinstance(row.get("created_at"), datetime):
        row["created_at"] = row["created_at"].isoformat(timespec="seconds")
    row["tax_status"] = get_tax_status(row.get("plate_date"))
    row["vehicle"] = find_vehicle_by_plate(row.get("plate_number"))
    return row


def fetch_vehicles(limit=100, filters=None):
    filters = filters or {}
    where = []
    params = []
    query = (filters.get("q") or "").strip()
    if query:
        where.append("(plate_number LIKE %s OR owner_name LIKE %s)")
        like = f"%{query}%"
        params.extend([like, like])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT v.id, v.plate_key, v.plate_number, v.owner_name, v.plate_date,
                       v.created_at, v.updated_at,
                       COUNT(d.id) AS detection_count,
                       MAX(d.created_at) AS last_detected_at
                FROM vehicles v
                LEFT JOIN detections d
                    ON REPLACE(REPLACE(UPPER(d.plate_number), ' ', ''), '-', '') = v.plate_key
                {where_sql}
                GROUP BY v.id, v.plate_key, v.plate_number, v.owner_name, v.plate_date, v.created_at, v.updated_at
                ORDER BY v.updated_at DESC, v.id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()

    for row in rows:
        for key in ("created_at", "updated_at", "last_detected_at"):
            if isinstance(row.get(key), datetime):
                row[key] = row[key].isoformat(timespec="seconds")
        row["tax_status"] = get_tax_status(row.get("plate_date"))

    status_filter = (filters.get("tax_status") or "").strip()
    if status_filter:
        rows = [row for row in rows if row["tax_status"]["class"] == status_filter]
    return rows


def update_vehicle(vehicle_id, plate_number, owner_name, plate_date):
    plate_number = (plate_number or "").strip().upper()
    owner_name = (owner_name or "").strip()
    plate_date = (plate_date or "").strip()
    new_plate_key = normalize_plate_key(plate_number)

    if not new_plate_key:
        raise ValueError("Nomor plat wajib diisi")
    if not owner_name:
        raise ValueError("Nama pemilik wajib diisi")

    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM vehicles WHERE id = %s LIMIT 1", (vehicle_id,))
            existing = cursor.fetchone()
            if existing is None:
                return False

            old_plate_key = existing["plate_key"]
            cursor.execute(
                """
                UPDATE vehicles
                SET plate_key = %s, plate_number = %s, owner_name = %s, plate_date = %s, updated_at = %s
                WHERE id = %s
                """,
                (
                    new_plate_key,
                    plate_number,
                    owner_name,
                    plate_date if plate_date not in ("", "-") else None,
                    datetime.now(),
                    vehicle_id,
                ),
            )
            cursor.execute(
                """
                UPDATE detections
                SET plate_number = %s, owner_name = %s, plate_date = %s
                WHERE REPLACE(REPLACE(UPPER(plate_number), ' ', ''), '-', '') = %s
                """,
                (
                    plate_number,
                    owner_name,
                    plate_date if plate_date not in ("", "-") else None,
                    old_plate_key,
                ),
            )
    return True


def delete_vehicle(vehicle_id):
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT plate_key FROM vehicles WHERE id = %s LIMIT 1", (vehicle_id,))
            existing = cursor.fetchone()
            if existing is None:
                return False
            plate_key = existing["plate_key"]
            cursor.execute(
                "DELETE FROM detections WHERE REPLACE(REPLACE(UPPER(plate_number), ' ', ''), '-', '') = %s",
                (plate_key,),
            )
            cursor.execute("DELETE FROM vehicles WHERE id = %s", (vehicle_id,))
    return True


def update_detection(detection_id, plate_number, owner_name, plate_date):
    plate_number = (plate_number or "").strip().upper()
    owner_name = (owner_name or "").strip()
    plate_date = (plate_date or "").strip()

    if not plate_number:
        raise ValueError("Nomor plat wajib diisi")
    if not owner_name:
        raise ValueError("Nama pemilik wajib diisi")

    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, plate_number, source, image_url, threshold
                FROM detections
                WHERE id = %s
                LIMIT 1
                """,
                (detection_id,),
            )
            existing = cursor.fetchone()
            if existing is None:
                return False

            cursor.execute(
                """
                UPDATE detections
                SET plate_number = %s, owner_name = %s, plate_date = %s
                WHERE id = %s
                """,
                (
                    plate_number,
                    owner_name,
                    plate_date if plate_date not in ("", "-") else None,
                    detection_id,
                ),
            )

            old_plate_key = normalize_plate_key(existing["plate_number"])
            new_plate_key = normalize_plate_key(plate_number)
            if old_plate_key and old_plate_key != new_plate_key:
                cursor.execute(
                    "SELECT COUNT(*) AS total FROM detections WHERE REPLACE(REPLACE(UPPER(plate_number), ' ', ''), '-', '') = %s",
                    (old_plate_key,),
                )
                remaining = cursor.fetchone()["total"]
                if remaining == 0:
                    cursor.execute("DELETE FROM vehicles WHERE plate_key = %s", (old_plate_key,))

    sync_vehicle_from_detection(plate_number, owner_name, plate_date)
    return True


def delete_detection(detection_id):
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT plate_number FROM detections WHERE id = %s LIMIT 1",
                (detection_id,),
            )
            existing = cursor.fetchone()
            if existing is None:
                return False

            plate_key = normalize_plate_key(existing["plate_number"])
            cursor.execute("DELETE FROM detections WHERE id = %s", (detection_id,))
            cursor.execute(
                "SELECT COUNT(*) AS total FROM detections WHERE REPLACE(REPLACE(UPPER(plate_number), ' ', ''), '-', '') = %s",
                (plate_key,),
            )
            remaining = cursor.fetchone()["total"]
            if remaining == 0 and plate_key:
                cursor.execute("DELETE FROM vehicles WHERE plate_key = %s", (plate_key,))

    return True


def get_tax_status(plate_date):
    if not plate_date:
        return {
            "label": "Tidak Diketahui",
            "class": "unknown",
        }

    match = re.search(r'(\d{1,2})[-./](\d{2,4})', str(plate_date))
    if not match:
        return {
            "label": "Tidak Diketahui",
            "class": "unknown",
        }

    month = int(match.group(1))
    year = int(match.group(2))
    if year < 100:
        year += 2000

    if month < 1 or month > 12:
        return {
            "label": "Tidak Diketahui",
            "class": "unknown",
        }

    last_day = calendar.monthrange(year, month)[1]
    tax_expiry = datetime(year, month, last_day, 23, 59, 59)
    if datetime.now() > tax_expiry:
        return {
            "label": "Pajak Mati",
            "class": "expired",
        }

    return {
        "label": "Pajak Aktif",
        "class": "active",
    }


def fetch_detection_stats():
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM detections")
            total = cursor.fetchone()["total"]
            cursor.execute(
                "SELECT COUNT(*) AS total FROM detections WHERE DATE(created_at) = CURDATE()"
            )
            today = cursor.fetchone()["total"]
            cursor.execute("SELECT plate_number FROM detections ORDER BY id DESC LIMIT 1")
            latest = cursor.fetchone()
            return {
                "total": total,
                "today": today,
                "latest": latest["plate_number"] if latest else "-",
            }


class WebPlateDetector:
    def __init__(self, model_path, camera_index=0, threshold=0.55, interval_ms=700, cuda=False):
        self.model_path = model_path
        self.camera_index = camera_index
        self.threshold = threshold
        self.interval_ms = interval_ms
        self.cuda = cuda
        self.ocr_mode = "accurate"

        self.capture = None
        self.detector = None
        self.running = False
        self.detecting = False
        self.detection_requested = False
        self.loading_model = False
        self.live_worker_busy = False
        self.image_processing = False
        self.status = "Kamera belum aktif"
        self.backend_name = "-"
        self.plate_text = "-"
        self.plate_date = "-"
        self.vehicle_info = None
        self.registration_status = "idle"
        self.fps = 0.0
        self.frame_count = 0
        self.fps_started_at = time.time()
        self.last_inference_at = 0
        self.latest_frame = None
        self.latest_output_frame = None
        self.last_image_url = ""
        self.pending_detection = None
        self.camera_thread = None
        self.lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.detector_lock = threading.Lock()

    def configure(self, camera=None, threshold=None, interval=None, cuda=None, ocr_mode=None):
        with self.lock:
            if camera is not None and not self.running:
                self.camera_index = int(camera)
            if threshold is not None:
                self.threshold = float(threshold)
            if interval is not None:
                self.interval_ms = max(int(interval), 100)
            if cuda is not None and self.detector is None:
                self.cuda = bool(cuda)
            if ocr_mode in ("fast", "balanced", "accurate"):
                self.ocr_mode = ocr_mode

    def start_camera(self):
        with self.lock:
            if self.running:
                return True, self.status
            camera_index = self.camera_index

        capture, first_frame, backend_name = self._open_camera(camera_index)
        if capture is None:
            with self.lock:
                self.status = f"Kamera index {camera_index} gagal dibuka"
            return False, self.status

        with self.lock:
            self.capture = capture
            self.latest_frame = first_frame
            self.latest_output_frame = None
            self.plate_text = "-"
            self.plate_date = "-"
            self.vehicle_info = None
            self.registration_status = "idle"
            self.fps = 0.0
            self.backend_name = backend_name
            self.running = True
            self.status = f"Kamera aktif ({backend_name})"
            self.frame_count = 0
            self.fps_started_at = time.time()
            self.camera_thread = threading.Thread(target=self._camera_reader_loop, daemon=True)
            self.camera_thread.start()
        return True, self.status

    def _open_camera(self, camera_index):
        backends = [
            ("DirectShow", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("Default", cv2.CAP_ANY),
        ]

        for backend_name, backend in backends:
            capture = cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            self._apply_camera_settings(capture, camera_index)

            for _ in range(20):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    return capture, frame, backend_name
                time.sleep(0.03)

            capture.release()

        return None, None, None

    def _apply_camera_settings(self, capture, camera_index):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FPS, 30)

        if camera_index == 1:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def _camera_reader_loop(self):
        consecutive_failures = 0

        while True:
            with self.lock:
                capture = self.capture
                running = self.running

            if not running or capture is None:
                break

            frame = None
            with self.camera_lock:
                capture.grab()
                ok, grabbed_frame = capture.retrieve()
                if ok and grabbed_frame is not None and grabbed_frame.size:
                    frame = grabbed_frame

            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 8:
                    with self.lock:
                        self.status = "Kamera aktif, tapi frame USB belum stabil"
                time.sleep(0.04)
                continue

            consecutive_failures = 0
            self._schedule_detection(frame)
            self._update_fps()

            with self.lock:
                self.latest_frame = frame

            time.sleep(0.02)

    def list_cameras(self, max_index=8):
        cameras = []
        for index in range(max_index + 1):
            camera = self._probe_camera(index)
            if camera is not None:
                cameras.append(camera)

        selected_exists = any(camera["index"] == self.camera_index for camera in cameras)
        if not selected_exists:
            cameras.insert(0, {
                "index": self.camera_index,
                "label": f"Kamera {self.camera_index}",
                "available": False,
                "backend": "-",
                "resolution": "-",
            })

        return cameras

    def _probe_camera(self, camera_index):
        backends = [
            ("DirectShow", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("Default", cv2.CAP_ANY),
        ]

        for backend_name, backend in backends:
            capture = cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok_frame = False
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            for _ in range(6):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    ok_frame = True
                    height, width = frame.shape[:2]
                    break
                time.sleep(0.02)

            capture.release()
            if ok_frame:
                resolution = f"{width}x{height}" if width and height else "-"
                return {
                    "index": camera_index,
                    "label": f"Kamera {camera_index}",
                    "available": True,
                    "backend": backend_name,
                    "resolution": resolution,
                }

        return None

    def stop_camera(self):
        with self.lock:
            self.detecting = False
            self.detection_requested = False
            self.running = False
            capture = self.capture
            self.capture = None
            self.latest_frame = None
            self.latest_output_frame = None
            self.camera_thread = None
            self.status = "Kamera berhenti"
            self.vehicle_info = None
            self.registration_status = "idle"

        if capture is not None:
            with self.camera_lock:
                capture.release()

    def start_detection(self):
        with self.lock:
            if not self.running:
                return False, "Mulai kamera dulu"
            if self.image_processing:
                return False, "Tunggu proses gambar selesai"
            self.detection_requested = True
            if self.detecting:
                return True, self.status
            if self.loading_model:
                return True, "Model sedang dimuat"
            if self.detector is not None:
                self.detecting = True
                self.status = "Deteksi berjalan"
                return True, self.status
            self.loading_model = True
            self.status = "Memuat model deteksi dan OCR..."

        threading.Thread(target=self._load_model_worker, daemon=True).start()
        return True, "Memuat model deteksi dan OCR..."

    def _load_model_worker(self):
        try:
            detector = self._build_detector()
            with self.lock:
                self.detector = detector
                self.detecting = self.running and self.detection_requested
                self.loading_model = False
                self.status = "Model siap, deteksi berjalan" if self.detecting else "Model siap"
        except Exception as exc:
            with self.lock:
                self.loading_model = False
                self.status = f"Gagal memuat model: {exc}"

    def _build_detector(self):
        from plate_recognition import PlateRecognition
        from super_resolution import SuperResolution

        enhancer = SuperResolution()
        return PlateRecognition(self.model_path, enhancer, self.cuda)

    def _ensure_detector(self):
        should_load = False

        with self.lock:
            if self.detector is not None:
                return self.detector
            if not self.loading_model:
                self.loading_model = True
                self.status = "Memuat model deteksi dan OCR..."
                should_load = True

        if not should_load:
            while True:
                time.sleep(0.2)
                with self.lock:
                    if self.detector is not None:
                        return self.detector
                    if not self.loading_model:
                        raise RuntimeError("Model belum siap")

        try:
            detector = self._build_detector()
            with self.lock:
                self.detector = detector
                self.loading_model = False
                self.status = "Model siap"
            return detector
        except Exception:
            with self.lock:
                self.loading_model = False
            raise

    def stop_detection(self):
        with self.lock:
            self.detection_requested = False
            self.detecting = False
            if self.running:
                self.status = "Preview kamera aktif"

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "detecting": self.detecting,
                "loadingModel": self.loading_model,
                "processingImage": self.image_processing,
                "camera": self.camera_index,
                "threshold": self.threshold,
                "interval": self.interval_ms,
                "ocrMode": self.ocr_mode,
                "cuda": self.cuda,
                "detectorLoaded": self.detector is not None,
                "backend": self.backend_name,
                "status": self.status,
                "plate": self.plate_text,
                "date": self.plate_date,
                "vehicle": self.vehicle_info,
                "registrationStatus": self.registration_status,
                "fps": round(self.fps, 1),
                "lastImageUrl": self.last_image_url,
                "hasPendingDetection": self.pending_detection is not None,
            }

    def frames(self):
        while True:
            frame = self._next_frame()
            if frame is None:
                frame = self._placeholder_frame()

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.015)

    def _next_frame(self):
        with self.lock:
            running = self.running
            source = self.latest_frame
            output = self.latest_output_frame if self.detecting and self.latest_output_frame is not None else source
            frame = output.copy() if running and output is not None else None
        return frame

    def _schedule_detection(self, frame):
        with self.lock:
            if self.image_processing or not self.detecting or self.detector is None or self.live_worker_busy:
                return
            now = time.time()
            if now - self.last_inference_at < self.interval_ms / 1000:
                return
            self.live_worker_busy = True
            self.last_inference_at = now
            threshold = self.threshold
            detector = self.detector

        threading.Thread(target=self._detect_worker, args=(detector, frame.copy(), threshold), daemon=True).start()

    def _detect_worker(self, detector, frame, threshold):
        try:
            with self.detector_lock:
                result = detector.anpr(frame, threshold, ocr_mode="fast")
            if len(result) == 3:
                output, plate_text, plate_date = result
            else:
                output, plate_text = result
                plate_date = "-"

            vehicle, registration_status = self._lookup_vehicle(plate_text)
            with self.lock:
                self.latest_output_frame = output
                self.plate_text = plate_text or "-"
                self.plate_date = plate_date or "-"
                self.vehicle_info = vehicle
                self.registration_status = registration_status
                if vehicle:
                    self.status = f"Plat terdaftar: {vehicle['owner_name']}"
                elif self._has_plate_text(plate_text):
                    self.status = "Plat belum terdaftar"
        except Exception as exc:
            with self.lock:
                self.status = f"Deteksi error: {exc}"
        finally:
            with self.lock:
                self.live_worker_busy = False

    def capture_frame(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is None:
            return False, {"ok": False, "message": "Frame kamera belum tersedia"}

        data = self.process_image(frame, "capture")
        return bool(data.get("ok")), data

    def process_upload(self, file_storage):
        data = file_storage.read()
        encoded = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or not frame.size:
            return False, {"ok": False, "message": "File gambar tidak bisa dibaca"}

        data = self.process_image(frame, "upload")
        return bool(data.get("ok")), data

    def process_image(self, frame, source):
        detector = self._ensure_detector()
        already_processing = False
        with self.lock:
            if self.image_processing:
                already_processing = True
            else:
                self.image_processing = True
                threshold = self.threshold
                ocr_mode = self.ocr_mode
                self.status = f"Memproses gambar {source}..."

        if already_processing:
            return {
                **self.get_status(),
                "ok": False,
                "message": "Masih memproses gambar sebelumnya",
            }

        try:
            with self.detector_lock:
                result = detector.anpr(frame, threshold, ocr_mode=ocr_mode)
                if len(result) == 3:
                    output, plate_text, plate_date = result
                else:
                    output, plate_text = result
                    plate_date = "-"

                if not self._has_confident_plate(detector, plate_text, plate_date, ocr_mode):
                    retry_threshold = min(threshold, 0.25)
                    result = detector.anpr(frame, retry_threshold, ocr_mode=ocr_mode)
                    if len(result) == 3:
                        output, plate_text, plate_date = result
                    else:
                        output, plate_text = result
                        plate_date = "-"

                if ocr_mode != "fast" and not self._has_confident_plate(detector, plate_text, plate_date, ocr_mode):
                    fallback_plate, fallback_date = detector.fallback_recognition(frame, ocr_mode=ocr_mode)
                    if fallback_plate or fallback_date:
                        output = frame.copy()
                        plate_text = fallback_plate or "-"
                        plate_date = fallback_date or "-"
                        self._draw_fallback_result(output, plate_text, plate_date)

            filename = f"{source}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.jpg"
            output_path = OUTPUT_DIR / filename
            cv2.imwrite(str(output_path), output)
            image_url = f"/outputs/{filename}"
            vehicle, registration_status = self._lookup_vehicle(plate_text)

            with self.lock:
                if source == "capture":
                    self.latest_output_frame = output
                self.plate_text = plate_text or "-"
                self.plate_date = plate_date or "-"
                self.vehicle_info = vehicle
                self.registration_status = registration_status
                self.last_image_url = image_url
                if vehicle:
                    self.status = f"Plat terdaftar: {vehicle['owner_name']}"
                elif self._has_plate_text(plate_text):
                    self.status = "Plat belum terdaftar, lengkapi data pemilik"
                else:
                    self.status = "Gambar selesai diproses"

            pending = None
            if source in ("capture", "upload") and self._has_plate_text(plate_text) and vehicle is None:
                pending = {
                    "plate_number": plate_text,
                    "plate_date": plate_date,
                    "source": source,
                    "image_url": image_url,
                    "threshold": threshold,
                    "ocr_mode": ocr_mode,
                }

            response = {
                "ok": True,
                "message": "Gambar selesai diproses",
                "imageUrl": image_url,
                "saved": False,
            }
            with self.lock:
                self.pending_detection = pending
        finally:
            with self.lock:
                self.image_processing = False

        return {
            **self.get_status(),
            **response,
        }

    def save_pending_detection(self, owner_name=None, plate_date=None, plate_number=None):
        with self.lock:
            pending = dict(self.pending_detection) if self.pending_detection else None

        if pending is None:
            return False, {
                **self.get_status(),
                "ok": False,
                "message": "Tidak ada hasil valid untuk ditambahkan",
            }

        owner_name = (owner_name or "").strip()
        if not owner_name:
            return False, {
                **self.get_status(),
                "ok": False,
                "message": "Nama pemilik wajib diisi",
            }

        selected_plate = (plate_number or pending["plate_number"]).strip().upper()
        if not selected_plate:
            return False, {
                **self.get_status(),
                "ok": False,
                "message": "Nomor plat wajib diisi",
            }

        selected_date = plate_date if plate_date not in (None, "") else pending["plate_date"]
        vehicle = save_vehicle(selected_plate, owner_name, selected_date)
        save_detection(
            selected_plate,
            selected_date,
            pending["source"],
            pending["image_url"],
            pending["threshold"],
            owner_name=owner_name,
        )

        with self.lock:
            self.pending_detection = None
            self.vehicle_info = vehicle
            self.registration_status = "registered"
            self.plate_text = selected_plate
            self.plate_date = selected_date or "-"
            self.status = "Data kendaraan berhasil ditambahkan"

        return True, {
            **self.get_status(),
            "ok": True,
            "message": "Data kendaraan berhasil ditambahkan",
            "saved": True,
        }

    def _lookup_vehicle(self, plate_text):
        if not self._has_plate_text(plate_text):
            return None, "idle"
        vehicle = find_vehicle_by_plate(plate_text)
        if vehicle is not None:
            return vehicle, "registered"
        return None, "unregistered"

    def _has_plate_text(self, plate_text):
        value = (plate_text or "").strip().lower()
        return value not in ("", "-", "not detected")

    def _has_confident_plate(self, detector, plate_text, plate_date, ocr_mode):
        if not self._has_plate_text(plate_text):
            return False
        valid_date = plate_date if plate_date not in ("", "-") else ""
        score = detector.score_plate_candidate(plate_text, valid_date, confidence=1.0)
        return detector.is_confident_ocr(plate_text, valid_date, score, ocr_mode)

    def _draw_fallback_result(self, frame, plate_text, plate_date):
        label = plate_text if plate_date in ("", "-") else f"{plate_text} | {plate_date}"
        cv2.rectangle(frame, (16, 16), (min(frame.shape[1] - 16, 760), 76), (0, 0, 0), -1)
        cv2.putText(frame, label, (28, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    def _update_fps(self):
        with self.lock:
            self.frame_count += 1
            elapsed = time.time() - self.fps_started_at
            if elapsed >= 1:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_started_at = time.time()

    def _placeholder_frame(self):
        frame = cv2.UMat(720, 1280, cv2.CV_8UC3).get()
        frame[:] = (9, 12, 16)
        cv2.putText(frame, "Tekan Mulai Kamera", (390, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 229, 238), 2)
        return frame


try:
    init_db()
except pymysql.MySQLError as exc:
    print(f"MySQL connection failed: {exc}")
    print(
        "Set PLATE_DB_HOST, PLATE_DB_PORT, PLATE_DB_USER, "
        "PLATE_DB_PASSWORD, and PLATE_DB_NAME before running web_app.py."
    )
    raise


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_mysql_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("index"))

        return render_template("login.html", error="Username atau password salah")

    if "user_id" in session:
        return redirect(url_for("index"))
    return render_template("login.html", error="")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        username=session.get("username", "admin"),
        stats=fetch_detection_stats(),
    )


@app.route("/history")
@login_required
def history_page():
    filters = {
        "q": request.args.get("q", ""),
        "source": request.args.get("source", ""),
        "tax_status": request.args.get("tax_status", ""),
        "date_from": request.args.get("date_from", ""),
        "date_to": request.args.get("date_to", ""),
    }
    return render_template(
        "history.html",
        username=session.get("username", "admin"),
        stats=fetch_detection_stats(),
        detections=fetch_recent_detections(limit=100, filters=filters),
        filters=filters,
    )


@app.route("/vehicles")
@login_required
def vehicles_page():
    filters = {
        "q": request.args.get("q", ""),
        "tax_status": request.args.get("tax_status", ""),
    }
    return render_template(
        "vehicles.html",
        username=session.get("username", "admin"),
        stats=fetch_detection_stats(),
        vehicles=fetch_vehicles(limit=100, filters=filters),
        filters=filters,
    )


@app.route("/detections/<int:detection_id>")
@login_required
def detection_detail_page(detection_id):
    detection = fetch_detection_detail(detection_id)
    if detection is None:
        return redirect(url_for("history_page"))
    return render_template(
        "detail.html",
        username=session.get("username", "admin"),
        detection=detection,
    )


@app.route("/video")
@login_required
def video():
    return Response(detector_state.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/outputs/<path:filename>")
@login_required
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/status")
@login_required
def status():
    return jsonify(detector_state.get_status())


@app.route("/api/cameras")
@login_required
def cameras():
    return jsonify({
        "selected": detector_state.get_status()["camera"],
        "cameras": detector_state.list_cameras(),
    })


@app.route("/api/history")
@login_required
def history():
    return jsonify({
        "stats": fetch_detection_stats(),
        "detections": fetch_recent_detections(),
    })


@app.route("/api/config", methods=["POST"])
@login_required
def config():
    data = request.get_json(force=True) or {}
    detector_state.configure(
        camera=data.get("camera"),
        threshold=data.get("threshold"),
        interval=data.get("interval"),
        ocr_mode=data.get("ocrMode"),
        cuda=data.get("cuda"),
    )
    return jsonify(detector_state.get_status())


@app.route("/api/camera/start", methods=["POST"])
@login_required
def camera_start():
    data = request.get_json(silent=True) or {}
    detector_state.configure(
        camera=data.get("camera"),
        threshold=data.get("threshold"),
        interval=data.get("interval"),
        ocr_mode=data.get("ocrMode"),
        cuda=data.get("cuda"),
    )
    ok, message = detector_state.start_camera()
    return jsonify({**detector_state.get_status(), "ok": ok, "message": message})


@app.route("/api/camera/stop", methods=["POST"])
@login_required
def camera_stop():
    detector_state.stop_camera()
    return jsonify(detector_state.get_status())


@app.route("/api/detection/start", methods=["POST"])
@login_required
def detection_start():
    data = request.get_json(silent=True) or {}
    detector_state.configure(
        threshold=data.get("threshold"),
        interval=data.get("interval"),
        ocr_mode=data.get("ocrMode"),
        cuda=data.get("cuda"),
    )
    ok, message = detector_state.start_detection()
    return jsonify({**detector_state.get_status(), "ok": ok, "message": message})


@app.route("/api/detection/stop", methods=["POST"])
@login_required
def detection_stop():
    detector_state.stop_detection()
    return jsonify(detector_state.get_status())


@app.route("/api/capture", methods=["POST"])
@login_required
def capture():
    data = request.get_json(silent=True) or {}
    detector_state.configure(
        threshold=data.get("threshold"),
        interval=data.get("interval"),
        ocr_mode=data.get("ocrMode"),
    )
    ok, data = detector_state.capture_frame()
    return jsonify(data), 200 if ok else 400


@app.route("/api/upload", methods=["POST"])
@login_required
def upload():
    image = request.files.get("image")
    if image is None or image.filename == "":
        return jsonify({"ok": False, "message": "Pilih file gambar dulu"}), 400

    detector_state.configure(
        threshold=request.form.get("threshold"),
        interval=request.form.get("interval"),
        ocr_mode=request.form.get("ocrMode"),
    )
    ok, data = detector_state.process_upload(image)
    return jsonify(data), 200 if ok else 400


@app.route("/api/detections/add", methods=["POST"])
@login_required
def add_detection():
    data = request.get_json(silent=True) or {}
    ok, data = detector_state.save_pending_detection(
        plate_number=data.get("plateNumber"),
        owner_name=data.get("ownerName"),
        plate_date=data.get("plateDate"),
    )
    return jsonify(data), 200 if ok else 400


@app.route("/api/detections/<int:detection_id>", methods=["PUT"])
@login_required
def edit_detection(detection_id):
    data = request.get_json(silent=True) or {}
    try:
        updated = update_detection(
            detection_id,
            data.get("plateNumber"),
            data.get("ownerName"),
            data.get("plateDate"),
        )
    except ValueError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400

    if not updated:
        return jsonify({"ok": False, "message": "Data tidak ditemukan"}), 404

    return jsonify({
        "ok": True,
        "message": "Data berhasil diperbarui",
        "stats": fetch_detection_stats(),
        "detections": fetch_recent_detections(limit=100),
    })


@app.route("/api/detections/<int:detection_id>", methods=["DELETE"])
@login_required
def remove_detection(detection_id):
    deleted = delete_detection(detection_id)
    if not deleted:
        return jsonify({"ok": False, "message": "Data tidak ditemukan"}), 404

    return jsonify({
        "ok": True,
        "message": "Data berhasil dihapus",
        "stats": fetch_detection_stats(),
        "detections": fetch_recent_detections(limit=100),
    })


@app.route("/api/vehicles/<int:vehicle_id>", methods=["PUT"])
@login_required
def edit_vehicle(vehicle_id):
    data = request.get_json(silent=True) or {}
    try:
        updated = update_vehicle(
            vehicle_id,
            data.get("plateNumber"),
            data.get("ownerName"),
            data.get("plateDate"),
        )
    except (ValueError, pymysql.err.IntegrityError) as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400

    if not updated:
        return jsonify({"ok": False, "message": "Data kendaraan tidak ditemukan"}), 404

    return jsonify({
        "ok": True,
        "message": "Data kendaraan berhasil diperbarui",
        "stats": fetch_detection_stats(),
        "vehicles": fetch_vehicles(limit=100),
    })


@app.route("/api/vehicles/<int:vehicle_id>", methods=["DELETE"])
@login_required
def remove_vehicle(vehicle_id):
    deleted = delete_vehicle(vehicle_id)
    if not deleted:
        return jsonify({"ok": False, "message": "Data kendaraan tidak ditemukan"}), 404

    return jsonify({
        "ok": True,
        "message": "Data kendaraan berhasil dihapus",
        "stats": fetch_detection_stats(),
        "vehicles": fetch_vehicles(limit=100),
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Web UI untuk deteksi plat nomor.")
    parser.add_argument("--host", default="127.0.0.1", help="Host web server.")
    parser.add_argument("--port", type=int, default=5000, help="Port web server.")
    parser.add_argument("--model", default="./model/best.onnx", help="Path model ONNX.")
    parser.add_argument("--camera", type=int, default=0, help="Index kamera laptop/webcam.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Threshold awal deteksi.")
    parser.add_argument("--interval", type=int, default=700, help="Jeda inferensi dalam milidetik.")
    parser.add_argument("--cuda", action="store_true", help="Gunakan CUDA/GPU.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector_state = WebPlateDetector(
        model_path=args.model,
        camera_index=args.camera,
        threshold=args.threshold,
        interval_ms=args.interval,
        cuda=args.cuda,
    )
    app.run(host=args.host, port=args.port, threaded=True)
