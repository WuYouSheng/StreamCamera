import grpc
import cv2
import time
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import uuid
import signal
import sys
import traceback
import gc
import weakref
import numpy as np
from ultralytics import YOLO

# 使用您現有的 protobuf 檔案
import simple_video_pb2
import simple_video_pb2_grpc


class YOLOv8PoseProcessor:
    """YOLOv8 骨架偵測處理器"""

    def __init__(self, model_path='yolov8n-pose.pt'):
        print("正在載入 YOLOv8 模型...")
        import torch
        self.device = 'cpu'
        self.cuda_reason = ''
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"已啟用 CUDA 加速 (NVIDIA 顯示卡: {torch.cuda.get_device_name(0)})")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
                print("已啟用 MPS 加速 (Apple M 系列處理器)")
            else:
                if not torch.version.cuda:
                    self.cuda_reason = "PyTorch 未安裝 CUDA 版本"
                elif torch.cuda.device_count() == 0:
                    self.cuda_reason = "找不到任何 NVIDIA 顯卡"
                else:
                    self.cuda_reason = "未知原因，請檢查 CUDA 驅動與安裝"
                print(f"使用 CPU 運算，原因：{self.cuda_reason}")
        except Exception as e:
            self.cuda_reason = f"初始化 CUDA 檢查時發生錯誤: {e}"
            print(f"使用 CPU 運算，原因：{self.cuda_reason}")
        self.model = YOLO(model_path)
        # 不需要 self.model.to(self.device)

        # 骨架點連線定義 (COCO 格式)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 頭部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (5, 11), (6, 12), (11, 12),  # 軀幹
            (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
        ]

        # 骨架點顏色 (BGR 格式)
        self.keypoint_colors = [
            (0, 0, 255),  # 鼻子 - 紅色
            (0, 255, 0),  # 眼睛 - 綠色
            (0, 255, 0),  # 眼睛 - 綠色
            (255, 0, 0),  # 耳朵 - 藍色
            (255, 0, 0),  # 耳朵 - 藍色
            (0, 255, 255),  # 肩膀 - 黃色
            (0, 255, 255),  # 肩膀 - 黃色
            (255, 0, 255),  # 手肘 - 紫色
            (255, 0, 255),  # 手肘 - 紫色
            (128, 0, 128),  # 手腕 - 紫色
            (128, 0, 128),  # 手腕 - 紫色
            (0, 128, 255),  # 臀部 - 橘色
            (0, 128, 255),  # 臀部 - 橘色
            (255, 128, 0),  # 膝蓋 - 青色
            (255, 128, 0),  # 膝蓋 - 青色
            (128, 255, 0),  # 腳踝 - 淺綠色
            (128, 255, 0),  # 腳踝 - 淺綠色
        ]

        print("YOLOv8 模型載入完成")

    def draw_skeleton(self, frame, keypoints, confidence_threshold=0.5):
        """
        繪製骨架

        Args:
            frame: 輸入影像
            keypoints: 骨架關鍵點
            confidence_threshold: 信心度門檻
        """
        # 繪製骨架連線
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints)):
                x1, y1, conf1 = keypoints[pt1_idx]
                x2, y2, conf2 = keypoints[pt2_idx]

                # 只有當兩個點的信心度都足夠時才繪製連線
                if conf1 > confidence_threshold and conf2 > confidence_threshold:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                             (255, 255, 255), 2)

        # 繪製關鍵點
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > confidence_threshold:
                color = self.keypoint_colors[i] if i < len(self.keypoint_colors) else (255, 255, 255)
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 0), 1)

    def process_frame(self, frame):
        """
        處理單一幀影像，進行骨架偵測

        Args:
            frame: 輸入影像

        Returns:
            processed_frame: 處理後的影像
            person_count: 偵測到的人數
        """
        try:
            # 使用 YOLOv8 進行姿勢偵測，指定 device
            results = self.model(frame, device=self.device, verbose=False)

            # 複製影像以進行標注
            processed_frame = frame.copy()
            person_count = 0

            for result in results:
                # 檢查是否有姿勢偵測結果
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()

                    for person_keypoints in keypoints:
                        person_count += 1

                        # 繪製骨架
                        self.draw_skeleton(processed_frame, person_keypoints)

                        # 計算邊界框 (基於可見的關鍵點)
                        visible_points = person_keypoints[person_keypoints[:, 2] > 0.5]
                        if len(visible_points) > 0:
                            x_min = int(np.min(visible_points[:, 0]))
                            y_min = int(np.min(visible_points[:, 1]))
                            x_max = int(np.max(visible_points[:, 0]))
                            y_max = int(np.max(visible_points[:, 1]))

                            # 繪製邊界框
                            cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max),
                                          (0, 255, 0), 2)

            # 顯示偵測到的人數
            cv2.putText(processed_frame, f'People: {person_count}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return processed_frame, person_count

        except Exception as e:
            print(f"YOLO 處理錯誤: {e}")
            return frame, 0


class MemorySafeVideoSender:
    def __init__(self):
        self.config = self.load_config()
        self.client_id = str(uuid.uuid4())[:8]  # 短一點的ID
        self.streaming = False
        self.cap = None
        self.channel = None
        self.stub = None
        self.stream_thread = None
        self.generator_active = False
        self.frame_buffer = None

        # 初始化 YOLO 處理器
        self.yolo_processor = None
        self.yolo_enabled = True  # 控制是否啟用 YOLO

        # 更精準的 FPS 計算
        self.frame_count = 0
        self.fps_start_time = None
        self.current_fps = 0.0
        self.fps_lock = threading.Lock()
        self.recent_frame_times = []  # 存儲最近的幀時間戳
        self.max_fps_samples = 60  # 計算 FPS 的樣本數量

    def load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 針對 YOLO 處理調整設定
                config['camera']['fps'] = min(config['camera'].get('fps', 8), 10)  # 限制最大 FPS
                return config
        except:
            return {
                "client": {"target_host": "127.0.0.1", "target_port": 50051},
                "camera": {"device_id": 0, "width": 640, "height": 480, "fps": 8},  # 提高解析度以改善 YOLO 效果
                "video": {"quality": 70}  # 提高品質以保持 YOLO 結果清晰度
            }

    def initialize_yolo(self):
        """初始化 YOLO 處理器"""
        try:
            if self.yolo_enabled and self.yolo_processor is None:
                print("正在初始化 YOLOv8 骨架偵測...")
                self.yolo_processor = YOLOv8PoseProcessor()
                print("YOLOv8 初始化完成")
                return True
        except Exception as e:
            print(f"YOLO 初始化失敗: {e}")
            self.yolo_enabled = False
            return False
        return True

    def connect_to_server(self):
        try:
            target = f"{self.config['client']['target_host']}:{self.config['client']['target_port']}"
            options = [
                ('grpc.keepalive_time_ms', 60000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.max_send_message_length', 8 * 1024 * 1024),  # 8MB (因為 YOLO 處理後的影像可能較大)
                ('grpc.max_receive_message_length', 8 * 1024 * 1024),
            ]
            self.channel = grpc.insecure_channel(target, options=options)
            self.stub = simple_video_pb2_grpc.VideoStreamServiceStub(self.channel)

            grpc.channel_ready_future(self.channel).result(timeout=5)
            print("gRPC 連接成功")
            return True
        except Exception as e:
            print(f"連接錯誤: {e}")
            return False

    def start_camera(self):
        try:
            # 清理之前的資源
            if self.cap:
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
                gc.collect()  # 強制垃圾回收

            # 嘗試開啟攝影機
            self.cap = cv2.VideoCapture(self.config['camera']['device_id'])
            if not self.cap.isOpened():
                raise Exception("無法開啟攝影機")

            # 設定攝影機參數
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 預分配緩衝區
            expected_size = self.config['camera']['width'] * self.config['camera']['height'] * 3
            self.frame_buffer = bytearray(expected_size)

            # 測試讀取
            for i in range(3):
                ret, frame = self.cap.read()
                if ret:
                    break
                time.sleep(0.1)

            if not ret:
                raise Exception("攝影機無法讀取影像")

            # 立即釋放測試幀
            del frame
            gc.collect()

            print("攝影機初始化成功")
            return True

        except Exception as e:
            print(f"攝影機錯誤: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def update_fps_calculation(self):
        """更精準的 FPS 計算方式"""
        current_time = time.time()

        with self.fps_lock:
            # 添加當前時間戳
            self.recent_frame_times.append(current_time)

            # 只保留最近的樣本
            if len(self.recent_frame_times) > self.max_fps_samples:
                self.recent_frame_times.pop(0)

            # 計算 FPS（基於最近幀之間的平均時間間隔）
            if len(self.recent_frame_times) >= 2:
                time_span = self.recent_frame_times[-1] - self.recent_frame_times[0]
                frame_intervals = len(self.recent_frame_times) - 1

                if time_span > 0:
                    self.current_fps = frame_intervals / time_span
                else:
                    self.current_fps = 0.0
            else:
                self.current_fps = 0.0

            self.frame_count += 1

    def safe_message_generator(self):
        """記憶體安全的訊息生成器 (整合 YOLO 處理) - 適配現有協議"""
        try:
            self.generator_active = True

            # 重置 FPS 計算
            with self.fps_lock:
                self.frame_count = 0
                self.fps_start_time = time.time()
                self.current_fps = 0.0
                self.recent_frame_times = []  # 重置時間戳列表

            # 註冊訊息 - 使用您現有的協議格式
            register_msg = simple_video_pb2.VideoMessage()
            register_msg.client.client_type = "sender"
            register_msg.client.client_id = self.client_id
            yield register_msg

            yolo_status = "啟用" if self.yolo_enabled else "停用"
            print(f"已註冊為傳送端，開始發送影像 (YOLO: {yolo_status})...")

            frame_count = 0
            consecutive_failures = 0
            max_failures = 5
            last_gc_time = time.time()

            # 編碼參數
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config['video']['quality']]

            while self.streaming and self.generator_active:
                try:
                    current_time = time.time()

                    # 定期垃圾回收
                    if current_time - last_gc_time > 30:  # 每30秒清理一次
                        gc.collect()
                        last_gc_time = current_time
                        print("執行記憶體清理")

                    if not self.cap or not self.cap.isOpened():
                        print("攝影機連接丟失")
                        break

                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            print("連續讀取失敗，停止串流")
                            break
                        time.sleep(0.2)
                        continue

                    consecutive_failures = 0

                    # 確保幀尺寸正確
                    target_height = self.config['camera']['height']
                    target_width = self.config['camera']['width']

                    if frame.shape[:2] != (target_height, target_width):
                        frame = cv2.resize(frame, (target_width, target_height))

                    # YOLO 處理
                    if self.yolo_enabled and self.yolo_processor:
                        try:
                            processed_frame, person_count = self.yolo_processor.process_frame(frame)

                            # 添加 YOLO 狀態資訊
                            cv2.putText(processed_frame, f'YOLO: ON',
                                        (target_width - 80, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 1)

                            frame = processed_frame

                        except Exception as e:
                            print(f"YOLO 處理錯誤: {e}")
                            # 如果 YOLO 處理失敗，使用原始幀
                            cv2.putText(frame, f'YOLO: ERROR',
                                        (target_width - 100, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 255), 1)
                    else:
                        # 顯示 YOLO 關閉狀態
                        cv2.putText(frame, f'YOLO: OFF',
                                    (target_width - 80, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (128, 128, 128), 1)

                    # 編碼影像
                    success, buffer = cv2.imencode('.jpg', frame, encode_param)

                    # 立即釋放原始幀記憶體
                    del frame

                    if not success:
                        print("影像編碼失敗")
                        continue

                    frame_count += 1

                    # 更新精準的 FPS 計算（只在成功發送時計算）
                    self.update_fps_calculation()

                    # 創建訊息 - 使用您現有的協議格式
                    frame_msg = simple_video_pb2.VideoMessage()
                    frame_msg.frame.data = buffer.tobytes()
                    frame_msg.frame.timestamp = int(time.time() * 1000)
                    frame_msg.frame.width = target_width
                    frame_msg.frame.height = target_height

                    # 立即釋放 buffer
                    del buffer

                    yield frame_msg

                    # 立即釋放訊息
                    del frame_msg

                    # 輸出進度
                    if frame_count % 100 == 0:
                        with self.fps_lock:
                            print(f"已發送 {frame_count} 幀，當前 FPS: {self.current_fps:.2f}")

                    # 控制幀率
                    time.sleep(1.0 / self.config['camera']['fps'])

                    # 每發送一定數量的幀就進行小清理
                    if frame_count % 50 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"生成訊息錯誤: {e}")
                    time.sleep(0.5)
                    continue

        except Exception as e:
            print(f"訊息生成器錯誤: {e}")
            traceback.print_exc()
        finally:
            self.generator_active = False
            gc.collect()  # 最終清理
            print("訊息生成器已結束")

    def get_current_fps(self):
        """獲取當前 FPS"""
        with self.fps_lock:
            return self.current_fps, self.frame_count

    def get_yolo_stats(self):
        """獲取 YOLO 統計資訊"""
        cuda_reason = ""
        device = 'cpu'
        if self.yolo_processor:
            device = getattr(self.yolo_processor, 'device', 'cpu')
            cuda_reason = getattr(self.yolo_processor, 'cuda_reason', '')
        return {
            'enabled': self.yolo_enabled,
            'device': device,
            'cuda_reason': cuda_reason
        }

    def toggle_yolo(self):
        """切換 YOLO 狀態"""
        if self.streaming:
            print("串流進行中，無法切換 YOLO 狀態")
            return False

        self.yolo_enabled = not self.yolo_enabled
        if self.yolo_enabled:
            return self.initialize_yolo()
        else:
            self.yolo_processor = None
            gc.collect()
            return True

    def start_streaming(self):
        # 初始化 YOLO (如果啟用)
        if self.yolo_enabled:
            if not self.initialize_yolo():
                print("YOLO 初始化失敗，將以一般模式執行")

        if not self.connect_to_server():
            return False

        if not self.start_camera():
            return False

        self.streaming = True

        def stream_worker():
            try:
                print("開始 gRPC 串流...")
                responses = self.stub.StreamVideo(self.safe_message_generator())

                response_count = 0
                for response in responses:
                    if not self.streaming:
                        break

                    response_count += 1

                    # 處理伺服器回應 - 適配您現有的協議
                    if response.HasField('status'):
                        if not response.status.success:
                            print(f"伺服器錯誤: {response.status.message}")
                        elif response_count % 100 == 0:
                            print(f"已收到 {response_count} 個回應")

                    # 釋放回應物件
                    del response

                    # 定期清理
                    if response_count % 50 == 0:
                        gc.collect()

            except grpc.RpcError as e:
                if self.streaming:
                    print(f"gRPC 錯誤: {e.code()}")
            except Exception as e:
                if self.streaming:
                    print(f"串流錯誤: {e}")
            finally:
                print("串流工作線程結束")
                self.generator_active = False
                gc.collect()

        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()
        return True

    def stop_streaming(self):
        print("正在停止串流...")
        self.streaming = False
        self.generator_active = False

        # 重置 FPS 計算
        with self.fps_lock:
            self.frame_count = 0
            self.current_fps = 0.0
            self.fps_start_time = None
            self.recent_frame_times = []  # 重置時間戳列表

        # 等待線程結束
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=3)

        # 釋放攝影機
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
                print("攝影機已釋放")
            except Exception as e:
                print(f"釋放攝影機錯誤: {e}")

        # 關閉 gRPC 連接
        if self.channel:
            try:
                self.channel.close()
                self.channel = None
                print("gRPC 連接已關閉")
            except Exception as e:
                print(f"關閉連接錯誤: {e}")

        # 清理緩衝區
        if self.frame_buffer:
            self.frame_buffer = None

        # 強制垃圾回收
        gc.collect()
        print("串流已完全停止")


class MemorySafeVideoSenderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOv8 骨架偵測影像傳送端")
        self.root.geometry("700x650")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 設定信號處理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.sender = MemorySafeVideoSender()
        self.preview_active = False
        self.preview_ref = None  # 使用弱引用

        self.setup_ui()

        # 定期記憶體清理
        self.schedule_memory_cleanup()

    def schedule_memory_cleanup(self):
        """定期記憶體清理"""
        gc.collect()
        self.root.after(60000, self.schedule_memory_cleanup)  # 每分鐘清理一次

    def signal_handler(self, signum, frame):
        print(f"收到信號 {signum}，正在安全關閉...")
        self.on_closing()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 設定資訊
        config_frame = ttk.LabelFrame(main_frame, text="設定資訊", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))

        config_text = f"伺服器: {self.sender.config['client']['target_host']}:{self.sender.config['client']['target_port']}\n"
        config_text += f"攝影機: {self.sender.config['camera']['width']}x{self.sender.config['camera']['height']} @ {self.sender.config['camera']['fps']}fps\n"
        config_text += f"品質: {self.sender.config['video']['quality']}%\n"
        config_text += f"YOLOv8 骨架偵測: {'啟用' if self.sender.yolo_enabled else '停用'}"
        ttk.Label(config_frame, text=config_text).pack(anchor=tk.W)

        # YOLO 控制
        yolo_frame = ttk.LabelFrame(main_frame, text="YOLO 控制", padding="10")
        yolo_frame.pack(fill=tk.X, pady=(0, 10))

        self.yolo_status_var = tk.StringVar(value=f"YOLO 狀態: {'啟用' if self.sender.yolo_enabled else '停用'}")
        ttk.Label(yolo_frame, textvariable=self.yolo_status_var).pack(side=tk.LEFT)

        self.toggle_yolo_btn = ttk.Button(yolo_frame, text="切換 YOLO", command=self.toggle_yolo)
        self.toggle_yolo_btn.pack(side=tk.RIGHT)

        # 控制按鈕
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=(0, 10))

        self.start_btn = ttk.Button(control_frame, text="開始串流", command=self.start_streaming)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(control_frame, text="停止串流", command=self.stop_streaming, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.gc_btn = ttk.Button(control_frame, text="記憶體清理", command=self.manual_gc)
        self.gc_btn.pack(side=tk.LEFT)

        # 狀態
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_var = tk.StringVar(value="就緒")
        self.fps_var = tk.StringVar(value="FPS: 0.00")

        ttk.Label(status_frame, text="狀態:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(status_frame, textvariable=self.fps_var).pack(side=tk.RIGHT)

        # 簡化的預覽（減少記憶體使用）
        preview_frame = ttk.LabelFrame(main_frame, text="狀態預覽", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(preview_frame, text="串流狀態將在此顯示")
        self.preview_label.pack(expand=True)

    def toggle_yolo(self):
        """切換 YOLO 狀態"""
        if self.sender.toggle_yolo():
            status = "啟用" if self.sender.yolo_enabled else "停用"
            self.yolo_status_var.set(f"YOLO 狀態: {status}")
            self.log_message(f"YOLO 已{status}")
        else:
            self.log_message("YOLO 狀態切換失敗")

    def manual_gc(self):
        """手動垃圾回收"""
        gc.collect()
        self.log_message("手動記憶體清理完成")

    def log_message(self, message):
        """簡化的日誌"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def start_streaming(self):
        self.log_message("開始串流...")
        if self.sender.start_streaming():
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.toggle_yolo_btn.config(state=tk.DISABLED)  # 串流時禁用 YOLO 切換
            self.status_var.set("正在串流...")
            self.update_status_display()
        else:
            self.log_message("串流啟動失敗")

    def stop_streaming(self):
        self.log_message("停止串流...")
        self.sender.stop_streaming()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.toggle_yolo_btn.config(state=tk.NORMAL)  # 重新啟用 YOLO 切換
        self.status_var.set("已停止")
        self.fps_var.set("FPS: 0.00")
        self.preview_label.config(text="串流狀態將在此顯示")

    def update_status_display(self):
        """更新狀態顯示（不使用攝影機預覽以節省記憶體）"""
        if self.sender.streaming:
            current_time = time.strftime("%H:%M:%S")
            fps, frame_count = self.sender.get_current_fps()
            yolo_stats = self.sender.get_yolo_stats()

            # 更新 FPS 顯示
            self.fps_var.set(f"FPS: {fps:.2f}")

            # 更新狀態文字
            status_text = f"串流中...\n時間: {current_time}\n已發送: {frame_count} 幀"

            if yolo_stats['enabled']:
                status_text += f"\nYOLO: 啟用"
            else:
                status_text += "\nYOLO: 停用"

            if self.sender.generator_active:
                status_text += "\n狀態: 發送中"
            else:
                status_text += "\n狀態: 準備中"

            self.preview_label.config(text=status_text)

            # 串流時啟用停止按鈕
            if self.sender.generator_active:
                self.stop_btn.config(state=tk.NORMAL)

            self.root.after(1000, self.update_status_display)  # 每秒更新
        else:
            self.fps_var.set("FPS: 0.00")

    def on_closing(self):
        self.log_message("正在安全關閉...")
        self.preview_active = False
        self.sender.stop_streaming()

        # 清理所有引用
        if self.preview_ref:
            self.preview_ref = None

        gc.collect()
        time.sleep(1)

        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

        sys.exit(0)

    def run(self):
        try:
            self.log_message("YOLOv8 骨架偵測傳送端已啟動")
            self.root.mainloop()
        except Exception as e:
            print(f"GUI 錯誤: {e}")
        finally:
            self.on_closing()


if __name__ == "__main__":
    try:
        gui = MemorySafeVideoSenderGUI()
        gui.run()
    except Exception as e:
        print(f"程式錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)