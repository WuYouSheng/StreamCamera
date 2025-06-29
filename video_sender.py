#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import hashlib

# 確保先執行：python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. simple_video.proto
import simple_video_pb2
import simple_video_pb2_grpc


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

        # 攝影機統計
        self.frame_count = 0
        self.fps_start_time = None
        self.current_fps = 0.0
        self.fps_lock = threading.Lock()
        self.recent_frame_times = []
        self.max_fps_samples = 30

    def load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)

                # 如果沒有 frame_detection 設定，則加入預設值
                if 'frame_detection' not in config:
                    config['frame_detection'] = {
                        "method": "hash",  # 可選 "hash" 或 "diff"
                        "sensitivity": 0.02  # 變化敏感度
                    }

                return config
        except Exception as e:
            print(f"讀取配置檔錯誤: {e}")
            return {
                "client": {"target_host": "127.0.0.1", "target_port": 50051},
                "camera": {"device_id": 0, "width": 320, "height": 240, "fps": 30},
                "video": {"quality": 50},
                "frame_detection": {
                    "method": "none",  # 無過濾
                    "enable_content_check": False
                }
            }

    def connect_to_server(self):
        try:
            target = f"{self.config['client']['target_host']}:{self.config['client']['target_port']}"
            options = [
                ('grpc.keepalive_time_ms', 60000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.max_send_message_length', 4 * 1024 * 1024),  # 4MB
                ('grpc.max_receive_message_length', 4 * 1024 * 1024),
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

            # 設定攝影機參數 - 最大化效能
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config['camera']['width']))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config['camera']['height']))
            self.cap.set(cv2.CAP_PROP_FPS, float(self.config['camera']['fps']))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)  # 最小緩衝

            # 獲取實際設定值
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"攝影機設定: {actual_width}x{actual_height} @ {actual_fps}fps - 無過濾模式")

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
        """FPS 計算"""
        current_time = time.time()
        with self.fps_lock:
            self.recent_frame_times.append(current_time)
            if len(self.recent_frame_times) > self.max_fps_samples:
                self.recent_frame_times.pop(0)

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
        """純粹的攝影機串流 - 無任何過濾"""
        try:
            self.generator_active = True

            # 重置計數器
            with self.fps_lock:
                self.frame_count = 0
                self.fps_start_time = time.time()
                self.current_fps = 0.0
                self.recent_frame_times = []

            # 註冊訊息
            register_msg = simple_video_pb2.VideoMessage()
            register_msg.client.client_type = "sender"
            register_msg.client.client_id = self.client_id
            yield register_msg

            print("已註冊為傳送端，開始原始攝影機串流...")

            consecutive_failures = 0
            max_failures = 5
            last_gc_time = time.time()

            # 編碼參數
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config['video']['quality']]

            while self.streaming and self.generator_active:
                try:
                    current_time = time.time()

                    # 定期記憶體清理
                    if current_time - last_gc_time > 30:
                        gc.collect()
                        last_gc_time = current_time

                    if not self.cap or not self.cap.isOpened():
                        print("攝影機連接丟失")
                        break

                    # 直接讀取攝影機幀 - 無任何檢查或延遲
                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            print("連續讀取失敗，停止串流")
                            break
                        continue

                    consecutive_failures = 0

                    # 確保幀尺寸正確
                    target_height = self.config['camera']['height']
                    target_width = self.config['camera']['width']

                    if frame.shape[:2] != (target_height, target_width):
                        frame = cv2.resize(frame, (target_width, target_height))

                    # 直接編碼 - 無任何過濾
                    success, buffer = cv2.imencode('.jpg', frame, encode_param)
                    del frame  # 立即釋放幀

                    if not success:
                        print("影像編碼失敗")
                        continue

                    # 更新 FPS 計算
                    self.update_fps_calculation()

                    # 創建並發送訊息
                    frame_msg = simple_video_pb2.VideoMessage()
                    frame_msg.frame.data = buffer.tobytes()
                    frame_msg.frame.timestamp = int(time.time_ns() // 1_000_000)
                    frame_msg.frame.width = target_width
                    frame_msg.frame.height = target_height

                    del buffer  # 立即釋放 buffer
                    yield frame_msg
                    del frame_msg  # 立即釋放訊息

                    # 進度輸出
                    if (self.frame_count % 100 == 0) and (self.frame_count > 0):
                        with self.fps_lock:
                            print(f"已發送 {self.frame_count} 幀，當前 FPS: {self.current_fps:.2f}")

                    # 定期清理
                    if self.frame_count % 50 == 0:
                        gc.collect()

                    # 完全無延遲 - 讓攝影機以最大速度運行

                except Exception as e:
                    print(f"串流錯誤: {e}")
                    continue

        except Exception as e:
            print(f"訊息生成器錯誤: {e}")
            traceback.print_exc()
        finally:
            self.generator_active = False
            gc.collect()
            print("原始串流已結束")

    def get_current_fps(self):
        """獲取當前 FPS"""
        with self.fps_lock:
            return self.current_fps, self.frame_count

    def start_streaming(self):
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

        # 重置計數器
        with self.fps_lock:
            self.frame_count = 0
            self.current_fps = 0.0
            self.fps_start_time = None
            self.recent_frame_times = []

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
        self.root.title("原始攝影機串流傳送端")
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
        config_text += f"模式: 原始攝影機串流 (無過濾)"
        ttk.Label(config_frame, text=config_text).pack(anchor=tk.W)

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

        # 狀態顯示
        status_frame = ttk.LabelFrame(main_frame, text="串流狀態", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(status_frame, text="原始攝影機串流狀態將在此顯示")
        self.preview_label.pack(expand=True)

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
            self.status_var.set("正在串流...")
            self.update_status_display()
        else:
            self.log_message("串流啟動失敗")

    def stop_streaming(self):
        self.log_message("停止串流...")
        self.sender.stop_streaming()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("已停止")
        self.fps_var.set("FPS: 0.00")
        self.preview_label.config(text="原始攝影機串流狀態將在此顯示")

    def update_status_display(self):
        """更新狀態顯示"""
        if self.sender.streaming:
            current_time = time.strftime("%H:%M:%S")
            fps, frame_count = self.sender.get_current_fps()

            # 更新 FPS 顯示
            self.fps_var.set(f"FPS: {fps:.2f}")

            # 更新狀態文字
            status_text = f"原始攝影機串流中...\n時間: {current_time}\n已發送: {frame_count} 幀"
            if self.sender.generator_active:
                status_text += "\n狀態: 高速發送中"
            else:
                status_text += "\n狀態: 準備中"

            self.preview_label.config(text=status_text)
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
            self.log_message("原始攝影機串流傳送端已啟動")
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