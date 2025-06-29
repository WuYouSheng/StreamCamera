#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import grpc
import cv2
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import uuid
import json
import hashlib

# 確保先執行：python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. simple_video.proto
import simple_video_pb2
import simple_video_pb2_grpc


class ImprovedVideoReceiver:
    def __init__(self):
        self.config = self.load_config()
        self.client_id = str(uuid.uuid4())
        self.receiving = False
        self.channel = None
        self.stub = None
        self.latest_frame_data = None
        self.frame_lock = threading.Lock()
        self.last_frame_hash = None

        # 改進的幀檢測
        self.frame_hashes = set()  # 存儲最近的幀哈希
        self.total_frames_received = 0  # 總接收幀數
        self.unique_frames_received = 0  # 唯一幀數
        self.duplicate_frames_received = 0  # 重複幀數
        self.last_10_hashes = []  # 最近10個幀的哈希

    def load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {
                "client": {"target_host": "127.0.0.1", "target_port": 50051}
            }

    def connect_to_server(self):
        try:
            target = f"{self.config['client']['target_host']}:{self.config['client']['target_port']}"
            self.channel = grpc.insecure_channel(target)
            self.stub = simple_video_pb2_grpc.VideoStreamServiceStub(self.channel)

            # 測試連接
            grpc.channel_ready_future(self.channel).result(timeout=5)
            print(f"成功連接到服務器: {target}")
            return True
        except Exception as e:
            print(f"連接錯誤: {e}")
            return False

    def message_generator(self):
        """發送空幀來標識自己是接收端"""
        while self.receiving:
            empty_frame = simple_video_pb2.VideoMessage()
            empty_frame.client.client_type = "receiver"
            empty_frame.client.client_id = self.client_id
            yield empty_frame
            time.sleep(1)  # 每秒發送一次心跳

    def _is_duplicate_frame(self, frame_hash):
        """檢查是否為重複幀，使用滑動窗口檢測"""
        # 檢查是否在最近的幀中出現過
        if frame_hash in self.last_10_hashes:
            return True

        # 添加到最近幀列表
        self.last_10_hashes.append(frame_hash)

        # 只保留最近10個幀的哈希
        if len(self.last_10_hashes) > 10:
            self.last_10_hashes.pop(0)

        return False

    def get_receive_stats(self):
        """獲取接收統計"""
        with self.frame_lock:
            duplicate_rate = 0
            if self.total_frames_received > 0:
                duplicate_rate = (self.duplicate_frames_received / self.total_frames_received) * 100

            return {
                'total_received': self.total_frames_received,
                'unique_received': self.unique_frames_received,
                'duplicates_received': self.duplicate_frames_received,
                'duplicate_rate': duplicate_rate
            }

    def start_receiving(self):
        if not self.connect_to_server():
            return False

        self.receiving = True

        def receive_worker():
            try:
                print("開始接收串流...")
                print("監控重複幀檢測已啟用")

                # 作為接收端，發送空幀並接收回應
                responses = self.stub.StreamVideo(self.message_generator())

                for response in responses:
                    if not self.receiving:
                        break

                    if response.HasField('frame'):
                        # 收到影像幀
                        self.total_frames_received += 1

                        # 計算幀哈希
                        frame_hash = hashlib.md5(response.frame.data).hexdigest()

                        # 檢查是否為重複幀
                        is_duplicate = self._is_duplicate_frame(frame_hash)

                        with self.frame_lock:
                            # 只有唯一幀才更新 latest_frame_data
                            if not is_duplicate:
                                self.latest_frame_data = response.frame.data
                                self.last_frame_hash = frame_hash
                                self.unique_frames_received += 1

                                # 每30個唯一幀報告一次
                                if self.unique_frames_received % 30 == 0:
                                    duplicate_rate = (self.duplicate_frames_received / self.total_frames_received) * 100
                                    print(
                                        f"已收到 {self.unique_frames_received} 唯一幀，{self.duplicate_frames_received} 重複幀 ({duplicate_rate:.1f}%)")
                            else:
                                self.duplicate_frames_received += 1

                                # 每100個重複幀報告一次
                                if self.duplicate_frames_received % 100 == 0:
                                    print(f"警告: 已收到 {self.duplicate_frames_received} 個重複幀")

                    elif response.HasField('status'):
                        if not response.status.success:
                            print(f"伺服器狀態: {response.status.message}")

            except Exception as e:
                if self.receiving:
                    print(f"接收錯誤: {e}")
                    if hasattr(self, 'error_callback'):
                        self.error_callback(f"接收錯誤: {e}")

        self.receive_thread = threading.Thread(target=receive_worker, daemon=True)
        self.receive_thread.start()
        return True

    def stop_receiving(self):
        print("正在停止接收...")
        self.receiving = False

        if hasattr(self, 'receive_thread') and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2)

        if self.channel:
            try:
                self.channel.close()
                self.channel = None
            except Exception as e:
                print(f"關閉連接錯誤: {e}")

        # 顯示最終統計
        stats = self.get_receive_stats()
        if stats['total_received'] > 0:
            print(
                f"接收統計: 總計 {stats['total_received']} 幀，唯一 {stats['unique_received']} 幀，重複率 {stats['duplicate_rate']:.1f}%")

        print("接收已停止")

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame_data, self.last_frame_hash


class ImprovedVideoReceiverGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("修正版影像接收端 - 重複幀檢測")
        self.root.geometry("900x750")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.receiver = ImprovedVideoReceiver()
        self.receiver.error_callback = self.show_error

        self.current_frame = None

        # 更精準的 FPS 計算
        self.recent_frame_times = []  # 存儲最近的幀時間戳
        self.max_fps_samples = 30  # 計算 FPS 的樣本數量
        self.current_fps = 0.0
        self.fps_lock = threading.Lock()

        self.setup_ui()

    def show_error(self, message):
        self.root.after(0, lambda: messagebox.showerror("錯誤", message))

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

    def get_current_fps(self):
        """獲取當前 FPS"""
        with self.fps_lock:
            return self.current_fps

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 設定資訊
        config_frame = ttk.LabelFrame(main_frame, text="連接設定", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))

        config_text = f"連接伺服器: {self.receiver.config['client']['target_host']}:{self.receiver.config['client']['target_port']}"
        config_text += "\n功能: 自動檢測並過濾重複幀，顯示真實FPS"
        ttk.Label(config_frame, text=config_text).pack(anchor=tk.W)

        # 控制按鈕
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=(0, 10))

        self.start_btn = ttk.Button(control_frame, text="開始接收", command=self.start_receiving)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(control_frame, text="停止接收", command=self.stop_receiving, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.save_btn = ttk.Button(control_frame, text="截圖保存", command=self.save_frame, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stats_btn = ttk.Button(control_frame, text="詳細統計", command=self.show_detailed_stats, state=tk.DISABLED)
        self.stats_btn.pack(side=tk.LEFT)

        # 狀態顯示
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_var = tk.StringVar(value="就緒")
        self.fps_var = tk.StringVar(value="顯示FPS: 0.00 | 重複率: 0.0%")

        ttk.Label(status_frame, text="狀態:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(status_frame, textvariable=self.fps_var).pack(side=tk.RIGHT)

        # 統計資訊框
        stats_frame = ttk.LabelFrame(main_frame, text="即時統計", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        self.stats_text = tk.Text(stats_frame, height=4, wrap=tk.WORD, font=("Courier", 9))
        self.stats_text.pack(fill=tk.X)

        # 影像顯示
        video_frame = ttk.LabelFrame(main_frame, text="即時影像", padding="10")
        video_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame, text="等待影像串流...")
        self.video_label.pack(expand=True)

    def start_receiving(self):
        if self.receiver.start_receiving():
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.stats_btn.config(state=tk.NORMAL)
            self.status_var.set("正在接收...")
            self.start_display()

    def stop_receiving(self):
        self.receiver.stop_receiving()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.stats_btn.config(state=tk.DISABLED)
        self.status_var.set("已停止")
        self.fps_var.set("顯示FPS: 0.00 | 重複率: 0.0%")
        self.video_label.config(image="", text="等待影像串流...")

        # 重置 FPS 計算
        with self.fps_lock:
            self.recent_frame_times = []
            self.current_fps = 0.0

        # 重置接收統計
        with self.receiver.frame_lock:
            self.receiver.last_frame_hash = None
            self.receiver.total_frames_received = 0
            self.receiver.unique_frames_received = 0
            self.receiver.duplicate_frames_received = 0
            self.receiver.last_10_hashes = []

        # 清空統計顯示
        self.stats_text.delete(1.0, tk.END)

    def start_display(self):
        """顯示接收到的影像"""
        last_processed_hash = None  # 追蹤最後處理的幀哈希

        def update_display():
            nonlocal last_processed_hash

            if self.receiver.receiving:
                frame_data, current_hash = self.receiver.get_latest_frame()

                if frame_data and current_hash != last_processed_hash:
                    try:
                        # 解碼影像
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            # 轉換色彩空間
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # 調整大小以適應顯示
                            height, width = frame_rgb.shape[:2]
                            max_height = 400
                            if height > max_height:
                                scale = max_height / height
                                new_width = int(width * scale)
                                frame_rgb = cv2.resize(frame_rgb, (new_width, max_height))

                            # 轉換為 PIL 圖像並顯示
                            image = Image.fromarray(frame_rgb)
                            photo = ImageTk.PhotoImage(image)

                            self.video_label.config(image=photo, text="")
                            self.video_label.image = photo
                            self.current_frame = frame_rgb

                            # 只在處理新幀時更新 FPS 計算
                            self.update_fps_calculation()

                            # 更新最後處理的哈希
                            last_processed_hash = current_hash

                    except Exception as e:
                        print(f"解碼影像錯誤: {e}")
                else:
                    # 沒有新幀時顯示等待訊息
                    if not frame_data:
                        self.show_waiting_message()

                # 更新 FPS 顯示（包含接收統計）
                current_fps = self.get_current_fps()
                stats = self.receiver.get_receive_stats()

                self.fps_var.set(f"顯示FPS: {current_fps:.2f} | 重複率: {stats['duplicate_rate']:.1f}%")

                # 更新統計文字
                self.update_stats_display(stats, current_fps)

                # 繼續更新
                self.root.after(33, update_display)  # 約30fps

        update_display()

    def update_stats_display(self, stats, current_fps):
        """更新統計顯示"""
        stats_text = f"總接收: {stats['total_received']:,} | "
        stats_text += f"唯一: {stats['unique_received']:,} | "
        stats_text += f"重複: {stats['duplicates_received']:,} | "
        stats_text += f"顯示FPS: {current_fps:.2f}\n"

        if stats['duplicate_rate'] > 50:
            stats_text += "狀態: 檢測到大量重複幀 (>50%) - 網路層問題\n"
        elif stats['duplicate_rate'] > 20:
            stats_text += "狀態: 中等程度重複幀 (20-50%) - 輕微網路問題\n"
        elif stats['duplicate_rate'] > 5:
            stats_text += "狀態: 少量重複幀 (5-20%) - 正常範圍\n"
        else:
            stats_text += "狀態: 幾乎無重複幀 (<5%) - 傳輸良好\n"

        # 效能評估
        if stats['total_received'] > 0:
            efficiency = (stats['unique_received'] / stats['total_received']) * 100
            stats_text += f"傳輸效率: {efficiency:.1f}%"

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def show_waiting_message(self):
        """顯示等待訊息而不是雜訊"""
        # 創建一個簡單的等待畫面
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame.fill(50)  # 深灰色背景

        # 添加文字
        text = "Waiting for video stream..."
        text_cn = "等待影像串流..."
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(frame, text, (width // 2 - 150, height // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, text_cn, (width // 2 - 100, height // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, timestamp, (width // 2 - 100, height // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 轉換並顯示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (400, 300))

        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)

        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
        self.current_frame = frame_rgb

    def show_detailed_stats(self):
        """顯示詳細的接收統計"""
        stats = self.receiver.get_receive_stats()
        current_fps = self.get_current_fps()

        stats_window = tk.Toplevel(self.root)
        stats_window.title("詳細統計報告")
        stats_window.geometry("500x400")

        text_widget = tk.Text(stats_window, wrap=tk.WORD, padx=10, pady=10, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(stats_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        stats_text = f"""接收端統計報告
{"=" * 40}

接收統計:
  總接收幀數: {stats['total_received']:,}
  唯一幀數: {stats['unique_received']:,}
  重複幀數: {stats['duplicates_received']:,}
  重複率: {stats['duplicate_rate']:.1f}%

顯示統計:
  當前顯示FPS: {current_fps:.2f}
  FPS計算樣本: {len(self.recent_frame_times)}

傳輸效率:"""

        if stats['total_received'] > 0:
            efficiency = (stats['unique_received'] / stats['total_received']) * 100
            stats_text += f"\n  有效幀比率: {efficiency:.1f}%"

            if stats['duplicate_rate'] > 0:
                bandwidth_waste = stats['duplicate_rate']
                stats_text += f"\n  頻寬浪費率: {bandwidth_waste:.1f}%"

        stats_text += f"\n\n診斷結果:"

        if stats['duplicate_rate'] > 60:
            stats_text += "\n  嚴重問題: 接收到大量重複幀!"
            stats_text += "\n  建議檢查:"
            stats_text += "\n  • 發送端是否正常運作"
            stats_text += "\n  • 網路連接品質"
            stats_text += "\n  • gRPC設定參數"
        elif stats['duplicate_rate'] > 30:
            stats_text += "\n  中等問題: 重複幀比率偏高"
            stats_text += "\n  建議檢查網路穩定性"
        elif stats['duplicate_rate'] > 10:
            stats_text += "\n  輕微問題: 少量重複幀"
            stats_text += "\n  可能是正常的網路波動"
        else:
            stats_text += "\n  狀態良好: 重複幀率可接受"
            stats_text += "\n  傳輸品質正常"

        if current_fps > 0:
            stats_text += f"\n\n效能評估:"
            if current_fps >= 25:
                stats_text += "\n  顯示流暢度: 優秀 (>=25fps)"
            elif current_fps >= 15:
                stats_text += "\n  顯示流暢度: 良好 (15-25fps)"
            elif current_fps >= 10:
                stats_text += "\n  顯示流暢度: 普通 (10-15fps)"
            else:
                stats_text += "\n  顯示流暢度: 需改善 (<10fps)"

        text_widget.insert(tk.END, stats_text)
        text_widget.config(state=tk.DISABLED)

    def save_frame(self):
        if self.current_frame is not None:
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"

                frame_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, frame_bgr)

                messagebox.showinfo("保存成功", f"截圖已保存為: {filename}")
            except Exception as e:
                messagebox.showerror("保存失敗", f"無法保存截圖: {e}")
        else:
            messagebox.showwarning("無影像", "目前沒有可保存的影像")

    def on_closing(self):
        print("正在關閉接收端...")
        self.receiver.stop_receiving()
        self.root.destroy()

    def run(self):
        print("修正版影像接收端已啟動")
        print("功能: 自動檢測重複幀，顯示真實FPS")
        print("-" * 40)
        self.root.mainloop()


if __name__ == "__main__":
    try:
        gui = ImprovedVideoReceiverGUI()
        gui.run()
    except Exception as e:
        print(f"程式錯誤: {e}")
        import traceback

        traceback.print_exc()