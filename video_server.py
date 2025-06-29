#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import grpc
from concurrent import futures
import threading
import queue
import time
import json
import hashlib

# 確保先執行：python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. simple_video.proto
import simple_video_pb2
import simple_video_pb2_grpc


class ImprovedVideoStreamServer(simple_video_pb2_grpc.VideoStreamServiceServicer):
    def __init__(self):
        self.senders = {}  # 傳送端連接
        self.receivers = {}  # 接收端連接
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # 新增：幀去重機制
        self.last_frame_hash = None
        self.frame_sequence = 0
        self.duplicate_frames = 0
        self.unique_frames = 0

        # 統計資訊
        self.server_stats = {
            'total_frames_received': 0,
            'unique_frames_forwarded': 0,
            'duplicate_frames_dropped': 0,
            'active_senders': 0,
            'active_receivers': 0
        }

    def calculate_frame_hash(self, frame_data):
        """計算幀數據的哈希值"""
        return hashlib.md5(frame_data).hexdigest()

    def is_duplicate_frame(self, frame_data):
        """檢查是否為重複幀"""
        frame_hash = self.calculate_frame_hash(frame_data)

        if frame_hash == self.last_frame_hash:
            return True

        self.last_frame_hash = frame_hash
        return False

    def forward_frame_to_receivers(self, frame_msg):
        """只轉發唯一幀給接收端"""
        if not self.receivers:
            return

        forwarded_count = 0
        for receiver_id, receiver_queue in list(self.receivers.items()):
            try:
                # 清空舊幀，只保留最新的唯一幀
                while not receiver_queue.empty():
                    try:
                        receiver_queue.get_nowait()
                    except queue.Empty:
                        break

                # 放入新的唯一幀
                if not receiver_queue.full():
                    receiver_queue.put(frame_msg, block=False)
                    forwarded_count += 1
            except Exception as e:
                print(f"轉發到接收端 {receiver_id} 失敗: {e}")

        print(f"唯一幀已轉發給 {forwarded_count} 個接收端")

    def print_server_stats(self):
        """輸出服務器統計"""
        total = self.server_stats['total_frames_received']
        unique = self.server_stats['unique_frames_forwarded']
        dropped = self.server_stats['duplicate_frames_dropped']

        if total > 0:
            drop_rate = (dropped / total) * 100
            print(f"\n=== 服務器統計 ===")
            print(f"總接收幀: {total}")
            print(f"唯一幀轉發: {unique}")
            print(f"重複幀丟棄: {dropped}")
            print(f"重複率: {drop_rate:.1f}%")
            print(f"活躍發送端: {len(self.senders)}")
            print(f"活躍接收端: {len(self.receivers)}")
            print("=" * 18)

    def StreamVideo(self, request_iterator, context):
        client_id = f"client_{id(context)}"
        client_type = None
        client_queue = queue.Queue(maxsize=2)  # 減小隊列大小

        print(f"新客戶端連接: {client_id}")

        def handle_requests():
            nonlocal client_type
            try:
                for request in request_iterator:
                    if request.HasField('client'):
                        # 客戶端註冊
                        client_type = request.client.client_type
                        if client_type == "sender":
                            self.senders[client_id] = {
                                'queue': client_queue,
                                'last_seen': time.time()
                            }
                            self.server_stats['active_senders'] = len(self.senders)
                            print(f"傳送端註冊: {client_id}")
                        elif client_type == "receiver":
                            self.receivers[client_id] = client_queue
                            self.server_stats['active_receivers'] = len(self.receivers)
                            print(f"接收端註冊: {client_id}")

                    elif request.HasField('frame'):
                        # 處理影像幀（來自傳送端）
                        if client_type == "sender":
                            self.server_stats['total_frames_received'] += 1

                            # 檢查是否為重複幀
                            if self.is_duplicate_frame(request.frame.data):
                                self.server_stats['duplicate_frames_dropped'] += 1
                                self.duplicate_frames += 1

                                if self.duplicate_frames % 50 == 0:
                                    print(f"已丟棄 {self.duplicate_frames} 個重複幀")

                                # 不轉發重複幀，但要給sender回應
                                continue
                            else:
                                # 處理唯一幀
                                self.unique_frames += 1
                                self.server_stats['unique_frames_forwarded'] += 1

                                with self.frame_lock:
                                    self.latest_frame = request.frame
                                    self.frame_sequence += 1

                                frame_size = len(request.frame.data)
                                print(f"收到唯一幀 #{self.frame_sequence}: {frame_size} bytes")

                                # 只轉發唯一幀
                                self.forward_frame_to_receivers(request)

                                # 每100個唯一幀輸出統計
                                if self.unique_frames % 100 == 0:
                                    self.print_server_stats()

            except Exception as e:
                print(f"處理請求錯誤: {e}")

        # 啟動請求處理線程
        request_thread = threading.Thread(target=handle_requests, daemon=True)
        request_thread.start()

        try:
            # 發送回應
            while context.is_active():
                try:
                    if client_type == "sender":
                        # 給傳送端發送狀態確認
                        status_msg = simple_video_pb2.VideoMessage()
                        status_msg.status.success = True
                        status_msg.status.message = "Frame processed"
                        yield status_msg
                        time.sleep(0.1)

                    elif client_type == "receiver":
                        # 給接收端發送影像幀
                        try:
                            # 嘗試從隊列獲取幀，如果沒有就等待
                            frame_msg = client_queue.get(timeout=0.5)
                            yield frame_msg

                            # 立即檢查是否還有更多幀（避免積壓）
                            while not client_queue.empty():
                                try:
                                    additional_frame = client_queue.get_nowait()
                                    yield additional_frame
                                except queue.Empty:
                                    break

                        except queue.Empty:
                            # 沒有新幀時發送心跳
                            status_msg = simple_video_pb2.VideoMessage()
                            status_msg.status.success = True
                            status_msg.status.message = "Waiting for frames"
                            yield status_msg

                except Exception as e:
                    print(f"發送回應錯誤: {e}")
                    break

        except Exception as e:
            print(f"連接錯誤: {e}")
        finally:
            # 清理連接
            if client_id in self.senders:
                del self.senders[client_id]
                self.server_stats['active_senders'] = len(self.senders)
                print(f"傳送端斷開: {client_id}")
            if client_id in self.receivers:
                del self.receivers[client_id]
                self.server_stats['active_receivers'] = len(self.receivers)
                print(f"接收端斷開: {client_id}")

            # 如果是最後一個客戶端，輸出最終統計
            if len(self.senders) == 0 and len(self.receivers) == 0:
                print("\n=== 最終服務器統計 ===")
                self.print_server_stats()


def start_server():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {"server": {"host": "0.0.0.0", "port": 50051}}

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    simple_video_pb2_grpc.add_VideoStreamServiceServicer_to_server(
        ImprovedVideoStreamServer(), server
    )

    listen_addr = f"{config['server']['host']}:{config['server']['port']}"
    server.add_insecure_port(listen_addr)
    server.start()

    print(f"修正版 gRPC 影像伺服器已啟動")
    print(f"監聽地址: {listen_addr}")
    print("功能: 自動檢測並丟棄重複幀")
    print("等待客戶端連接...")
    print("按 Ctrl+C 停止伺服器")
    print("-" * 40)

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n正在停止伺服器...")
        server.stop(0)


if __name__ == "__main__":
    start_server()