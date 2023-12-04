import socket
import time

def start_listening(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen()
    return sock

host, port = '127.0.0.1', 12345

while True:
    # 启动监听
    sock = start_listening(host, port)
    print(f"Listening on {host}:{port}")

    # 这里可以添加接受连接和处理数据的代码
    # ...

    # 举例：我们让它监听一段时间后关闭
    time.sleep(10)  # 休眠 10 秒来模拟一些处理活动
    print("Closing socket and restarting...")
    sock.close()
