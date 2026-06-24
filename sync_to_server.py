#!/usr/bin/env python3
"""
同步本地 humanoid 目录到远程服务器（使用 paramiko + SFTP，递归）
"""
import os
import paramiko

# ---- 配置 ----
LOCAL_DIR = "/home/fff/noetix/noetix_n2_gym/humanoid"
REMOTE_BASE = "/root/autodl-tmp"
REMOTE_HOST = "connect.bjb1.seetacloud.com"
REMOTE_PORT = 22672
REMOTE_USER = "root"
REMOTE_PASS = "FzaTw4zlAP/a"
REMOTE_DEST = f"{REMOTE_BASE}/leggym/noetix_n2_gym/humanoid"

# ---- SSH 连接 ----
print(f"Connecting to {REMOTE_HOST}:{REMOTE_PORT} ...")
transport = paramiko.Transport((REMOTE_HOST, REMOTE_PORT))
transport.connect(username=REMOTE_USER, password=REMOTE_PASS)
sftp = paramiko.SFTPClient.from_transport(transport)

# 确保远程目录存在（支持多级创建）
def ensure_remote_dir(path):
    dirs = path.split("/")
    for i in range(1, len(dirs) + 1):
        partial = "/".join(dirs[:i])
        if partial:
            try:
                sftp.stat(partial)
            except FileNotFoundError:
                print(f"  [mkdir] {partial}")
                sftp.mkdir(partial)

# ---- 递归上传目录 ----
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
SKIP_EXTS = {'.pyc', '.so', '.a', '.o'}

def sync_dir(local_path, remote_path):
    """递归同步 local_path 下的所有文件/目录到 remote_path"""
    ensure_remote_dir(remote_path)
    count = 0
    entries = sorted(os.scandir(local_path), key=lambda e: e.name)
    for entry in entries:
        name = entry.name
        remote_file = f"{remote_path}/{name}"

        # 跳过忽略目录
        if entry.is_dir():
            if name in SKIP_DIRS:
                print(f"  [skip-dir] {name}")
                continue
            try:
                sftp.stat(remote_file)
                print(f"  [exists]  {name}/")
            except FileNotFoundError:
                print(f"  [mkdir]  {name}/")
                sftp.mkdir(remote_file)
            count += sync_dir(entry.path, remote_file)

        elif entry.is_file():
            ext = os.path.splitext(name)[1]
            if ext in SKIP_EXTS:
                print(f"  [skip]    {name}{ext}")
                continue
            print(f"  [upload]  {name}")
            sftp.put(entry.path, remote_file)
            count += 1

    return count

print(f"\nSyncing:")
print(f"  Local:  {LOCAL_DIR}")
print(f"  Remote: {REMOTE_DEST}")
print()

n = sync_dir(LOCAL_DIR, REMOTE_DEST)
print(f"\nDone. {n} file(s) uploaded.")

sftp.close()
transport.close()
