#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Description ：****** 
@File    ：AudioVideoCombiner.py
@IDE     ：PyCharm 
@Author  ：Sean Han
@Date    ：2025/7/19 10:22 
"""
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

# 主窗口类
class AVCombinerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("音视频合成器")
        self.root.geometry("500x300")
        self.root.configure(bg="#f0f0f0")

        self.files = {}  # {'video': path, 'audio': path}

        self.drop_area = tk.Label(root, text="拖入1个MP4视频和1个MP3音频", relief="groove",
                                  width=50, height=6, bg="white")
        self.drop_area.pack(pady=20)
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.on_drop)

        self.file_list_frame = tk.Frame(root, bg="#f0f0f0")
        self.file_list_frame.pack()

        self.combine_button = tk.Button(root, text="开始合成", command=self.on_combine, state='disabled')
        self.combine_button.pack(pady=10)

    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        for path in paths:
            path = path.strip('{').strip('}')
            ext = Path(path).suffix.lower()
            if ext == ".mp4" and "video" not in self.files:
                self.files["video"] = path
            elif ext == ".mp3" and "audio" not in self.files:
                self.files["audio"] = path
            else:
                messagebox.showwarning("文件错误", f"不支持的文件或已存在相同类型：{Path(path).name}")
        self.refresh_file_list()

    def refresh_file_list(self):
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()

        for ftype in ["video", "audio"]:
            if ftype in self.files:
                path = self.files[ftype]
                row = tk.Frame(self.file_list_frame, bg="#f0f0f0")
                row.pack(pady=2, anchor="w")
                label = tk.Label(row, text=f"[{ftype.upper()}] {Path(path).name}", bg="#f0f0f0")
                label.pack(side="left")
                del_btn = tk.Button(row, text="✕", command=lambda t=ftype: self.delete_file(t), bg="#f0f0f0", fg="red", relief="flat")
                del_btn.pack(side="left", padx=5)

        if "video" in self.files and "audio" in self.files:
            self.combine_button.config(state='normal')
        else:
            self.combine_button.config(state='disabled')

    def delete_file(self, ftype):
        if ftype in self.files:
            del self.files[ftype]
            self.refresh_file_list()

    def on_combine(self):
        try:
            video_path = self.files["video"]
            audio_path = self.files["audio"]
            combine_audio_video(video_path, audio_path)
            messagebox.showinfo("合成完成", "视频合成成功！输出文件已保存。")
        except Exception as e:
            messagebox.showerror("错误", str(e))

def combine_audio_video(video_path, audio_path, output_path=None):
    # 检查路径合法性
    if not os.path.isfile(video_path) or not video_path.lower().endswith('.mp4'):
        raise ValueError("无效的视频路径，请提供一个 .mp4 文件。")
    if not os.path.isfile(audio_path) or not audio_path.lower().endswith('.mp3'):
        raise ValueError("无效的音频路径，请提供一个 .mp3 文件。")

    # 加载视频和音频
    print("加载视频和音频...")
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # 设置音频长度为视频长度（如音频更长）
    audio_clip = audio_clip.subclipped(0, video_clip.duration)

    # 合成视频
    print("合成中...")
    final_clip = video_clip.with_audio(audio_clip)

    # 默认输出路径
    if output_path is None:
        video_dir = Path(video_path).parent
        output_path = video_dir / "output_合成版.mp4"
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / "output_合成版.mp4"

    # 写出视频
    print(f"保存到：{output_path}")
    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    # 释放资源
    video_clip.close()
    audio_clip.close()
    final_clip.close()
    print("合成完成！")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AVCombinerApp(root)
    root.mainloop()
