# -*- coding: utf-8 -*-
"""
人脸识别系统(GUI 版)
- 简洁界面,鼠标操作,适合普通用户
- 实时预览不卡顿;点"识别当前画面"只对比一帧,结果显示后自动恢复预览
- 支持:注册人脸 / 单帧识别 / 置信度阈值调节 / 删除已注册人员
- 所有错误写入 error.log,不会闪退
- 兼容旧版 save.json(未归一化的数据会自动归一化后比对)

依赖:insightface opencv-python numpy pillow(见 requirements.txt)
运行:python face_gui.py  或双击 start.bat
"""

import json
import os
import sys
import traceback
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ---------------------------------------------------------------- 路径与常量
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(APP_DIR, "save.json")
ERROR_LOG = os.path.join(APP_DIR, "error.log")

FONT_FILES = ["msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"]

# 主题色(简约黑白灰)
BG        = "#1c1c1e"   # 窗口背景
PANEL     = "#262628"   # 面板
CARD      = "#2e2e31"   # 卡片
FG        = "#f2f2f2"   # 主文字
MUTED     = "#9a9a9e"   # 次要文字
BTN       = "#3c3c40"   # 按钮
BTN_HV    = "#4a4a4f"   # 按钮悬停
SEL       = "#4a4a4f"   # 列表选中

CAM_W, CAM_H = 640, 480
RESULT_HOLD_MS = 2500  # 识别结果展示时长(毫秒)


def log_error():
    msg = traceback.format_exc()
    try:
        with open(ERROR_LOG, "w", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass
    return msg


def load_font(size):
    for name in FONT_FILES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_data():
    try:
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_data(data):
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize(emb):
    emb = np.asarray(emb, dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-6)


def make_button(parent, text, command, bg=BTN, fg="#ffffff"):
    btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                    relief="flat", bd=0, font=("Microsoft YaHei UI", 11),
                    cursor="hand2", activebackground=bg, activeforeground=fg,
                    pady=8)
    btn.bind("<Enter>", lambda e: btn.config(bg=BTN_HV))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        root.title("人脸识别系统")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.app = None
        self.cap = None
        self.mode = "preview"      # preview | capture | snap
        self.pending_name = ""
        self.capture_left = 0
        self.preview_paused = False  # 识别结果停留期间暂停实时刷新
        self.threshold = 0.75
        self.last_frame = None
        self.imgtk = None
        self.font_sm = load_font(18)

        self._build_ui()
        self._init_backend()

    # ================================================================ 界面
    def _build_ui(self):
        header = tk.Frame(self.root, bg=PANEL)
        header.pack(fill="x")
        tk.Label(header, text="人脸识别系统", bg=PANEL, fg=FG,
                 font=("Microsoft YaHei UI", 15, "bold")).pack(side="left", padx=16, pady=12)
        self.header_state = tk.Label(header, text="模型加载中...", bg=PANEL,
                                     fg=MUTED, font=("Microsoft YaHei UI", 10))
        self.header_state.pack(side="right", padx=16)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=12)

        # 左侧:摄像头画面 + 状态栏
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(left, bg="#000000", width=CAM_W, height=CAM_H)
        self.video_label.pack()
        self._show_placeholder("正在启动摄像头...")

        status = tk.Frame(left, bg=PANEL)
        status.pack(fill="x", pady=(8, 0))
        self.status_var = tk.StringVar(value="正在初始化...")
        tk.Label(status, textvariable=self.status_var, bg=PANEL, fg=FG,
                 font=("Microsoft YaHei UI", 10)).pack(side="left", padx=12, pady=8)

        # 右侧:控制面板
        right = tk.Frame(body, bg=BG, width=280)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        # 注册
        self._card(right, "注册人脸").pack(fill="x")
        reg = tk.Frame(right, bg=CARD)
        reg.pack(fill="x", pady=(0, 10))
        self.name_var = tk.StringVar()
        tk.Entry(reg, textvariable=self.name_var, bg=PANEL, fg=FG,
                 insertbackground=FG, relief="flat",
                 font=("Microsoft YaHei UI", 12)).pack(fill="x", padx=12, pady=(10, 6))
        make_button(reg, "注册人脸", self.register_face).pack(fill="x", padx=12, pady=(0, 12))

        # 识别
        self._card(right, "识别").pack(fill="x")
        rec = tk.Frame(right, bg=CARD)
        rec.pack(fill="x", pady=(0, 10))
        self.rec_btn = make_button(rec, "识别当前画面", self.snap_recognize)
        self.rec_btn.pack(fill="x", padx=12, pady=(10, 6))

        th = tk.Frame(rec, bg=CARD)
        th.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(th, text="匹配阈值", bg=CARD, fg=MUTED,
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        self.th_var = tk.StringVar(value="0.75")
        tk.Label(th, textvariable=self.th_var, bg=CARD, fg=FG,
                 font=("Microsoft YaHei UI", 10)).pack(side="right")
        self.th_slider = tk.Scale(rec, from_=0.50, to=0.95, resolution=0.01,
                                  orient="horizontal", bg=CARD, fg=FG,
                                  troughcolor=PANEL, highlightthickness=0, bd=0,
                                  activebackground=BTN_HV, command=self._on_threshold)
        self.th_slider.set(0.75)
        self.th_slider.pack(fill="x", padx=12, pady=(0, 12))

        self.last_var = tk.StringVar(value="最近识别:--")
        tk.Label(rec, textvariable=self.last_var, bg=CARD, fg=FG,
                 font=("Microsoft YaHei UI", 10)).pack(fill="x", padx=12, pady=(0, 12))

        # 已注册人员
        self._card(right, "已注册人员").pack(fill="x")
        lst = tk.Frame(right, bg=CARD)
        lst.pack(fill="both", expand=True, pady=(0, 10))
        self.listbox = tk.Listbox(lst, bg=PANEL, fg=FG, selectbackground=SEL,
                                  relief="flat", highlightthickness=0,
                                  font=("Microsoft YaHei UI", 11))
        self.listbox.pack(fill="both", expand=True, padx=12, pady=(10, 6))
        make_button(lst, "删除选中人员", self.delete_selected).pack(
            fill="x", padx=12, pady=(0, 12))

        # 退出
        make_button(right, "退出", self.on_close).pack(
            fill="x", side="bottom", pady=(10, 0))

    def _card(self, parent, title):
        card = tk.Frame(parent, bg=PANEL)
        tk.Label(card, text=title, bg=PANEL, fg=FG,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        return card

    def _show_placeholder(self, text):
        img = Image.new("RGB", (CAM_W, CAM_H), (20, 20, 32))
        draw = ImageDraw.Draw(img)
        draw.text((CAM_W // 2 - 100, CAM_H // 2), text, fill=(140, 140, 140),
                  font=self.font_sm)
        self._show_image(img)

    def _show_image(self, pil_img):
        self.imgtk = ImageTk.PhotoImage(pil_img)
        self.video_label.configure(image=self.imgtk)

    def set_status(self, text, color=FG):
        self.status_var.set(text)

    # ================================================================ 后端
    def _init_backend(self):
        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            ctx = 0 if "CUDAExecutionProvider" in providers else -1

            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis()
            self.app.prepare(ctx_id=ctx, det_size=(640, 640))
            self.header_state.config(text="模型就绪", fg=FG)

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = None
                self._show_placeholder("无法打开摄像头,请检查设备")
                self.set_status("摄像头不可用")
            else:
                self.set_status("输入姓名后点击[注册人脸]")
        except Exception:
            self.header_state.config(text="启动失败", fg=MUTED)
            self._show_placeholder("初始化失败,请查看 error.log")
            self.set_status("初始化失败,详见 error.log")
            log_error()

        self.refresh_list()
        self.root.after(20, self.update_frame)

    # ================================================================ 主循环
    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.mode == "capture":
                    self._do_capture(frame)
                    self._display(frame)
                elif self.mode == "snap":
                    self._do_recognize(frame)
                elif self.preview_paused and self.last_frame is not None:
                    self._display(self.last_frame)  # 识别结果停留
                else:
                    self._display(frame)  # 实时预览,不跑模型,始终流畅
        self.root.after(20, self.update_frame)

    def _display(self, frame_bgr):
        frame = cv2.resize(frame_bgr, (CAM_W, CAM_H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._show_image(Image.fromarray(rgb))

    # ================================================================ 识别(单帧)
    def snap_recognize(self):
        if self.mode != "preview":
            return
        if self.app is None:
            messagebox.showerror("错误", "模型未加载成功,请查看 error.log")
            return
        self.mode = "snap"
        self.rec_btn.config(text="识别中...")
        self.set_status("识别中,请正对摄像头...")

    def _do_recognize(self, frame_bgr):
        self.mode = "preview"
        self.rec_btn.config(text="识别当前画面")
        try:
            faces = self.app.get(frame_bgr)
            data = load_data()
            best_text = "最近识别:--"
            draw_frame = frame_bgr.copy()
            for face in faces:
                emb = normalize(face.embedding)
                best_name, best_sim = "陌生人", -1.0
                for item in data:
                    sim = float(np.dot(normalize(item["embedding"]), emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = item.get("name", "未知")
                known = best_sim >= self.threshold
                # 黑白灰:白色=认识的人,灰色=陌生人
                color = (255, 255, 255) if known else (150, 150, 150)
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 3)
                name = best_name if known else "陌生人"
                self._draw_label(draw_frame, f"{name} {best_sim * 100:.0f}%",
                                 x1, y1 - 30, color)
                if known:
                    best_text = f"最近识别:{name} ({best_sim * 100:.0f}%)"
            self.last_var.set(best_text)
            self.last_frame = draw_frame
            self.preview_paused = True
            self._display(draw_frame)
            if faces and "最近识别:--" not in best_text:
                self.set_status(best_text.replace("最近识别:", ""))
            elif faces:
                self.set_status("未识别到已注册人员")
            else:
                self.set_status("画面中未检测到人脸")
            self.root.after(RESULT_HOLD_MS, self._resume_preview)
        except Exception:
            log_error()
            self.set_status("识别出错,详见 error.log")

    def _resume_preview(self):
        self.preview_paused = False

    # ================================================================ 注册
    def register_face(self):
        if self.mode != "preview":
            return
        if self.app is None:
            messagebox.showerror("错误", "模型未加载成功,请查看 error.log")
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("提示", "请先在上方输入框填写姓名")
            return
        self.pending_name = name
        self.mode = "capture"
        self.capture_left = 25
        self.set_status(f"注册中:请正对摄像头...")

    def _do_capture(self, frame_bgr):
        self.capture_left -= 1
        try:
            faces = self.app.get(frame_bgr)
            if faces:
                emb = normalize(faces[0].embedding)
                data = load_data()
                data.append({"name": self.pending_name, "embedding": emb.tolist()})
                save_data(data)
                self.mode = "preview"
                self.refresh_list()
                self.set_status(f"注册成功:{self.pending_name}")
                messagebox.showinfo("注册成功", f"已保存人脸:{self.pending_name}")
            elif self.capture_left <= 0:
                self.mode = "preview"
                self.set_status("未检测到人脸,请调整位置后重试")
                messagebox.showwarning("注册失败", "没有检测到人脸,请正对摄像头再试")
        except Exception:
            log_error()
            self.mode = "preview"
            self.set_status("注册出错,详见 error.log")

    # ================================================================ 人员管理
    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        names = []
        for item in load_data():
            n = item.get("name", "未知")
            if n not in names:
                names.append(n)
        for n in names:
            self.listbox.insert(tk.END, n)
        if not names:
            self.listbox.insert(tk.END, "(暂无注册人员)")

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "请先在列表中选择要删除的人员")
            return
        name = self.listbox.get(sel[0])
        if name.startswith("(暂无"):
            return
        if not messagebox.askyesno("删除确认", f"确定删除 [{name}] 的全部人脸数据吗?"):
            return
        data = [d for d in load_data() if d.get("name", "未知") != name]
        save_data(data)
        self.refresh_list()
        self.set_status(f"已删除:{name}")

    def _on_threshold(self, val):
        self.threshold = float(val)
        self.th_var.set(f"{self.threshold:.2f}")

    # ================================================================ 绘图
    def _draw_label(self, frame_bgr, text, x, y, color_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        y = max(y, 2)
        bbox = draw.textbbox((x, y), text, font=self.font_sm)
        draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2],
                       fill=(10, 10, 10))
        draw.text((x, y), text, fill=color_bgr, font=self.font_sm)
        out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        frame_bgr[:] = out

    # ================================================================ 退出
    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1000x620")
    root.minsize(920, 580)
    try:
        FaceRecognitionApp(root)
    except Exception:
        msg = log_error()
        messagebox.showerror("启动失败", f"程序启动出错,详情见 error.log:\n\n{msg[-500:]}")
        return
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_error()
        try:
            messagebox.showerror("错误", "程序出错,详情已写入 error.log")
        except Exception:
            pass
