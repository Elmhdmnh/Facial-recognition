from insightface.app import FaceAnalysis
import cv2
import json
import os
import tkinter as tk
import numpy as np
from tkinter import messagebox, simpledialog

def appendText(text,position,frame):
    cv2.putText(frame,text,position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

# 读取/保存统一用脚本所在目录的 save.json,避免从别的目录运行时找不到文件而闪退
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save.json")

app = FaceAnalysis()

app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

show_result = False  # 是否已检测完成，保持显示结果帧
all_faces_data = []  # 保存所有人脸数据

while True:
    if not show_result:
        ret, frame = cap.read()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        if show_result:
            show_result = False
        else:
            break

    if key == ord('x') and not show_result:
        face = app.get(frame)
        face_num = len(face)
        all_faces_data = []  # 重置

        #绘制人脸
        for i in range(face_num):
            #绘制人脸方框
            x1,y1,x2,y2=face[i].bbox.astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),thickness=2)


            #获取人脸5点
            kps1,kps2,kps3,kps4,kps5 = face[i].kps.astype(int)
            cv2.circle(frame,(kps1[0],kps1[1]),1,(0,0,255),thickness=-1)
            cv2.circle(frame,(kps2[0],kps2[1]),1,(0,0,255),thickness=-1)
            cv2.circle(frame,(kps3[0],kps3[1]),1,(0,0,255),thickness=-1)
            cv2.circle(frame,(kps4[0],kps4[1]),1,(0,0,255),thickness=-1)
            cv2.circle(frame,(kps5[0],kps5[1]),1,(0,0,255),thickness=-1)

            #获取人脸68点三维关键点
            kps68 = face[i].landmark_3d_68.astype(int)
            for j in range(68):
                cv2.circle(frame,(kps68[j][0],kps68[j][1]),1,(255,0,0),thickness=-1)

            #性别预测
            gender1 = face[i].gender
            if gender1 == 0:
                gender = "女"
            else:
                gender = "男"
            appendText(gender,(x1,y1),frame)

            #年龄预测
            age = face[i].age
            appendText(str(age),(x1,y1+30),frame)

            #获取人脸特征向量
            embedding = face[i].embedding
            #归一化:向量长度变为1,点积结果才是真正的相似度(-1~1),0.75阈值才有效
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

            #比较
            try:
                with open(SAVE_PATH,"r",encoding="utf-8") as f:
                    data=json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data=[]  # 文件不存在/损坏时当作空,不闪退
            haveperson=False
            person_name=""
            for j in range(len(data)):
                saved_emb=np.asarray(data[j]["embedding"],dtype=np.float32)
                saved_emb=saved_emb/(np.linalg.norm(saved_emb)+1e-6)
                similarity=np.dot(saved_emb,embedding)
                if similarity>=0.75:
                    haveperson=True
                    person_name=data[j].get("name","未知")
                    break
            if haveperson:
                appendText(f"{person_name}置信度:{similarity:.3f}",(x1+30,y1),frame)

            # 收集人脸数据
            face_data = {
                "embedding": embedding.tolist(),
            }
            all_faces_data.append(face_data)

        show_result = True

    cv2.imshow("Face", frame)

# 释放摄像头和 OpenCV 窗口
cap.release()
cv2.destroyAllWindows()

# 退出时询问是否保存
if all_faces_data:
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.geometry("1x1+0+0")
    root.lift()
    root.focus_force()
    root.update()
    if messagebox.askyesno("保存", "是否要保存人脸数据？", parent=root):
        name = simpledialog.askstring("保存名称", "请输入姓名：", parent=root)
        if name:
            save_data = []
            for fd in all_faces_data:
                fd["name"] = name
                save_data.append(fd)

            filepath = SAVE_PATH
            existing = []
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

            existing.extend(save_data)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"已保存到 {filepath}")
    root.destroy()