import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

# مسار الداتا
DATA_DIR = "archive/train"

# 1. نبني قاموس: النص → قائمة فيديوهات
def build_text2video_map(data_dir):
    text2videos = {}
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        videos = [os.path.join(label_dir, f) for f in os.listdir(label_dir) 
                  if f.lower().endswith((".mp4", ".avi", ".mov"))]
        if videos:
            text2videos[label.lower()] = videos
    return text2videos

text2videos = build_text2video_map(DATA_DIR)

# 2. نعمل API
app = FastAPI()

# نعمل ماب لـ static folder (عشان نعرض الفيديوهات كـ لينكات)
app.mount("/videos", StaticFiles(directory=DATA_DIR), name="videos")

@app.get("/sign_video/")
def get_sign_video(text: str):
    text = text.lower()
    if text not in text2videos:
        raise HTTPException(status_code=404, detail=f"No sign video found for: {text}")
    
    # ناخد أول فيديو للكلمة
    video_path = text2videos[text][0]
    video_name = os.path.relpath(video_path, DATA_DIR).replace("\\", "/")  
    video_url = f"/videos/{video_name}"
    
    return {"text": text, "video_url": video_url}
