# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import base64
import csv
import os
import platform
import sys
from pathlib import Path
import time
import torch
import paho.mqtt.client as mqtt

import socketio
import asyncio
from aiohttp import web

import socketio

sio = socketio.AsyncClient()

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("esp32cheat/notif")

def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload.decode()}")
    
@sio.event
async def connect():
    print('connection established')

@sio.event
async def disconnect():
    print('disconnected from server')

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(username="emqx",password="public")
client.connect("localhost", 1883, 60)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

async def send(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        await sio.emit('frame', jpg_as_text)      
    except :
        print("Error sending frame")
        
async def notify(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        await sio.emit('cheat_notif', jpg_as_text)      
    except :
        print("Error sending frame")

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
project=ROOT / "runs/detect" 
name="exp" 
save_txt=False
weights = "best.pt"
source = str("simulasi_cheat.mp4")
save_img = True  
data=ROOT / "data.yaml", 
imgsz=(640, 640)
is_streaming = False

device = select_device("")
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  

model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
dataset = None

is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
screenshot = source.lower().startswith("screen")
if is_url and is_file:
    source = check_file(source)  

save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
(save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

bs = 1  # batch_size
if webcam:
    view_img = check_imshow(warn=True)
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    bs = len(dataset)
elif screenshot:
    dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
vid_path, vid_writer = [None] * bs, [None] * bs


@smart_inference_mode()
async def run(
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,  
    view_img=True,  
    save_txt=False,  
    save_crop=False, 
    classes=None,  
    agnostic_nms=False,  
    augment=False,  
    visualize=False,  
    update=False, 
    line_thickness=3,  
    hide_labels=False,  
    hide_conf=False, 
):
    global seen, device, model , imgsz, stride, source, weights, is_streaming, dt, window, dataset
    curtime = round(time.time() * 1000)

    for path, im, im0s, vid_cap, s in dataset:
        if not is_streaming : break 
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float() 
            im /= 255 
            if len(im.shape) == 3:
                im = im[None]  
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

       
        for i, det in enumerate(pred):  
            seen += 1
            if webcam:  
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  
            save_path = str(save_dir / p.name)  # im.jpg
            s += "%gx%g " % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            detected = 0
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " 
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls) 
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    if abs(curtime - round(time.time() * 1000)) > 3000 and (names[c]=="Cheating" or names[c]=="Mobile" )and confidence >= .35:
                        if not client.is_connected() :
                            detected += 1
                            client.reconnect()
                            client.publish("esp32cheat/notif", "detected")
                            client.publish("esp32cheat/kecurangan", str(c) +" - " + names[int(c)] + " conf : " + confidence_str)
                            loop = asyncio.get_event_loop()
                            curtime = round(time.time() * 1000)

                    if save_img or save_crop or view_img and confidence >= .3:
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

              

            # Stream results
            im0 = annotator.result()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = loop.create_task(send(im0))
                    await task
                else:
                    loop.run_until_complete(send(im0))
            except :
                pass
            
            if detected > 0:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = loop.create_task(notify(im0))
                    await task
                else:
                    loop.run_until_complete(notify(im0))
                
         
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


async def mains():
    await sio.connect('http://localhost:3000')

    await sio.wait()

task = None

@sio.event
async def start_stream(data):
    global is_streaming, task
    if not is_streaming :
        is_streaming = True
        print("Starting stream")
        try:
            task = asyncio.create_task(run())
        except:
            
            print("Error create task")
        
@sio.event
async def stop_stream(data):
    global is_streaming, task
    if is_streaming :
        is_streaming = False

if __name__ == "__main__":
    asyncio.run(mains())

    
client.loop_forever()
