from pathlib import Path
import cv2
import tkinter as tk
from tkinter import Tk, Canvas, Button, PhotoImage
import threading
#from ffpyplayer.player import MediaPlayer
import subprocess
from plot import start_plot
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import pycuda.autoinit
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
import argparse
import yaml
import time
import pafy


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = Path("assets")

def non_maximum_suppression_fast(boxes, overlapThresh=0.3):
    
    # If there is no bounding box, then return an empty list
    if len(boxes) == 0:
        return []
        
    # Initialize the list of picked indexes
    pick = []
    
    # Coordinates of bounding boxes
    x1 = boxes[:,0].astype("float")
    y1 = boxes[:,1].astype("float")
    x2 = boxes[:,2].astype("float")
    y2 = boxes[:,3].astype("float")
    
    # Calculate the area of bounding boxes
    bound_area = (x2-x1+1) * (y2-y1+1)
    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    sort_index = np.argsort(y2)
    
    
    # Looping until nothing left in sort_index
    while sort_index.shape[0] > 0:
        # Get the last index of sort_index
        # i.e. the index of bounding box having the biggest y2
        last = sort_index.shape[0]-1
        i = sort_index[last]
        
        # Add the index to the pick list
        pick.append(i)
        
        # Compared to every bounding box in one sitting
        xx1 = np.maximum(x1[i], x1[sort_index[:last]])
        yy1 = np.maximum(y1[i], y1[sort_index[:last]])
        xx2 = np.minimum(x2[i], x2[sort_index[:last]])
        yy2 = np.minimum(y2[i], y2[sort_index[:last]])        

        # Calculate the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # Compute the ratio of overlapping
        overlap = (w*h) / bound_area[sort_index[:last]]
        
        # Delete the bounding box with the ratio bigger than overlapThresh
        sort_index = np.delete(sort_index, 
                               np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    # return only the bounding boxes in pick list        
    # return boxes[pick]
    return pick

def load_engine(trt_runtime, plan_path):

    engine = trt_runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
    return engine

def allocate_buffers(engine, batch_size):

    inputs = []
    outputs = []
    bindings = []
    # data_type = engine.get_binding_dtype(0)

    for binding in engine:
        # print(engine.get_binding_dtype(binding))
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        dic = {
                "host_mem" : host_mem,
                "device_mem" : device_mem,
                "shape" : engine.get_binding_shape(binding),
                "dtype" : dtype
            }
        if engine.binding_is_input(binding):
            inputs.append(dic)
        else:
            outputs.append(dic)

    stream = cuda.Stream()
    return inputs , outputs , bindings , stream

def load_images_to_buffer(pics, pagelocked_buffer):
   
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed)

def do_inference(context, pics_1, inputs , outputs , bindings , stream, model_output_shape):

    start = time.perf_counter()
    load_images_to_buffer(pics_1, inputs[0]["host_mem"])

    [cuda.memcpy_htod_async(intput_dic['device_mem'], intput_dic['host_mem'], stream) for intput_dic in inputs]

    # Run inference.

    # context.profiler = trt.Profiler()
    context.execute(batch_size=1, bindings=bindings)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(output_dic["host_mem"], output_dic["device_mem"], stream) for output_dic in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return the host output.
    out = outputs[0]["host_mem"].reshape((outputs[0]['shape']))
    # out = h_output

    return out , time.perf_counter() - start


def draw_detect(img , x1 , y1 , x2 , y2 , conf , class_id , label , color_palette):
    # label = f'{CLASSES[class_id]} ({confidence:.2f})'
    global f
    T=time.localtime()
    result = time.strftime("%H:%M:%S ",T)
    color = color_palette[class_id]
    print(label[class_id])
    f.write(result)
    f.write(label[class_id])
    f.write(" ")
    f.write(f"{conf:0.3}")
    f.write('\n')
    # print(x1 , y1 , x2 , y2 , conf , class_id)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    cv2.putText(img, f"{label[class_id]} {conf:0.3}", (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def show_detect(img , preds , iou_threshold , conf_threshold, class_label , color_palette):
    boxes = []
    scores = []
    class_ids = []
    
    detect=False

    # print()
    max_conf = np.max(preds[0,4:,:] , axis=0)
    idx_list = np.where(max_conf > conf_threshold)[0]
    
    # for pred_idx in range(preds.shape[2]):
    for pred_idx in idx_list:

        pred = preds[0,:,pred_idx]
        conf = pred[4:]
        
        
        box = [pred[0] - 0.5*pred[2], pred[1] - 0.5*pred[3] , pred[0] + 0.5*pred[2] , pred[1] + 0.5*pred[3]]
        boxes.append(box)

        label = np.argmax(conf)
        
        scores.append(max_conf[pred_idx])
        class_ids.append(label)

    boxes = np.array(boxes)
    result_boxes = non_maximum_suppression_fast(boxes, overlapThresh=iou_threshold)
    

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        
        draw_detect(img, round(box[0]), round(box[1]),round(box[2]), round(box[3]),
            scores[index] , class_ids[index] , class_label , color_palette)
        detect=True
    
    return detect

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)
"""
def get_audio(file):
    from moviepy.editor import VideoFileClip
    # Load the MP4 file
    video = VideoFileClip(file + ".mp4")
    # Extract the audio
    audio = video.audio
    # Save the audio as an MP3 file
    audio.write_audiofile(file + ".mp3")

def play_video(video_path):
    video = cv2.VideoCapture(video_path + ".mp4")

    while True:
        grabbed, frame = video.read()
        frame = cv2.resize(frame, (800, 500))

        if not grabbed:
            print("End of video")
            break

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        current_time=video.get(cv2.CAP_PROP_POS_MSEC)/1000
        print(current_time)

    video.release()
    cv2.destroyAllWindows()
"""

def open_camera():
    global f
    f=open('test.txt','w')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.imshow('Camera', frame)
        start_time=time.perf_counter()

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = np.array(im, dtype=np.float32, order='C')
        im = im.transpose((2, 0, 1))
        im /=  255

        out , _ = do_inference(context, im, inputs , outputs , bindings, stream, model_output_shape)
        show_detect(frame , out , iou_threshold , conf_threshold , label , color_palette)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            f.close()
            pycuda.autoinit.context.pop()
            break

    cap.release()
    cv2.destroyAllWindows()


def movie():
    video_thread = threading.Thread(target=play_video, args=("dog_cat",))
    video_thread.start()
    open_camera()
    video_thread.join()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
engine_path = "emotion_re/best.engine"
engine = load_engine(trt_runtime, engine_path)
iou_threshold = 0.45
conf_threshold = 0.25

yaml_path = "emotion.yaml"

with open(yaml_path, 'r') as stream:
    data = yaml.load(stream)

label = data['names']
color_palette = np.random.uniform(0, 255, size=(len(label), 3))

inputs , outputs , bindings , stream = allocate_buffers(engine, 1)
context = engine.create_execution_context()

WIDTH = inputs[0]["shape"][2]
HEIGHT = inputs[0]["shape"][3]

model_output_shape = outputs[0]['shape']

f=open('test.txt','w')

window = Tk()

window.geometry("800x500")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 500,
    width = 800,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    400.0,
    250.0,
    image=image_image_1
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=movie,
    relief="flat"
)
button_1.place(
    x=124.0,
    y=196.0,
    width=219.0,
    height=54.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=open_camera,
    relief="flat"
)
button_2.place(
    x=455.0,
    y=196.0,
    width=219.0,
    height=54.0
)
button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=start_plot,
    relief="flat"
)
button_3.place(
    x=290.0,
    y=301.0,
    width=219.0,
    height=54.0
)
window.resizable(False, False)
window.mainloop()
