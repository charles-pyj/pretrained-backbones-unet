from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import backbones_unet
import cv2
from time import time

import torch
model = torch.load("../ckpts/convnext_base_ckpt5.pth")


def preprocess(img):
    img = img.resize((224, 224)) 
    img = torch.Tensor(np.array(img, dtype=np.uint8).transpose((2, 0, 1)))
    img = img.float() / 255.0
    return img

def getVideo(in_path,out_path):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    frames = []
    start_time = time()
    try:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if there are no frames left
            
            # Optionally resize or preprocess the frame here if necessary
            frames.append(frame)
    finally:
        cap.release()  # Make sure to release the video capture objec
    print(frames[0].shape)
    H, W, C = frames[0].shape
    segmented_frames = []
    cnt = 0
    #print(len(frames))
    val_every_x_frame = 5
    output = None
    for i in range(len(frames)):
        frame = frames[i]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(Image.fromarray(rgb_frame))
        if i % val_every_x_frame == 0:
            frame = frames[i]
            #print(input_tensor.shape)
            input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
            cnt += 1
            #plt.imshow(input_batch[0].permute(1,2,0))
            #plt.show()
            # Move the input and model to GPU for faster computation
            input_batch = input_batch.to('cuda')
            model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
                #print(output.shape)
        out = input_tensor * output[0].cpu().clamp(0,1)
        # Convert frame to RGB (OpenCV uses BGR format by default)
        #plt.imshow(out.permute(1,2,0))
        #plt.show()
        # Assuming output is a segmented tensor, converting it to an image:
        # This step depends heavily on what your model outputs
        output_image = out.cpu().data.squeeze().numpy()  # Example conversion, adjust as necessary
        segmented_frames.append(output_image)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # or (*'XVID') depending on the desired output format
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (224,224))
    assert(len(segmented_frames) == len(frames))
    for frame in segmented_frames:
        frame = frame.transpose((1, 2, 0))
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        #plt.imshow(frame)
        #plt.show()
        #print(frame.shape)
        # Convert your processed frame back to BGR from RGB if necessary
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()  # Release everything if job is finished
    end_time = time()
    print(f'{end_time - start_time} seconds spent')
    


getVideo("./videos/fast.mp4","./videos/fast_out.mp4")