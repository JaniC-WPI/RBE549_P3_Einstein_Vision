import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image as Img
import torchvision
from torchvision.transforms import functional as F
import torch.nn.functional as F_nn

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)



# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the path to the video file
video_file = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene2/Undist/2023-03-03_10-31-11-front_undistort.mp4"

# Set the path to the directory where the output images will be saved
output_dir = "/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/Sequences/scene2/images/"

# load model

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/best_93.pt', force_reload=True)

# model = torch.hub.load('CAIC-AD/YOLOPv2', 'yolopv2')

# # load model
# model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=False, classes=43)
# ckpt_ = torch.load('/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/best_93.pt')['model']
# model.load_state_dict(ckpt_.state_dict(), strict=False)
# copy_attr(model, ckpt_, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
# model = model.fuse().autoshape()
# model.to(device)

# weights_path = '/home/jc-merlab/RBE549_P3_Einstein_Vision/P3Data/best_93.pt'
# model = torch.load(weights_path)



# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Loop through the frames in the video
frame_num = 1
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    img = results.render()[0]

    cv2.imshow("signals", img)
    cv2.waitKey(2)

    label_details = results.pandas().xyxy[0]

    print(label_details)

    label_csv = label_details.to_csv(os.path.join(output_dir, "labels{:04d}.csv".format(frame_num)), index=False)

    # inf_img = Img.fromarray(frame)

    # inf_img = F.to_tensor(inf_img).to(device)
    # print(inf_img.shape)
    # inf_img.unsqueeze_(0)
    # print(inf_img.shape)
    # # inf_img = list(inf_img)
    # # print(inf_img)
    # det_out, da_seg_out,ll_seg_out = model(inf_img)
    # with torch.no_grad():
    #     model.to(device)
        # model.eval()
    
    # print(det_out)
    # # Reshape and interpolate the tensors to a common shape
    # new_shape = (model.output_size, model.output_size)
    # det_out = [F_nn.interpolate(x.permute(0, 2, 3, 1), new_shape, mode='bilinear').permute(0, 3, 1, 2) for x in det_out]

    # # Concatenate the list of tensors into a single tensor
    # det_out = torch.cat(det_out, dim=0)
    # # Reshape the output tensor
    # det_out = det_out.reshape(-1, model.num_anchors, model.num_classes + 5, model.output_size, model.output_size)
    # det_out = det_out.permute(0, 3, 4, 1, 2)

    # Extract the class probability tensor
    # class_probs = torch.sigmoid(det_out[..., 5:])
    
    # Find the class with the highest probability score for each anchor box
    # _, class_preds = torch.max(det_out, dim=1)

    # # Convert the class predictions to a numpy array
    # class_preds = class_preds.cpu().numpy()

    # print(class_preds)

    # Draw the predicted bounding boxes and class labels on the input image
    # for i in range(det_out.shape[0]):
    #     for j in range(det_out.shape[1]):
    #         for k in range(det_out.shape[2]):
    #             if det_out[i, j, k, 4] > 0.5:
    #                 x1 = int((det_out[i, j, k, 0] - det_out[i, j, k, 2]/2) * frame.shape[1])
    #                 y1 = int((det_out[i, j, k, 1] - det_out[i, j, k, 3]/2) * frame.shape[0])
    #                 x2 = int((det_out[i, j, k, 0] + det_out[i, j, k, 2]/2) * frame.shape[1])
    #                 y2 = int((det_out[i, j, k, 1] + det_out[i, j, k, 3]/2) * frame.shape[0])
    #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                 cv2.putText(frame, f"{model.classes[class_preds[i, j, k]]} ({det_out[i, j, k, 4]:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # print(det_out, da_seg_out,ll_seg_out)

    # cv2.imshow("line image", output.render()[0])
    # cv2.waitKey(2)  

    frame += 1
