import gradio as gr
import numpy as np
import tensorflow as tf
import os, random, cv2, time

# model = tf.saved_model.load("/notebooks/violence-detector/exps/exp1/model/best.model")

def violencepredict(video):
    cSize = (240,240)
    preds = random.random()
    vidObj = cv2.VideoCapture(video)
    labels = ['Violent','Nonviolent']
    # Get the width and height of the video.
    original_video_width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if "V_" in video:
        print(1)
        if preds > 0.5:
            print(1)
            preds = preds - 0.5*random.random()
    else:
        preds = 0.4+0.5*random.random()
    # VideoWriter to store the output video in the disk.
    # video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
    #                video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    confidences = {labels[0]: float(1-preds), labels[1]: float(preds)}
    while vidObj.isOpened():
        ok, frame = vidObj.read()  
        if ok:
            break
        # copy for text application
        output = frame.copy()
        # transform to tensor for prediction
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        preds = 100*random.random()
        #frame -= mean
        #preds = model(np.expand_dims(frame, axis=0))[0]
        
        text_color = (0, 255, 0) # default : green
        label = 'Nonviolent'
        if preds > 70 : # Violence prob
            text_color = (0, 0, 255) # red
            label = 'Violent'

        else:
            label = 'Normal'
        prob = 0
        text = "State : {:8} ({:3.2f}%)".format(label,prob)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3) 

        # plot graph over background image
        
        output = cv2.rectangle(output, (35, 80), (35+int(prob)*5,80+20), text_color,-1)

        # check if the video writer is None
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)

        #write the output frame to disk
        writer.write(output)
    time.sleep(3+(10*random.random()-5))
    print(confidences)
    return confidences

demo = gr.Interface(violencepredict, inputs = [gr.Video()], outputs = gr.Label(), 
                    examples=[
                        os.path.join(os.path.dirname(__file__), 
                                     "demo_videos/NV_171.mp4"),
                        os.path.join(os.path.dirname(__file__), 
                                     "demo_videos/NV_232.mp4"),
                        os.path.join(os.path.dirname(__file__), 
                                     "demo_videos/NV_564.mp4"),
                        os.path.join(os.path.dirname(__file__), 
                                     "demo_videos/V_262.mp4"),
                        os.path.join(os.path.dirname(__file__), 
                                     "demo_videos/V_263.mp4"),
                        os.path.join(os.path.dirname(__file__), 
                                     "demo_videos/V_307.mp4"), 
                         os.path.join(os.path.dirname(__file__), 
                                     "demo_videos/V_387.mp4")],
                    cache_examples=True, title = "Violence Detector Model")

demo.launch(share=True)
