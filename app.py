from flask import Flask,render_template,request,Response
app = Flask(__name__)
import cv2, numpy as np, glob
import os
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def gen_frames_watch(id):  #watch cam
    img1 = cv2.imread('circle.png') #mianitem
    win_name = 'Camera Matching'
    MIN_MATCH = 10
    types = ['*.png','*.jpg']
    images = glob.glob("*.png") + glob.glob("*.jpg")
    print(images)
    currentImage=0  
    if (id=="302"):
        replaceImg=cv2.imread(images[1])
    else:
        replaceImg =cv2.imread(images[2])

    
    rows,cols,ch = replaceImg.shape
    pts1 = np.float32([[0, 0],[0,rows],[(cols),(rows)],[cols,0]])
    zoomLevel = 0   
    processing = True   
    maskThreshold=10

    detector = cv2.ORB_create(1000)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,
                    key_size = 12,
                    multi_probe_level = 1)
    search_params=dict(checks=32)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    cap = cv2.VideoCapture(0)
  
    #---------
    while True:
      
        success, frame = cap.read()
        if img1 is None:
            res = frame
        else:
            img2 = frame
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            kp1, desc1 = detector.detectAndCompute(gray1, None)
            kp2, desc2 = detector.detectAndCompute(gray2, None)
            matches = matcher.knnMatch(desc1, desc2, 2)
            ratio = 0.75
            good_matches = [m[0] for m in matches \
                                if len(m) == 2 and m[0].distance < m[1].distance * ratio]
            matchesMask = np.zeros(len(good_matches)).tolist()
            if len(good_matches) > MIN_MATCH:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
   
                if mask.sum() > MIN_MATCH:
                    matchesMask = mask.ravel().tolist()
                    h,w, = img1.shape[:2]
                    pts = np.float32([ [[0,0]],[[0,h]],[[w,h]],[[w,0]] ])
                    dst = cv2.perspectiveTransform(pts,mtrx)
    
                    dst = cv2.getPerspectiveTransform(pts1,dst)
                   
                    rows, cols, ch = frame.shape
                    distance = cv2.warpPerspective(replaceImg,dst,(cols,rows))
                    rt, mk = cv2.threshold(cv2.cvtColor(distance, cv2.COLOR_BGR2GRAY), maskThreshold, 1,cv2.THRESH_BINARY_INV)
                  
                    # mk = cv2.erode(mk, (3, 3))
                    # mk = cv2.dilate(mk, (3, 3))
                    
                    for c in range(0, 3):
                        frame[:, :, c] = distance[:,:,c]*(1-mk[:,:]) + frame[:,:,c]*mk[:,:]
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'


                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=1, fy=1)
    h, w, _ = overlay.shape 
    rows, cols, _ = src.shape  
    y, x = pos[0], pos[1]  

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0) 
            
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

     
def gen_frames(ide,color):  # generate frame by frame from camera
    if (ide==1 and color=="black"):
        specs_ori = cv2.imread('static/images/newglass2.png', -1)
    #change this portion abhiii
    elif(ide==1 and color=="yellow"):
        specs_ori = cv2.imread('static/images/colorednewglass2.png', -1)

    elif(ide==2 and color=="black"):
        specs_ori = cv2.imread('static/images/newglass1.png', -1)
    elif(ide==2 and color=="yellow"):
        specs_ori = cv2.imread('static/images/colorednewglass1.png', -1)
#C:\Users\dhoni\augmentedsys\static\images\colorednewglass1.png

    while True:
    
        ret, frame = camera.read() 
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in face_rects:
            glass_symin = int(y + 1.5 * h / 6)
            glass_symax = int(y + 2.5 * h / 6)
            #print(glass_symin,glass_symax)
            sh_glass = glass_symax-glass_symin
          
    
            face_glass_roi_color = frame[glass_symin:glass_symax+y, x:x+w]
            # face_glass_roi_color = frame[glass_symin:glass_symax+y, x:x+w]
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
            specs = cv2.resize(specs_ori, (w, int(2*sh_glass)))
      
            transparentOverlay(face_glass_roi_color,specs)
   
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

@app.route("/spectacles")
def index():
    return render_template("home.html")
@app.route("/watches")
def watches():
    return render_template("watches.html")

@app.route("/product", methods=['GET'])
def product():
    val = request.args['id']
    iden  = request.args['val'] #Identifier
    color = request.args['color']
    return render_template("product.html",prod=val,item="specs",identifier=iden,color=color)
    

@app.route("/productwatch", methods=['GET'])
def productwatch():
    val = request.args['id']
    iden  = request.args['val'] #Identifier
    return render_template("prodwatch.html",prod=iden,item="watch2w")

@app.route('/video_feed', methods=['GET'])
def spec_video_feed():
    val  = request.args['id']
    color = request.args['color']
    if(val == "201"):
        return Response(gen_frames(1,color), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_frames(2,color), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/watch_feed",methods=["GET"])
def watch_video_feed():
    id = request.args['id']
    print(id)
    return Response(gen_frames_watch(id), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)