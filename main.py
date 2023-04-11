from flask import Flask, render_template, request,send_file
import pathlib,cv2
import my_detect

app=Flask(__name__,static_folder='static',template_folder='templates')
IMG_PATH = pathlib.Path(__file__).parent

@app.route('/',methods=['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path=IMG_PATH / f'images/{imagefile.filename}' 
    imagefile.save(image_path)
    pred_img_arr = my_detect.predict(str(image_path))
    pred_image_path = IMG_PATH / f'pred_images/{imagefile.filename}' 
    cv2.imwrite(str(pred_image_path), pred_img_arr)
    return send_file(pred_image_path,mimetype='image/PNG')

if __name__=='__main__':
    app.run(port=3000, debug=True)