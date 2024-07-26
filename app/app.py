import json
import os
import base64
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.form:
        return jsonify({'error': 'No image provided'}), 400
    
    image_data = request.form['image']
    
    # Decode the image
    try:
        image_data = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({'error': 'Invalid image data'}), 400

    # Save the image to the server
    image_path = os.path.join(UPLOAD_FOLDER, str(request.form['name'])+'.jpg')
    with open(image_path, 'wb') as image_file:
        image_file.write(image_data)

    return jsonify({'success': 'Image uploaded successfully', 'path': image_path})

@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")

@socketio.on('connect')
def handle_connect():
    print('connection established')

@socketio.on('disconnect')
def handle_disconnect():
    print('disconnected from server')

@socketio.on('my_message')
def handle_message(data):
    print('message received with ', data)
    emit('response', {'data': 'Message received!'})

@socketio.on('get_video')
def handle_get_video(data):
    print('Menerima data video')

@socketio.on('frame_data')
def handle_frame_data(data):
    print("frame")

if __name__ == "__main__":
    socketio.run(app, debug=True, host="localhost", port=5000)
