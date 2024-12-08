import time
from pylsl import StreamInlet, resolve_stream
import torch
from torch import nn
from torch.nn import functional as F
from train import InceptionLikeV2
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import threading
import requests
class_names = ['Blink','Girt', 'Tongue', 'Rest']

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 存储最新预测结果的全局变量
latest_prediction = {
    'timestamp': None,
    'class': None,
    'class_name': None
}

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    return jsonify(latest_prediction)

def run_flask():
    app.run(host='0.0.0.0', port=5000)

print("looking for a stream...")
# First resolve a Motion stream on the lab network
streams = resolve_stream('type', 'EEG')
print(streams)

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# Define parameters
sr = 128  # Hz
piece_duration = 1  # seconds
window_slide = 0.25  # seconds
window_size = sr * piece_duration  # Number of samples in a 3-second window
channel_indices = list(range(3, 17))  # Select channels from 4th to 17th (0-based index)

buffer = []  # To store the EEG data for 3 seconds


# 加载模型
model = InceptionLikeV2()
# Load the trained model
model_path = 'model/blink_girt_tongue_4s_piece1s_still_10min_4classifications_1202/seed1/inception_like_v2.pth'
model=torch.load(model_path)
model.eval()



if __name__ == '__main__':
    # 在单独的线程中启动Flask服务器
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # 继续运行主循环
    
    while True:
        # Pull sample
        sample, timestamp = inlet.pull_sample()

        if timestamp is not None:
            # Extract relevant EEG channels (from 4th to 17th element)
            eeg_data = [sample[i] for i in channel_indices]
            print(eeg_data)
            buffer.append(eeg_data)

            # If we have collected enough data for 3 seconds, classify the data
            if len(buffer) >= window_size:
                buffer = buffer[-window_size:]
                # Convert buffer to tensor for model input
                data_tensor = torch.tensor(buffer).unsqueeze(0)  # Shape: (1, 1, 384, 14)
                data_tensor = data_tensor.permute(0, 2, 1)  # Shape: [1, 14, 384]
                # Perform classification
                with torch.no_grad():
                    output = model(data_tensor)
                    _, predicted_class = torch.max(output, 1)
                    pred_class = predicted_class.item()
                    # 更新最新预测结果
                    latest_prediction.update({
                        'timestamp': time.time(),
                        'class': pred_class,
                        'class_name': class_names[pred_class]
                    })
                    print(f"Predicted Class: {class_names[pred_class]}")
                    # 发送预测结果到Flask服务器
                    try:
                        response = requests.post('http://localhost:5001/receive_prediction', 
                                                json=latest_prediction)
                    except requests.exceptions.RequestException as e:
                        print(f"Error sending prediction to Flask server: {e}")
                # Clear the buffer for the next sliding window
                buffer = buffer[int(window_slide * sr):]

        # Add a small sleep to avoid overwhelming the CPU
        time.sleep(1 / sr)