import socket
import joblib
import pandas as pd

HOST = "127.0.0.1"
PORT = 5000
print("Loading model...")
model = joblib.load(r"C:\Users\garvi\OneDrive\Documents\Code\Real-Time Stress and Emotion Detection System\stress_model.pkl")
scaler = joblib.load(r"C:\Users\garvi\OneDrive\Documents\Code\Real-Time Stress and Emotion Detection System\feature_scaler.pkl")
print("Model loaded.")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print("Waiting for C++ client...")
conn, addr = server.accept()
print("Connected:", addr)
feature_columns = ["0" , "1" , "2" , "3" , "4" , "5" , "6" , "7" , "8" , "9" , "10" , "11" , "eye_mouth_ratio", "jaw_face_ratio", "brow_ratio", "eye_diff", "mouth_jaw_ratio"]

while True:
    data = conn.recv(1024).decode()

    if not data:
        break

    try:
        base = list(map(float, data.split(",")))
        ear , mar , eye_w_l , eye_w_r , mouth_w , lip_open , brow_dist , brow_eye , jaw_open , face_h , face_w , face_ratio = base
        eye_mouth_ratio = ear/(mar+1e-6)
        jaw_face_ratio = jaw_open/(face_h+1e-6)
        brow_ratio = brow_dist/(brow_eye+1e-6)
        eye_diff = abs(eye_w_l-eye_w_r)
        mouth_jaw_ratio = mouth_w/(jaw_open+1e-6)
        features = [ear , mar , eye_w_l , eye_w_r , mouth_w , lip_open , brow_dist , brow_eye , jaw_open , face_h , face_w , face_ratio , eye_mouth_ratio , jaw_face_ratio , brow_ratio , eye_diff , mouth_jaw_ratio]
        df = pd.DataFrame([features], columns=feature_columns)
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        conn.send(str(pred).encode())

    except Exception as e:
        print("Error:", e)
        conn.send(b"0")

conn.close()