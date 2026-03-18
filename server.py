import socket #Used for network communication
import joblib #Used to load machine learning models
import pandas as pd #Used to create DataFrames

HOST = "127.0.0.1" #Used to define on which IP address the server will run on
PORT = 5000 #Used to define which port the server will use for communication
print("Loading model...")
model = joblib.load([INSERT LOCATION OF THE TRAINED MODEL]) #Used to load the trained Random Forest Model
scaler = joblib.load([INSERT LOCATION OF THE TRAINED MODEL]) #Used to load the trained StandardScaler Model
print("Model loaded.")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Used to create a TCP Socket
server.bind((HOST, PORT)) #Used to bind the server to the previously specified IP Address and Port Number
server.listen(1) #Used to start listening for a maximum of only 1 incoming connection from the client side
print("Waiting for C++ client...")
conn, addr = server.accept() #Used to wait till the client successfully connects
print("Connected:", addr)
feature_columns = ["0" , "1" , "2" , "3" , "4" , "5" , "6" , "7" , "8" , "9" , "10" , "11" , "eye_mouth_ratio", "jaw_face_ratio", "brow_ratio", "eye_diff", "mouth_jaw_ratio"] #Used to define the names of the base features as well as the engineered features made when training the model

while True: #Used to run the server till the time the application is active
    data = conn.recv(1024).decode() #Used to receive the data from the "main.cpp" file with the maximum bytes to read being 1024 and converting the bytes to string

    if not data: #Enters this block when the connection is lost
        break

    try:
        base = list(map(float, data.split(","))) #Used to split the string using commas, convert each value to float and storing these values in a list
        ear , mar , eye_w_l , eye_w_r , mouth_w , lip_open , brow_dist , brow_eye , jaw_open , face_h , face_w , face_ratio = base #Used to unpack the items from the list into their own variables
        eye_mouth_ratio = ear/(mar+1e-6) #Used to calculate and store the ratio of the openness of the eye and the openness of the mouth while also adding a constant, "1e-6", to avoid division by 0
        jaw_face_ratio = jaw_open/(face_h+1e-6) #Used to calculate and store the ratio of the openness of the jaw and the height of the mouth while also adding a constant, "1e-6", to avoid division by 0
        brow_ratio = brow_dist/(brow_eye+1e-6) #Used to calculate and store the ratio of the distance between the eyebrows and the distance between the eye & the eyebrow while also adding a constant, "1e-6", to avoid division by 0
        eye_diff = abs(eye_w_l-eye_w_r) #Used to calculate and store the difference between the eyes
        mouth_jaw_ratio = mouth_w/(jaw_open+1e-6) #Used to calculate and store the ratio of the width of the mouth and the openness of the jaw while also adding a constant, "1e-6", to avoid division by 0
        features = [ear , mar , eye_w_l , eye_w_r , mouth_w , lip_open , brow_dist , brow_eye , jaw_open , face_h , face_w , face_ratio , eye_mouth_ratio , jaw_face_ratio , brow_ratio , eye_diff , mouth_jaw_ratio] #Used to create a list containing all the features
        df = pd.DataFrame([features], columns=feature_columns) #Used to create a DataFrame of all the features
        scaled = scaler.transform(df) #Used to normalize all the features
        pred = model.predict(scaled)[0] #Used to apply the trained model 
        conn.send(str(pred).encode()) #Used to convert the number to string, string to bytes and send the data over the TCP Socket

    except Exception as e:
        print("Error:", e)

conn.close() #Used to close the connection once the loop breaks
