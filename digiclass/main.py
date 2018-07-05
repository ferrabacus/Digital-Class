import cv2
import os
import shutil
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import datetime
from PIL import Image
from io import BytesIO
import json
import smtplib
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

#
##
### Following methods are related to facial detection & recognition
##
#

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    part = []
    for (x, y, w, h) in faces:
        part.append(gray[y:y+h, x:x+w])
    return part, faces

def detect_train(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], faces[0]

def prepare_training_data(data_folder_path):
    dirs = sorted(os.listdir(data_folder_path))
    faces = []
    labels = []
    label_count = 0
    for dir_name in dirs:
        if dir_name.startswith("."):
            continue
        label = label_count
        label_count += 1
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_train(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    return faces, labels

def predict(test_img):
    img = test_img.copy()
    gray, faces = detect_face(img)
    i = 0
    for(x, y, w, h) in faces:
        label, confidence = face_recognizer.predict(gray[i])
        label_text = students[label]
        confidence = round(confidence, 2)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            saveFace(label_text, gray[i])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if(confidence < 100):
            cv2.putText(img, label_text + " " + str(confidence), (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
        if(confidence < 50):
            take_attendance()
        i+=1
    return img

#
##
### Following methods are related to storing info or images to files
##
#

#def storeJson():

def store_dict(dict, report_type, path):
    df = pd.DataFrame.from_dict(dict, orient="index", columns=[report_type])
    date = datetime.datetime.now()
    df.to_csv(path + str(date) + '.csv',index=True, header=True)

def saveFace(label, img):
    date = datetime.datetime.now()
    cv2.imwrite("picture-data/test-data/" + label + "/" + str(date) + '.jpg', img)

def mashImages():
    path = "picture-data/test-data/"
    dir = sorted(os.listdir(path))
    image_list = []
    width_list = []
    height_list = []

    for student in dir:
        if student.startswith('.'):
            continue
        student_path = "picture-data/test-data/" + student
        images = os.listdir(student_path)
        for image in images:
            if image.startswith('.'):
                continue
            img1 = Image.open(image)
            image_list.append(img1)
            width, height = img1.size
            width_list.append(width)
            height_list.append(height)

        for i in width_list:
            width_total += i
        height_total = max(height_list)

        result = Image.new('RGB', (width_total, height_total))

        for i in image_list:
            result.paste(im=i)

        date = datetime.datetime.now()
        cv2.imwrite("picture-data/emote-data/" + student + "/" + str(date) + '.jpg', result)

def store_attendance():
    store_dict(attendance_data, "Attendance")

#
##
### Following methods are related to reports_handler()
##
#

def emotion(label):
    image_path = 'picture-data/test-data/' + label
    dirs = os.listdir(image_path)
    picture = []
    for image in dirs:
        if image.startswith("."):
            continue
        picture.append(image_path + '/' + image)

    img = cv2.imread(picture[0], 90)
    date = datetime.datetime.now()
    cv2.imwrite(image_path + '/' + str(date) + '.jpg', img)
    path = image_path + '/' + str(date) + '.jpg'

    #insert microsoft face API key here
    subscription_key = ''
    assert subscription_key

    face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'

    headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key
        }
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
        'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
        }
    data = open(path, 'rb').read()
    response = requests.post(face_api_url, params=params, headers=headers, data=data)
    faces = response.json()

    # image = Image.open(BytesIO(data))
    # plt.figure(figsize=(8, 8))
    # ax = plt.imshow(image, alpha=0.6)
    # for face in faces:
    #     fr = face["faceRectangle"]
    #     fa = face["faceAttributes"]
    #     origin = (fr["left"], fr["top"])
    #     p = patches.Rectangle(
    #         origin, fr["width"], fr["height"], fill=False, linewidth=2, color='b')
    #     ax.axes.add_patch(p)
    #     plt.text(origin[0], origin[1], "%s, %d"%(fa["gender"].capitalize(), fa["age"]),
    #          fontsize=20, weight="bold", va="bottom")
    #     _ = plt.axis("off")
    # date = datetime.datetime.now()
    # plt.savefig("emote-data/" + label + "/" + str(date) + '.png')

    date = datetime.datetime.now()
    with open("picture-data/emote-data/" + label + "/" + str(date) + '.txt', 'w') as outfile:
        json.dump(faces, outfile, indent = 4, sort_keys=True)

def emotion_analysis():

    answer = input("Emotion analysis for all or individual student?: ")

    if(answer == "all"):
        for i in students:
            emotion(i)

    elif(answer == "individual"):
        for i in students:
            print(i)

        which = input("Enter student name to analyze: ")
        for i in students:
            if(which == i):
                print("Analyzing: " + i)
                emotion(i)

def take_attendance(label):
    attendance[students[label]] = "Present"

def email_emotion():
    pass
def email_attendance():
    pass
#Enter your own email and passcode here
MY_ADDRESS = ''
PASSWORD = ''

def get_contacts(filename):
    """
    Return two lists names, emails containing names and email addresses
    read from a file specified by filename.
    """

    names = []
    emails = []
    with open(filename, mode='r', encoding='utf-8') as contacts_file:
        for a_contact in contacts_file:
            names.append(a_contact.split()[0])
            emails.append(a_contact.split()[1])
    return names, emails

def read_template(filename):
    """
    Returns a Template object comprising the contents of the
    file specified by filename.
    """

    with open(filename, 'r', encoding='utf-8') as template_file:
        template_file_content = template_file.read()
    return Template(template_file_content)

def main_email():
    names, emails = get_contacts('mycontacts.txt') # read contacts
    message_template = read_template('message.txt')

    # set up the SMTP server
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(MY_ADDRESS, PASSWORD)

    # For each contact, send the email:
    for name, email in zip(names, emails):
        msg = MIMEMultipart()       # create a message

        # add in the actual person name to the message template
        message = message_template.substitute(PERSON_NAME=name.title())

        # Prints out the message body for our sake
        print(message)

        # setup the parameters of the message
        msg['From']=MY_ADDRESS
        msg['To']=email
        msg['Subject']="This is TEST"

        # add in the message body
        msg.attach(MIMEText(message, 'plain'))


        # send the message via the server set up earlier.
        s.send_message(msg)
        del msg

    # Terminate the SMTP session and close the connection
    s.quit()

#
##
### Following methods are used in classroom_handler()
##
#
def capture():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        img = predict(frame)
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()



def training_all():
    print("")
    print(OKGREEN + "****DIRECTIONS****" + ENDC)
    print("Program will loop through and gather photos on each student")
    print("Press q when finished with one student")
    print("Window will close and then prompt you for the next student")
    print(OKGREEN + "******************" + ENDC)
    print("")

    for i in students:
        answer = input("Press enter when ready to train for" + i)

        while True:
            video_capture = cv2.VideoCapture(0)
            ret, frame = video_capture.read()
            if not ret:
                break
            gray, face = detect_train(frame)
            cv2.imshow(i, frame)
            saveFace(i, gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

def training_by_input(new_subjects):

        print("")
        print(OKGREEN + "****DIRECTIONS****" + ENDC)
        print("Program will loop through each new student for training")
        print("Press q when done taking photos of student")
        print("Window will close and then prompt you for the next student")
        print("Press enter when ready for new student")
        print(OKGREEN + "******************" + ENDC)
        print("")

        for i in new_subjects:

            answer = input("Press enter when ready to train " + i)

            while True:
                video_capture = cv2.VideoCapture(0)
                ret, frame = video_capture.read()
                if not ret:
                    break
                gray, face = detect_train(frame)
                cv2.imshow(i, frame)
                saveFace(i, gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            video_capture.release()
            cv2.destroyAllWindows()

#
##
### Following methods are used in subject_handler()
##
#

def load_subjects():
    subject_path = "picture-data/training-data/"
    subject_dirs = sorted(os.listdir(subject_path))
    subjects = []

    for i in subject_dirs:
        if i.startswith("."):
            continue
        subjects.append(i)
        attendance[i] = "Absent" #Defaults students to absent

    return subjects

def create_new_subjects():
    test_path = "picture-data/test-data/"
    emote_path = "picture-data/emote-data/"
    training_path = "picture-data/training-data/"
    new_subjects = []

    while(True):
        number_to_create = int(input("How many subjects would you like to create?: "))

        for i in range(number_to_create):
            name = input(str(i) + ": Enter new subject name: ")
            new_subjects.append(name)

            if not os.path.exists(test_path + name):
                os.makedirs(test_path + name)
            if not os.path.exists(emote_path + name):
                os.makedirs(emote_path + name)
            if not os.path.exists(training_path + name):
                os.makedirs(training_path + name)

        training_by_input(new_subjects)

        break

def delete_subjects():
    test_path = "picture-data/test-data/"
    emote_path = "picture-data/emote-data/"
    training_path = "picture-data/training-data"

    while(True):
        number_to_delete = int(input("How many subjects would you like to delete?: "))

        for i in range(number_to_delete):
            name = input(str(i) + ": Enter subject to delete: ")

            if os.path.exists(test_path + name):
                shutil.rmtree(test_path + name)
            if os.path.exists(emote_path + name):
                shutil.rmtree(emote_path + name)
            if os.path.exists(training_path + name):
                shutil.rmtree(training_path + name)
        break

#
##
### Following methods are used in training_handler()
##
#

def train_recognizer():
    print("******************")
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    print("******************")

    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write('trainer.yml')

def load_recognizer():
    face_recognizer.read('trainer.yml')

#
##
### Following methods are the main program loops handlers
##
#
def subject_handler():
    retrain = False
    while(True):
        answer = input("Would you like to add, delete, or keep subjects: ")

        if(answer == "add"):
            create_new_subjects()
            subjects = load_subjects()
            retrain = True
            break
        elif(answer == "delete"):
            delete_subjects()
            subjects = load_subjects()
            retrain = True
            break
        elif(answer == "keep"):
            subjects = load_subjects()
            retrain = False
            break
        else:
            print("I do not understand your input. Try again!")

    return retrain, subjects

def training_handler(retrain):
    if(retrain == True):
        print("You either added or deleted subjects and we need to collect photos & retrain!")
        training_all()

    elif(retrain == False):
        print("You did not add or delete any subjects")

        print("")
        print(OKBLUE + "******INPUT*******" + ENDC)
        print("add -> allows you to add photos to a current subject, create a trainer.yml, and load the recognizer")
        print("load -> allows you to load a trainer.yml file to the recognizer")
        print("train -> writes a trainer.yml from training-data and loads to the recognizer")
        print(OKBLUE + "******************" + ENDC)
        print("")
        answer = input("Would you like to add, load, or train: ")

        if(answer == "add"):
            training_all()
            train_recognizer()
        elif(answer == "load"):
            if os.path.isfile("trainer.yml"):
                print("I have found a training yml for the face recognizer and will load")
                load_recognizer()
            else:
                print("I have found no training yml for the face recognizer and will train it from training-data")
                train_recognizer()
        elif(answer == "train"):
            train_recognizer()

def classroom_handler():
    print("")
    print(OKGREEN + "****DIRECTIONS****" + ENDC)
    print("Gathering attendance data and photos for emotion recognition")
    print("Press c to take faceshot for emotional analysis")
    print("Press q when done")
    print(OKGREEN + "******************" + ENDC)
    print("")
    capture()
    print("******************")
    print("Attendance data stored and ready for emotion recognition")
    emotion_analysis()

def reports_handler():

    #mashImages()

    print("")
    print(OKBLUE + "******INPUT*******" + ENDC)
    print("emote -> Mash together faceshots taken and stored in test-data, then send for analysis")
    print("emotion -> Email emotion.csv to teacher")
    print("attendance -> Email attendance.csv to teacher")
    print(OKBLUE + "******************" + ENDC)
    print("")
    answer = input("Enter emote, emotion, or attendance: ")

    if(answer == "emote"):
        emotion_analysis()
    elif(answer == "emotion"):
        email_emotion()
    elif(answer == "attendance"):
        email_attendance()
#
##
### MAIN PROGRAM LOOP
##
# List for students
attendance = {}
students = []
email =

while(True):
    retrain = False
    students = load_subjects()

    print("")
    print(OKBLUE + "******INPUT*******" + ENDC)
    print("students -> Adding/Deleting Students")
    print("training -> Gathering photos/Training recognizer")
    print("classroom -> Live video feed to recognize and take attendance")
    print("reports -> Emotional analysis & email reports")
    print(OKBLUE + "******************" + ENDC)
    print("")

    answer = input("Where would you like to start: students, training, classroom, reports, or exit: ")

    #add, delete, keep current subjects
    #returns boolean value if we need to retrain the recognizer
    #returns list of all subjects
    if(answer == "student"):
        retrain, students = subject_handler()
        if(retrain == True):
            training_handler(retrain)

    #Re-train face recognizer or load data for face recognizer
    elif(answer == "training"):
        training_handler(retrain)

    #Determine attendance & gather data for emotion recognition
    elif(answer == "classroom"):
        load_recognizer()
        classroom_handler()

    #Reports handling
    elif(answer == "reports"):
        reports_handler()

    elif(answer == "exit"):
        print("Adios")
        break

    else:
        print("I did not understand that. Let's try again")
