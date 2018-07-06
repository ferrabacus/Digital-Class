# Digital Classroom

RoadMap(order doesn't correlate to next task)
  End dependency on Azure, and bring local support for emotion recognition.
  Using classes to better group functionality.
  GUI interface so software can be used by more teachers
  TensorFlow for better profile building and recognition
  Better Reports.
  Multi-Camera Support
  Raspberry Pi Support

TIPS:

* Training necessary before key features will work

Before Running

1. Install OpenCV Python3 bindings

  If using Windows, go to: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html
  If using Mac, go to: https://medium.com/init27-labs/installation-of-opencv-using-anaconda-mac-faded05a4ef6
  If using Linus, go to: https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html

2. Verify OpenCV works

  From Terminal or Command Prompt:
    Enter python shell by entering "python"
    Type "import cv2"

    If it fails, restart your computer and try again.
    If it still fails, try the tutorial again and check that you installed OpenCV for python3

3. Clone Github project

4. Get A Microsoft Azure Face Trial Key

    Go to: https://azure.microsoft.com/en-us/try/cognitive-services/
    This will give you a 7 day trial key without having to enter in a credit card.

    You may make an account and get a free key without being charged, but you will have to enter in a CC incase you use paid features.

    Enter that subscription key in main.py -> emotion(label) -> subscription_key = ''

5. Email Reports

    Above the main loop toward the bottom of main.py, there is a commented line that creates the email_handling.py object.

    You need to send it the correct parameters: your email, password, and smtp server.

    Then, go to the email-templates folder. The mycontacts.txt is where you will put the names and emails of who should receive email reports.

6. Quick Overview

  Folders
  -> email-templates : basic templates used to send prefixed messages and contacts
  -> picture-data : where training images, emotion images, screenshots are stored.
  -> reports-data : attendance & emotion reports are stored.

  Files
  -> email_handling.py : all code related to sending reports.
  -> main.py : all code related to running the main app, facial detection, facial recognition.
  -> haarcascade_frontalface_default.xml : Data file used to train OpenCV face_recognizer on what a face looks like.


7. Testing

  To run, open terminal or command line.
    If using terminal, enter: python3 main.py
    If using command line, enter: python main.py

  From there, follow console instructions.
