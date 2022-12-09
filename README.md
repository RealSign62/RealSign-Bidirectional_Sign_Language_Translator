# RealSign-Bidirectional_Sign_Language_Translator

This project aims to close the gap of communication between those who are hard of hearing and those who are not. Real-time bidirectional Indian sign language (ISL) can be carried out with the help of this application. 

### Checklist before running the code
1. Make sure you have a saved model trained on a sign language database that you can use to predict *Static* sign language gestures. 
2. Since we predict fingerspelled ISL gestures, we uploaded the letters in the letters folder. 
3. A speech to text library. We used deepspeech model for our project. But any other libraries/APIs can be used. 

### The Application file structure (Present in RealSign folder) is as follows:
RealSign.py has the base application code. 
  - Letters - This folder contains the fingerspelled ISL alphabet images. 
  - Pages - This folder contains two code files.
    - ssl.py - This file has the speech to sign language conversion code.
    - slr.py - This file has the sign language recognition code that converts sign language to text.

The Application has been created using streamlit. To run the code, clone the repository. Run the following command:

`Streamlit run RealSign.py`

Our team has also published a systematic literature review on sign language translation systems. You can find the paper [here](https://www.igi-global.com/gateway/article/311448).
