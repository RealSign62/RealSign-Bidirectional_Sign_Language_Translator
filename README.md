# RealSign-Bidirectional_Sign_Language_Translator

This project aims to close the gap of communication between those who are hard of hearing and those who are not. Real-time bidirectional Indian sign language (ISL) can be carried out with the help of this application. 

### Checklist before running the code
1. A saved Deep Learning model that is trained on an image dataset.
2. A speech-to-text library. We used DeepSpeech model for our project. But any other libraries/APIs can be used. 

### The Application file structure (Present in RealSign folder) is as follows:
RealSign.py has the base application code. 
  - Letters - This folder contains the fingerspelled ISL alphabet images. 
  - Pages - This folder contains two code files.
    - ssl.py - This file has the speech to sign language conversion code.
    - slr.py - This file has the sign language recognition code that converts sign language to text.

The Application has been created using streamlit. To run the code, clone the repository. Run the following command:

`streamlit run RealSign.py`

Our team has also published a systematic literature review on sign language translation systems. You can find the paper [here](https://www.igi-global.com/gateway/article/311448).

### Our team members include:
[Ankith Boggaram](https://github.com/Ankithboggaram/) 

[Aaptha Boggaram](https://github.com/Aaptha0204)

[Aryan Sharma](https://github.com/aryan2090)

[Ashwin Srinivasa Ramanujan](https://github.com/ashwin-0911) 


