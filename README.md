# CSC405-Group14-SpeechRecognition
A project on Speech Recognition using Hidden Markov Model

SPEECH RECOGNITION SYSTEM - GROUP 14

SYSTEM REQUIREMENTS:
- Python 3.11
- Internet connection (required only for the initial setup to download libraries)

ONLINE REPOSITORY:
https://github.com/Hirosus/CSC405-Group14-SpeechRecognition

ONLINE LINK TO ACCESS THE APP:
https://csc405-group14-speechrecognition-9i4mga9ohvulsuksbh4tvc.streamlit.app

HOW TO SET UP & RUN THE APPLICATION ON A SYSTEM IF NOT USING THE ONLINE LINK:

Step 1: Setup
   - Copy the entire folder to your Desktop.
   - Open a terminal (Command Prompt or PowerShell) inside this folder.
     (Tip: Right-click inside the folder > "Open in Terminal")

Step 2: Create Environment
   - Run this command to create a virtual environment:
     python -m venv venv

Step 3: Activate Environment
   - Windows: venv\Scripts\activate
   - Mac/Linux: source venv/bin/activate

Step 4: Install Dependencies
   - Run this command to install the necessary libraries:
     pip install -r requirements.txt

Step 5: Run Application
   - Start the system with this command:
     streamlit run app.py

Step 6: Usage
   - The application will open automatically in your web browser.
   - You can record your voice or upload the provided test files from the 'dataset' folder.

NOTE:
- The trained models are already included in the 'models/' folder.
- You do not need to run 'train_model.py' unless you wish to retrain the system from scratch.
