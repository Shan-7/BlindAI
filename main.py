import cv2
import pytesseract
from gtts import gTTS
import os
from googletrans import Translator
import speech_recognition as sr
import google.generativeai as genai

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Configure the generative AI package with your API key
genai.configure(api_key="AIzaSyCq200-3L_QK4peqhN8zmD6Mb67gJmfprw")

# Set up the model with generation configuration and safety settings
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

translator = Translator()


def translate_to_hindi_and_speech(text):
    try:
        # Translate text to Hindi
        translated_text = translator.translate(text, src='en', dest='hi').text
        print("Translated Text (Hindi):")
        print(translated_text)

        # Convert translated text to speech
        tts = gTTS(text=translated_text, lang='hi')
        tts.save("translated_output.mp3")
        os.system("afplay translated_output.mp3")  # macOS specific command to play the audio
    except Exception as e:
        print("Translation failed:", e)


def capture_and_ocr():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale for better OCR accuracy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the captured frame
        cv2.imshow('Frame', frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit the loop
            break
        elif key == ord('f'):  # Press 'f' to perform OCR
            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(gray)

            # Display the OCR result
            print("OCR Result:")
            print(text)

            # Check if there is text to speak
            if text:
                # Convert text to speech
                tts = gTTS(text=text, lang='en')
                tts.save("output.mp3")
                os.system("afplay output.mp3")  # macOS specific command to play the audio

        elif key == ord('g'):  # Press 'g' to send OCR text to Gemini
            if text:
                # Start a conversation with the Gemini model
                convo = model.start_chat(history=[])
                # Send OCR text to the model
                convo.send_message(text)
                # Retrieve and print the response
                print("Gemini Response:")
                gemini_response = convo.last.text
                print(gemini_response)

                # Speak the first 500 characters of Gemini response
                gemini_response_first_500 = gemini_response[:500]
                tts = gTTS(text=gemini_response_first_500, lang='en')
                tts.save("gemini_output.mp3")
                os.system("afplay gemini_output.mp3")

        elif key == ord('H'):  # Press 'H' to translate OCR text to Hindi and speak
            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(gray)

            # Display the OCR result
            print("OCR Result:")
            print(text)

            # Check if there is text to translate
            if text:
                # Translate OCR text to Hindi and speak
                translate_to_hindi_and_speech(text)

        elif key == ord('j'):  # Press 'j' to perform speech-to-text and send to Gemini
            # Initialize recognizer class (for recognizing the speech)
            r = sr.Recognizer()

            # Reading Microphone as source
            with sr.Microphone() as source:
                print("Speak...")
                # Listen to the phrase and convert it to text
                audio = r.listen(source)

                try:
                    # Using google to recognize audio
                    text = r.recognize_google(audio)
                    print("You said :", text)

                    # Start a conversation with the Gemini model
                    convo = model.start_chat(history=[])
                    # Send speech-to-text to the model
                    convo.send_message(text)
                    # Retrieve and print the response
                    print("Gemini Response:")
                    gemini_response = convo.last.text
                    print(gemini_response)

                    # Speak the first 500 characters of Gemini response
                    gemini_response_first_500 = gemini_response[:500]
                    tts = gTTS(text=gemini_response_first_500, lang='en')
                    tts.save("gemini_output.mp3")
                    os.system("afplay gemini_output.mp3")

                except Exception as e:
                    print("Error :", e)

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_ocr()
