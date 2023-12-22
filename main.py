from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
import cv2
from keras.models import load_model
import numpy as np
import webbrowser
import time



class EmoRythm(App):
    def build(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model('.\\emotion_model.hdf5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        self.music_recommended = False
        self.music_recommendation_timer = None

        # UI components
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)
        self.label = Label(text='Emotion: ',size_hint_y=0.2)
        self.layout.add_widget(self.label)
        self.button = Button(text='Recommend Music', size_hint_y=0.2)
        self.button.bind(on_press=self.recommend_music_button_pressed)
        self.layout.add_widget(self.button)

        # Open the webcam (use 0 for the default camera)
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 60.0)  # Update at 30 FPS

        return self.layout

    def update(self, dt):
        ret, frame = self.cap.read()

        if ret:
            frame = self.detect_emotion(frame)
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (64, 64), interpolation=cv2.INTER_AREA)
            face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0) / 255.0

            emotion_probabilities = self.emotion_model.predict(face_roi)
            emotion_index = np.argmax(emotion_probabilities)
            emotion = self.emotion_labels[emotion_index]

            # Draw rectangle around the face and display emotion text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

            if not self.music_recommended and self.music_recommendation_timer is None:
                self.music_recommendation_timer = time.time()

            if self.music_recommendation_timer is not None and time.time() - self.music_recommendation_timer > 2:
                self.recommend_music(emotion)
                self.music_recommended = True
                self.music_recommendation_timer = None

            self.label.text = f'Emotion: {emotion}'

        return frame

    def recommend_music_button_pressed(self, instance):
        self.music_recommended = False

    def recommend_music(self, emotion):
        music_url = self.get_music_url(emotion)
        webbrowser.open(music_url)

    def get_music_url(self, emotion):
        music_mapping = {
            'Angry': 'https://www.youtube.com/watch?v=YKLX3QbKBg0',
            'Disgust': 'https://www.youtube.com/watch?v=I-QfPUz1es8',
            'Fear': 'https://www.youtube.com/watch?v=GVUqZC7lNiw',
            'Happy': 'https://www.youtube.com/watch?v=dhYOPzcsbGM',
            'Sad': 'https://www.youtube.com/watch?v=50VNCymT-Cs',
            'Surprise': 'https://www.youtube.com/watch?v=7ufkMTshjz8',
            'Neutral': 'https://www.youtube.com/watch?v=TBsKCT4rsPw'
        }
        return music_mapping.get(emotion, 'https://example.com/default-music-playlist')

    def on_stop(self):
        # Release the camera when the app is closed
        self.cap.release()

if __name__ == "__main__":
    Window.fullscreen = False
    EmoRythm().run()
