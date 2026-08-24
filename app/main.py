import cv2
import threading
import time

from collections import deque

from kivy.app import App
from kivy.clock import Clock

from kivy.graphics import (
    Color,
    Rectangle,
    RoundedRectangle,
    Line
)

from kivy.graphics.texture import Texture

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button

from kivy.metrics import dp, sp
from kivy.core.window import Window

from arabic_reshaper import reshape
from bidi.algorithm import get_display

import mrealtime


# ============================================================
# WINDOW
# ============================================================

Window.clearcolor = (
    0.035,
    0.04,
    0.055,
    1
)


# ============================================================
# PERSIAN
# ============================================================

def fa(text):
    """
    Prepare Persian/Arabic text for Kivy.
    """

    return get_display(
        reshape(text)
    )


# ============================================================
# FONT
# ============================================================

FONT_PATH = "Titr.TTF"


# ============================================================
# COLORS
# ============================================================

BACKGROUND = (
    0.035,
    0.04,
    0.055,
    1
)

CARD_COLOR = (
    0.07,
    0.08,
    0.105,
    1
)

CARD_COLOR_DARK = (
    0.055,
    0.06,
    0.08,
    1
)

WHITE = (
    0.96,
    0.97,
    1.0,
    1
)

TEXT_SECONDARY = (
    0.62,
    0.66,
    0.73,
    1
)

ACCENT = (
    0.25,
    0.55,
    1.0,
    1
)

ACCENT_DARK = (
    0.16,
    0.34,
    0.68,
    1
)

GREEN = (
    0.20,
    0.85,
    0.50,
    1
)

RED = (
    0.95,
    0.28,
    0.32,
    1
)

DEVELOPER_COLOR = (
    0.55,
    0.38,
    0.95,
    1
)


# ============================================================
# UI SETTINGS
# ============================================================

MAX_SENTENCE_WORDS = 8

STABILITY_TIME = 1.25

MIN_CONFIDENCE_FOR_SENTENCE = 0.7


# ============================================================
# ROUNDED CARD
# ============================================================

class RoundedCard(FloatLayout):

    def __init__(
        self,
        background_color=CARD_COLOR,
        radius=18,
        **kwargs
    ):

        super().__init__(**kwargs)

        with self.canvas.before:

            self._color = Color(
                *background_color
            )

            self._background = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[
                    (dp(radius), dp(radius)),
                    (dp(radius), dp(radius)),
                    (dp(radius), dp(radius)),
                    (dp(radius), dp(radius))
                ]
            )

        self.bind(
            pos=self._update_background,
            size=self._update_background
        )

    def _update_background(
        self,
        instance,
        value
    ):

        self._background.pos = self.pos
        self._background.size = self.size


# ============================================================
# APP
# ============================================================

class PSLApp(App):

    def build(self):

        # ====================================================
        # STATE
        # ====================================================

        self.recognition_running = False

        self.recognition_thread = None

        self.developer_mode = False

        self.current_prediction = None

        self.current_confidence = 0.0

        self.stable_prediction = None

        self.stable_since = None

        self.last_added_prediction = None

        self.sentence_words = deque(
            maxlen=MAX_SENTENCE_WORDS
        )

        self.last_frame = None

        # ====================================================
        # ROOT
        # ====================================================

        self.layout = FloatLayout()

        with self.layout.canvas.before:

            self.background_color = Color(
                *BACKGROUND
            )

            self.background = Rectangle(
                pos=self.layout.pos,
                size=self.layout.size
            )

        self.layout.bind(
            pos=self.update_background,
            size=self.update_background
        )

        # ====================================================
        # TITLE
        # ====================================================

        self.title_label = Label(

            text=fa(
                "هوش مصنوعی\n"
                "زبان اشاره فارسی"
            ),

            font_name=FONT_PATH,

            font_size=sp(25),

            color=WHITE,

            halign="center",

            valign="middle",

            size_hint=(0.72, 0.13),

            pos_hint={
                "x": 0.14,
                "y": 0.84
            }
        )

        self.title_label.bind(
            size=self.update_text_size
        )

        self.layout.add_widget(
            self.title_label
        )

        # ====================================================
        # DEVELOPER MODE BUTTON
        # ====================================================

        self.developer_button = Button(

            text=fa(
                "حالت توسعه"
            ),

            font_name=FONT_PATH,

            font_size=sp(11),

            color=TEXT_SECONDARY,

            background_normal="",

            background_color=(0, 0, 0, 0),

            size_hint=(0.20, 0.045),

            pos_hint={
                "x": 0.76,
                "y": 0.925
            }
        )

        with self.developer_button.canvas.before:

            self.developer_button_color = Color(
                *CARD_COLOR
            )

            self.developer_button_background = RoundedRectangle(
                pos=self.developer_button.pos,
                size=self.developer_button.size,
                radius=[
                    (dp(10), dp(10)),
                    (dp(10), dp(10)),
                    (dp(10), dp(10)),
                    (dp(10), dp(10))
                ]
            )

        self.developer_button.bind(
            pos=self.update_developer_background,
            size=self.update_developer_background,
            on_release=self.toggle_developer_mode
        )

        self.layout.add_widget(
            self.developer_button
        )

        # ====================================================
        # CAMERA CARD
        # ====================================================

        self.camera_card = RoundedCard(
            background_color=CARD_COLOR,
            radius=20,

            size_hint=(0.90, 0.47),

            pos_hint={
                "x": 0.05,
                "y": 0.34
            }
        )

        self.layout.add_widget(
            self.camera_card
        )

        # ====================================================
        # CAMERA
        # ====================================================

        self.camera_image = Image(

            size_hint=(0.96, 0.94),

            pos_hint={
                "x": 0.02,
                "y": 0.03
            },

            allow_stretch=True,

            keep_ratio=True
        )

        self.camera_card.add_widget(
            self.camera_image
        )

        # ====================================================
        # CAMERA PLACEHOLDER
        # ====================================================

        self.camera_placeholder = Label(

            text=fa(
                "برای شروع تشخیص\n"
                "دکمه شروع را بزنید"
            ),

            font_name=FONT_PATH,

            font_size=sp(17),

            color=TEXT_SECONDARY,

            halign="center",

            valign="middle",

            size_hint=(1, 1),

            pos_hint={
                "x": 0,
                "y": 0
            }
        )

        self.camera_placeholder.bind(
            size=self.update_text_size
        )

        self.camera_card.add_widget(
            self.camera_placeholder
        )

        # ====================================================
        # CURRENT PREDICTION LABEL
        # ====================================================

        self.current_title = Label(

            text=fa(
                "پیش‌بینی فعلی"
            ),

            font_name=FONT_PATH,

            font_size=sp(13),

            color=TEXT_SECONDARY,

            halign="center",

            valign="middle",

            size_hint=(0.90, 0.035),

            pos_hint={
                "x": 0.05,
                "y": 0.295
            }
        )

        self.current_title.bind(
            size=self.update_text_size
        )

        self.layout.add_widget(
            self.current_title
        )

        # ====================================================
        # CURRENT PREDICTION
        # ====================================================

        self.result_label = Label(

            text=fa(
                "---"
            ),

            font_name=FONT_PATH,

            font_size=sp(23),

            color=WHITE,

            halign="center",

            valign="middle",

            size_hint=(0.90, 0.07),

            pos_hint={
                "x": 0.05,
                "y": 0.235
            }
        )

        self.result_label.bind(
            size=self.update_text_size
        )

        self.layout.add_widget(
            self.result_label
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        self.confidence_label = Label(

            text=fa(
                "اطمینان: ---"
            ),

            font_name=FONT_PATH,

            font_size=sp(13),

            color=TEXT_SECONDARY,

            halign="center",

            valign="middle",

            opacity=0,

            size_hint=(0.90, 0.04),

            pos_hint={
                "x": 0.05,
                "y": 0.205
            }
        )

        self.confidence_label.bind(
            size=self.update_text_size
        )

        self.layout.add_widget(
            self.confidence_label
        )

        # ====================================================
        # SENTENCE CARD
        # ====================================================

        self.sentence_card = RoundedCard(
            background_color=CARD_COLOR_DARK,
            radius=16,

            size_hint=(0.90, 0.105),

            pos_hint={
                "x": 0.05,
                "y": 0.105
            }
        )

        self.layout.add_widget(
            self.sentence_card
        )

        # ====================================================
        # SENTENCE TITLE
        # ====================================================

        self.sentence_title = Label(

            text=fa(
                "جمله"
            ),

            font_name=FONT_PATH,

            font_size=sp(12),

            color=TEXT_SECONDARY,

            halign="right",

            valign="middle",

            size_hint=(0.20, 0.28),

            pos_hint={
                "x": 0.74,
                "y": 0.68
            }
        )

        self.sentence_title.bind(
            size=self.update_text_size
        )

        self.sentence_card.add_widget(
            self.sentence_title
        )

        # ====================================================
        # SENTENCE
        # ====================================================

        self.sentence_label = Label(

            text=fa(
                "هنوز کلمه‌ای ثبت نشده است"
            ),

            font_name=FONT_PATH,

            font_size=sp(16),

            color=WHITE,

            halign="center",

            valign="middle",

            size_hint=(0.92, 0.62),

            pos_hint={
                "x": 0.04,
                "y": 0.08
            }
        )

        self.sentence_label.bind(
            size=self.update_text_size
        )

        self.sentence_card.add_widget(
            self.sentence_label
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = Label(

            text=fa(
                "آماده"
            ),

            font_name=FONT_PATH,

            font_size=sp(12),

            color=TEXT_SECONDARY,

            halign="center",

            valign="middle",

            size_hint=(0.45, 0.045),

            pos_hint={
                "x": 0.05,
                "y": 0.06
            }
        )

        self.status_label.bind(
            size=self.update_text_size
        )

        self.layout.add_widget(
            self.status_label
        )

        # ====================================================
        # CLEAR BUTTON
        # ====================================================

        self.clear_button = Button(

            text=fa(
                "پاک کردن"
            ),

            font_name=FONT_PATH,

            font_size=sp(11),

            color=TEXT_SECONDARY,

            background_normal="",

            background_color=(0, 0, 0, 0),

            size_hint=(0.23, 0.045),

            pos_hint={
                "x": 0.52,
                "y": 0.06
            }
        )

        with self.clear_button.canvas.before:

            self.clear_button_color = Color(
                *CARD_COLOR
            )

            self.clear_button_background = RoundedRectangle(
                pos=self.clear_button.pos,
                size=self.clear_button.size,
                radius=[
                    (dp(10), dp(10)),
                    (dp(10), dp(10)),
                    (dp(10), dp(10)),
                    (dp(10), dp(10))
                ]
            )

        self.clear_button.bind(
            pos=self.update_clear_background,
            size=self.update_clear_background,
            on_release=self.clear_sentence
        )

        self.layout.add_widget(
            self.clear_button
        )

        # ====================================================
        # START / STOP
        # ====================================================

        self.start_button = Button(

            text=fa(
                "شروع تشخیص"
            ),

            font_name=FONT_PATH,

            font_size=sp(16),

            color=WHITE,

            background_normal="",

            background_color=(0, 0, 0, 0),

            size_hint=(0.30, 0.055),

            pos_hint={
                "x": 0.35,
                "y": 0.005
            }
        )

        with self.start_button.canvas.before:

            self.start_button_color = Color(
                *ACCENT
            )

            self.start_button_background = RoundedRectangle(
                pos=self.start_button.pos,
                size=self.start_button.size,
                radius=[
                    (dp(14), dp(14)),
                    (dp(14), dp(14)),
                    (dp(14), dp(14)),
                    (dp(14), dp(14))
                ]
            )

        self.start_button.bind(
            pos=self.update_start_button_background,
            size=self.update_start_button_background,
            on_release=self.toggle_recognition
        )

        self.layout.add_widget(
            self.start_button
        )

        return self.layout

    # ========================================================
    # BACKGROUND
    # ========================================================

    def update_background(
        self,
        instance,
        value
    ):

        self.background.pos = instance.pos
        self.background.size = instance.size

    # ========================================================
    # TEXT SIZE
    # ========================================================

    def update_text_size(
        self,
        instance,
        value
    ):

        instance.text_size = instance.size

    # ========================================================
    # DEVELOPER BUTTON BACKGROUND
    # ========================================================

    def update_developer_background(
        self,
        instance,
        value
    ):

        self.developer_button_background.pos = (
            instance.pos
        )

        self.developer_button_background.size = (
            instance.size
        )

    # ========================================================
    # CLEAR BUTTON BACKGROUND
    # ========================================================

    def update_clear_background(
        self,
        instance,
        value
    ):

        self.clear_button_background.pos = (
            instance.pos
        )

        self.clear_button_background.size = (
            instance.size
        )

    # ========================================================
    # START BUTTON BACKGROUND
    # ========================================================

    def update_start_button_background(
        self,
        instance,
        value
    ):

        self.start_button_background.pos = (
            instance.pos
        )

        self.start_button_background.size = (
            instance.size
        )

    # ========================================================
    # DEVELOPER MODE
    # ========================================================

    def toggle_developer_mode(
        self,
        instance
    ):

        self.developer_mode = not self.developer_mode

        if self.developer_mode:

            self.developer_button_color.rgba = (
                DEVELOPER_COLOR
            )

            self.developer_button.color = WHITE

            self.confidence_label.opacity = 1

            self.status_label.text = fa(
                "حالت توسعه فعال است"
            )

        else:

            self.developer_button_color.rgba = (
                CARD_COLOR
            )

            self.developer_button.color = (
                TEXT_SECONDARY
            )

            self.confidence_label.opacity = 0

            if self.recognition_running:

                self.status_label.text = fa(
                    "تشخیص فعال است"
                )

            else:

                self.status_label.text = fa(
                    "آماده"
                )

    # ========================================================
    # TOGGLE RECOGNITION
    # ========================================================

    def toggle_recognition(
        self,
        instance
    ):

        if not self.recognition_running:

            self.start_recognition()

        else:

            self.stop_recognition()

    # ========================================================
    # START
    # ========================================================

    def start_recognition(self):

        if self.recognition_running:
            return

        self.recognition_running = True

        self.start_button.text = fa(
            "توقف تشخیص"
        )

        self.start_button_color.rgba = RED

        self.status_label.text = fa(
            "در حال شروع دوربین..."
        )

        self.camera_placeholder.text = fa(
            "در حال شروع دوربین..."
        )

        self.result_label.text = fa(
            "---"
        )

        self.current_prediction = None
        self.current_confidence = 0.0

        self.stable_prediction = None
        self.stable_since = None

        # ----------------------------------------------------
        # Callback
        # ----------------------------------------------------

        mrealtime.set_prediction_callback(
            self.receive_frame
        )

        # ----------------------------------------------------
        # Thread
        # ----------------------------------------------------

        self.recognition_thread = threading.Thread(

            target=mrealtime.run,

            args=(False,),

            daemon=True
        )

        self.recognition_thread.start()

    # ========================================================
    # STOP
    # ========================================================

    def stop_recognition(self):

        if not self.recognition_running:
            return

        self.recognition_running = False

        # ----------------------------------------------------
        # Disconnect callback.
        #
        # mrealtime.py is intentionally untouched.
        # ----------------------------------------------------

        mrealtime.set_prediction_callback(
            None
        )

        self.start_button.text = fa(
            "شروع تشخیص"
        )

        self.start_button_color.rgba = ACCENT

        self.status_label.text = fa(
            "تشخیص متوقف شد"
        )

        self.camera_placeholder.text = fa(
            "برای شروع تشخیص\n"
            "دکمه شروع را بزنید"
        )

        self.result_label.text = fa(
            "---"
        )

        self.confidence_label.text = fa(
            "اطمینان: ---"
        )

        self.camera_image.texture = None

        self.current_prediction = None
        self.current_confidence = 0.0

        self.stable_prediction = None
        self.stable_since = None

    # ========================================================
    # RECEIVE FRAME
    # ========================================================

    def receive_frame(
        self,
        prediction,
        confidence,
        frame
    ):

        # ----------------------------------------------------
        # Ignore callbacks after Stop.
        # ----------------------------------------------------

        if not self.recognition_running:
            return

        Clock.schedule_once(

            lambda dt: self.update_screen(
                prediction,
                confidence,
                frame
            )
        )

    # ========================================================
    # UPDATE SCREEN
    # ========================================================

    def update_screen(
        self,
        prediction,
        confidence,
        frame
    ):

        if not self.recognition_running:
            return

        # ====================================================
        # CURRENT PREDICTION
        # ====================================================

        self.current_prediction = prediction

        self.current_confidence = confidence

        # ====================================================
        # CAMERA
        # ====================================================

        if frame is not None:

            self.last_frame = frame

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            height, width, channels = (
                frame_rgb.shape
            )

            texture = Texture.create(

                size=(
                    width,
                    height
                ),

                colorfmt="rgb"
            )

            texture.blit_buffer(

                frame_rgb.tobytes(),

                colorfmt="rgb",

                bufferfmt="ubyte"
            )

            texture.flip_vertical()

            self.camera_image.texture = texture

            self.camera_placeholder.text = ""

        # ====================================================
        # PREDICTION
        # ====================================================

        if prediction is None:

            self.result_label.text = fa(
                "---"
            )

            return

        sign_name = mrealtime.indexes_in_clsmap.get(

            int(prediction),

            str(prediction)
        )

        # ----------------------------------------------------
        # CURRENT PREDICTION
        # ----------------------------------------------------

        self.result_label.text = fa(
            sign_name
        )

        # ----------------------------------------------------
        # DEVELOPER CONFIDENCE
        # ----------------------------------------------------

        if self.developer_mode:

            self.confidence_label.text = fa(
                f"اطمینان: {confidence * 100:.1f}%"
            )

        # ====================================================
        # STABILITY TRACKING
        # ====================================================

        now = time.monotonic()

        if (
            self.stable_prediction
            != prediction
        ):

            self.stable_prediction = prediction

            self.stable_since = now

            self.status_label.text = fa(
                "در حال تأیید علامت..."
            )

            return

        # ====================================================
        # CHECK 0.5 SECOND
        # ====================================================

        if self.stable_since is None:
            return

        stable_duration = (
            now - self.stable_since
        )

        if stable_duration >= STABILITY_TIME:

            # ------------------------------------------------
            # Don't add the exact same prediction repeatedly.
            # ------------------------------------------------

            if (
                self.last_added_prediction
                != prediction
            ):

                self.add_word(
                    sign_name
                )

                self.last_added_prediction = (
                    prediction
                )

            self.status_label.text = fa(
                "تشخیص فعال است"
            )

    # ========================================================
    # ADD WORD
    # ========================================================

    def add_word(
        self,
        word
    ):

        # ----------------------------------------------------
        # Avoid consecutive duplicates.
        # ----------------------------------------------------

        if (
            len(self.sentence_words) > 0
            and self.sentence_words[-1] == word
        ):

            return

        # ----------------------------------------------------
        # Add.
        # ----------------------------------------------------

        self.sentence_words.append(
            word
        )

        self.update_sentence()

    # ========================================================
    # SENTENCE
    # ========================================================

    def update_sentence(self):

        if not self.sentence_words:

            self.sentence_label.text = fa(
                "هنوز کلمه‌ای ثبت نشده است"
            )

            return

        sentence = " ".join(
            self.sentence_words
        )

        self.sentence_label.text = fa(
            sentence
        )

    # ========================================================
    # CLEAR SENTENCE
    # ========================================================

    def clear_sentence(
        self,
        instance=None
    ):

        self.sentence_words.clear()

        self.last_added_prediction = None

        self.update_sentence()

    # ========================================================
    # STOP APP
    # ========================================================

    def on_stop(self):

        # ----------------------------------------------------
        # Disconnect callback.
        # ----------------------------------------------------

        mrealtime.set_prediction_callback(
            None
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    PSLApp().run()