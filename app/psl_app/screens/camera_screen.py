"""Android-first live camera and Persian Sign Language recognition screen."""

import cv2
from collections import deque
from kivy.clock import Clock
from kivy.graphics import Color, Line, Ellipse
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.uix.widget import Widget

from widgets import PSLActionButton, IconButton, StatusPill, SurfaceCard
import app_config as config
from engine import SignRecognitionEngine


class CameraScreen(Screen):
    processing_active = BooleanProperty(False)
    developer_mode = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = FloatLayout()
        self._root = root

        self.image_widget = Image(
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
            allow_stretch=True,
            keep_ratio=True,
            mipmap=False,
        )
        root.add_widget(self.image_widget)

        # No global translucent overlay: it was obscuring the camera and UI.

        self.skeleton_widget = Widget(size_hint=(1, 1))
        root.add_widget(self.skeleton_widget)

        # Top status area ------------------------------------------------
        top_bar = SurfaceCard(
            size_hint=(0.94, None),
            height=dp(68),
            pos_hint={"center_x": 0.5, "top": 0.975},
            bg_color=(0.035, 0.045, 0.070, 0.82),
            border_color=(0.20, 0.25, 0.35, 0.78),
            radius=dp(22),
        )
        top_layout = BoxLayout(
            orientation="horizontal",
            padding=(dp(12), dp(7)),
            spacing=dp(8),
        )
        top_bar.add_widget(top_layout)
        top_bar.bind(
            pos=lambda *_: setattr(top_layout, "pos", top_bar.pos),
            size=lambda *_: setattr(top_layout, "size", top_bar.size),
        )
        top_layout.pos = top_bar.pos
        top_layout.size = top_bar.size

        title_col = BoxLayout(
            orientation="vertical",
            spacing=dp(7),
            padding=(0, dp(2)),
        )
        top_layout.add_widget(title_col)

        title_col.add_widget(Label(
            text="PSL recognition",
            color=config.COLOR_TEXT_PRIMARY,
            font_size=sp(16),
            bold=True,
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(25),
            text_size=(None, dp(25)),
        ))

        self.status_pill = StatusPill(
            text="Camera ready",
            size_hint_y=None,
            height=dp(28),
            size_hint_x=None,
            width=dp(126),
        )
        title_col.add_widget(self.status_pill)

        self.dev_button = IconButton(
            text="Developer",
            size_hint=(None, None),
            size=(dp(92), dp(46)),
        )
        self.dev_button.bind(on_release=self._on_toggle_dev_mode)
        top_layout.add_widget(self.dev_button)

        self.flip_button = IconButton(
            text="Flip",
            size_hint=(None, None),
            size=(dp(62), dp(46)),
        )
        self.flip_button.bind(on_release=self._on_flip_camera)
        top_layout.add_widget(self.flip_button)

        root.add_widget(top_bar)

        # Single combined prediction/control card ----------------------
        # This replaces the separate middle card that only appeared after Start.
        self.control_card = SurfaceCard(
            size_hint=(0.94, None),
            height=dp(170),
            pos_hint={"center_x": 0.5, "y": 0.035},
            bg_color=(0.035, 0.045, 0.070, 0.95),
            border_color=(0.20, 0.25, 0.35, 0.90),
            radius=dp(24),
        )
        control_col = BoxLayout(
            orientation="vertical",
            padding=(dp(15), dp(10)),
            spacing=dp(4),
        )
        self.control_card.add_widget(control_col)
        self.control_card.bind(
            pos=lambda *_: setattr(control_col, "pos", self.control_card.pos),
            size=lambda *_: setattr(control_col, "size", self.control_card.size),
        )
        control_col.pos = self.control_card.pos
        control_col.size = self.control_card.size

        self.current_sign_label = Label(
            text="SENTENCE",
            color=config.COLOR_TEXT_SECONDARY,
            font_size=sp(10),
            bold=True,
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(18),
            text_size=(None, dp(18)),
        )
        control_col.add_widget(self.current_sign_label)

        # Recent stable predictions are kept as a word sequence so the user
        # can build a sentence naturally instead of seeing only the latest word.
        self.prediction_history_label = Label(
            text="Ready",
            color=config.COLOR_TEXT_PRIMARY,
            font_size=sp(22),
            bold=True,
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(42),
            shorten=True,
            shorten_from="left",
        )
        self.prediction_history_label.bind(width=lambda *_: setattr(
            self.prediction_history_label,
            "text_size",
            (self.prediction_history_label.width, None)
        ))
        control_col.add_widget(self.prediction_history_label)

        self.prediction_label = Label(
            text="",
            color=config.COLOR_TEXT_SECONDARY,
            font_size=sp(13),
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(24),
        )
        self.prediction_label.bind(width=lambda *_: setattr(
            self.prediction_label,
            "text_size",
            (self.prediction_label.width, None)
        ))
        control_col.add_widget(self.prediction_label)

        bottom_row = BoxLayout(
            orientation="horizontal",
            spacing=dp(10),
            size_hint_y=None,
            height=dp(54),
        )
        control_col.add_widget(bottom_row)

        hint = Label(
            text="Show your hand and hold the sign steady",
            color=config.COLOR_TEXT_SECONDARY,
            font_size=sp(12),
            halign="left",
            valign="middle",
        )
        hint.bind(width=lambda *_: setattr(hint, "text_size", (hint.width, None)))
        bottom_row.add_widget(hint)

        self.start_button = PSLActionButton(
            text="Start",
            size_hint=(None, 1),
            width=dp(122),
            bg_color=config.COLOR_PRIMARY,
            pressed_color=config.COLOR_PRIMARY_PRESSED,
            font_size=sp(17),
            radius=dp(18),
        )
        self.start_button.bind(on_release=self._on_start_pressed)
        bottom_row.add_widget(self.start_button)

        root.add_widget(self.control_card)

        self.error_label = Label(
            text="",
            color=config.COLOR_ERROR,
            font_size=sp(13),
            bold=True,
            size_hint=(0.86, None),
            height=dp(40),
            pos_hint={"center_x": 0.5, "y": 0.235},
            halign="center",
            valign="middle",
            opacity=0,
        )
        self.error_label.bind(width=lambda *_: setattr(
            self.error_label, "text_size", (self.error_label.width, None)
        ))
        root.add_widget(self.error_label)

        self.camera_index = 0
        self.capture = None
        self.engine = None
        self._event = None

        # Sentence-building history. Only stable, changed predictions are added.
        self._prediction_history = deque(maxlen=8)
        self._last_history_prediction = None

        self.add_widget(root)

    def on_pre_enter(self, *_):
        if self.engine is None:
            try:
                self.engine = SignRecognitionEngine()
            except FileNotFoundError as exc:
                print(f"[CameraScreen] Engine not started: {exc}")
                self.engine = None
                self.status_pill.text = "Camera only"
                self.status_pill.dot_color = config.COLOR_WARNING
                self._show_error("Recognition model files are not available yet.")
        if self.capture is None:
            self._open_camera(self.camera_index)
        if self._event is None:
            self._event = Clock.schedule_interval(self._update_frame, 1.0 / 30.0)

    def on_leave(self, *_):
        # Keep the camera warm for fast navigation within the app.
        pass

    def _open_camera(self, index):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            self.status_pill.text = "Camera unavailable"
            self.status_pill.dot_color = config.COLOR_ERROR
            self._show_error("Unable to open the camera. Check Android camera permission.")
        else:
            self.status_pill.text = "Camera ready"
            self.status_pill.dot_color = config.COLOR_SUCCESS
            self._hide_error()

    def _on_flip_camera(self, *_):
        self.camera_index = 1 - self.camera_index
        self._open_camera(self.camera_index)

    def _on_toggle_dev_mode(self, *_):
        self.developer_mode = not self.developer_mode
        self.dev_button.text = "Dev on" if self.developer_mode else "Developer"
        if not self.developer_mode:
            self.skeleton_widget.canvas.clear()
            self.prediction_label.text = ""
        elif self.processing_active:
            self.prediction_label.text = "Confidence will update while recognizing."

    def _on_start_pressed(self, *_):
        self.processing_active = not self.processing_active
        if self.processing_active:
            if self.engine is not None:
                self.engine.reset_session()
            self._last_history_prediction = None
            self.start_button.text = "Stop"
            self.start_button.bg_color = config.COLOR_ERROR
            self.start_button.pressed_color = (0.85, 0.22, 0.26, 1)
            self.status_pill.text = "Recognizing"
            self.status_pill.dot_color = config.COLOR_SUCCESS
            self.prediction_label.text = "Listening…"
            self.prediction_history_label.text = (
                " ".join(self._prediction_history)
                if self._prediction_history else "Listening…"
            )
            self._hide_error()
        else:
            self.start_button.text = "Start"
            self.start_button.bg_color = config.COLOR_PRIMARY
            self.start_button.pressed_color = config.COLOR_PRIMARY_PRESSED
            self.status_pill.text = "Camera ready"
            self.prediction_history_label.text = (
                " ".join(self._prediction_history)
                if self._prediction_history else "Ready"
            )
            self.prediction_label.text = ""
            self.skeleton_widget.canvas.clear()

    def _update_frame(self, _dt):
        if self.capture is None or not self.capture.isOpened():
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        if self.engine is None:
            self._blit_frame(frame)
            return
        try:
            state = self.engine.process_frame(
                frame,
                run_inference=self.processing_active,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._blit_frame(frame)
            self._show_error(f"Recognition paused: {type(exc).__name__}")
            return

        self._hide_error()
        self._blit_frame(frame)

        if self.developer_mode:
            self._draw_skeleton(state)
        elif self.skeleton_widget.canvas.children:
            self.skeleton_widget.canvas.clear()

        if self.processing_active:
            self._update_prediction_ui(state)

    def _blit_frame(self, frame):
        buf = cv2.flip(frame, 0).tobytes()
        h, w = frame.shape[:2]
        if (
            self.image_widget.texture is None
            or self.image_widget.texture.width != w
            or self.image_widget.texture.height != h
        ):
            self.image_widget.texture = Texture.create(
                size=(w, h),
                colorfmt="bgr",
            )
        self.image_widget.texture.blit_buffer(
            buf,
            colorfmt="bgr",
            bufferfmt="ubyte",
        )
        self.image_widget.canvas.ask_update()

    def _draw_skeleton(self, state):
        self.skeleton_widget.canvas.clear()

        # The Kivy Image uses keep_ratio=True, so the camera frame may be
        # letterboxed. Drawing normalized landmarks over the entire widget
        # causes a small but visible offset. Map them into the actual image
        # rectangle instead.
        widget_w, widget_h = self.image_widget.size
        image_x, image_y = self.image_widget.pos

        texture = self.image_widget.texture
        if texture is None or texture.width <= 0 or texture.height <= 0:
            return

        frame_w = float(texture.width)
        frame_h = float(texture.height)

        scale = min(widget_w / frame_w, widget_h / frame_h)
        displayed_w = frame_w * scale
        displayed_h = frame_h * scale
        offset_x = image_x + (widget_w - displayed_w) / 2.0
        offset_y = image_y + (widget_h - displayed_h) / 2.0

        def to_widget_coords(nx, ny):
            # Camera frame has already been horizontally mirrored before
            # inference, and _blit_frame vertically flips it for Kivy.
            px = offset_x + nx * displayed_w
            py = offset_y + (1.0 - ny) * displayed_h
            return px, py

        with self.skeleton_widget.canvas:
            # High-contrast cyan/yellow-green is substantially easier to see
            # against both dark and bright camera backgrounds.
            Color(0.20, 1.0, 0.92, 1.0)

            for landmarks in (
                state["left_landmarks"],
                state["right_landmarks"],
            ):
                if not landmarks:
                    continue

                points = [to_widget_coords(x, y) for (x, y) in landmarks]

                for a, b in state["connections"]:
                    if a < len(points) and b < len(points):
                        Line(
                            points=[*points[a], *points[b]],
                            width=2.2,
                        )

                for px, py in points:
                    Ellipse(
                        pos=(px - dp(3.2), py - dp(3.2)),
                        size=(dp(6.4), dp(6.4)),
                    )

    def _update_prediction_ui(self, state):
        pred = state["prediction"]

        # Add only stable, changed predictions. This prevents the same sign
        # from being appended dozens of times while the user holds it.
        if (
            pred
            and state["stable_duration"] >= config.STABLE_REQUIRED_TIME
            and pred != self._last_history_prediction
        ):
            self._prediction_history.append(pred)
            self._last_history_prediction = pred

        if self._prediction_history:
            self.prediction_history_label.text = "  ".join(
                self._prediction_history
            )
        else:
            self.prediction_history_label.text = pred if pred else "Listening…"

        if self.developer_mode:
            self.prediction_label.text = (
                f"Confidence  •  {state['confidence'] * 100:.1f}%"
            )
        elif pred:
            self.prediction_label.text = (
                "Sign detected  •  hold to add it"
                if state["stable_duration"] > 0
                else "Looking for a stable sign…"
            )
        else:
            self.prediction_label.text = "Looking for a stable sign…"

    def _show_error(self, message):
        self.error_label.text = message
        self.error_label.opacity = 1

    def _hide_error(self):
        self.error_label.opacity = 0

    def on_stop(self):
        if self._event is not None:
            self._event.cancel()
            self._event = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.engine is not None:
            self.engine.close()
            self.engine = None
