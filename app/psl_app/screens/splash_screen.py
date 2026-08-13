"""Landing screen for the Persian Sign Language recognizer."""

from kivy.animation import Animation
from kivy.graphics import Color, Ellipse, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.uix.widget import Widget

from widgets import PSLActionButton
import app_config as config


class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        root = FloatLayout()
        with root.canvas.before:
            Color(*config.COLOR_BACKGROUND)
            self._bg = RoundedRectangle(pos=root.pos, size=root.size, radius=[0])
            Color(0.12, 0.30, 0.62, 0.11)
            self._orb1 = Ellipse(size=(dp(260), dp(260)))
            Color(0.25, 0.62, 1.0, 0.07)
            self._orb2 = Ellipse(size=(dp(180), dp(180)))

        root.bind(pos=self._sync_bg, size=self._sync_bg)

        content = BoxLayout(
            orientation="vertical",
            spacing=dp(14),
            padding=(dp(24), dp(30), dp(24), dp(12)),
            size_hint=(0.92, 0.92),
            pos_hint={"center_x": 0.5, "center_y": 0.50},
        )
        root.add_widget(content)

        brand = Label(
            text="PSL AI",
            color=config.COLOR_PRIMARY,
            font_size=sp(15),
            bold=True,
            size_hint_y=None,
            height=dp(30),
            halign="left",
            valign="middle",
            text_size=(None, dp(30)),
        )
        content.add_widget(brand)

        # No fixed-height text box: the label gets the full available width,
        # preventing the title from looking squeezed into a narrow rectangle.
        title = Label(
            text="Persian Sign\nLanguage,\nmade simpler.",
            color=config.COLOR_TEXT_PRIMARY,
            font_size=sp(34),
            bold=True,
            halign="left",
            valign="top",
            size_hint_y=None,
            height=dp(132),
            shorten=False,
        )
        title.bind(width=lambda *_: setattr(title, "text_size", (title.width, None)))
        content.add_widget(title)

        subtitle = Label(
            text="Use your camera to recognize Persian signs in real time with a lightweight AI model.",
            color=config.COLOR_TEXT_SECONDARY,
            font_size=sp(16),
            halign="left",
            valign="top",
            size_hint_y=None,
            height=dp(68),
            shorten=False,
        )
        subtitle.bind(width=lambda *_: setattr(subtitle, "text_size", (subtitle.width, None)))
        content.add_widget(subtitle)

        content.add_widget(Widget(size_hint_y=1))

        # Removed the unnecessary card above the button.
        self.go_button = PSLActionButton(
            text="Let's go",
            size_hint_y=None,
            height=dp(58),
            bg_color=config.COLOR_PRIMARY,
            pressed_color=config.COLOR_PRIMARY_PRESSED,
            font_size=sp(17),
            radius=dp(18),
        )
        self.go_button.bind(on_release=self._on_go_pressed)
        content.add_widget(self.go_button)

        footer = Label(
            text="Point the camera at your hands to begin.",
            color=config.COLOR_TEXT_SECONDARY,
            font_size=sp(12),
            size_hint_y=None,
            height=dp(34),
            halign="center",
            valign="middle",
            shorten=False,
        )
        footer.bind(width=lambda *_: setattr(footer, "text_size", (footer.width, None)))
        content.add_widget(footer)

        self.add_widget(root)
        self._root = root

    def _sync_bg(self, *_):
        self._bg.pos = self._root.pos
        self._bg.size = self._root.size
        self._orb1.pos = (self._root.right - dp(160), self._root.top - dp(150))
        self._orb2.pos = (self._root.x - dp(70), self._root.y + dp(90))

    def on_pre_enter(self, *_):
        self.go_button.opacity = 0
        Animation(opacity=1, duration=0.28, t="out_quad").start(self.go_button)

    def _on_go_pressed(self, *_):
        if self.manager:
            self.manager.transition.direction = "left"
            self.manager.current = "camera"
