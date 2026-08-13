"""Reusable lightweight UI components for the Android-first PSL app."""

from kivy.animation import Animation
from kivy.graphics import Color, Ellipse, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.label import Label
from kivy.uix.widget import Widget

import app_config as config


class SurfaceCard(Widget):
    """A lightweight rounded surface with an optional border."""

    bg_color = ListProperty(config.COLOR_SURFACE)
    border_color = ListProperty(config.COLOR_BORDER)
    radius = NumericProperty(dp(22))
    border_width = NumericProperty(dp(1))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            self._fill = Color(*self.bg_color)
            self._rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[self.radius])
            self._border = Color(*self.border_color)
            self._line = RoundedRectangle(
                pos=(self.x + self.border_width / 2, self.y + self.border_width / 2),
                size=(max(0, self.width - self.border_width), max(0, self.height - self.border_width)),
                radius=[max(0, self.radius - self.border_width / 2)],
            )
        self.bind(pos=self._sync, size=self._sync, bg_color=self._sync_fill,
                  border_color=self._sync_border, radius=self._sync_radius,
                  border_width=self._sync)

    def _sync(self, *_):
        self._rect.pos = self.pos
        self._rect.size = self.size
        self._line.pos = (self.x + self.border_width / 2, self.y + self.border_width / 2)
        self._line.size = (max(0, self.width - self.border_width), max(0, self.height - self.border_width))

    def _sync_fill(self, *_):
        self._fill.rgba = self.bg_color

    def _sync_border(self, *_):
        self._border.rgba = self.border_color

    def _sync_radius(self, *_):
        self._rect.radius = [self.radius]
        self._line.radius = [max(0, self.radius - self.border_width / 2)]


class PSLActionButton(Widget):
    """Touch-first button with pressed feedback and no desktop hover behaviour."""

    text = StringProperty("")
    disabled = BooleanProperty(False)
    bg_color = ListProperty(config.COLOR_PRIMARY)
    text_color = ListProperty(config.COLOR_ON_PRIMARY)
    pressed_color = ListProperty(config.COLOR_PRIMARY_PRESSED)
    radius = NumericProperty(dp(18))
    font_size = NumericProperty(sp(16))
    __events__ = ("on_release",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            self._color = Color(*self.bg_color)
            self._rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[self.radius])
        self.label = Label(
            text=self.text,
            color=self.text_color,
            font_size=self.font_size,
            bold=True,
            halign="center",
            valign="middle",
        )
        self.add_widget(self.label)
        self._released_color = list(self.bg_color)
        self.bind(pos=self._sync, size=self._sync, text=self._sync_text,
                  bg_color=self._sync_color, text_color=self._sync_text_color,
                  radius=self._sync_radius, font_size=self._sync_font)

    def _sync(self, *_):
        self._rect.pos = self.pos
        self._rect.size = self.size
        self.label.pos = self.pos
        self.label.size = self.size
        self.label.text_size = self.size

    def _sync_text(self, *_):
        self.label.text = self.text

    def _sync_color(self, *_):
        self._released_color = list(self.bg_color)
        self._color.rgba = self.bg_color

    def _sync_text_color(self, *_):
        self.label.color = self.text_color

    def _sync_radius(self, *_):
        self._rect.radius = [self.radius]

    def _sync_font(self, *_):
        self.label.font_size = self.font_size

    def _set_pressed(self, pressed):
        target = self.pressed_color if pressed else self._released_color
        Animation.cancel_all(self._color)
        Animation(rgba=target, duration=0.08).start(self._color)
        if pressed:
            Animation(opacity=0.84, duration=0.08).start(self)
        else:
            Animation(opacity=1, duration=0.12).start(self)

    def on_touch_down(self, touch):
        if self.disabled or not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        self._set_pressed(True)
        self._touch = touch
        return True

    def on_touch_up(self, touch):
        if getattr(self, "_touch", None) is not touch:
            return super().on_touch_up(touch)
        self._set_pressed(False)
        self._touch = None
        if self.collide_point(*touch.pos):
            self.dispatch("on_release")
        return True

    def on_release(self, *_args):
        pass


class IconButton(PSLActionButton):
    """Compact action button intended for camera controls."""

    font_size = NumericProperty(sp(15))
    radius = NumericProperty(dp(16))
    bg_color = ListProperty(config.COLOR_GLASS)
    pressed_color = ListProperty(config.COLOR_GLASS_PRESSED)
    text_color = ListProperty(config.COLOR_TEXT_PRIMARY)


class StatusPill(Widget):
    """Small status indicator used for camera/recognition state."""

    text = StringProperty("")
    dot_color = ListProperty(config.COLOR_SUCCESS)
    bg_color = ListProperty(config.COLOR_GLASS)
    text_color = ListProperty(config.COLOR_TEXT_PRIMARY)
    font_size = NumericProperty(sp(12))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            self._bg_color = Color(*self.bg_color)
            self._bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(14)])
            self._dot_color = Color(*self.dot_color)
            self._dot = Ellipse(size=(dp(7), dp(7)))
        self.label = Label(
            text=self.text,
            color=self.text_color,
            font_size=self.font_size,
            bold=True,
            halign="left",
            valign="middle",
        )
        self.add_widget(self.label)
        self.bind(pos=self._sync, size=self._sync, text=self._sync_text,
                  bg_color=self._sync_bg_color, dot_color=self._sync_dot_color,
                  text_color=self._sync_text_color, font_size=self._sync_font)

    def _sync(self, *_):
        self._bg.pos = self.pos
        self._bg.size = self.size
        dot_size = dp(7)
        self._dot.pos = (self.x + dp(10), self.center_y - dot_size / 2)
        self.label.pos = (self.x + dp(25), self.y)
        self.label.size = (max(0, self.width - dp(30)), self.height)
        self.label.text_size = self.label.size

    def _sync_text(self, *_):
        self.label.text = self.text

    def _sync_bg_color(self, *_):
        self._bg_color.rgba = self.bg_color

    def _sync_dot_color(self, *_):
        self._dot_color.rgba = self.dot_color

    def _sync_text_color(self, *_):
        self.label.color = self.text_color

    def _sync_font(self, *_):
        self.label.font_size = self.font_size


class RippleButton(PSLActionButton):
    """Backward-compatible alias for the original project API."""


class RoundedHeaderBox(SurfaceCard):
    """Backward-compatible header component with polished typography."""

    text = StringProperty("")
    text_color = ListProperty(config.COLOR_TEXT_PRIMARY)
    font_size_sp = NumericProperty(22)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label = Label(
            text=self.text,
            color=self.text_color,
            font_size=sp(self.font_size_sp),
            bold=True,
            halign="center",
            valign="middle",
            padding=(dp(10), dp(8)),
        )
        self.add_widget(self.label)
        self.bind(pos=self._sync_label, size=self._sync_label, text=self._sync_text,
                  text_color=self._sync_text_color, font_size_sp=self._sync_font)

    def _sync_label(self, *_):
        self.label.pos = self.pos
        self.label.size = self.size
        self.label.text_size = self.size

    def _sync_text(self, *_):
        self.label.text = self.text

    def _sync_text_color(self, *_):
        self.label.color = self.text_color

    def _sync_font(self, *_):
        self.label.font_size = sp(self.font_size_sp)
