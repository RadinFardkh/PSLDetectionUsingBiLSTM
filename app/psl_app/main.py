"""Application entry point for the Android-first PSL recognizer."""

from kivy.config import Config

Config.set("graphics", "resizable", True)

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.screenmanager import FadeTransition, ScreenManager

from screens.splash_screen import SplashScreen
from screens.camera_screen import CameraScreen


class PSLApp(App):
    def build(self):
        self.title = "PSL AI"
        self._request_camera_permission()
        sm = ScreenManager(transition=FadeTransition(duration=0.22))
        self.splash_screen = SplashScreen(name="splash")
        self.camera_screen = CameraScreen(name="camera")
        sm.add_widget(self.splash_screen)
        sm.add_widget(self.camera_screen)
        sm.current = "splash"
        Window.bind(on_keyboard=self._on_keyboard)
        return sm

    @staticmethod
    def _request_camera_permission():
        try:
            from android.permissions import Permission, request_permissions
            request_permissions([Permission.CAMERA])
        except (ImportError, RuntimeError):
            # Desktop/dev environments do not provide Android's permission API.
            pass

    def _on_keyboard(self, _window, key, *_args):
        # Android back: stop recognition first, then return to the launch screen.
        if key == 27 and self.root and self.root.current == "camera":
            if self.camera_screen.processing_active:
                self.camera_screen._on_start_pressed()
            else:
                self.root.transition.direction = "right"
                self.root.current = "splash"
            return True
        return False

    def on_start(self):
        # Gives the first frame a chance to settle before expensive camera setup.
        Clock.schedule_once(lambda _dt: None, 0)

    def on_stop(self):
        self.camera_screen.on_stop()
        Window.unbind(on_keyboard=self._on_keyboard)


if __name__ == "__main__":
    PSLApp().run()
