"""OS utility functions for window management and keyboard input."""
import platform

# ---------------------------------------------------------------------------
# keyboard wrapper — the 'keyboard' library requires root on Linux.
# If it can't initialise, fall back to Qt key-event tracking (works without
# root as long as a PyQtGraph / Qt window has focus).
# ---------------------------------------------------------------------------
try:
    import keyboard as _kb
    # Force early init so the ImportError surfaces now, not mid-loop
    _kb.is_pressed('shift')
    _keyboard_available = True
except (ImportError, Exception):
    _kb = None
    _keyboard_available = False

# ── Qt-based key tracker (fallback for Linux without root) ────────────────
_qt_key_tracker = None


class _QtKeyTracker:
    """Track currently held keys via a Qt event filter on QApplication."""

    _KEY_MAP = None

    def __init__(self):
        from pyqtgraph.Qt import QtCore, QtWidgets
        self._pressed = set()
        self._QtCore = QtCore

        if _QtKeyTracker._KEY_MAP is None:
            _QtKeyTracker._KEY_MAP = {
                'q': QtCore.Qt.Key_Q, 'space': QtCore.Qt.Key_Space,
                '[': QtCore.Qt.Key_BracketLeft, ']': QtCore.Qt.Key_BracketRight,
                '-': QtCore.Qt.Key_Minus, '=': QtCore.Qt.Key_Equal,
                'escape': QtCore.Qt.Key_Escape, 'enter': QtCore.Qt.Key_Return,
            }
            for c in 'abcdefghijklmnoprstuvwxyz':
                _QtKeyTracker._KEY_MAP.setdefault(
                    c, getattr(QtCore.Qt, f'Key_{c.upper()}', None))

        app = QtWidgets.QApplication.instance()
        if app is not None:
            FilterCls = _make_event_filter_class()
            self._filter = FilterCls(self)
            app.installEventFilter(self._filter)

    def key_down(self, qt_key):
        self._pressed.add(qt_key)

    def key_up(self, qt_key):
        self._pressed.discard(qt_key)

    def is_pressed(self, name: str) -> bool:
        qt_key = (_QtKeyTracker._KEY_MAP or {}).get(name.lower())
        if qt_key is None:
            return False
        return qt_key in self._pressed


def _make_event_filter_class():
    """Build the event-filter class lazily so the import happens at call time."""
    from pyqtgraph.Qt import QtCore

    class _QtEventFilter(QtCore.QObject):
        """QObject event filter that feeds key events to _QtKeyTracker."""

        def __init__(self, tracker, parent=None):
            super().__init__(parent)
            self._tracker = tracker
            self._KeyPress = QtCore.QEvent.KeyPress
            self._KeyRelease = QtCore.QEvent.KeyRelease

        def eventFilter(self, obj, event):
            t = event.type()
            if t == self._KeyPress and not event.isAutoRepeat():
                self._tracker.key_down(event.key())
            elif t == self._KeyRelease and not event.isAutoRepeat():
                self._tracker.key_up(event.key())
            return False

    return _QtEventFilter


def _ensure_qt_tracker():
    global _qt_key_tracker
    if _qt_key_tracker is None:
        try:
            _qt_key_tracker = _QtKeyTracker()
        except Exception:
            pass
    return _qt_key_tracker


def is_key_pressed(key: str) -> bool:
    """Check if *key* is currently held down.

    Uses the ``keyboard`` library if available (Windows, or Linux with root).
    Otherwise falls back to Qt key-event tracking (requires the PyQtGraph
    window to have focus).
    """
    if _keyboard_available:
        return _kb.is_pressed(key)
    tracker = _ensure_qt_tracker()
    if tracker is not None:
        return tracker.is_pressed(key)
    return False


if platform.system() == "Windows":
    import win32gui
    import win32con

    def maximize_by_title(title):
        """Maximize a window by its title."""
        hwnd = win32gui.FindWindow(None, title)
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return True
        return False

    def minimise_by_title(title):
        """Minimize a window by its title."""
        hwnd = win32gui.FindWindow(None, title)
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return True
        return False

    def window_exists(title: str) -> bool:
        """Check if a window with the given title exists."""
        return win32gui.FindWindow(None, title) != 0

else:
    def maximize_by_title(title):
        """No-op on non-Windows platforms."""
        return False

    def minimise_by_title(title):
        """No-op on non-Windows platforms."""
        return False

    def window_exists(title: str) -> bool:
        """No-op on non-Windows platforms."""
        return False
