from tutor_bot.ui.app import _VISIBLE_APP_MODES
from tutor_bot.ui.app_mode import AppMode


def test_visible_app_modes_follow_required_order() -> None:
    assert _VISIBLE_APP_MODES == [
        AppMode.BROWSE_NOTES,
        AppMode.ADD_NOTE,
        AppMode.DATABASES,
        AppMode.TEST_NOTES,
        AppMode.LLMS,
        AppMode.PREPARE_FOR_VACANCY,
        AppMode.QUESTIONS,
    ]
