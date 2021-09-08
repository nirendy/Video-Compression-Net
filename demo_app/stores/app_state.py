from demo_app.stores import AppStateKeys
from demo_app.stores import BaseState


class AppState(BaseState[AppStateKeys]):
    def state_prefix(self) -> str:
        return "app_"
