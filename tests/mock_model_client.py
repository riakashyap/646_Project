"""
Copyright:

  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file provides a mock large language model.

Code:
"""

from src.model_clients import ModelClient
from typing import Tuple


class MockModelClient(ModelClient):

    ret: str = ""
    last_user_prompt: str = ""
    last_system_prompt: str = ""

    def set_ret(self, ret: str) -> None:
        self.ret = ret

    def get_prompts(self) -> Tuple[str, str]:
        return self.last_user_prompt, self.last_system_prompt

    def send_query(self, user_prompt: str,
                   system_prompt: str | None = None) -> str:
        self.last_user_prompt = user_prompt
        self.last_system_prompt = system_prompt
        return self.ret
