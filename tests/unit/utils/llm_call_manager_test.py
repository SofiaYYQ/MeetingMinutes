

from utils.llm_call_manager import LLMCallManager


class TestLLMCallManager:
    def test_singleton(self):
        manager1 = LLMCallManager()
        manager2 = LLMCallManager()

        assert id(manager1) == id(manager2)
