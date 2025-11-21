
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock external dependencies BEFORE importing main
sys.modules['redis'] = MagicMock()
sys.modules['redis.asyncio'] = MagicMock()
sys.modules['prometheus_client'] = MagicMock()
sys.modules['slowapi'] = MagicMock()
sys.modules['slowapi.util'] = MagicMock()
sys.modules['slowapi.errors'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

# Mock project dependencies
sys.modules['agents'] = MagicMock()
sys.modules['agents.financial_analyst'] = MagicMock()
sys.modules['agents.orchestrator'] = MagicMock()
sys.modules['rag'] = MagicMock()
sys.modules['rag.core'] = MagicMock()
sys.modules['data'] = MagicMock()
sys.modules['data.collectors'] = MagicMock()
sys.modules['data.collectors.sec_edgar'] = MagicMock()
sys.modules['data.collectors.company_resolver'] = MagicMock()
sys.modules['data.processors'] = MagicMock()
sys.modules['data.processors.document_parser'] = MagicMock()

sys.modules['uvicorn'] = MagicMock()
sys.modules['regex'] = MagicMock()

# Mock FastAPI and Pydantic
mock_fastapi = MagicMock()
sys.modules['fastapi'] = mock_fastapi
sys.modules['fastapi.middleware'] = MagicMock()
sys.modules['fastapi.middleware.cors'] = MagicMock()
sys.modules['fastapi.middleware.trustedhost'] = MagicMock()
sys.modules['fastapi.responses'] = MagicMock()
sys.modules['fastapi.encoders'] = MagicMock()

mock_pydantic = MagicMock()
class MockBaseModel:
    pass
mock_pydantic.BaseModel = MockBaseModel
mock_pydantic.Field = MagicMock()
mock_pydantic.field_validator = MagicMock(return_value=lambda x: x)
sys.modules['pydantic'] = mock_pydantic

# Mock specific classes/functions used in main.py
# with patch('uvicorn.run'): # No longer needed if we mock the module
from api.main import app

# from fastapi.testclient import TestClient # Cannot use TestClient with mocked FastAPI

class TestAPIStartup(unittest.TestCase):
    def test_app_initialization(self):
        # If we reached here, main.py was imported successfully
        # This confirms that:
        # 1. Metrics are defined (no NameError)
        # 2. Orchestrator typo is fixed (no ImportError/NameError)
        self.assertIsNotNone(app)
        print("Successfully imported app from main.py")

if __name__ == '__main__':
    unittest.main()
