
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock dependencies
sys.modules['langchain_chroma'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.documents'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()

# Mock rag.core since it imports the above
mock_rag_core = MagicMock()
sys.modules['rag.core'] = mock_rag_core

from rag import rag_usage

class TestRagUsage(unittest.TestCase):
    def test_huggingface_rag(self):
        # Mock the ProductionRAGSystem class
        mock_system_instance = MagicMock()
        mock_rag_core.ProductionRAGSystem.return_value = mock_system_instance
        mock_system_instance.add_company_data.return_value = ["id1", "id2"]
        mock_system_instance.query.return_value = []

        # Run the function
        rag_usage.Huggingface_rag()

        # Verify it called ProductionRAGSystem
        mock_rag_core.ProductionRAGSystem.assert_called_with(
            persist_directory="./data/vector_stores/financial_hf",
            embedding_type="huggingface"
        )
        # Verify it called add_company_data (not add_documents)
        mock_system_instance.add_company_data.assert_called()

if __name__ == '__main__':
    unittest.main()
