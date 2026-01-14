"""
Comprehensive module import test
Tests all modules in the codebase for import errors
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

modules_to_test = [
    # Agents
    'agents.orchestrator',
    'agents.financial_analyst',
    'agents.market_analyst',
    'agents.legal_reviewer',
    'agents.memory_manager',
    
    # Tools
    'tools.financial_tools',
    'tools.market_tools',
    'tools.legal_tools',
    
    # Data
    'data.collectors.sec_edgar',
    'data.collectors.company_resolver',
    'data.processors.document_parser',
    
    # RAG
    'rag.core',
    'rag.rag_usage',
    
    # API
    'api.main',
    
    # Workflows
    'workflows.due_diligence',
    
    # Utils
    'utils.config',
    'utils.logger',
    
    # Core
    'core.config',
    
    # UI
    'ui.app',
]

def test_import(module_name):
    """Test importing a single module"""
    try:
        __import__(module_name)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 80)
    print("MODULE IMPORT TEST")
    print("=" * 80)
    
    results = {'passed': [], 'failed': []}
    
    for module in modules_to_test:
        print(f"\nTesting: {module}", end=" ... ")
        success, error = test_import(module)
        
        if success:
            print("PASS")
            results['passed'].append(module)
        else:
            print(f"FAIL")
            print(f"  Error: {error}")
            results['failed'].append((module, error))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Passed: {len(results['passed'])}/{len(modules_to_test)}")
    print(f"Failed: {len(results['failed'])}/{len(modules_to_test)}")
    
    if results['failed']:
        print("\n" + "=" * 80)
        print("FAILED MODULES")
        print("=" * 80)
        for module, error in results['failed']:
            print(f"\n{module}:")
            print(f"  {error}")
    
    return len(results['failed']) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
