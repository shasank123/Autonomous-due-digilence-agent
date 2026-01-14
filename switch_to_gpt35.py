#!/usr/bin/env python
"""
Quick script to switch model defaults to use environment variable
Run with: python switch_to_gpt35.py
"""
import os
import re

def update_create_function(filepath):
    """Update create_*_team functions to use OPENAI_MODEL env var"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern 1: Update function signature
    content = re.sub(
        r'async def create_\w+_team\(\s*model:\s*str\s*=\s*"gpt-4o"',
        r'async def create_\w+_team(\n        model: Optional[str] = None',
        content
    )
    
    # Pattern 2: Add model env var check after other env var checks
    # Find the pattern where api_key is fetched from env
    content = re.sub(
        r'(\s+)if not api_key:\s+api_key = os\.getenv\("OPENAI_API_KEY"\)',
        r'\1# Model environment variable fallback\n\1if not model:\n\1    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")\n\1\n\1if not api_key:\n\1    api_key = os.getenv("OPENAI_API_KEY")',
        content
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Updated {os.path.basename(filepath)}")

def main():
    """Update all agent files"""
    base_dir = "src/agents"
    files_to_update = [
        "financial_analyst.py",
        # Add others as needed
    ]
    
    for filename in files_to_update:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            try:
                update_create_function(filepath)
            except Exception as e:
                print(f"✗ Error updating {filename}: {e}")
        else:
            print(f"✗ File not found: {filepath}")
    
    print("\n✓ All files updated successfully!")
    print("Model will now use OPENAI_MODEL env var (defaults to gpt-3.5-turbo)")

if __name__ == "__main__":
    main()
