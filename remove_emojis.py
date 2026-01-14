#!/usr/bin/env python
"""
Remove all emojis from the project to fix Windows console encoding issues.
This script replaces Unicode emojis with ASCII equivalent text.
"""
import os
import re
from pathlib import Path

# Define emoji replacements (emoji -> text)
EMOJI_REPLACEMENTS = {
    '[OK]': '[OK]',
    '[ERROR]': '[ERROR]',
    '[DATA]': '[DATA]',
    '[TIME]': '[TIME]',
    '[NOTE]': '[NOTE]',
    '[SEARCH]': '[SEARCH]',
    '[MONEY]': '[MONEY]',
    '[WARN]': '[WARN]',
    '[TIP]': '[TIP]',
    '[FIX]': '[FIX]',
    '[TARGET]': '[TARGET]',
    '[LAUNCH]': '[LAUNCH]',
    '[FOLDER]': '[FOLDER]',
    '[SAVE]': '[SAVE]',
    '[SKIP]': '[SKIP]',
    '[DONE]': '[DONE]',
    '[REFRESH]': '[REFRESH]',
}

def remove_emojis_from_file(filepath):
    """Remove emojis from a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Replace each emoji with its text equivalent
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            if emoji in content:
                count = content.count(emoji)
                content = content.replace(emoji, replacement)
                changes_made.append(f"  - Replaced {count}x '{emoji}' with '{replacement}'")
        
        # Write back if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made
        
        return False, []
    
    except Exception as e:
        print(f"  [ERROR] Failed to process {filepath}: {e}")
        return False, []

def scan_and_fix_project(project_root):
    """Scan entire project and remove emojis"""
    
    print("Emoji Removal Tool for Windows Compatibility\n")
    print(f"Scanning project: {project_root}\n")
    
    # File extensions to process
    extensions = ['.py', '.md', '.txt', '.json', '.yaml', '.yml']
    
    # Directories to skip
    skip_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', 
                 'due_diligence_venv', '.pytest_cache', 'logs'}
    
    files_processed = 0
    files_modified = 0
    total_changes = 0
    
    for root, dirs, files in os.walk(project_root):
        # Remove skip directories from the walk
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            # Check if file has one of our target extensions
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, project_root)
                
                files_processed += 1
                modified, changes = remove_emojis_from_file(filepath)
                
                if modified:
                    files_modified += 1
                    total_changes += len(changes)
                    print(f"\n[MODIFIED] {relative_path}")
                    for change in changes:
                        print(change)
    
    print(f"\n\nSummary:")
    print(f"  - Files scanned: {files_processed}")
    print(f"  - Files modified: {files_modified}")
    print(f"  - Total replacements: {total_changes}")
    print(f"\n[COMPLETE] All emojis removed!\n")

if __name__ == "__main__":
    # Get project root (parent of this script's directory)
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    scan_and_fix_project(str(project_root))
