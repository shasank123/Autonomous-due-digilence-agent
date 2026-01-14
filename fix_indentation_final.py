"""
Robust fix for output formatting and indentation
"""
import sys

file_path = r"c:\Users\polam\OneDrive\Desktop\agentic projects\autonomous due diligence agent\src\agents\financial_analyst.py"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

fixed_count = 0

# Fix line 250 (approximate location)
# Look for the return statement with source prefix
for i, line in enumerate(lines):
    if 'return f" {source}: {content[:500]}..."' in line:
        # Get indentation from the line itself
        indentation = line[:line.find('return')]
        
        # Create new line with same indentation
        new_line = indentation + 'return content\n'
        
        lines[i] = new_line
        fixed_count += 1
        print(f"Fixed line {i+1}: Removed source prefix")

# Fix line 260 (approximate location)
for i, line in enumerate(lines):
    if 'return f" {source}: {content[:400]}"' in line:
        # Get indentation from the line itself
        indentation = line[:line.find('return')]
        
        # Create new line with same indentation
        new_line = indentation + 'return content\n'
        
        lines[i] = new_line
        fixed_count += 1
        print(f"Fixed line {i+1}: Removed source prefix")

if fixed_count > 0:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\nSUCCESS: Updated {fixed_count} lines.")
        print("Restart backend to apply changes.")
    except Exception as e:
        print(f"Error writing file: {e}")
else:
    print("\nWARNING: No lines matched for fixing.")
    # Print lines around 260 to debug
    if len(lines) > 260:
        print("\nContext around line 260:")
        for j in range(255, 265):
            if j < len(lines):
                print(f"{j+1}: {repr(lines[j])}")

