"""
Final fix for output formatting with correct indentation
"""

file_path = r"c:\Users\polam\OneDrive\Desktop\agentic projects\autonomous due diligence agent\src\agents\financial_analyst.py"

# Read file
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_count = 0

# Line 250: Inside if statement - needs 28 spaces
original_250 = '                            return f" {source}: {content[:500]}..."\n'
replacement_250 = '                            return content\n'

if 249 < len(lines) and lines[249] == original_250:
    lines[249] = replacement_250
    fixed_count += 1
    print("Fixed line 250 - removed source prefix")

#Line 260: Inside if statement - needs 18 spaces (not 19!)  
original_260 = '                  return f" {source}: {content[:400]}"\n'
replacement_260 = '                  return content\n'

if 259 < len(lines) and lines[259] == original_260:
    lines[259] = replacement_260
    fixed_count += 1
    print("Fixed line 260 - removed source prefix")

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

if fixed_count == 2:
    print("\nSUCCESS: Both lines fixed!")
else:
    print(f"\nWARNING: Only {fixed_count}/2 lines fixed")

print("Restart backend now")
