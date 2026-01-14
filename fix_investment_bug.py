"""
Quick fix for the investment summary bug in financial_tools.py
Removes the incorrect len() call on line 817
"""

file_path = r"c:\Users\polam\OneDrive\Desktop\agentic projects\autonomous due diligence agent\src\tools\financial_tools.py"

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 817 (0-indexed, so line 816)
# Change: 'has_sufficient_data': len(high_confidence_documents) >= 5 and...
# To: 'has_sufficient_data': high_confidence_documents >= 5 and...

if 816 < len(lines):
    original_line = lines[816]
    if "len(high_confidence_documents)" in original_line:
        fixed_line = original_line.replace("len(high_confidence_documents)", "high_confidence_documents")
        lines[816] = fixed_line
        print("FIXED line 817")
        print(f"Old: {original_line.strip()}")
        print(f"New: {fixed_line.strip()}")
    else:
        print("WARNING: Line 817 doesn't contain expected text")
        print(f"Found: {original_line.strip()}")
else:
    print("ERROR: File doesn't have 817 lines")

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\nFile updated successfully!")
