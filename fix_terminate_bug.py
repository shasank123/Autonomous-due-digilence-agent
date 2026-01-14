#!/usr/bin/env python3
"""Fix the TERMINATE keyword in the prompt that causes premature termination"""

file_path = r"c:\Users\polam\OneDrive\Desktop\agentic projects\autonomous due diligence agent\src\agents\financial_analyst.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic line
old_line = '3. REVIEWER: Validate the findings. If valid, summarize and type "TERMINATE".'
new_line = '3. REVIEWER: Validate the findings. If valid, provide your final summary and complete the review.'

content = content.replace(old_line, new_line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed TERMINATE keyword in financial_analyst.py")
print(f"Changed: {old_line}")
print(f"To: {new_line}")
