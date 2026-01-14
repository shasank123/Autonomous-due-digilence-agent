#!/usr/bin/env python3
"""Fix the misleading validation logic in financial_tools.py"""

file_path = r"c:\Users\polam\OneDrive\Desktop\agentic projects\autonomous due diligence agent\src\tools\financial_tools.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the misleading conflict logic
old_validation_logic = """    def _analyze_findings_validation(self, docs: List[Tuple], findings: str, company: str) -> Dict:
        # Simple overlap logic restored
        supporting, conflicting, neutral = [], [], []
        terms = findings.lower().split()
        for doc, score in docs:
            matches = sum(1 for t in terms if t in doc.page_content.lower())
            relevance = matches / len(terms) if terms else 0
            item = {'content': doc.page_content, 'conf': min(score, relevance)}
            
            if relevance > 0.6: supporting.append(item)
            elif relevance < 0.2: conflicting.append(item)
            else: neutral.append(item)
            
        return {'supporting': supporting, 'conflicting': conflicting, 'neutral': neutral}

    def _format_validation_report(self, res: Dict, company: str, total: int) -> str:
        report = [f" [REPORT] Validation: {company}"]
        if res['supporting']:
            report.append(f" [SUPPORT] {len(res['supporting'])} docs confirm findings.")
        if res['conflicting']:
            report.append(f" [CONFLICT] {len(res['conflicting'])} docs conflict.")
        return "\\n".join(report)"""

new_validation_logic = """    def _analyze_findings_validation(self, docs: List[Tuple], findings: str, company: str) -> Dict:
        # Improved validation logic - low relevance != conflict
        supporting, irrelevant, neutral = [], [], []
        terms = findings.lower().split()
        
        # Filter out common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'to'}
        meaningful_terms = [t for t in terms if t not in stop_words and len(t) > 2]
        if not meaningful_terms:
            meaningful_terms = terms
        
        for doc, score in docs:
            content_lower = doc.page_content.lower()
            matches = sum(1 for t in meaningful_terms if t in content_lower)
            relevance = matches / len(meaningful_terms) if meaningful_terms else 0
            item = {'content': doc.page_content, 'conf': score, 'relevance': relevance}
            
            if relevance >= 0.5: supporting.append(item)
            elif relevance < 0.15: irrelevant.append(item)
            else: neutral.append(item)
            
        return {'supporting': supporting, 'conflicting': [], 'neutral': neutral, 'irrelevant': irrelevant}

    def _format_validation_report(self, res: Dict, company: str, total: int) -> str:
        report = [f" [REPORT] Validation Summary for {company}:"]
        report.append(f" [INFO] Analyzed {total} documents")
        
        if res['supporting']:
            report.append(f" [SUPPORT] {len(res['supporting'])} docs support findings")
        if res.get('neutral'):
            report.append(f" [NEUTRAL] {len(res['neutral'])} docs with partial relevance")
        if res.get('irrelevant'):
            report.append(f" [INFO] {len(res['irrelevant'])} docs not directly relevant")
        
        # Assessment
        if len(res['supporting']) >= 3:
            report.append(" [VERDICT] Findings are well-supported")
        elif len(res['supporting']) > 0:
            report.append(" [VERDICT] Findings have some support")
        else:
            report.append(" [VERDICT] Limited supporting evidence")
        
        return "\\n".join(report)"""

if old_validation_logic in content:
    content = content.replace(old_validation_logic, new_validation_logic)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Fixed validation logic in financial_tools.py")
    print("Changed: Low relevance documents are now marked as 'irrelevant' instead of 'conflicting'")
    print("This will prevent agents from getting stuck on false conflicts")
else:
    print("ERROR: Could not find the target code to replace")
    print("The file may have already been modified or has different formatting")
