import re
from transformers import pipeline

# 1. Load Legal-BERT with explicit local configuration
# Using a dedicated device (0 for GPU, -1 for CPU) ensures stability in production
try:
    classifier = pipeline(
        "text-classification", 
        model="nlpaueb/legal-bert-base-uncased",
        device=-1  # Default to CPU for maximum compatibility
    )
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

def analyze_contract(text):
    """
    Analyzes contract text using a hybrid approach of Regex pattern matching
    and Transformer-based contextual validation.
    """
    
    # 2. Refined Rules with Regex Patterns
    # Uses word boundaries (\b) to prevent accidental matches like "international" for "internal"
    RULES = {
        "arbitration": {
            "pattern": r"\barbitration\b", 
            "risk": "High", 
            "msg": "Forced Arbitration: You lose your right to a court trial."
        },
        "auto_renew": {
            "pattern": r"\bautomatically\s+renews?\b|\bauto-renew\b", 
            "risk": "Medium", 
            "msg": "Auto-Renewal: Contract extends without manual consent."
        },
        "unilateral": {
            "pattern": r"\bunilateral\b|\bat\s+any\s+time\s+without\s+notice\b", 
            "risk": "High", 
            "msg": "One-sided Modification: They can change terms anytime."
        },
        "refunds": {
            "pattern": r"\bno\s+refunds?\b|\bnon-refundable\b", 
            "risk": "High", 
            "msg": "Restrictive Refund: Getting your money back will be difficult."
        },
        "data_sharing": {
            "pattern": r"\bthird-party\b|\bsharing\s+with\s+partners\b", 
            "risk": "Low", 
            "msg": "Data Sharing: Your data may be shared with partners."
        },
        "indemnify": {
            "pattern": r"\bindemnify\b|\bhold\s+harmless\b", 
            "risk": "Medium", 
            "msg": "Indemnification: You may be liable for their legal costs."
        }
    }

    # 3. Text Pre-processing
    # Split into sentences while removing empty strings and excessive whitespace
    sentences = [s.strip() for s in re.split(r'[.\n]', text) if len(s.strip()) > 25]
    
    findings = []
    seen_issues = set()  # Prevent duplicate flagging of the same issue type
    total_risk_score = 0

    for sent in sentences:
        for key, data in RULES.items():
            # Check for pattern match
            if re.search(data['pattern'], sent, re.IGNORECASE):
                
                # Contextual AI Validation: Ensure BERT is loaded and sentence isn't too long
                # We slice to 500 chars to stay safely under BERT's 512-token limit
                is_significant = True
                if classifier:
                    try:
                        ai_check = classifier(sent[:500])[0]
                        # Only flag if the AI is reasonably confident this is legal/formal text
                        if ai_check['score'] < 0.25:
                            is_significant = False
                    except:
                        is_significant = True # Fallback to regex if AI fails

                if is_significant:
                    # Prevent penalizing the score multiple times for the same rule
                    if data['msg'] not in seen_issues:
                        points = 35 if data['risk'] == "High" else (15 if data['risk'] == "Medium" else 5)
                        total_risk_score += points
                        seen_issues.add(data['msg'])

                    # We still collect the finding instance (up to 10 for UI clarity)
                    if len(findings) < 10:
                        findings.append({
                            "label": data['risk'],
                            "issue": data['msg'],
                            "text": sent[:160] + "..." # Clean snippet
                        })
                    break 

    # 4. Final Verdict Logic
    # Weighted calculation for a more realistic assessment
    final_score = min(total_risk_score, 100)
    verdict = "⚠️ UNSAFE" if final_score > 50 else "✅ SAFE"
    
    return {
        "verdict": verdict, 
        "score": final_score,
        "flags": findings
    }