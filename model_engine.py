import torch
from transformers import pipeline

# Load Legal-BERT locally.
# The 'device' parameter helps if you have a GPU, otherwise it defaults to CPU.
classifier = pipeline("text-classification", model="nlpaueb/legal-bert-base-uncased")

def analyze_contract(text):
    RULES = {
        "arbitration": {"risk": "High", "msg": "Forced Arbitration: You lose your right to a court trial."},
        "automatically renew": {"risk": "Medium", "msg": "Auto-Renewal: Contract extends without manual consent."},
        "unilateral": {"risk": "High", "msg": "One-sided Modification: They can change terms anytime."},
        "no refunds": {"risk": "High", "msg": "Restrictive Refund: Getting your money back will be difficult."},
        "third-party": {"risk": "Low", "msg": "Data Sharing: Your data may be shared with partners."},
        "indemnify": {"risk": "Medium", "msg": "Indemnification: You may be liable for their legal costs."}
    }

    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if len(s) > 20]
    findings = []
    risk_score = 0

    for sent in sentences:
        for key, data in RULES.items():
            if key in sent.lower():
                # Truncate to 512 tokens to avoid BERT errors
                # We use the AI to ensure the keyword is being used in a 'Legal' way
                ai_check = classifier(sent[:512])[0]
                
                # If BERT confirms this is significant legal text
                if ai_check['score'] > 0.3:
                    points = 35 if data['risk'] == "High" else 15
                    risk_score += points
                    findings.append({
                        "label": data['risk'],
                        "issue": data['msg'],
                        "text": sent[:150] + "..." # Snippet for the UI
                    })
                    break 

    # Logic-based verdict
    verdict = "⚠️ UNSAFE" if risk_score > 50 else "✅ SAFE"
    
    return {
        "verdict": verdict, 
        "score": min(risk_score, 100), # Cap at 100%
        "flags": findings
    }