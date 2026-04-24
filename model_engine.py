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
            "pattern": r"\barbitration\b|\bbinding\s+arbitration\b", 
            "risk": "High", 
            "msg": "Forced Arbitration: You lose your right to a court trial."
        },
        "auto_renew": {
            "pattern": r"\bautomatically\s+renews?\b|\bauto-renew\b|\bautomatic\s+renewal\b", 
            "risk": "Medium", 
            "msg": "Auto-Renewal: Contract extends without manual consent."
        },
        "unilateral": {
            "pattern": r"\bunilateral\b|\bat\s+any\s+time\s+without\s+notice\b|\bsole\s+discretion\b|\bright\s+to\s+modify\b", 
            "risk": "High", 
            "msg": "One-sided Modification: They can change terms anytime at their sole discretion."
        },
        "refunds": {
            "pattern": r"\bno\s+refunds?\b|\bnon-refundable\b|\ball\s+sales\s+are\s+final\b", 
            "risk": "High", 
            "msg": "Restrictive Refund: Getting your money back will be difficult or impossible."
        },
        "data_sharing": {
            "pattern": r"\bthird-party\b|\bsharing\s+with\s+partners\b|\bsell\s+your\s+data\b", 
            "risk": "Low", 
            "msg": "Data Sharing: Your data may be shared with or sold to partners/third-parties."
        },
        "indemnify": {
            "pattern": r"\bindemnify\b|\bhold\s+harmless\b|\bindemnification\b", 
            "risk": "Medium", 
            "msg": "Indemnification: You may be held liable for their legal costs or damages."
        },
        "non_compete": {
            "pattern": r"\bnon-compete\b|\bnot\s+to\s+compete\b|\bnon-competition\b",
            "risk": "High",
            "msg": "Non-Compete: Restricts your ability to work or operate in similar fields."
        },
        "liability_cap": {
            "pattern": r"\blimitation\s+of\s+liability\b|\bcap\s+on\s+liability\b|\bnot\s+be\s+liable\s+for\s+any\s+indirect\b",
            "risk": "High",
            "msg": "Liability Limitation: Severely caps the damages you can claim if they wrong you."
        },
        "jury_waiver": {
            "pattern": r"\bwaiver\s+of\s+jury\s+trial\b|\bwaive[a-z]*\s+jury\b",
            "risk": "High",
            "msg": "Jury Trial Waiver: You surrender your constitutional right to a jury."
        },
        "warranty_disclaimer": {
            "pattern": r"\bas\s+is\b|\bdisclaim[a-z]*\s+all\s+warranties\b|\bwithout\s+warranty\b",
            "risk": "Medium",
            "msg": "Warranty Disclaimer: The service/product is provided 'as is' with no guarantees."
        },
        "ip_assignment": {
            "pattern": r"\birrevocable\s+license\b|\bassignment\s+of\s+intellectual\s+property\b|\bperpetual,\s+worldwide\b",
            "risk": "Medium",
            "msg": "IP Rights: You may be granting them permanent rights to your created content."
        },
        "termination_convenience": {
            "pattern": r"\bterminat[a-z]*\s+for\s+convenience\b|\bterminat[a-z]*\s+at\s+any\s+time\b",
            "risk": "Medium",
            "msg": "Termination for Convenience: They can cancel the contract at any time for no reason."
        },
        "liquidated_damages": {
            "pattern": r"\bliquidated\s+damages\b|\bpenalty\s+fee\b",
            "risk": "Medium",
            "msg": "Liquidated Damages: Pre-set financial penalties you must pay for breaching the contract."
        },
        "governing_law_venue": {
            "pattern": r"\bgoverning\s+law\b|\bexclusive\s+jurisdiction\b|\bexclusive\s+venue\b",
            "risk": "Low",
            "msg": "Jurisdiction Constraint: You may be forced to resolve disputes in an inconvenient state or country."
        },
        "non_disparagement": {
            "pattern": r"\bnon-disparagement\b|\bnot\s+disparage\b|\bnegative\s+comments\b",
            "risk": "Medium",
            "msg": "Non-Disparagement: You are legally forbidden from speaking negatively about the company."
        },
        "class_action_waiver": {
            "pattern": r"\bclass\s+action\s+waiver\b|\bwaive\s+right\s+to\s+participate\s+in\s+a\s+class\s+action\b",
            "risk": "High",
            "msg": "Class Action Waiver: You cannot join with others to sue the company."
        },
        "survival": {
            "pattern": r"\bsurvival\s+of\s+terms\b|\bshall\s+survive\s+termination\b",
            "risk": "Medium",
            "msg": "Survival: Certain restrictive terms continue to apply even after you cancel."
        },
        "entire_agreement": {
            "pattern": r"\bentire\s+agreement\b|\bsupersedes\s+all\s+prior\b",
            "risk": "Low",
            "msg": "Entire Agreement: Any verbal promises made outside this text are legally void."
        },
        "force_majeure": {
            "pattern": r"\bforce\s+majeure\b|\bact\s+of\s+god\b|\bbeyond\s+reasonable\s+control\b",
            "risk": "Low",
            "msg": "Force Majeure: They can delay or cancel services during unforeseen emergencies."
        },
        "assignment_without_consent": {
            "pattern": r"\bassign\s+this\s+agreement\b|\bfreely\s+assignable\b|\bwithout\s+your\s+consent\b",
            "risk": "Medium",
            "msg": "Assignment: They can sell or transfer your contract to another company without asking you."
        },
        "right_of_first_refusal": {
            "pattern": r"\bright\s+of\s+first\s+refusal\b",
            "risk": "Medium",
            "msg": "First Refusal: You must offer them a deal before taking it to a competitor."
        },
        "confidentiality": {
            "pattern": r"\bconfidential\s+information\b|\bnon-disclosure\b|\bstrictly\s+confidential\b",
            "risk": "Medium",
            "msg": "Confidentiality/NDA: You are strictly barred from sharing details about this arrangement."
        },
        "automatic_price_increase": {
            "pattern": r"\bautomatic\s+rent\s+increase\b|\bincrease\s+annually\b|\bprice\s+may\s+increase\b",
            "risk": "High",
            "msg": "Automatic Price Increase: Fees can go up on a schedule without renegotiation."
        },
        "subcontracting": {
            "pattern": r"\bright\s+to\s+subcontract\b|\bengage\s+subcontractors\b",
            "risk": "Medium",
            "msg": "Subcontracting: They can hire unknown third parties to do the work you paid them for."
        },
        "exclusivity": {
            "pattern": r"\bexclusive\s+provider\b|\bsole\s+supplier\b|\bshall\s+not\s+engage\s+any\s+other\b",
            "risk": "High",
            "msg": "Exclusivity: You are forbidden from hiring or using competing services."
        },
        "acceleration": {
            "pattern": r"\bacceleration\s+of\s+payments\b|\ball\s+amounts\s+become\s+immediately\s+due\b",
            "risk": "High",
            "msg": "Payment Acceleration: If you default, they can demand the entire contract value immediately."
        },
        "liability_shift": {
            "pattern": r"\bassume\s+all\s+risk\b|\bat\s+your\s+own\s+risk\b",
            "risk": "High",
            "msg": "Liability Shift: You take full responsibility for any injuries, damages, or losses."
        },
        "marketing_use": {
            "pattern": r"\buse\s+your\s+name\s+and\s+logo\b|\bfor\s+marketing\s+purposes\b",
            "risk": "Low",
            "msg": "Marketing Consent: They can use your name or company logo in their advertisements."
        },
        "fee_shifting": {
            "pattern": r"\bprevailing\s+party\b|\bcosts\s+and\s+attorney[s]?\s+fees\b",
            "risk": "Medium",
            "msg": "Fee Shifting: If you lose a lawsuit against them, you pay their expensive legal bills."
        }
    }

    # 3. Text Pre-processing
    # Replace newlines with spaces so sentences aren't cut in half, then split by periods
    text = text.replace('\n', ' ')
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    
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
                    if data['msg'] not in seen_issues:
                        # Scoring Curve: 
                        # 1st High Risk = 25 pts. 
                        # Subsequent High Risks = 30 pts (ensures 2 Highs = 55, tripping the >50 UI).
                        # Medium = 15 pts, Low = 5 pts.
                        if data['risk'] == "High":
                            has_high = any(f['label'] == 'High' for f in findings)
                            points = 30 if has_high else 25
                        elif data['risk'] == "Medium":
                            points = 15
                        else:
                            points = 5
                            
                        total_risk_score += points
                        seen_issues.add(data['msg'])

                        # Collect the finding instance (up to 10 for UI clarity)
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