"""The 10 test questions + a ground-truth rubric used for judging retrieval.

`facts`      : list of fact groups; a group is satisfied if ANY of its strings
               appears (case-insensitive) in the retrieved text.
`min_needed` : how many groups must be satisfied for a 'yes'.
               Some found but fewer than min_needed -> 'partial'. None -> 'no'.
`answerable` : False if the reference document does not contain the answer.
"""
QUESTIONS = [
    {"q": "What are the two main diagnostic criteria for Type 2 diabetes based on HbA1c and fasting plasma glucose?",
     "facts": [["6.5%"], ["126 mg/dl"]], "min_needed": 2, "answerable": True,
     "gold": "HbA1c >= 6.5% and fasting plasma glucose >= 126 mg/dL"},
    {"q": "What HbA1c range indicates prediabetes?",
     "facts": [["5.7%"], ["6.4%"]], "min_needed": 2, "answerable": True,
     "gold": "5.7%-6.4%"},
    {"q": "Name three first-line oral medications used to treat Type 2 diabetes.",
     "facts": [["metformin"], ["sulfonylurea"], ["dpp-4"], ["sglt2"], ["glp-1"]],
     "min_needed": 3, "answerable": True,
     "gold": "Doc names metformin as first-line; lists sulfonylureas, DPP-4, SGLT2, GLP-1 as other oral options"},
    {"q": "What are the classic symptoms of hyperglycemia?",
     "facts": [["thirst"], ["urination"], ["hunger"], ["fatigue"], ["blurred vision"]],
     "min_needed": 3, "answerable": True,
     "gold": "Thirst, frequent urination, hunger, weight loss, fatigue, blurred vision..."},
    {"q": "What lifestyle modifications are recommended for managing Type 2 diabetes?",
     "facts": [["weight loss"], ["physical activity"], ["whole grains"], ["smoking cessation"], ["alcohol"]],
     "min_needed": 4, "answerable": True,
     "gold": "Weight loss, 150 min/week exercise, healthy diet, quit smoking, limit alcohol"},
    {"q": "Name two microvascular complications of diabetes.",
     "facts": [["retinopathy"], ["nephropathy"], ["neuropathy"]], "min_needed": 2, "answerable": True,
     "gold": "Retinopathy, nephropathy, neuropathy"},
    {"q": "Name two macrovascular complications of diabetes.",
     "facts": [["coronary artery disease"], ["cerebrovascular"], ["peripheral vascular"]],
     "min_needed": 2, "answerable": True,
     "gold": "Coronary artery disease, cerebrovascular disease, peripheral vascular disease"},
    {"q": "How often should HbA1c be monitored in someone with stable Type 2 diabetes?",
     "facts": [["3-6 months"]], "min_needed": 1, "answerable": True,
     "gold": "Every 3-6 months"},
    {"q": "What blood glucose target range is recommended for most adults with Type 2 diabetes?",
     "facts": [], "min_needed": 1, "answerable": False,
     "gold": "NOT IN DOCUMENT (the guide gives diagnostic thresholds, not treatment targets). Correct behaviour = abstain."},
    {"q": "What symptoms should prompt someone to seek emergency medical care?",
     "facts": [["300 mg/dl"], ["loss of consciousness", "seizures"], ["chest pain"], ["foot wound"]],
     "min_needed": 3, "answerable": True,
     "gold": "Glucose >300 + vomiting/confusion; severe hypoglycemia; chest pain/stroke signs; worsening foot wound"},
]
