"""A small hand-written eval set over the sample corpus. Each item names the
source document the answer should come from and a keyword that should appear
in a correct answer -- crude but transparent proxies, good enough for a POC
comparison rather than a rigorous benchmark."""

EVAL_QUESTIONS = [
    {
        "question": "How many days of paid annual leave do employees get per year?",
        "expected_doc_id": "hr_leave_policy",
        "expected_keywords": ["22"],
    },
    {
        "question": "How many unused annual leave days can I carry over into next year?",
        "expected_doc_id": "hr_leave_policy",
        "expected_keywords": ["5"],
    },
    {
        "question": "How many weeks of parental leave do primary caregivers get?",
        "expected_doc_id": "hr_leave_policy",
        "expected_keywords": ["16"],
    },
    {
        "question": "What is the daily meal per-diem for business travel in the US?",
        "expected_doc_id": "expense_policy",
        "expected_keywords": ["75", "$75"],
    },
    {
        "question": "How many days do I have to submit an expense report after the expense was incurred?",
        "expected_doc_id": "expense_policy",
        "expected_keywords": ["30"],
    },
    {
        "question": "How soon after starting must I enable multi-factor authentication?",
        "expected_doc_id": "security_policy",
        "expected_keywords": ["7"],
    },
    {
        "question": "How often should passwords be rotated for production system access?",
        "expected_doc_id": "security_policy",
        "expected_keywords": ["180"],
    },
    {
        "question": "What software do new engineers need installed for local development?",
        "expected_doc_id": "engineering_onboarding",
        "expected_keywords": ["Docker", "Python", "Node"],
    },
    {
        "question": "What is the maximum payload capacity of the RoboArm X1?",
        "expected_doc_id": "product_faq",
        "expected_keywords": ["5 kilogram", "5kg", "5 kg"],
    },
    {
        "question": "What is the warranty period on the RoboArm X1?",
        "expected_doc_id": "product_faq",
        "expected_keywords": ["2-year", "2 year", "two year", "two-year"],
    },
]
