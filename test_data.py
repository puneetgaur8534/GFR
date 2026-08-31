from typing import List, Dict

# Evaluation Dataset for GFR 2017 RAG System
# Ground truths are extracted directly from official GFR 2017 provisions.

TEST_DATASET: List[Dict[str, str]] = [
    {
        "question": "What is the monetary limit for purchasing goods without inviting quotations or bids under GFR 2017?",
        "ground_truth": "According to Rule 154 of GFR 2017, purchase of goods up to the value of Rs. 25,000 (Rupees Twenty Five Thousand) only on each occasion may be made without inviting quotations or bids on the basis of a certificate recorded by the competent authority."
    },
    {
        "question": "When is procurement of goods and services through Government e-Marketplace (GeM) mandatory under GFR 2017?",
        "ground_truth": "According to Rule 149 of GFR 2017, procurement of Goods and Services by Ministries or Departments will be mandatory for Goods or Services available on the Government e-Marketplace (GeM)."
    }
]

def get_test_dataset() -> List[Dict[str, str]]:
    """Return the ground truth test dataset."""
    return TEST_DATASET

if __name__ == "__main__":
    print(f"📋 Loaded {len(TEST_DATASET)} ground truth evaluation records:")
    for idx, item in enumerate(TEST_DATASET, 1):
        print(f"\n[{idx}] Question: {item['question']}")
        print(f"    Ground Truth: {item['ground_truth']}")