"""
Ground truth Q&A pairs for RAGAS evaluation.
Written from the actual content of your uploaded GFR.pdf (General Financial
Rules 2017, updated up to 31.07.2024) so scores reflect real accuracy.

Feel free to add more pairs — 15-25 gives a more reliable average score.
"""

ground_truth_data = [
    {
        "question": "What does 'Competent Authority' mean under the GFR?",
        "ground_truth": "Competent Authority means, in respect of the power to be exercised under any of the GFR rules, the President or such other authority to which the power is delegated by or under these Rules, the Delegation of Financial Power Rules, or any other general or special orders issued by the Government of India.",
    },
    {
        "question": "What does 'Controlling Officer' mean under the GFR?",
        "ground_truth": "Controlling Officer means an officer entrusted by a Department of the Central Government with the responsibility of controlling the incurring of expenditure and/or the collection of revenue. The term includes a Head of Department and also an Administrator.",
    },
]