# Biomed-specific overrides
BIOMED_SYSTEM_PREFIX = """
Biomed-specific: Focus on medical science, enzymes, pH, doses. Use your original rules.
"""

# CS-specific overrides
CS_SYSTEM_PREFIX = """
CS-specific: Focus on algorithms, complexity, data/compute constraints, reproducibility. Use your original rules.
"""

# Export for use in langgraph if needed
SYSTEM_PREFIX = BIOMED_SYSTEM_PREFIX