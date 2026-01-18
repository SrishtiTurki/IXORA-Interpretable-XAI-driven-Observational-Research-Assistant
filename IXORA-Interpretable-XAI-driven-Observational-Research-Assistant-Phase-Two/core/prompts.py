# Domain-specific system prompts
BIOMED_SYSTEM_PREFIX = """
You are a biomedical research assistant. Focus on medical science, enzymes, pH, doses, and related biomedical concepts.
Provide accurate, evidence-based information and cite relevant medical literature when possible.
"""

CS_SYSTEM_PREFIX = """
You are a computer science research assistant. Focus on algorithms, programming, data structures, 
artificial intelligence, and related technical topics. Provide clear, concise, and technically 
accurate information with code examples when relevant.
"""

def get_system_prompt(domain: str = "biomed") -> str:
    """Get the appropriate system prompt based on domain.
    
    Args:
        domain: The domain to get the prompt for ('biomed' or 'cs'/'computerscience')
        
    Returns:
        str: The system prompt for the specified domain
    """
    domain = domain.lower()
    if domain in ["cs", "computerscience", "ai"]:
        return CS_SYSTEM_PREFIX
    return BIOMED_SYSTEM_PREFIX

# Default export for backward compatibility
SYSTEM_PREFIX = BIOMED_SYSTEM_PREFIX