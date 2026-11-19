"""Standalone LiteLLM access for the three planned investigation specialists."""
import os


ROLES = {
    "health_assessment": "Assess tyre health using the supplied measurements and evidence.",
    "root_cause_analysis": "Identify plausible causes and distinguish hypotheses from verified facts.",
    "maintenance_planning": "Draft a maintenance plan for human review using the supplied evidence.",
}


def request_text(role: str, prompt: str) -> str:
    """Call the configured provider only when explicitly invoked."""
    if role not in ROLES:
        raise ValueError(f"Unknown specialist role: {role}")
    if not prompt.strip():
        raise ValueError("Supply a nonempty prompt.")
    model = os.getenv(f"TYREMIND_LLM_MODEL_{role.upper()}") or os.getenv("TYREMIND_LLM_MODEL")
    if not model or not model.strip():
        raise ValueError("Set TYREMIND_LLM_MODEL to a LiteLLM provider/model identifier.")
    from litellm import completion

    response = completion(model=model.strip(), messages=[
        {"role": "system", "content": ROLES[role]},
        {"role": "user", "content": prompt},
    ], timeout=30, num_retries=0)
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise RuntimeError("Provider did not return a complete text reply.")
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Provider returned an empty text reply.")
    return content.strip()
