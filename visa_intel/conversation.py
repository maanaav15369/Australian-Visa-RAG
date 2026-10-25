"""Intent routing for natural conversation and immigration questions."""
import re


_REPLIES = {
    "hi": "Hi! How can I help with your immigration plans today?",
    "hii": "Hi! How can I help with your immigration plans today?",
    "hiii": "Hi! How can I help with your immigration plans today?",
    "hello": "Hello! What would you like help with today?",
    "helloo": "Hello! What would you like help with today?",
    "hey": "Hey! How can I help with your immigration plans?",
    "heyy": "Hey! How can I help with your immigration plans?",
    "hiya": "Hi! What can I help you explore today?",
    "hi there": "Hi there! How can I help with your immigration plans today?",
    "hello there": "Hello! What would you like help with today?",
    "hey there": "Hey! How can I help?",
    "good morning": "Good morning! How can I help with your immigration plans?",
    "good afternoon": "Good afternoon! What would you like help with today?",
    "good evening": "Good evening! How can I help?",
    "how are you": (
        "I'm here and ready to help, thanks for asking. "
        "What would you like to explore today?"
    ),
    "how are you doing": (
        "I'm here and ready to help, thanks for asking. "
        "What would you like to explore?"
    ),
    "hows it going": "All set to help. What would you like to explore today?",
    "whats up": "I'm here to help. What is on your mind?",
    "thanks": "You're welcome! What would you like to look at next?",
    "thank you": "You're welcome! What would you like to explore next?",
    "thanks a lot": "You're welcome! Happy to help with your next question.",
    "thank you so much": "You're welcome! What else would you like to explore?",
    "who are you": (
        "I'm an immigration information assistant. I explain official "
        "requirements in plain language and help organize next steps. "
        "Australia is currently the only country with indexed evidence."
    ),
    "what can you do": (
        "I can help explain immigration information, organize your goals, "
        "and work through next steps. Australia is currently the only country "
        "with indexed evidence in this system."
    ),
}


def _normalise(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


def try_small_talk_reply(text: str) -> str | None:
    """Handle exact short greetings and pleasantries."""
    return _REPLIES.get(_normalise(text))


_RULE_MARKERS = (
    "requirement", "eligib", "qualify", "qualification", "criteria",
    "application fee", "application charge", "base application", "visa fee",
    "visa fees", "processing time", "processing times", "work rights",
    "work limit", "points test", "points score", "documents required",
    "required documents", "visa condition", "visa conditions",
    "application process", "how to apply", "apply for a visa",
    "can i apply for", "current rule", "current rules", "legal rule",
    "legal rules", "legislation", "what are the rules", "what is the rule",
    "what is a student visa", "what is the student visa",
    "what is subclass", "how much is the visa", "how much does the visa",
    "how long does the visa", "can i work on", "can i stay on",
    "can i bring", "am i eligible", "do i qualify", "is it legal",
    "visa cost", "visa costs",
)

_PLANNING_MARKERS = (
    "guide me", "where do i start", "where should i start", "where to start",
    "dont know where to start", "do not know where to start",
    "help me plan", "help me choose", "help me figure out",
    "thinking about", "thinking of", "planning to", "looking to",
    "explore my options", "what options might suit", "best path for me",
    "not sure where", "not sure what to do", "i need guidance",
    "can you guide", "i want to", "i would like to", "id like to",
    "help me with my plans", "help me get started",
)

_TARGET_PATTERNS = {
    "AU": (
        r"\b(?:in|to|for)\s+australia\b",
        r"\bfor\s+australia\b",
        r"\baustralian\s+(?:visa|subclass|student visa|work visa|immigration)\b",
        r"\b(?:visa|immigration|student visa|work visa)\s+(?:in|for|to)\s+australia\b",
        r"\baustralia\s+subclass\s+\d{3}\b",
    ),
    "CA": (
        r"\b(?:study|studying|work|working|move|moving|visit|visiting|"
        r"immigrate|immigrating|settle|settling)\s+(?:in|to)\s+canada\b",
        r"\b(?:visa|immigration|study permit|work permit|student visa|"
        r"permanent residence)\s+(?:in|for|to)\s+canada\b",
        r"\bcanadian\s+(?:visa|study permit|work permit|student visa|"
        r"immigration|permanent residence)\b",
        r"\bcanada\s+(?:student\s+)?(?:visa|study permit|work permit|"
        r"immigration|permanent residence)\b",
    ),
    "NZ": (
        r"\b(?:study|studying|work|working|move|moving|visit|visiting|"
        r"immigrate|immigrating|settle|settling)\s+(?:in|to)\s+new zealand\b",
        r"\b(?:visa|immigration|student visa|work visa|residence)\s+"
        r"(?:in|for|to)\s+new zealand\b",
        r"\bnew zealand\s+(?:visa|immigration|student visa|work visa|"
        r"residence|visa requirements|visa rules)\b",
        r"\bnz\s+(?:visa|immigration|student visa|work visa)\b",
    ),
    "US": (
        r"\b(?:study|studying|work|working|move|moving|visit|visiting|"
        r"immigrate|immigrating|settle|settling)\s+(?:in|to)\s+"
        r"(?:the\s+)?(?:united states|u\s+s|usa|america)\b",
        r"\b(?:united states|u\s+s|usa|america)\s+"
        r"(?:student\s+)?(?:visa|immigration|visa requirements|visa rules)\b",
        r"\b(?:visa|immigration|student visa|work visa)\s+"
        r"(?:in|for|to)\s+(?:the\s+)?(?:united states|u\s+s|usa|america)\b",
        r"\bamerican\s+(?:visa|immigration|student visa|work visa)\b",
    ),

    "GB": (
        r"\b(?:study|studying|work|working|move|moving|visit|visiting|"
        r"immigrate|immigrating|settle|settling)\s+(?:in|to)\s+"
        r"(?:the\s+)?(?:uk|united kingdom|great britain|britain)\b",
        r"\b(?:visa|immigration|student visa|work visa|study visa|"
        r"skilled worker visa|family visa|settlement)\s+(?:in|for|to)\s+"
        r"(?:the\s+)?(?:uk|united kingdom|great britain|britain)\b",
        r"\b(?:uk|united kingdom|great britain|britain)\s+"
        r"(?:student\s+)?(?:visa|immigration|visa requirements|visa rules|"
        r"work visa|study visa|skilled worker visa|family visa|settlement)\b",
    ),

}


def detect_target_jurisdictions(text: str) -> set[str]:
    """Detect clearly targeted destinations, not citizenship alone."""
    normalized = _normalise(text)
    return {
        code
        for code, patterns in _TARGET_PATTERNS.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    }


def try_counsellor_intake_reply(text: str) -> str | None:
    """Handle broad planning requests without inventing legal conclusions."""
    q = _normalise(text)

    # Factual/legal questions always use the evidence-backed RAG path.
    if any(marker in q for marker in _RULE_MARKERS):
        return None

    if not any(marker in q for marker in _PLANNING_MARKERS):
        return None

    words = set(q.split())
    study = bool(words & {
        "study", "studying", "student", "university", "college",
        "bachelor", "vocational", "school",
    }) or "master degree" in q or "masters degree" in q

    work = bool(words & {
        "work", "working", "job", "occupation", "career", "employment",
    })

    visit = bool(words & {
        "visit", "visitor", "tourism", "tourist", "holiday", "travel",
    })

    targets = detect_target_jurisdictions(text)
    unavailable = targets - {"AU"}
    if unavailable:
        names = {"CA": "Canada", "NZ": "New Zealand", "US": "the United States", "GB": "the United Kingdom"}
        countries = ", ".join(names[c] for c in sorted(unavailable))
        return (
            f"I can help organize your goals, but this version does not yet "
            f"have indexed official immigration evidence for {countries}. "
            "I don't want to guess about its rules. What would you like to "
            "clarify first?"
        )

    if study and ("australia" in q or "australian" in q):
        return (
            "Of course—we can work through it step by step. What level of "
            "study are you considering in Australia: a bachelor's degree, "
            "a master's degree, vocational study, or are you still exploring? "
            "Once I know that, we can check the relevant official requirements."
        )

    if study:
        return (
            "Of course—we can work through it step by step. Which country "
            "would you like to focus on first? I can currently verify "
            "Australian immigration requirements; other countries' evidence "
            "has not yet been indexed in this system."
        )

    if work:
        return (
            "Let's narrow it down together. What kind of occupation or work "
            "are you hoping to pursue?"
        )

    if visit:
        return (
            "Of course. What is the main purpose of your visit—tourism, "
            "visiting family, business, or something else?"
        )

    if any(word in words for word in ("move", "migrate", "immigrate", "settle")):
        return (
            "We can work through this step by step. Is your main goal to "
            "move for work, study, join family, or explore longer-term residence?"
        )

    return (
        "We can work through it step by step. Which country and main goal "
        "would you like to explore first—study, work, visit, or longer-term migration?"
    )
