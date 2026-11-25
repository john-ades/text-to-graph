"""Version 1 of the shared predicate vocabulary and its directed projections."""

# Canonical predicate, grammatical subject role, grammatical object role.
FRAMES = {
    "write": ("AUTHORED", "AUTHOR", "WORK"),
    "edit": ("EDITED", "EDITOR", "WORK"),
    "lend": ("LOAN", "LENDER", "ITEM"),
    "prêter": ("LOAN", "LENDER", "ITEM"),
    "prestar": ("LOAN", "LENDER", "ITEM"),
    "borrow": ("LOAN", "BORROWER", "ITEM"),
    "sell": ("SALE", "SELLER", "GOODS"),
    "buy": ("SALE", "BUYER", "GOODS"),
    "acheter": ("SALE", "BUYER", "GOODS"),
    "comprar": ("SALE", "BUYER", "GOODS"),
    "give": ("GIFT", "GIVER", "ITEM"),
    "hang": ("HANG", "AGENT", "ITEM"),
    "fold": ("FOLD", "AGENT", "ITEM"),
    "put": ("PLACE", "AGENT", "ITEM"),
    "measure": ("MEASURE", "INSTRUMENT", "PROPERTY"),
    "record": ("READING", "INSTRUMENT", "VALUE"),
    "stop": ("STOP", "SUBJECT", "ACTIVITY"),
    "transmit": ("TRANSMIT", "SUBJECT", "SIGNAL"),
    "cause": ("CAUSE", "CAUSE", "EFFECT"),
    "reopen": ("REOPEN", "SUBJECT", "OBJECT"),
    "pass": ("PASS", "SUBJECT", "TEST"),
    "remain": ("STATE", "SUBJECT", "VALUE"),
    "say": ("REPORT", "SPEAKER", "CONTENT"),
    "report": ("REPORT", "SPEAKER", "CONTENT"),
    "tell": ("REPORT", "SPEAKER", "LISTENER"),
    "name": ("IDENTIFY", "SOURCE", "SUBJECT"),
    "translate": ("TRANSLATE", "TRANSLATOR", "ORIGINAL"),
    "win": ("AWARD", "WINNER", "PRIZE"),
    "find": ("FIND", "FINDER", "ITEM"),
    "belong": ("OWNERSHIP", "ITEM", "OWNER"),
    "found": ("FOUND", "FOUNDER", "ORGANIZATION"),
    "design": ("DESIGN", "DESIGNER", "ITEM"),
    "test": ("TEST", "TESTER", "ITEM"),
    "offer": ("OFFER", "OFFEROR", "CONTENT"),
    "follow": ("BEFORE", "LATER", "EARLIER"),
    "precede": ("BEFORE", "EARLIER", "LATER"),
}
PROJECTIONS = {
    "SALE": [("BUYER", "BOUGHT", "GOODS"), ("SELLER", "SOLD", "GOODS")],
    "LOAN": [("BORROWER", "BORROWED", "ITEM"), ("LENDER", "LENT", "ITEM")],
    "GIFT": [("RECIPIENT", "RECEIVED", "ITEM"), ("GIVER", "GAVE", "ITEM")],
    "AUTHORED": [("AUTHOR", "AUTHORED", "WORK")],
    "EDITED": [("EDITOR", "EDITED", "WORK")],
    "WROTE_TO": [("SENDER", "WROTE_TO", "ADDRESSEE")],
    "AWARD": [("WINNER", "WON", "PRIZE")],
    "OWNERSHIP": [("OWNER", "OWNS", "ITEM")],
    "BEFORE": [("EARLIER", "BEFORE", "LATER")],
    "CAUSE": [("CAUSE", "CAUSES", "EFFECT")],
    "TRANSLATE": [("TRANSLATION", "TRANSLATION_OF", "ORIGINAL")],
}
WEEKDAYS = {}
for labels in zip(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
    ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
):
    WEEKDAYS.update((label.lower(), labels[0].upper()) for label in labels)
CONCEPTS = dict.fromkeys(("telescope", "télescope", "telescopio"), "TELESCOPE")
CONCEPTS.update(dict.fromkeys(("book", "livre", "libro"), "BOOK"))

# Validated noun forms in the supported native loan constructions.
LOAN_ITEMS = {
    "ar": {"كِتَاب": "BOOK", "كتاب": "BOOK", "كتابًا": "BOOK"},
    "da": {"bog": "BOOK"},
    "de": {"buch": "BOOK"},
    "id": {"buku": "BOOK"},
    "it": {"libro": "BOOK"},
    "ms": {"buku": "BOOK"},
    "pt": {"livro": "BOOK"},
    "ru": {"книга": "BOOK", "книгу": "BOOK"},
}

# Definitions are supplied to the multilingual model in the shared label language.
SENSES = {
    "AUTHORED": "Creating a written work as its author",
    "WROTE_TO": "Addressing written communication to a recipient",
    "EDITED": "Revising or editing a work",
    "LOAN": "Lending or borrowing an item for temporary use",
    "SALE": "Buying or selling goods in a commercial transaction",
    "GIFT": "Giving an item to a recipient",
    "REPORT": "A speaker says or reports a proposition",
    "TRANSLATE": "Rendering an original work in another language",
    "READING": "Recording an instrument reading with a measured value",
    "BEFORE": "One event occurring earlier than another",
    "CAUSE": "One event explicitly causing another",
    "OTHER": "A different meaning",
}
