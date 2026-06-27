"""Hand-curated scenario tables for the semantic-quality tests."""

from __future__ import annotations


TOPICAL_SCENARIOS: list[tuple[str, list[str], int]] = [
    ("how do I reset my password", ["password", "reset", "passwort", "zurücksetzen"], 3),
    ("I forgot my password", ["password", "passwort", "forgot", "vergess"], 3),
    ("can't log into my account", ["login", "log in", "anmelde", "konto"], 3),
    ("login button does nothing", ["login", "button", "anmelde"], 3),
    ("two-factor authentication not working", ["two-factor", "2fa", "authentication", "code"], 3),
    ("I want a refund", ["refund", "rückerstattung", "money back", "erstatt"], 3),
    ("how to cancel my subscription", ["cancel", "subscription", "kündig", "abo"], 3),
    ("my invoice is wrong", ["invoice", "rechnung", "billing", "charge"], 3),
    ("charged twice for the same order", ["charge", "double", "twice", "rechnung", "order"], 3),
    ("where is my package", ["package", "shipping", "delivery", "lieferung", "paket", "versand"], 3),
    ("delivery is late", ["delivery", "shipping", "late", "lieferung", "verspät"], 3),
    ("change my email address", ["email", "address", "e-mail", "adresse", "ändern"], 3),
    ("update billing address", ["billing", "address", "rechnung", "adresse"], 3),
    ("app keeps crashing on startup", ["crash", "startup", "app", "absturz", "start"], 3),
    ("error message when uploading", ["error", "upload", "fehler", "hochlad"], 3),
    ("can't connect to the server", ["connect", "server", "verbind"], 3),
    ("how do I export my data", ["export", "data", "download", "daten"], 3),
    ("delete my account", ["delete", "account", "löschen", "konto"], 3),
    ("warranty claim for my product", ["warranty", "claim", "garantie"], 3),
    ("payment method declined", ["payment", "declined", "zahlung", "abgelehnt"], 3),
]


CROSS_LINGUAL_SCENARIOS: list[tuple[str, str, list[str]]] = [
    # (query, target_language, expected_keywords in target hit)
    ("Passwort zurücksetzen", "en", ["password", "reset", "forgot"]),
    ("Rechnung falsch", "en", ["invoice", "billing", "wrong", "incorrect"]),
    ("Anmeldung funktioniert nicht", "en", ["login", "log in", "sign in", "can't"]),
    ("password reset", "de", ["passwort", "zurücksetzen", "vergessen"]),
    ("invoice problem", "de", ["rechnung", "problem", "falsch"]),
    ("login broken", "de", ["anmelde", "anmeldung", "geht nicht", "funktioniert"]),
]


HARD_NEGATIVE_SCENARIOS: list[tuple[str, list[str], list[str]]] = [
    # (query, topic_A_keywords (should appear), topic_B_keywords (should NOT dominate top-3))
    ("password reset", ["password", "reset", "passwort"], ["shipping", "package", "delivery", "lieferung"]),
    ("invoice billing problem", ["invoice", "billing", "rechnung"], ["password", "login", "passwort", "anmelde"]),
    ("package not delivered", ["package", "delivery", "shipping", "paket", "lieferung"], ["password", "billing", "invoice", "rechnung"]),
    ("login error", ["login", "log in", "anmelde"], ["refund", "shipping", "rückerstattung", "lieferung"]),
    ("refund request", ["refund", "rückerstattung", "money back"], ["password", "login", "passwort", "anmelde"]),
]
