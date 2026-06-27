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


# 30 EN + 30 DE free-text queries for the no-filter language-purity baseline.
# Mixed lengths (1-5 words) so the mean is not dominated by one query shape.
PURITY_QUERIES_EN: list[str] = [
    "login problem reset password",
    "forgot my password",
    "cannot log in",
    "two factor authentication code",
    "account locked out",
    "change email address",
    "delete my account",
    "export my data",
    "refund request",
    "cancel subscription",
    "invoice incorrect",
    "billing question",
    "charged twice",
    "payment declined",
    "credit card",
    "package not delivered",
    "shipping delay",
    "tracking number missing",
    "return item",
    "warranty claim",
    "app crashing on startup",
    "error 500",
    "cannot connect to server",
    "upload failing",
    "slow performance",
    "mobile app bug",
    "feature request",
    "documentation unclear",
    "how to integrate API",
    "rate limit exceeded",
]

PURITY_QUERIES_DE: list[str] = [
    "Passwort zurücksetzen",
    "Passwort vergessen",
    "Anmeldung funktioniert nicht",
    "Zwei-Faktor-Authentifizierung Code",
    "Konto gesperrt",
    "E-Mail Adresse ändern",
    "Konto löschen",
    "Daten exportieren",
    "Rückerstattung beantragen",
    "Abonnement kündigen",
    "Rechnung falsch",
    "Frage zur Abrechnung",
    "doppelt belastet",
    "Zahlung abgelehnt",
    "Kreditkarte",
    "Paket nicht angekommen",
    "Lieferung verspätet",
    "Sendungsnummer fehlt",
    "Artikel zurücksenden",
    "Garantieanspruch",
    "App stürzt beim Start ab",
    "Fehler 500",
    "keine Verbindung zum Server",
    "Hochladen schlägt fehl",
    "langsame Geschwindigkeit",
    "mobile App Fehler",
    "Funktionswunsch",
    "Dokumentation unklar",
    "wie integriere ich die API",
    "Ratenlimit überschritten",
]


HARD_NEGATIVE_SCENARIOS: list[tuple[str, list[str], list[str]]] = [
    # (query, topic_A_keywords (should appear), topic_B_keywords (should NOT dominate top-3))
    ("password reset", ["password", "reset", "passwort"], ["shipping", "package", "delivery", "lieferung"]),
    ("invoice billing problem", ["invoice", "billing", "rechnung"], ["password", "login", "passwort", "anmelde"]),
    ("package not delivered", ["package", "delivery", "shipping", "paket", "lieferung"], ["password", "billing", "invoice", "rechnung"]),
    ("login error", ["login", "log in", "anmelde"], ["refund", "shipping", "rückerstattung", "lieferung"]),
    ("refund request", ["refund", "rückerstattung", "money back"], ["password", "login", "passwort", "anmelde"]),
]
