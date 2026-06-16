"""Deterministic synthetic ticket fixture. Run once; output committed.

`uv run python -m tests.fixtures.build_fixture` to regenerate.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

QUEUES = ["Billing", "Technical", "Account", "Returns", "Shipping", "Other"]
PRIORITIES = ["low", "medium", "high", "critical", "info"]
LANGUAGES = ["en", "de"]
TYPES = ["question", "incident", "request", "problem"]
TAGS_POOL = ["login", "password", "refund", "invoice", "shipping", "urgent",
             "api", "ui", "crash", "billing", "payment", "feature"]

SUBJECTS_EN = [
    ("Login broken on iOS", "I can't sign in from my iPhone after the update", "Please try clearing app cache and re-login."),
    ("Refund not processed", "I requested a refund 3 weeks ago and haven't received it", "Your refund has been issued; allow 5-7 business days."),
    ("Wrong invoice amount", "The invoice charges me for 5 seats but I only have 3", "We've corrected the invoice. New copy attached."),
    ("App crashes on startup", "Every time I open the app it closes immediately", "Please update to v2.4.1 which fixes the startup crash."),
    ("Can't reset password", "Reset link in email is expired", "Reset links are valid for 1 hour; here's a fresh one."),
]
SUBJECTS_DE = [
    ("Anmeldung funktioniert nicht", "Ich kann mich nach dem Update nicht mehr anmelden", "Bitte App-Cache leeren und erneut anmelden."),
    ("Rueckerstattung fehlt", "Ich warte seit 3 Wochen auf meine Rueckerstattung", "Die Rueckerstattung wurde veranlasst; 5-7 Werktage."),
    ("Falscher Rechnungsbetrag", "Die Rechnung berechnet 5 Lizenzen statt 3", "Wir haben die Rechnung korrigiert."),
    ("App stuerzt ab", "Die App stuerzt beim Start ab", "Bitte aktualisieren Sie auf Version 2.4.1."),
    ("Passwort-Reset geht nicht", "Der Reset-Link ist abgelaufen", "Reset-Links sind 1 Stunde gueltig; hier ein neuer Link."),
]


def build(n: int = 200, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        lang = rng.choice(LANGUAGES)
        subject, body, answer = rng.choice(SUBJECTS_EN if lang == "en" else SUBJECTS_DE)
        # vary every row so vector search has signal
        body = f"{body} (case #{i})"
        tags = rng.sample(TAGS_POOL, k=rng.randint(1, 4))
        # pad to 6 slots
        padded = tags + [""] * (6 - len(tags))
        # ~10% of rows have empty answer (to test draft_reply filter)
        if rng.random() < 0.10:
            answer = ""
        rows.append({
            "subject": subject,
            "body": body,
            "answer": answer,
            "type": rng.choice(TYPES),
            "queue": rng.choice(QUEUES),
            "priority": rng.choice(PRIORITIES),
            "language": lang,
            "version": f"1.{rng.randint(0, 5)}",
            "tag_1": padded[0],
            "tag_2": padded[1],
            "tag_3": padded[2],
            "tag_4": padded[3],
            "tag_5": padded[4],
            "tag_6": padded[5],
        })
    return rows


def main() -> None:
    rows = build()
    out = Path(__file__).parent / "tickets.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
