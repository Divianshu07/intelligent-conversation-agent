from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


def make_client(tmp_path) -> TestClient:
    settings = Settings(database_path=str(tmp_path / "contexts.db"))
    return TestClient(create_app(settings))


def reply_payload(conversation_id: str, message: str, turn_number: int, **overrides) -> dict:
    payload = {
        "conversation_id": conversation_id,
        "merchant_id": "m_001",
        "customer_id": None,
        "from_role": "merchant",
        "message": message,
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": turn_number,
    }
    payload.update(overrides)
    return payload


def test_normal_reply_produces_send_or_wait(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/reply",
            json=reply_payload("conv_normal", "Sounds interesting, tell me more?", 1),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in ("send", "wait")
    assert body["rationale"]


def test_repeated_auto_reply_eventually_stops(tmp_path):
    canned = "Thank you for contacting us! Our team will respond shortly."
    with make_client(tmp_path) as client:
        responses = [
            client.post("/v1/reply", json=reply_payload("conv_auto", canned, turn))
            for turn in (1, 2, 3)
        ]
    for response in responses:
        assert response.status_code == 200
    actions = [response.json()["action"] for response in responses]
    # It must not keep generating fresh outbound messages forever, and per
    # challenge-brief.md Pattern B ("gold standard" auto-reply handling) it
    # must exit gracefully with a short closing message, not go silent.
    assert actions[-1] == "end"
    last_body = responses[-1].json()
    assert last_body["body"]
    assert "automated" in last_body["body"].lower()


def test_commitment_message_transitions_to_action(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/reply",
            json=reply_payload("conv_commit", "Ok lets do it. Whats next?", 1),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "send"
    assert body["body"]
    assert "next step" in body["body"].lower() or "move ahead" in body["body"].lower()


def test_generic_replies_never_repeat_body_verbatim(tmp_path):
    # Grading penalizes any body sent verbatim twice in the same
    # conversation. Different genuine merchant replies across turns must
    # not fall through to the same static acknowledgement text.
    messages = [
        "Sounds interesting, tell me more?",
        "What does that involve exactly?",
        "Okay, and how long does it take?",
        "Got it, anything else I should know?",
    ]
    with make_client(tmp_path) as client:
        responses = [
            client.post("/v1/reply", json=reply_payload("conv_varied", msg, turn))
            for turn, msg in enumerate(messages, start=1)
        ]
    bodies = []
    for response in responses:
        assert response.status_code == 200
        payload = response.json()
        assert payload["action"] == "send"
        assert payload["body"]
        bodies.append(payload["body"])
    assert len(bodies) == len(set(bodies)), f"expected all-unique bodies, got {bodies}"


def test_repeated_commitment_message_does_not_repeat_body(tmp_path):
    with make_client(tmp_path) as client:
        first = client.post(
            "/v1/reply",
            json=reply_payload("conv_commit_twice", "Ok lets do it. Whats next?", 1),
        )
        second = client.post(
            "/v1/reply",
            json=reply_payload("conv_commit_twice", "Great, lets go then!", 2),
        )
    assert first.status_code == 200 and second.status_code == 200
    first_body = first.json()["body"]
    second_body = second.json()["body"]
    assert first_body and second_body
    assert first_body != second_body


_INTERNAL_OPS_PHRASES = (
    "i'll follow up",
    "next step",
    "circle back",
    "sorted",
    "i'll get",
    "i'll take a look",
    "i'll have the",
    "i'll sort",
)


def test_customer_generic_reply_avoids_internal_ops_language(tmp_path):
    messages = [
        "Sounds good, thanks!",
        "When is my appointment?",
        "Can you confirm the time?",
    ]
    with make_client(tmp_path) as client:
        responses = [
            client.post(
                "/v1/reply",
                json=reply_payload(
                    "conv_customer_generic",
                    msg,
                    turn,
                    from_role="customer",
                    merchant_id="m_001",
                    customer_id="c_001",
                ),
            )
            for turn, msg in enumerate(messages, start=1)
        ]
    bodies = []
    for response in responses:
        assert response.status_code == 200
        payload = response.json()
        assert payload["action"] == "send"
        assert payload["body"]
        bodies.append(payload["body"])

    # No internal-ops wording should leak into customer-facing copy.
    for body in bodies:
        lowered = body.lower()
        for phrase in _INTERNAL_OPS_PHRASES:
            assert phrase not in lowered, f"internal-ops phrase {phrase!r} leaked into customer body: {body!r}"

    # Repeat-prevention still applies to the customer-facing pool.
    assert len(bodies) == len(set(bodies)), f"expected all-unique bodies, got {bodies}"


def test_customer_commitment_reply_avoids_internal_ops_language(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/reply",
            json=reply_payload(
                "conv_customer_commit",
                "Sounds good, sign me up!",
                1,
                from_role="customer",
                merchant_id="m_001",
                customer_id="c_001",
            ),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "send"
    lowered = body["body"].lower()
    for phrase in _INTERNAL_OPS_PHRASES:
        assert phrase not in lowered, f"internal-ops phrase {phrase!r} leaked into customer body: {body['body']!r}"


def test_merchant_generic_reply_still_uses_merchant_voice(tmp_path):
    # Same content-neutral message, but from_role="merchant" — must keep
    # using the original merchant/ops-facing pool (unchanged behavior).
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/reply",
            json=reply_payload("conv_merchant_generic", "Sounds interesting, tell me more?", 1),
        )
    assert response.status_code == 200
    body = response.json()["body"]
    assert body in (
        "Thanks for letting me know — noted, and I'll follow up with the next step.",
        "Got it, appreciate the reply — I'll take a look and come back to you shortly.",
        "Noted, thanks for sharing that — I'll follow up again soon with more.",
        "Thanks for the update — keeping this in mind and will circle back shortly.",
    )


def test_hostile_opt_out_ends_conversation_politely(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/reply",
            json=reply_payload("conv_hostile", "Stop messaging me. This is useless spam.", 1),
        )
        # A follow-up turn on the same conversation must stay ended, not
        # re-engage or repeat itself.
        follow_up = client.post(
            "/v1/reply",
            json=reply_payload("conv_hostile", "Anything else?", 2),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "end"
    assert "sorry" in body["body"].lower()

    assert follow_up.status_code == 200
    assert follow_up.json()["action"] == "end"
