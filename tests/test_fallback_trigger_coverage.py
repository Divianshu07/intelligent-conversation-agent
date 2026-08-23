from __future__ import annotations

import pytest

from app.composer import AIComposer
from app.context_resolver import ContextResolver
from app.context_store import ContextStore
from app.trigger_router import TriggerRouter


def build_brief(
    kind: str,
    trigger_payload: dict,
    *,
    scope: str = "merchant",
    customer_name: str | None = None,
    consent_scopes: list[str] | None = None,
):
    """Seed a minimal category/merchant/(customer)/trigger set and build the
    MessageBrief the live TickService path would produce for one trigger."""
    store = ContextStore(":memory:")
    store.put("category", "cat1", 1, {"slug": "general", "voice": {"tone": "friendly"}})
    store.put(
        "merchant",
        "m1",
        1,
        {
            "category_slug": "cat1",
            "identity": {"name": "Test Merchant", "city": "Pune", "locality": "Kothrud", "languages": ["en"]},
            "offers": [],
            "performance": {},
            "signals": [],
        },
    )

    customer_id = None
    if scope == "customer":
        customer_id = "c1"
        store.put(
            "customer",
            "c1",
            1,
            {
                "identity": {"name": customer_name or "Test Customer", "language_pref": "en"},
                "state": {},
                "relationship": {},
                "preferences": {"channel": "whatsapp"},
                "consent": {"scope": consent_scopes or []},
            },
        )

    trigger_top_level = {
        "kind": kind,
        "urgency": 1,
        "suppression_key": f"{kind}:m1",
        "merchant_id": "m1",
        "scope": scope,
        "payload": trigger_payload,
    }
    if customer_id:
        trigger_top_level["customer_id"] = customer_id

    store.put("trigger", "trig1", 1, trigger_top_level)

    resolver = ContextResolver(store)
    router = TriggerRouter()
    brief = router.build_brief(resolver.resolve("trig1"), request_id="trig1")
    store.close()
    return brief


def compose(brief):
    # provider=None -> the exact live-path fallback used when Gemini is
    # unavailable / no GEMINI_API_KEY is configured.
    return AIComposer(provider=None).compose(brief)


def test_perf_spike_produces_grounded_send(tmp_path=None):
    brief = build_brief("perf_spike", {"metric": "calls", "delta_pct": 0.15, "window": "7d"})
    output = compose(brief)
    assert output.should_send is True
    assert output.action == "send"
    assert "calls" in output.message
    assert "15%" in output.message
    assert "7d" in output.message


def test_festival_upcoming_produces_grounded_send():
    brief = build_brief("festival_upcoming", {"festival": "Diwali", "date": "2026-10-31", "days_until": 188})
    output = compose(brief)
    assert output.should_send is True
    assert "Diwali" in output.message
    assert "2026-10-31" in output.message


def test_active_planning_intent_produces_grounded_send():
    brief = build_brief("active_planning_intent", {"intent_topic": "corporate_bulk_thali_package"})
    output = compose(brief)
    assert output.should_send is True
    assert "corporate bulk thali package" in output.message


def test_milestone_reached_imminent_never_claims_achieved():
    brief = build_brief("milestone_reached", {"metric": "review_count", "milestone_value": 150, "is_imminent": True})
    output = compose(brief)
    assert output.should_send is True
    lowered = output.message.lower()
    assert "150" in output.message
    assert "reached" not in lowered
    assert "congratulations" not in lowered


def test_milestone_reached_actual_can_celebrate():
    brief = build_brief("milestone_reached", {"metric": "review_count", "milestone_value": 150, "is_imminent": False})
    output = compose(brief)
    assert output.should_send is True
    assert "150" in output.message


def test_customer_lapsed_hard_requires_promotional_consent():
    brief_with_consent = build_brief(
        "customer_lapsed_hard",
        {"days_since_last_visit": 57},
        scope="customer",
        customer_name="Rashmi",
        consent_scopes=["promotional_offers"],
    )
    output = compose(brief_with_consent)
    assert output.should_send is True
    assert "Rashmi" in output.message
    assert "57" in output.message

    brief_without_consent = build_brief(
        "customer_lapsed_hard",
        {"days_since_last_visit": 57},
        scope="customer",
        customer_name="Rashmi",
        consent_scopes=[],
    )
    output_no_consent = compose(brief_without_consent)
    assert output_no_consent.should_send is False
    assert output_no_consent.action == "wait"


def test_gbp_unverified_produces_grounded_send():
    brief = build_brief("gbp_unverified", {"verified": False, "verification_path": "postcard_or_phone_call"})
    output = compose(brief)
    assert output.should_send is True
    assert "postcard or phone call" in output.message


def test_category_seasonal_produces_grounded_send():
    brief = build_brief("category_seasonal", {"season": "summer_2026", "trends": ["ORS_demand_+40"]})
    output = compose(brief)
    assert output.should_send is True
    assert "summer 2026" in output.message


def test_cde_opportunity_produces_grounded_send():
    brief = build_brief("cde_opportunity", {"digest_item_id": "d1", "credits": 2, "fee": "free_for_members"})
    output = compose(brief)
    assert output.should_send is True
    assert "2" in output.message
    assert "free for members" in output.message


def test_competitor_opened_produces_grounded_send():
    brief = build_brief(
        "competitor_opened",
        {"competitor_name": "Smile Studio", "distance_km": 1.3, "opened_date": "2026-04-08"},
    )
    output = compose(brief)
    assert output.should_send is True
    assert "Smile Studio" in output.message


def test_dormant_with_vera_produces_grounded_send():
    brief = build_brief(
        "dormant_with_vera",
        {"days_since_last_merchant_message": 38, "last_topic": "subscription_expiry"},
    )
    output = compose(brief)
    assert output.should_send is True
    assert "38" in output.message
    assert "subscription expiry" in output.message


def test_curious_ask_due_produces_grounded_send():
    brief = build_brief("curious_ask_due", {"ask_template": "what_service_in_demand_this_week"})
    output = compose(brief)
    assert output.should_send is True
    assert "what's a service that's been in demand this week" in output.message


def test_renewal_due_produces_grounded_send():
    brief = build_brief("renewal_due", {"plan": "Pro", "days_remaining": 12, "renewal_amount": 4999})
    output = compose(brief)
    assert output.should_send is True
    assert "Pro" in output.message
    assert "12" in output.message
    assert "4999" in output.message


def test_review_theme_emerged_produces_grounded_send():
    brief = build_brief(
        "review_theme_emerged",
        {"theme": "delivery_late", "occurrences_30d": 4, "trend": "rising"},
    )
    output = compose(brief)
    assert output.should_send is True
    assert "delivery late" in output.message
    assert "4" in output.message


def test_seasonal_perf_dip_reassures_when_expected():
    brief = build_brief(
        "seasonal_perf_dip",
        {
            "metric": "views",
            "delta_pct": -0.3,
            "window": "7d",
            "is_expected_seasonal": True,
            "season_note": "post_resolution_window_apr_jun",
        },
    )
    output = compose(brief)
    assert output.should_send is True
    assert "30%" in output.message
    assert "seasonal" in output.message.lower()


def test_seasonal_perf_dip_waits_when_not_flagged_expected():
    brief = build_brief(
        "seasonal_perf_dip",
        {"metric": "views", "delta_pct": -0.3, "window": "7d", "is_expected_seasonal": False},
    )
    output = compose(brief)
    assert output.should_send is False
    assert output.action == "wait"


def test_trial_followup_requires_appointment_consent():
    payload = {
        "trial_date": "2026-04-22",
        "next_session_options": [{"iso": "2026-05-03T08:00:00+05:30", "label": "Sat 3 May, 8am"}],
    }
    brief_with_consent = build_brief(
        "trial_followup",
        payload,
        scope="customer",
        customer_name="Karthik",
        consent_scopes=["appointment_reminders"],
    )
    output = compose(brief_with_consent)
    assert output.should_send is True
    assert "Sat 3 May, 8am" in output.message

    brief_without_consent = build_brief(
        "trial_followup",
        payload,
        scope="customer",
        customer_name="Karthik",
        consent_scopes=[],
    )
    output_no_consent = compose(brief_without_consent)
    assert output_no_consent.should_send is False
    assert output_no_consent.action == "wait"


def test_winback_eligible_produces_grounded_send():
    brief = build_brief(
        "winback_eligible",
        {"days_since_expiry": 38, "lapsed_customers_added_since_expiry": 24},
    )
    output = compose(brief)
    assert output.should_send is True
    assert "38" in output.message
    assert "24" in output.message


def test_wedding_package_followup_requires_appointment_consent():
    payload = {
        "wedding_date": "2026-11-08",
        "days_to_wedding": 196,
        "next_step_window_open": "skin_prep_program_30day",
    }
    brief_with_consent = build_brief(
        "wedding_package_followup",
        payload,
        scope="customer",
        customer_name="Kavya",
        consent_scopes=["appointment_reminders"],
    )
    output = compose(brief_with_consent)
    assert output.should_send is True
    assert "196" in output.message
    assert "skin prep program 30day" in output.message

    brief_without_consent = build_brief(
        "wedding_package_followup",
        payload,
        scope="customer",
        customer_name="Kavya",
        consent_scopes=[],
    )
    output_no_consent = compose(brief_without_consent)
    assert output_no_consent.should_send is False
    assert output_no_consent.action == "wait"


@pytest.mark.parametrize(
    "kind,payload,scope",
    [
        ("perf_spike", {"metric": "calls", "window": "7d"}, "merchant"),  # missing delta_pct
        ("festival_upcoming", {"festival": "Diwali"}, "merchant"),  # missing date
        ("renewal_due", {"plan": "Pro"}, "merchant"),  # missing days_remaining/renewal_amount
        ("gbp_unverified", {"verified": False}, "merchant"),  # missing verification_path
    ],
)
def test_missing_required_fields_fall_back_to_wait(kind, payload, scope):
    brief = build_brief(kind, payload, scope=scope)
    output = compose(brief)
    assert output.should_send is False
    assert output.action == "wait"


def test_chronic_refill_due_requires_refill_consent():
    payload = {
        "molecule_list": ["metformin", "atorvastatin"],
        "last_refill": "2026-03-26",
        "stock_runs_out_iso": "2026-04-28T00:00:00+05:30",
        "delivery_address_saved": True,
    }
    brief_with_consent = build_brief(
        "chronic_refill_due",
        payload,
        scope="customer",
        customer_name="Mr. Sharma",
        consent_scopes=["refill_reminders"],
    )
    output = compose(brief_with_consent)
    assert output.should_send is True
    assert "Mr. Sharma" in output.message
    assert "metformin" in output.message
    assert "atorvastatin" in output.message
    assert "2026-04-28" in output.message
    # Never mentions dose, price, or availability -- not supplied.
    assert "dose" not in output.message.lower()
    assert "price" not in output.message.lower()

    brief_without_consent = build_brief(
        "chronic_refill_due",
        payload,
        scope="customer",
        customer_name="Mr. Sharma",
        consent_scopes=[],
    )
    output_no_consent = compose(brief_without_consent)
    assert output_no_consent.should_send is False
    assert output_no_consent.action == "wait"


def test_chronic_refill_due_missing_molecule_list_falls_back_to_wait():
    brief = build_brief(
        "chronic_refill_due",
        {"stock_runs_out_iso": "2026-04-28T00:00:00+05:30"},
        scope="customer",
        customer_name="Mr. Sharma",
        consent_scopes=["refill_reminders"],
    )
    output = compose(brief)
    assert output.should_send is False
    assert output.action == "wait"


def test_ipl_match_today_produces_grounded_send():
    brief = build_brief(
        "ipl_match_today",
        {
            "match": "DC vs MI",
            "venue": "Arun Jaitley Stadium",
            "city": "Delhi",
            "match_time_iso": "2026-04-26T19:30:00+05:30",
            "is_weeknight": False,
        },
    )
    output = compose(brief)
    assert output.should_send is True
    assert "DC vs MI" in output.message
    assert "Arun Jaitley Stadium" in output.message
    # Never claims a demand/covers impact that wasn't supplied.
    assert "more customers" not in output.message.lower()


def test_ipl_match_today_missing_venue_falls_back_to_wait():
    brief = build_brief("ipl_match_today", {"match": "DC vs MI"})
    output = compose(brief)
    assert output.should_send is False
    assert output.action == "wait"


def test_appointment_tomorrow_requires_appointment_reminder_consent():
    payload = {"appointment_time": "10:00 AM", "service": "cleaning", "staff_member": "Dr. Meera"}
    brief_with_consent = build_brief(
        "appointment_tomorrow",
        payload,
        scope="customer",
        customer_name="Priya",
        consent_scopes=["appointment_reminders"],
    )
    output = compose(brief_with_consent)
    assert output.should_send is True
    assert "Priya" in output.message
    assert "cleaning" in output.message
    assert "10:00 AM" in output.message

    brief_without_consent = build_brief(
        "appointment_tomorrow",
        payload,
        scope="customer",
        customer_name="Priya",
        consent_scopes=[],
    )
    output_no_consent = compose(brief_without_consent)
    assert output_no_consent.should_send is False
    assert output_no_consent.action == "wait"


def test_appointment_tomorrow_missing_service_falls_back_to_wait():
    brief = build_brief(
        "appointment_tomorrow",
        {"appointment_time": "10:00 AM"},
        scope="customer",
        customer_name="Priya",
        consent_scopes=["appointment_reminders"],
    )
    output = compose(brief)
    assert output.should_send is False
    assert output.action == "wait"


def test_customer_lapsed_hard_recognizes_winback_offers_as_promotional_consent():
    brief = build_brief(
        "customer_lapsed_hard",
        {"days_since_last_visit": 57},
        scope="customer",
        customer_name="Rashmi",
        consent_scopes=["winback_offers"],
    )
    output = compose(brief)
    assert output.should_send is True
    assert brief.consent_state.promotional_consent is True
    assert "Rashmi" in output.message
    assert "57" in output.message
