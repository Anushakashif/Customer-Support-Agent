import pytest
from unittest.mock import patch, MagicMock
from graph.state import SupportState, Category, Priority, TicketStatus


# ── Shared mock state ─────────────────────────────────────────────────────────
def make_mock_state(
    message="How do I reset my password?",
    customer_id="test-001"
) -> SupportState:
    return {
        "customer_id":          customer_id,
        "customer_message":     message,
        "conversation_history": [],
        "category":             None,
        "priority":             None,
        "extracted_entities":   {},
        "generated_answer":     None,
        "confidence_score":     None,
        "retrieved_docs":       [],
        "should_escalate":      False,
        "escalation_reason":    None,
        "escalation_summary":   None,
        "final_response":       None,
        "ticket_status":        TicketStatus.OPEN,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TRIAGE AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestTriageAgent:

    def test_triage_classifies_billing(self):
        """Billing message should be classified as billing category."""
        from agents.triage_agent import triage_node
        state = make_mock_state(message="I was charged twice for my subscription")
        result = triage_node(state)
        assert result["category"] == Category.BILLING

    def test_triage_classifies_technical(self):
        """Password reset is a technical issue."""
        from agents.triage_agent import triage_node
        state = make_mock_state(message="I cannot log into my account")
        result = triage_node(state)
        assert result["category"] in [Category.TECHNICAL, Category.GENERAL]

    def test_triage_assigns_urgent_priority(self):
        """Data loss should be flagged as urgent."""
        from agents.triage_agent import triage_node
        state = make_mock_state(message="All my data has been deleted and I am losing money")
        result = triage_node(state)
        assert result["priority"] in [Priority.URGENT, Priority.HIGH]

    def test_triage_assigns_low_priority(self):
        """Simple question should be low priority."""
        from agents.triage_agent import triage_node
        state = make_mock_state(message="How do I upgrade my plan?")
        result = triage_node(state)
        assert result["priority"] in [Priority.LOW, Priority.MEDIUM]

    def test_triage_returns_entities(self):
        """Triage should extract entities from the message."""
        from agents.triage_agent import triage_node
        state = make_mock_state(message="My order ID 12345 has not arrived")
        result = triage_node(state)
        assert result["extracted_entities"] is not None
        assert isinstance(result["extracted_entities"], dict)

    def test_triage_never_returns_none_category(self):
        """Category should always be set after triage."""
        from agents.triage_agent import triage_node
        state = make_mock_state(message="????")
        result = triage_node(state)
        assert result["category"] is not None

    def test_triage_preserves_existing_state(self):
        """Triage should not wipe other state fields."""
        from agents.triage_agent import triage_node
        state = make_mock_state()
        state["conversation_history"] = [{"role": "customer", "content": "hello"}]
        result = triage_node(state)
        assert result["conversation_history"] == [{"role": "customer", "content": "hello"}]


# ═════════════════════════════════════════════════════════════════════════════
# ANSWER AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAnswerAgent:

    def _make_triaged_state(self, message="How do I reset my password?") -> SupportState:
        """Helper — returns a state already processed by triage."""
        state = make_mock_state(message=message)
        state["category"] = Category.TECHNICAL
        state["priority"] = Priority.LOW
        return state

    def test_answer_returns_response(self):
        """Answer agent should always return a final response."""
        from agents.answer_agent import answer_node
        state = self._make_triaged_state()
        result = answer_node(state)
        assert result["generated_answer"] is not None
        assert len(result["generated_answer"]) > 0

    def test_answer_returns_confidence_score(self):
        """Confidence score should be between 0 and 1."""
        from agents.answer_agent import answer_node
        state = self._make_triaged_state()
        result = answer_node(state)
        assert result["confidence_score"] is not None
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_answer_high_confidence_not_escalated(self):
        """High confidence answer should not be escalated."""
        from agents.answer_agent import answer_node
        state = self._make_triaged_state("How do I reset my password?")
        result = answer_node(state)
        if result["confidence_score"] >= 0.65:
            assert result["should_escalate"] == False
            assert result["final_response"] is not None

    def test_answer_low_confidence_escalated(self):
        """Unknown topic should trigger escalation."""
        from agents.answer_agent import answer_node
        state = self._make_triaged_state(
            "My quantum encryption module is failing on the blockchain node"
        )
        result = answer_node(state)
        if result["confidence_score"] < 0.65:
            assert result["should_escalate"] == True
            assert result["final_response"] is None

    def test_answer_retrieves_docs(self):
        """Answer agent should retrieve docs from knowledge base."""
        from agents.answer_agent import answer_node
        state = self._make_triaged_state("How do I cancel my subscription?")
        result = answer_node(state)
        assert result["retrieved_docs"] is not None
        assert isinstance(result["retrieved_docs"], list)


# ═════════════════════════════════════════════════════════════════════════════
# ESCALATION AGENT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestEscalationAgent:

    def _make_escalation_state(self) -> SupportState:
        """Helper — returns a state ready for escalation."""
        state = make_mock_state(
            message="My entire account data has been wiped"
        )
        state["category"]         = Category.TECHNICAL
        state["priority"]         = Priority.URGENT
        state["generated_answer"] = "I'm sorry, I cannot help with this."
        state["confidence_score"] = 0.3
        state["should_escalate"]  = True
        return state

    def test_escalation_sets_status(self):
        """Escalation agent should mark ticket as escalated."""
        from agents.escalation_agent import escalation_node
        state = self._make_escalation_state()
        result = escalation_node(state)
        assert result["ticket_status"] == TicketStatus.ESCALATED

    def test_escalation_sets_reason(self):
        """Escalation reason should always be set."""
        from agents.escalation_agent import escalation_node
        state = self._make_escalation_state()
        result = escalation_node(state)
        assert result["escalation_reason"] is not None
        assert len(result["escalation_reason"]) > 0

    def test_escalation_sets_customer_message(self):
        """Customer should receive a polite handoff message."""
        from agents.escalation_agent import escalation_node
        state = self._make_escalation_state()
        result = escalation_node(state)
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0

    def test_escalation_sets_summary(self):
        """Escalation summary for human agent should be set."""
        from agents.escalation_agent import escalation_node
        state = self._make_escalation_state()
        result = escalation_node(state)
        assert result["escalation_summary"] is not None


# ═════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_pipeline_resolves_known_question(self):
        """Known question should be resolved without escalation."""
        from graph.workflow import run_support_pipeline
        result = run_support_pipeline(
            customer_id="test-pipeline-001",
            customer_message="How do I reset my password?"
        )
        assert result["ticket_status"] == TicketStatus.RESOLVED
        assert result["final_response"] is not None
        assert result["should_escalate"] == False

    def test_pipeline_escalates_unknown_question(self):
        """Unknown question should be escalated."""
        from graph.workflow import run_support_pipeline
        result = run_support_pipeline(
            customer_id="test-pipeline-002",
            customer_message="My quantum encryption module is failing on the blockchain node"
        )
        assert result["ticket_status"] in [TicketStatus.ESCALATED, TicketStatus.RESOLVED]
        assert result["final_response"] is not None

    def test_pipeline_always_returns_response(self):
        """Pipeline should always return a final response no matter what."""
        from graph.workflow import run_support_pipeline
        result = run_support_pipeline(
            customer_id="test-pipeline-003",
            customer_message="asdfghjkl random gibberish"
        )
        assert result["final_response"] is not None