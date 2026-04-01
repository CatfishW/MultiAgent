from agent_swarm_port.ids import format_agent_id, generate_request_id, parse_agent_id, parse_request_id


def test_agent_id_roundtrip():
    agent_id = format_agent_id("worker-1", "alpha-team")
    parsed = parse_agent_id(agent_id)
    assert agent_id == "worker-1@alpha-team"
    assert parsed is not None
    assert parsed.agent_name == "worker-1"
    assert parsed.team_name == "alpha-team"


def test_request_id_roundtrip():
    request_id = generate_request_id("shutdown", "worker-1@alpha-team")
    parsed = parse_request_id(request_id)
    assert parsed is not None
    assert parsed.request_type == "shutdown"
    assert parsed.agent_id == "worker-1@alpha-team"
    assert parsed.timestamp > 0
