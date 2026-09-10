from multi_agent_rag.health import check_readiness


def test_readiness_reports_ready_when_all_required_services_respond() -> None:
    report = check_readiness(
        ["mongodb", "qdrant"],
        probes={
            "mongodb": lambda _timeout: "MongoDB ready.",
            "qdrant": lambda _timeout: "Qdrant ready.",
        },
    )

    assert report.ready is True
    assert report.status == "ready"
    assert report.required_services == ["mongodb", "qdrant"]
    assert [item.status for item in report.dependencies] == ["ready", "ready"]


def test_readiness_reports_unavailable_without_leaking_exception_details() -> None:
    def fail(_timeout: float) -> str:
        raise RuntimeError("mongodb://user:secret@private-host")

    report = check_readiness(["mongodb"], probes={"mongodb": fail})

    assert report.ready is False
    assert report.status == "not_ready"
    assert report.dependencies[0].status == "unavailable"
    assert report.dependencies[0].details == "Probe failed with RuntimeError."
    assert "secret" not in report.dependencies[0].details


def test_readiness_reports_unknown_required_service() -> None:
    report = check_readiness(["unknown"], probes={})

    assert report.status == "not_ready"
    assert report.dependencies[0].details == "No health probe is registered for this service."


def test_readiness_rejects_non_positive_timeout() -> None:
    try:
        check_readiness(["mongodb"], timeout_seconds=0, probes={"mongodb": lambda _timeout: "ready"})
    except ValueError as exc:
        assert str(exc) == "Health probe timeout must be greater than zero."
    else:
        raise AssertionError("Expected a timeout validation error.")


def test_readiness_supports_task_queue_probe() -> None:
    report = check_readiness(
        ["task_queue"],
        probes={"task_queue": lambda _timeout: "Redis and Celery workers are ready."},
    )

    assert report.ready is True
    assert report.dependencies[0].details == "Redis and Celery workers are ready."
