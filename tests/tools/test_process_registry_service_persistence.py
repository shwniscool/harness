"""A service declaration must survive a gateway restart, and only while alive.

Two invariants, both regressions found against the merged #17 service-node work:

1. **Round-trip.** ``_write_checkpoint`` → ``recover_from_checkpoint`` must
   preserve the five ``service_*`` fields. Without them a background service is
   adopted after a gateway restart as an anonymous process: ``service_name``
   comes back empty, ``collect_service_declarations()`` skips it, and the
   service silently disappears from the cron interflow graph while its process
   is still running. The process is the lease, so the declaration has to outlive
   the gateway exactly as long as the process does.

2. **Liveness is still probed.** Fixing (1) creates the mirror hazard: a
   recovered session's ``exited`` flag is stale (there is no waitable handle),
   so a service whose process died while the gateway was down would be reported
   live forever. ``collect_service_declarations`` must reconcile detached
   sessions against the real PID like every other read path does.

These assert the contract between the two halves (what is persisted must be what
is restored, and liveness must reflect the OS), not any particular field list.
"""
import json
import time
from unittest.mock import patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture
def registry():
    return ProcessRegistry()


def _service_session(sid="proc_dash", pid=4242):
    s = ProcessSession(
        id=sid,
        command="python3 dashboard.py",
        task_id="t1",
        started_at=time.time(),
        pid=pid,
        pid_scope="host",
        host_start_time=int(time.time()),
    )
    s.service_name = "Compendium Dashboard"
    s.service_description = "FastAPI dashboard on :8700 over the compendium."
    s.service_inputs = ["postgres:agentic_payments.transfers"]
    s.service_outputs = ["file:/tmp/dash-cache.json"]
    s.service_side_effects = ["notify:ops"]
    return s


class TestServiceDeclarationSurvivesRestart:
    def test_checkpoint_round_trip_preserves_declaration(self, registry, tmp_path):
        """The declaration a service registered with must be the one it comes
        back with — otherwise it vanishes from the graph on restart."""
        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()

            # Persisted at all?
            entry = json.loads(checkpoint.read_text())[0]
            assert entry["service_name"] == original.service_name
            assert entry["service_description"] == original.service_description
            assert entry["service_inputs"] == original.service_inputs
            assert entry["service_outputs"] == original.service_outputs
            assert entry["service_side_effects"] == original.service_side_effects

            # Restored into a NEW registry, with the process still alive.
            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                assert fresh.recover_from_checkpoint() == 1

            revived = fresh._running[original.id]
            assert revived.service_name == original.service_name
            assert revived.service_description == original.service_description
            assert revived.service_inputs == original.service_inputs
            assert revived.service_outputs == original.service_outputs
            assert revived.service_side_effects == original.service_side_effects

    def test_recovered_service_still_appears_in_the_graph(self, registry, tmp_path):
        """End-to-end of the actual symptom: after a restart the still-running
        service must still be collected for build_cron_graph."""
        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()

            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                fresh.recover_from_checkpoint()

            with patch.object(fresh, "_host_pid_is_ours", return_value=True):
                services = fresh.collect_service_declarations()

        assert len(services) == 1
        assert services[0]["label"] == "Compendium Dashboard"
        assert services[0]["inputs"] == ["postgres:agentic_payments.transfers"]
        assert services[0]["description"]  # required by normalize_service_declaration

    def test_declaration_shape_matches_graph_builder(self, registry, tmp_path):
        """A recovered service must feed build_cron_graph and converge with a
        cron on the shared resource node — the whole point of declaring it."""
        from cron.jobs import build_cron_graph

        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()
            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                fresh.recover_from_checkpoint()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True):
                services = fresh.collect_service_declarations()

        jobs = [{
            "id": "indexer",
            "name": "indexer",
            "outputs": ["postgres:agentic_payments.transfers"],
        }]
        graph = build_cron_graph(jobs=jobs, services=services)
        shared = "postgres:agentic_payments.transfers"

        # The shared ref must be ONE node, not one per producer/consumer.
        assert sum(1 for n in graph["nodes"] if n["id"] == shared) == 1
        assert {(e["source"], e["target"], e["type"]) for e in graph["edges"]} >= {
            ("indexer", shared, "writes"),
            (shared, original.id, "reads"),
        }


class TestRecoveredServiceLiveness:
    def test_dead_recovered_service_is_not_reported_live(self, registry, tmp_path):
        """The mirror hazard of persisting the declaration: if the process died
        while the gateway was down, the service must NOT still be in the graph.
        Presence in _running is not evidence of liveness for detached sessions."""
        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()

            fresh = ProcessRegistry()
            # Alive at recovery time...
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                fresh.recover_from_checkpoint()
            assert fresh._running[original.id].detached is True

            # ...but the PID is gone (or recycled) by the time we build a graph.
            with patch.object(fresh, "_host_pid_is_ours", return_value=False):
                services = fresh.collect_service_declarations()

        assert services == [], "a dead service must not appear as a live node"

    def test_live_recovered_service_is_reported(self, registry, tmp_path):
        """Symmetry check — the liveness probe must not drop a service whose
        process genuinely survived, or the fix would hide every service."""
        checkpoint = tmp_path / "procs.json"
        original = _service_session()
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[original.id] = original
            registry._write_checkpoint()
            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True), \
                 patch.object(ProcessRegistry, "_safe_host_start_time",
                              return_value=original.host_start_time):
                fresh.recover_from_checkpoint()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True):
                services = fresh.collect_service_declarations()

        assert [s["label"] for s in services] == ["Compendium Dashboard"]


class TestBackwardCompatibility:
    def test_old_checkpoint_without_service_keys_recovers(self, tmp_path):
        """A checkpoint written by a build predating the service fields must
        recover as a plain background process, not raise."""
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_legacy",
            "command": "sleep 999",
            "pid": 5150,
            "pid_scope": "host",
            "host_start_time": int(time.time()),
            "cwd": "/tmp",
            "started_at": time.time(),
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", return_value=True):
                assert fresh.recover_from_checkpoint() == 1
                revived = fresh._running["proc_legacy"]
                assert revived.service_name == ""
                assert revived.service_inputs == []
                # Not a service, so it contributes no graph node.
                assert fresh.collect_service_declarations() == []


class TestRegistrationPersistsImmediately:
    """The spawn helpers write the checkpoint as their LAST action, so attaching
    the declaration afterwards leaves an empty service_name on disk until some
    unrelated later write refreshes it. A gateway restart in that window adopts
    the process without its identity — the exact loss the service_* fields exist
    to prevent. register_service_declaration must close that window."""

    def test_registration_is_visible_on_disk_immediately(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        s = _service_session(sid="proc_dash2", pid=7777)
        # Arrive as a bare session, exactly as a spawn helper leaves it.
        for f in ("service_name", "service_description"):
            setattr(s, f, "")
        s.service_inputs, s.service_outputs, s.service_side_effects = [], [], []

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._running[s.id] = s
            registry._write_checkpoint()          # the spawn helper's write
            assert json.loads(checkpoint.read_text())[0]["service_name"] == ""

            assert registry.register_service_declaration(s.id, {
                "name": "Compendium Dashboard",
                "description": "FastAPI dashboard on :8700.",
                "inputs": ["postgres:agentic_payments.transfers"],
                "outputs": [],
                "side_effects": [],
            }) is True

            # Persisted WITHOUT waiting for any further checkpoint write.
            entry = json.loads(checkpoint.read_text())[0]
            assert entry["service_name"] == "Compendium Dashboard"
            assert entry["service_inputs"] == ["postgres:agentic_payments.transfers"]

    def test_registering_an_unknown_or_dead_session_reports_failure(self, registry, tmp_path):
        """A caller must not believe a dead session was registered."""
        checkpoint = tmp_path / "procs.json"
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            decl = {"name": "X", "description": "d", "inputs": [],
                    "outputs": [], "side_effects": []}
            assert registry.register_service_declaration("proc_nope", decl) is False

            dead = _service_session(sid="proc_dead", pid=8888)
            dead.exited = True
            registry._running[dead.id] = dead
            assert registry.register_service_declaration("proc_dead", decl) is False
