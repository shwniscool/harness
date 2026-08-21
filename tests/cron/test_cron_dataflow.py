"""Tests for cron dataflow metadata (inputs / outputs / side_effects).

Phase 1 of the cron interflow graph: crons never dispatch to each other — they
communicate through data, so each job declares typed ``scheme:value`` resource
lists and cron→cron edges are inferred from ``cron-output:<id>`` inputs. These
tests cover normalization, the declared-vs-derived cross-checks, referential
integrity, acyclicity, backfill, and the store-wide ``validate_store`` sweep.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


class TestResourceListNormalization:
    def test_string_and_list_accepted_and_sorted_deduped(self):
        from cron.jobs import _normalize_resource_list, _INPUT_SCHEMES

        out = _normalize_resource_list(
            ["wiki:b", "file:a", "wiki:b", " file:a "],
            allowed_schemes=_INPUT_SCHEMES,
            field_name="inputs",
        )
        assert out == ["file:a", "wiki:b"]

        single = _normalize_resource_list(
            "https://example.com/x",
            allowed_schemes=_INPUT_SCHEMES,
            field_name="inputs",
        )
        assert single == ["https://example.com/x"]

    def test_none_and_empty_become_empty_list(self):
        from cron.jobs import _normalize_resource_list, _OUTPUT_SCHEMES

        assert _normalize_resource_list(
            None, allowed_schemes=_OUTPUT_SCHEMES, field_name="outputs"
        ) == []
        assert _normalize_resource_list(
            ["", "   "], allowed_schemes=_OUTPUT_SCHEMES, field_name="outputs"
        ) == []

    def test_scheme_lowercased(self):
        from cron.jobs import _normalize_resource_list, _OUTPUT_SCHEMES

        assert _normalize_resource_list(
            "WIKI:Reports/Daily", allowed_schemes=_OUTPUT_SCHEMES, field_name="outputs"
        ) == ["wiki:Reports/Daily"]

    def test_missing_colon_rejected(self):
        from cron.jobs import _normalize_resource_list, _INPUT_SCHEMES

        with pytest.raises(ValueError, match="not a typed reference"):
            _normalize_resource_list(
                "just-a-value", allowed_schemes=_INPUT_SCHEMES, field_name="inputs"
            )

    def test_unknown_scheme_rejected(self):
        from cron.jobs import _normalize_resource_list, _OUTPUT_SCHEMES

        with pytest.raises(ValueError, match="unknown scheme"):
            _normalize_resource_list(
                "telegram:me", allowed_schemes=_OUTPUT_SCHEMES, field_name="outputs"
            )

    def test_missing_value_rejected(self):
        from cron.jobs import _normalize_resource_list, _INPUT_SCHEMES

        with pytest.raises(ValueError, match="missing a value"):
            _normalize_resource_list(
                "wiki:   ", allowed_schemes=_INPUT_SCHEMES, field_name="inputs"
            )

    def test_non_string_entry_rejected(self):
        from cron.jobs import _normalize_resource_list, _INPUT_SCHEMES

        with pytest.raises(ValueError, match="must be 'scheme:value'"):
            _normalize_resource_list(
                [123], allowed_schemes=_INPUT_SCHEMES, field_name="inputs"
            )


class TestCreateStoresDataflow:
    def test_fields_stored_and_normalized(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="collect",
            schedule="every 1h",
            inputs=["https://api.example.com", "wiki:notes"],
            outputs="wiki:reports/daily",
            side_effects=["telegram:me"],
        )
        assert job["inputs"] == ["https://api.example.com", "wiki:notes"]
        assert job["outputs"] == ["wiki:reports/daily"]
        assert job["side_effects"] == ["telegram:me"]

        loaded = get_job(job["id"])
        assert loaded["inputs"] == ["https://api.example.com", "wiki:notes"]
        assert loaded["outputs"] == ["wiki:reports/daily"]
        assert loaded["side_effects"] == ["telegram:me"]

    def test_defaults_to_empty_lists(self, cron_env):
        from cron.jobs import create_job

        job = create_job(prompt="hello", schedule="every 1h")
        assert job["inputs"] == []
        assert job["outputs"] == []
        assert job["side_effects"] == []

    def test_output_scheme_rejected_in_inputs(self, cron_env):
        # telegram is a side-effect scheme; not valid as an input.
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="unknown scheme 'telegram'"):
            create_job(prompt="x", schedule="every 1h", inputs=["telegram:me"])


class TestDeliverSideEffectCrossCheck:
    def test_external_deliver_requires_matching_side_effect(self, cron_env):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="side_effects declares no 'telegram:'"):
            create_job(
                prompt="x",
                schedule="every 1h",
                deliver="telegram",
                side_effects=["email:team@x.com"],
            )

    def test_matching_side_effect_passes(self, cron_env):
        from cron.jobs import create_job

        job = create_job(
            prompt="x",
            schedule="every 1h",
            deliver="telegram",
            side_effects=["telegram:me"],
        )
        assert job["deliver"] == "telegram"

    def test_undeclared_side_effects_not_blocked(self, cron_env):
        # Empty declaration is the legacy path — do not hard-block existing
        # callers that deliver externally without declaring dataflow.
        from cron.jobs import create_job

        job = create_job(prompt="x", schedule="every 1h", deliver="telegram")
        assert job["side_effects"] == []


class TestContextFromCrossCheck:
    def test_context_from_must_be_declared_input(self, cron_env):
        from cron.jobs import create_job

        upstream = create_job(prompt="produce", schedule="every 1h")
        with pytest.raises(ValueError, match="not declared as inputs"):
            create_job(
                prompt="consume",
                schedule="every 2h",
                context_from=upstream["id"],
                inputs=["wiki:something-else"],
            )

    def test_context_from_with_matching_input_passes(self, cron_env):
        from cron.jobs import create_job

        upstream = create_job(prompt="produce", schedule="every 1h")
        downstream = create_job(
            prompt="consume",
            schedule="every 2h",
            context_from=upstream["id"],
            inputs=[f"cron-output:{upstream['id']}"],
        )
        assert downstream["context_from"] == [upstream["id"]]

    def test_context_from_without_declared_inputs_not_blocked(self, cron_env):
        from cron.jobs import create_job

        upstream = create_job(prompt="produce", schedule="every 1h")
        downstream = create_job(
            prompt="consume", schedule="every 2h", context_from=upstream["id"]
        )
        assert downstream["context_from"] == [upstream["id"]]
        assert downstream["inputs"] == []


class TestReferentialIntegrityAndCycles:
    def test_cron_output_input_must_exist(self, cron_env):
        from cron.jobs import create_job

        with pytest.raises(ValueError, match="unknown job"):
            create_job(
                prompt="x",
                schedule="every 1h",
                inputs=["cron-output:doesnotexist"],
            )

    def test_self_reference_rejected_via_update(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="x", schedule="every 1h")
        with pytest.raises(ValueError, match="cycle"):
            update_job(job["id"], {"inputs": [f"cron-output:{job['id']}"]})

    def test_two_node_cycle_rejected(self, cron_env):
        from cron.jobs import create_job, update_job

        a = create_job(prompt="a", schedule="every 1h")
        b = create_job(
            prompt="b",
            schedule="every 1h",
            inputs=[f"cron-output:{a['id']}"],
        )
        # Now make A read B's output → A→B→A cycle.
        with pytest.raises(ValueError, match="cycle"):
            update_job(a["id"], {"inputs": [f"cron-output:{b['id']}"]})

    def test_linear_chain_allowed(self, cron_env):
        from cron.jobs import create_job

        a = create_job(prompt="a", schedule="every 1h")
        b = create_job(
            prompt="b", schedule="every 1h", inputs=[f"cron-output:{a['id']}"]
        )
        c = create_job(
            prompt="c", schedule="every 1h", inputs=[f"cron-output:{b['id']}"]
        )
        assert c["inputs"] == [f"cron-output:{b['id']}"]


class TestUpdateJobDataflow:
    def test_update_normalizes_and_stores(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="x", schedule="every 1h")
        updated = update_job(job["id"], {"outputs": ["WIKI:reports/x", "wiki:reports/x"]})
        assert updated["outputs"] == ["wiki:reports/x"]

    def test_update_clears_with_empty_list(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(
            prompt="x", schedule="every 1h", outputs=["wiki:reports/x"]
        )
        updated = update_job(job["id"], {"outputs": []})
        assert updated["outputs"] == []

    def test_update_bad_scheme_rejected(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="x", schedule="every 1h")
        with pytest.raises(ValueError, match="unknown scheme"):
            update_job(job["id"], {"inputs": ["bogus:thing"]})


class TestBackfill:
    def test_legacy_record_backfilled_on_read(self, cron_env):
        import json

        from cron.jobs import JOBS_FILE, create_job, get_job

        job = create_job(prompt="x", schedule="every 1h")
        # Simulate a pre-feature record by stripping the fields on disk.
        data = json.loads(JOBS_FILE.read_text())
        for record in data["jobs"]:
            record.pop("inputs", None)
            record.pop("outputs", None)
            record.pop("side_effects", None)
        JOBS_FILE.write_text(json.dumps(data))

        loaded = get_job(job["id"])
        assert loaded["inputs"] == []
        assert loaded["outputs"] == []
        assert loaded["side_effects"] == []


class TestValidateStore:
    def test_clean_store_reports_no_issues(self, cron_env):
        from cron.jobs import create_job, validate_store

        a = create_job(prompt="a", schedule="every 1h")
        create_job(prompt="b", schedule="every 1h", inputs=[f"cron-output:{a['id']}"])
        assert validate_store() == []

    def test_dangling_producer_reported(self, cron_env):
        import json

        from cron.jobs import JOBS_FILE, create_job, validate_store

        a = create_job(prompt="a", schedule="every 1h")
        create_job(prompt="b", schedule="every 1h", inputs=[f"cron-output:{a['id']}"])
        # Delete the producer directly on disk (drift create/update can't catch).
        data = json.loads(JOBS_FILE.read_text())
        data["jobs"] = [r for r in data["jobs"] if r["id"] != a["id"]]
        JOBS_FILE.write_text(json.dumps(data))

        issues = validate_store()
        assert any("unknown job" in i for i in issues)

    def test_cycle_formed_on_disk_reported(self, cron_env):
        import json

        from cron.jobs import JOBS_FILE, create_job, validate_store

        a = create_job(prompt="a", schedule="every 1h")
        b = create_job(prompt="b", schedule="every 1h")
        # Hand-edit both to reference each other → cycle in the stored graph.
        data = json.loads(JOBS_FILE.read_text())
        for record in data["jobs"]:
            if record["id"] == a["id"]:
                record["inputs"] = [f"cron-output:{b['id']}"]
            elif record["id"] == b["id"]:
                record["inputs"] = [f"cron-output:{a['id']}"]
        JOBS_FILE.write_text(json.dumps(data))

        issues = validate_store()
        assert any("cycle" in i for i in issues)


class TestBuildCronGraph:
    def test_empty_store(self, cron_env):
        from cron.jobs import build_cron_graph

        graph = build_cron_graph()
        assert graph == {"nodes": [], "edges": []}

    def test_cron_node_shape(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        job = create_job(name="collector", prompt="x", schedule="every 1h")
        graph = build_cron_graph()
        cron_nodes = [n for n in graph["nodes"] if n["kind"] == "cron"]
        assert len(cron_nodes) == 1
        node = cron_nodes[0]
        assert node["id"] == job["id"]
        assert node["type"] == "cron"
        assert node["label"] == "collector"
        assert node["uses_llm"] is True

    def test_no_agent_job_marks_uses_llm_false(self, cron_env):
        import os

        from cron.jobs import build_cron_graph, create_job

        script_dir = cron_env / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        (script_dir / "w.sh").write_text("echo hi\n")
        os.chmod(script_dir / "w.sh", 0o755)
        create_job(prompt="", schedule="every 1h", no_agent=True, script="w.sh")
        graph = build_cron_graph()
        cron_nodes = [n for n in graph["nodes"] if n["kind"] == "cron"]
        assert cron_nodes[0]["uses_llm"] is False

    def test_source_node_and_reads_edge(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        job = create_job(
            prompt="x", schedule="every 1h", inputs=["https://api.example.com"]
        )
        graph = build_cron_graph()
        source = [n for n in graph["nodes"] if n["kind"] == "source"]
        assert source == [
            {
                "id": "https://api.example.com",
                "kind": "source",
                "type": "https",
                "label": "//api.example.com",
            }
        ]
        assert {
            "source": "https://api.example.com",
            "target": job["id"],
            "type": "reads",
        } in graph["edges"]

    def test_artifact_links_producer_to_consumer(self, cron_env):
        # A writes wiki:reports/daily; B reads it → shared artifact node with a
        # writes edge in and a reads edge out (outputs make edges).
        from cron.jobs import build_cron_graph, create_job

        a = create_job(prompt="a", schedule="every 1h", outputs=["wiki:reports/daily"])
        b = create_job(prompt="b", schedule="every 2h", inputs=["wiki:reports/daily"])
        graph = build_cron_graph()

        artifacts = [n for n in graph["nodes"] if n["kind"] == "artifact"]
        assert [n["id"] for n in artifacts] == ["wiki:reports/daily"]

        edges = graph["edges"]
        assert {"source": a["id"], "target": "wiki:reports/daily", "type": "writes"} in edges
        assert {"source": "wiki:reports/daily", "target": b["id"], "type": "reads"} in edges

    def test_cron_output_input_makes_feeds_edge(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        a = create_job(prompt="a", schedule="every 1h")
        b = create_job(
            prompt="b", schedule="every 1h", inputs=[f"cron-output:{a['id']}"]
        )
        graph = build_cron_graph()
        assert {"source": a["id"], "target": b["id"], "type": "feeds"} in graph["edges"]
        # cron-output does not spawn a resource node — it's a direct cron→cron edge.
        assert all(n["kind"] == "cron" for n in graph["nodes"])

    def test_side_effect_makes_sink_and_scheme_typed_edge(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        job = create_job(
            prompt="x",
            schedule="every 1h",
            deliver="telegram",
            side_effects=["telegram:me"],
        )
        graph = build_cron_graph()
        sinks = [n for n in graph["nodes"] if n["kind"] == "sink"]
        assert sinks == [
            {"id": "telegram:me", "kind": "sink", "type": "telegram", "label": "me"}
        ]
        assert {
            "source": job["id"],
            "target": "telegram:me",
            "type": "telegram",
        } in graph["edges"]

    def test_postgres_output_links_writer_to_reader(self, cron_env):
        # A cron that writes a postgres table and another that reads it share one
        # artifact node — postgres is a first-class data store like wiki/file.
        from cron.jobs import build_cron_graph, create_job

        w = create_job(
            prompt="ingest", schedule="every 1h", outputs=["postgres:analytics.events"]
        )
        r = create_job(
            prompt="report", schedule="every 2h", inputs=["postgres:analytics.events"]
        )
        graph = build_cron_graph()

        artifacts = [n for n in graph["nodes"] if n["kind"] == "artifact"]
        assert artifacts == [
            {
                "id": "postgres:analytics.events",
                "kind": "artifact",
                "type": "postgres",
                "label": "analytics.events",
            }
        ]
        edges = graph["edges"]
        assert {
            "source": w["id"],
            "target": "postgres:analytics.events",
            "type": "writes",
        } in edges
        assert {
            "source": "postgres:analytics.events",
            "target": r["id"],
            "type": "reads",
        } in edges

    def test_produced_resource_outranks_source(self, cron_env):
        # If a ref is both read (by one job) and written (by another), the node
        # is an artifact, not a source.
        from cron.jobs import build_cron_graph, create_job

        create_job(prompt="reader", schedule="every 1h", inputs=["file:/data/x"])
        create_job(prompt="writer", schedule="every 2h", outputs=["file:/data/x"])
        graph = build_cron_graph()
        res = [n for n in graph["nodes"] if n["id"] == "file:/data/x"]
        assert len(res) == 1
        assert res[0]["kind"] == "artifact"

    def test_service_shares_store_node_with_cron(self, cron_env):
        # A live service reading postgres:analytics.events meets the cron that
        # writes it on ONE shared artifact node — service and cron converge.
        from cron.jobs import build_cron_graph, create_job

        writer = create_job(
            prompt="ingest",
            schedule="every 1h",
            outputs=["postgres:analytics.events"],
        )
        services = [{
            "id": "proc_dash1",
            "label": "Analytics Dashboard",
            "description": "# Dashboard\nRenders analytics from the events table.",
            "inputs": ["postgres:analytics.events"],
            "outputs": [],
            "side_effects": [],
        }]
        graph = build_cron_graph(services=services)

        assert [n for n in graph["nodes"] if n["kind"] == "service"] == [{
            "id": "proc_dash1",
            "kind": "service",
            "type": "service",
            "label": "Analytics Dashboard",
            "description": "# Dashboard\nRenders analytics from the events table.",
        }]
        stores = [n for n in graph["nodes"] if n["id"] == "postgres:analytics.events"]
        assert len(stores) == 1 and stores[0]["kind"] == "artifact"
        edges = graph["edges"]
        assert {
            "source": writer["id"],
            "target": "postgres:analytics.events",
            "type": "writes",
        } in edges
        assert {
            "source": "postgres:analytics.events",
            "target": "proc_dash1",
            "type": "reads",
        } in edges

    def test_no_services_leaves_no_service_nodes(self, cron_env):
        from cron.jobs import build_cron_graph, create_job

        create_job(prompt="x", schedule="every 1h")
        graph = build_cron_graph(services=[])
        assert not any(n["kind"] == "service" for n in graph["nodes"])


class TestServiceDeclaration:
    def test_valid_declaration_normalizes_and_dedupes(self):
        from cron.jobs import normalize_service_declaration

        decl = normalize_service_declaration(
            name="  Analytics Dashboard  ",
            description="  Renders analytics.  ",
            inputs=["postgres:analytics.events", "postgres:analytics.events"],
        )
        # Assert the normalization CONTRACT (trim, dedupe, empty defaults) rather
        # than freezing the exact key set — an exact-dict compare here is a
        # change-detector test (see AGENTS.md) and breaks on every new optional
        # field, which is what happened when `hosts` was added.
        assert decl["name"] == "Analytics Dashboard"
        assert decl["description"] == "Renders analytics."
        assert decl["inputs"] == ["postgres:analytics.events"]
        for empty in ("outputs", "side_effects", "hosts"):
            assert decl[empty] == []

    def test_description_required(self):
        from cron.jobs import normalize_service_declaration

        for bad in (None, "", "   "):
            with pytest.raises(ValueError, match="description is required"):
                normalize_service_declaration(name="dash", description=bad)

    def test_name_required(self):
        from cron.jobs import normalize_service_declaration

        with pytest.raises(ValueError, match="name is required"):
            normalize_service_declaration(name="   ", description="a description")

    def test_bad_scheme_rejected(self):
        from cron.jobs import normalize_service_declaration

        with pytest.raises(ValueError, match="unknown scheme"):
            normalize_service_declaration(
                name="dash", description="d", inputs=["telegram:me"]
            )


class TestCronGraphRPC:
    def test_handler_returns_ok_envelope(self, cron_env):
        from cron.jobs import create_job

        create_job(prompt="x", schedule="every 1h", side_effects=["notify:desktop"])

        # Invoke the registered handler directly with a stubbed _ok/_err.
        import tui_gateway.methods_tools as mt

        handler = dict(mt._registry._pending)["cron.graph"]
        captured = {}

        def _ok(rid, result):
            captured["rid"] = rid
            captured["result"] = result
            return {"result": result}

        def _err(rid, code, msg):  # pragma: no cover - failure path
            raise AssertionError(f"handler errored: {code} {msg}")

        handler.__globals__["_ok"] = _ok
        handler.__globals__["_err"] = _err
        handler.__globals__.setdefault("logger", __import__("logging").getLogger("t"))

        handler(7, {})
        assert captured["rid"] == 7
        assert set(captured["result"].keys()) == {"nodes", "edges"}
        assert any(n["kind"] == "cron" for n in captured["result"]["nodes"])


class TestHostsContainmentEdge:
    """`hosts` says a service RUNS a store. It is containment, not dataflow: the
    Postgres container does not produce the rows its tables hold — the indexer
    cron does. Declaring the tables as `outputs` (the only option before this)
    drew a `writes` edge indistinguishable from the cron's, collapsing the
    runtime/data distinction."""

    def _pg_service(self, hosts):
        from cron.jobs import normalize_service_declaration

        decl = normalize_service_declaration(
            name="Compendium Postgres",
            description="Postgres 16 store for agentic-payments.",
            hosts=hosts,
        )
        return {"id": "docker:a1ce2ec76fb9", "label": decl["name"],
                "description": decl["description"], "inputs": decl["inputs"],
                "outputs": decl["outputs"], "side_effects": decl["side_effects"],
                "hosts": decl["hosts"]}

    def test_hosts_emits_a_hosts_edge_not_a_writes_edge(self):
        ref = "postgres:agentic_payments.transfers"
        from cron.jobs import build_cron_graph

        jobs = [{"id": "indexer", "name": "indexer", "outputs": [ref]}]
        g = build_cron_graph(jobs=jobs, services=[self._pg_service([ref])])

        typed = {(e["source"], e["target"], e["type"]) for e in g["edges"]}
        # The cron is the sole producer.
        assert ("indexer", ref, "writes") in typed
        # The container states containment, and never claims to write.
        assert ("docker:a1ce2ec76fb9", ref, "hosts") in typed
        assert ("docker:a1ce2ec76fb9", ref, "writes") not in typed

    def test_hosted_ref_dedupes_onto_the_cron_node(self):
        """Containment must meet dataflow on ONE node, or the graph shows the
        store twice and the whole point is lost."""
        ref = "postgres:agentic_payments.transfers"
        from cron.jobs import build_cron_graph

        jobs = [{"id": "indexer", "name": "indexer", "outputs": [ref]},
                {"id": "reader", "name": "reader", "inputs": [ref]}]
        g = build_cron_graph(jobs=jobs, services=[self._pg_service([ref])])
        assert sum(1 for n in g["nodes"] if n["id"] == ref) == 1

    def test_hosting_alone_does_not_make_a_ref_an_artifact(self):
        """Hosting is not producing. A table only a container hosts has no
        provenance, so it must stay a `source` — otherwise the graph asserts the
        container produced data it merely stores."""
        from cron.jobs import build_cron_graph

        ref = "postgres:agentic_payments.seller_meta"
        g = build_cron_graph(jobs=[], services=[self._pg_service([ref])])
        node = next(n for n in g["nodes"] if n["id"] == ref)
        assert node["kind"] == "source"

    def test_a_real_writer_still_promotes_a_hosted_ref_to_artifact(self):
        """Order independence: whichever is declared first, a ref with a real
        cron writer is an artifact."""
        from cron.jobs import build_cron_graph

        ref = "postgres:agentic_payments.transfers"
        jobs = [{"id": "indexer", "name": "indexer", "outputs": [ref]}]
        g = build_cron_graph(jobs=jobs, services=[self._pg_service([ref])])
        node = next(n for n in g["nodes"] if n["id"] == ref)
        assert node["kind"] == "artifact"

    def test_hosts_rejects_non_store_schemes(self):
        """You can host a data store, not a URL or a telegram chat."""
        from cron.jobs import normalize_service_declaration

        for bad in ["https:example.com/x", "telegram:me", "cron-output:abc123"]:
            with pytest.raises(ValueError):
                normalize_service_declaration(
                    name="X", description="d", hosts=[bad])

    def test_hosts_defaults_empty_and_is_backward_compatible(self):
        """A service declared without `hosts` behaves exactly as before."""
        from cron.jobs import build_cron_graph, normalize_service_declaration

        decl = normalize_service_declaration(name="X", description="d")
        assert decl["hosts"] == []
        svc = {"id": "proc_1", "label": "X", "description": "d",
               "inputs": [], "outputs": [], "side_effects": []}   # no 'hosts' key
        g = build_cron_graph(jobs=[], services=[svc])
        assert [e for e in g["edges"] if e["type"] == "hosts"] == []
