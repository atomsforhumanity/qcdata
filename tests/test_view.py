from html.parser import HTMLParser

import pytest

from qcdata import OptimizationData, ProgramInput, ProgramOutput, ProgramSpec, ScanData
from qcdata.view import generate_optimization_plot, generate_output_table


@pytest.mark.parametrize(
    "wall_time, expected", [(None, "No timing data"), (0, "00.00s"), (65, "01m:05.00s")]
)
def test_output_table_execution_and_producer(prog_output, wall_time, expected):
    payload = prog_output.model_dump()
    payload["execution"] = {"wall_time": wall_time}
    payload["results"]["provenance"] = {
        "program": "actual-producer",
        "program_version": "1.2.3",
    }
    output = ProgramOutput.model_validate(payload)
    rendered = generate_output_table(output)
    assert expected in rendered
    assert "actual-producer 1.2.3" in rendered
    assert output.input_data.program in rendered


def test_generate_optimization_plot_with_single_prog_output_failure(
    prog_input_factory, results_failure
):
    opt_input = prog_input_factory("optimization")

    prog_output = ProgramOutput[ProgramInput, OptimizationData](
        input_data=opt_input,
        success=False,
        traceback="Traceback...",
        results=OptimizationData(
            provenance={"program": "qcdata-test-suite"}, trajectory=[results_failure]
        ),
        execution={},
    )
    generate_optimization_plot(prog_output)


def test_recursive_output_table(water, prog_output):
    inp = ProgramInput(
        program="workflow-request",
        calctype="scan",
        structure=water,
        files={"workflow.in": "hidden root content"},
        subprograms=[
            ProgramSpec(
                program="geometric",
                calctype="optimization",
                subprograms=[
                    ProgramSpec(
                        program="terachem<script>",
                        calctype="gradient",
                        model={"method": "wb97x-d3", "basis": "def2-svp"},
                        keywords={"convthre": 1e-6},
                        files={"guess.bin": b"hidden binary content"},
                        cmdline_args=["--verbose"],
                        extras={"note": "child metadata"},
                    )
                ],
            )
        ],
    )
    output = ProgramOutput(
        input_data=inp,
        success=False,
        results=ScanData(provenance={"program": "actual-executor"}),
        traceback="test failure",
        execution={},
    )
    rendered = generate_output_table(prog_output, output)
    for value in [
        "Requested Program",
        "Result Program",
        "workflow-request",
        "actual-executor",
        "Subprograms",
        "geometric",
        "optimization",
        "terachem&lt;script&gt;",
        "gradient",
        "wb97x-d3",
        "def2-svp",
        "convthre",
        "guess.bin",
        "--verbose",
        "child metadata",
    ]:
        assert value in rendered
    assert "<script>" not in rendered
    assert "hidden binary content" not in rendered
    assert "hidden root content" not in rendered
    assert rendered.count("<ul>") == 2

    # Nested detail tables must not affect the outer table's column counts.
    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth = 0
            self.columns = []

        def handle_starttag(self, tag, attrs):
            if tag == "table":
                self.depth += 1
            if self.depth == 1:
                if tag == "tr":
                    self.columns.append(0)
                elif tag in {"td", "th"}:
                    self.columns[-1] += 1

        def handle_endtag(self, tag):
            if tag == "table":
                self.depth -= 1

    parser = TableParser()
    parser.feed(rendered)
    assert len(parser.columns) == 3
    assert len(set(parser.columns)) == 1


def test_display_empty_failures(prog_input_factory, file_input, monkeypatch):
    from qcdata import ConformerSearchData, FileData, SinglePointData, view

    rendered = []
    monkeypatch.setattr(view, "display", rendered.append)
    for input_data, cls in [
        (file_input, FileData),
        (prog_input_factory("energy"), SinglePointData),
        (prog_input_factory("optimization"), OptimizationData),
        (prog_input_factory("scan"), ScanData),
        (prog_input_factory("conformer_search"), ConformerSearchData),
    ]:
        output = ProgramOutput(
            input_data=input_data,
            results=cls(provenance={"program": "producer"}),
            success=False,
            traceback="Failed before producing values",
        )
        view.program_outputs(output)
    assert len(rendered) == 5
    assert "no optimization steps" in rendered[2].data
    assert "No optimization energies available" in rendered[2].data
