from typing import Optional
from pathlib import Path
from wasabi import msg
import typer
import logging
import sys

from ._util import app, Arg, Opt, parse_config_overrides, show_validation_error
from ._util import import_code, setup_gpu
from ..training.distill_loop import distill
from ..training.initialize import init_nlp
from .. import util


@app.command(
    "distill", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def distill_cli(
    # fmt: off
    ctx: typer.Context,  # This is only used to read additional arguments
    teacher_model: Path = Arg(..., help="Path to teacher model", exists=True, allow_dash=False),
    student_config_path: Path = Arg(..., help="Path to student config file", exists=True, allow_dash=False),
    output_path: Optional[Path] = Opt(None, "--output", "--output-path", "-o", help="Output directory to store trained pipeline in"),
    code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
    verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU")
    # fmt: on
):
    """
    Distill a spaCy pipeline.

    DOCS: https://spacy.io/api/cli#distill
    """
    util.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Make sure all files and paths exists if they are needed
    if not student_config_path or not student_config_path.exists():
        msg.fail("Student config file not found", student_config_path, exits=1)
    if not output_path:
        msg.info("No output directory provided")
    else:
        if not output_path.exists():
            output_path.mkdir(parents=True)
            msg.good(f"Created output directory: {output_path}")
        msg.info(f"Saving to output directory: {output_path}")
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    setup_gpu(use_gpu)
    with show_validation_error(student_config_path):
        student_config = util.load_config(student_config_path, overrides=overrides, interpolate=False)

    msg.divider("Loading teacher pipeline")
    teacher = util.load_model(teacher_model)
    msg.good("Loaded teacher pipeline")
    msg.divider("Initializing student pipeline")
    with show_validation_error(student_config_path, hint_fill=False):
        student = init_nlp(student_config, use_gpu=use_gpu)
    msg.good("Initialized student pipeline")
    msg.divider("Training pipeline")
    distill(teacher, student, output_path, use_gpu=use_gpu, stdout=sys.stdout, stderr=sys.stderr)
