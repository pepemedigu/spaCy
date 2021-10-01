from typing import List, Callable, Tuple, Dict, Iterable, Union, Any, IO
from typing import Optional, TYPE_CHECKING
from pathlib import Path
from timeit import default_timer as timer
from thinc.api import Optimizer, Config, constant, fix_random_seed, set_gpu_allocator
from wasabi import Printer
import random
import sys
import shutil

from .loop import create_evaluation_callback, create_train_batches, subdivide_batch
from .loop import update_meta, create_before_to_disk_callback, clean_output_dir
from ..schemas import ConfigSchemaTraining
from ..util import resolve_dot_names, registry

if TYPE_CHECKING:
    from ..language import Language  # noqa: F401


DIR_MODEL_BEST = "model-best"
DIR_MODEL_LAST = "model-last"


def distill(
    teacher: "Language",
    student: "Language",
    output_path: Optional[Path] = None,
    *,
    use_gpu: int = -1,
    stdout: IO = sys.stdout,
    stderr: IO = sys.stderr,
) -> Tuple["Language", Optional[Path]]:
    """Train a pipeline.

    teacher (Language): The loaded teacher nlp object with the full config.
    student (Language): The initialized student nlp object with the full config.
    output_path (Path): Optional output path to save trained model to.
    use_gpu (int): Whether to train on GPU. Make sure to call require_gpu
        before calling this function.
    stdout (file): A file-like object to write output messages. To disable
        printing, set to io.StringIO.
    stderr (file): A second file-like object to write output messages. To disable
        printing, set to io.StringIO.

    RETURNS (tuple): The final nlp object and the path to the exported model.
    """
    # We use no_print here so we can respect the stdout/stderr options.
    msg = Printer(no_print=True)
    # Create iterator, which yields out info after each optimization step.
    config = student.config.interpolate()
    if config["training"]["seed"] is not None:
        fix_random_seed(config["training"]["seed"])
    allocator = config["training"]["gpu_allocator"]
    if use_gpu >= 0 and allocator:
        set_gpu_allocator(allocator)
    T = registry.resolve(config["training"], schema=ConfigSchemaTraining)
    dot_names = [T["train_corpus"], T["dev_corpus"]]
    train_corpus, dev_corpus = resolve_dot_names(config, dot_names)
    optimizer = T["optimizer"]
    score_weights = T["score_weights"]
    batcher = T["batcher"]
    train_logger = T["logger"]
    before_to_disk = create_before_to_disk_callback(T["before_to_disk"])

    # Helper function to save checkpoints. This is a closure for convenience,
    # to avoid passing in all the args all the time.
    def save_checkpoint(is_best):
        with student.use_params(optimizer.averages):
            before_to_disk(student).to_disk(output_path / DIR_MODEL_LAST)
        if is_best:
            # Avoid saving twice (saving will be more expensive than
            # the dir copy)
            if (output_path / DIR_MODEL_BEST).exists():
                shutil.rmtree(output_path / DIR_MODEL_BEST)
            shutil.copytree(output_path / DIR_MODEL_LAST, output_path / DIR_MODEL_BEST)

    # Components that shouldn't be updated during training
    frozen_components = T["frozen_components"]
    # Components that should set annotations on update
    annotating_components = T["annotating_components"]
    # Create iterator, which yields out info after each optimization step.
    training_step_iterator = distill_while_improving(
        teacher,
        student,
        optimizer,
        create_train_batches(student, train_corpus, batcher, T["max_epochs"]),
        create_evaluation_callback(student, dev_corpus, score_weights),
        dropout=T["dropout"],
        accumulate_gradient=T["accumulate_gradient"],
        patience=T["patience"],
        max_steps=T["max_steps"],
        eval_frequency=T["eval_frequency"],
        exclude=frozen_components,
        annotating_components=annotating_components,
    )
    clean_output_dir(output_path)
    stdout.write(msg.info(f"Pipeline: {student.pipe_names}") + "\n")
    if frozen_components:
        stdout.write(msg.info(f"Frozen components: {frozen_components}") + "\n")
    if annotating_components:
        stdout.write(
            msg.info(f"Set annotations on update for: {annotating_components}") + "\n"
        )
    stdout.write(msg.info(f"Initial learn rate: {optimizer.learn_rate}") + "\n")
    with student.select_pipes(disable=frozen_components):
        log_step, finalize_logger = train_logger(student, stdout, stderr)
    try:
        for batch, info, is_best_checkpoint in training_step_iterator:
            if is_best_checkpoint is not None:
                with student.select_pipes(disable=frozen_components):
                    update_meta(T, student, info)
                if output_path is not None:
                    save_checkpoint(is_best_checkpoint)
                    info["output_path"] = str(output_path / DIR_MODEL_LAST)
            log_step(info if is_best_checkpoint is not None else None)
    except Exception as e:
        if output_path is not None:
            stdout.write(
                msg.warn(
                    f"Aborting and saving the final best model. "
                    f"Encountered exception: {repr(e)}"
                )
                + "\n"
            )
        raise e
    finally:
        finalize_logger()
        if output_path is not None:
            save_checkpoint(False)
    # This will only run if we did't hit an error
    if optimizer.averages:
        student.use_params(optimizer.averages)
    if output_path is not None:
        stdout.write(
            msg.good("Saved pipeline to output directory", output_path / DIR_MODEL_LAST)
            + "\n"
        )
        return (student, output_path / DIR_MODEL_LAST)
    else:
        return (student, None)


def distill_while_improving(
    teacher: "Language",
    student: "Language",
    optimizer: Optimizer,
    train_data,
    evaluate,
    *,
    dropout: float,
    eval_frequency: int,
    accumulate_gradient: int,
    patience: int,
    max_steps: int,
    exclude: List[str],
    annotating_components: List[str],
):
    """Train until an evaluation stops improving. Works as a generator,
    with each iteration yielding a tuple `(batch, info, is_best_checkpoint)`,
    where info is a dict, and is_best_checkpoint is in [True, False, None] --
    None indicating that the iteration was not evaluated as a checkpoint.
    The evaluation is conducted by calling the evaluate callback.

    Positional arguments:
        nlp: The spaCy pipeline to evaluate.
        optimizer: The optimizer callable.
        train_data (Iterable[Batch]): A generator of batches, with the training
            data. Each batch should be a Sized[Tuple[Input, Annot]]. The training
            data iterable needs to take care of iterating over the epochs and
            shuffling.
        evaluate (Callable[[], Tuple[float, Any]]): A callback to perform evaluation.
            The callback should take no arguments and return a tuple
            `(main_score, other_scores)`. The main_score should be a float where
            higher is better. other_scores can be any object.

    Every iteration, the function yields out a tuple with:

    * batch: A list of Example objects.
    * info: A dict with various information about the last update (see below).
    * is_best_checkpoint: A value in None, False, True, indicating whether this
        was the best evaluation so far. You should use this to save the model
        checkpoints during training. If None, evaluation was not conducted on
        that iteration. False means evaluation was conducted, but a previous
        evaluation was better.

    The info dict provides the following information:

        epoch (int): How many passes over the data have been completed.
        step (int): How many steps have been completed.
        score (float): The main score from the last evaluation.
        other_scores: : The other scores from the last evaluation.
        losses: The accumulated losses throughout training.
        checkpoints: A list of previous results, where each result is a
            (score, step, epoch) tuple.
    """
    if isinstance(dropout, float):
        dropouts = constant(dropout)
    else:
        dropouts = dropout
    results = []
    losses = {}
    words_seen = 0
    start_time = timer()
    for step, (epoch, batch) in enumerate(train_data):
        dropout = next(dropouts)
        for subbatch in subdivide_batch(batch, accumulate_gradient):
            student.distill(
                teacher,
                subbatch,
                drop=dropout,
                losses=losses,
                sgd=False,
                exclude=exclude,
                annotates=annotating_components,
            )
        # TODO: refactor this so we don't have to run it separately in here
        for name, proc in student.pipeline:
            if (
                name not in exclude
                and hasattr(proc, "is_trainable")
                and proc.is_trainable
                and proc.model not in (True, False, None)
            ):
                proc.finish_update(optimizer)
        optimizer.step_schedules()
        if not (step % eval_frequency):
            if optimizer.averages:
                with student.use_params(optimizer.averages):
                    score, other_scores = evaluate()
            else:
                score, other_scores = evaluate()
            results.append((score, step))
            is_best_checkpoint = score == max(results)[0]
        else:
            score, other_scores = (None, None)
            is_best_checkpoint = None
        words_seen += sum(len(eg) for eg in batch)
        info = {
            "epoch": epoch,
            "step": step,
            "score": score,
            "other_scores": other_scores,
            "losses": losses,
            "checkpoints": results,
            "seconds": int(timer() - start_time),
            "words": words_seen,
        }
        yield batch, info, is_best_checkpoint
        if is_best_checkpoint is not None:
            losses = {}
        # Stop if no improvement in `patience` updates (if specified)
        # Negate step value so that the earliest best step is chosen for the
        # same score, i.e. (1.0, 100) is chosen over (1.0, 200)
        best_result = max((r_score, -r_step) for r_score, r_step in results)
        best_step = -best_result[1]
        if patience and (step - best_step) >= patience:
            break
        # Stop if we've exhausted our max steps (if specified)
        if max_steps and step >= max_steps:
            break
