"""Command line: python -m valorant_voice <command> --help"""

import argparse
import sys

from . import config
from .agents import AGENTS, require_agent


def _agents_from(args) -> list[str]:
    if args.agent:
        return [require_agent(args.agent)]
    return [slug for slug in AGENTS if (config.RAW_DIR / slug).is_dir()]


def _fetch(args):
    from .data.fetch import NO_PUBLIC_DATA, fetch_agent
    from .data.stats import record

    for agent in [require_agent(args.agent)] if args.agent else [a for a in AGENTS if a not in NO_PUBLIC_DATA]:
        count = fetch_agent(agent)
        record(config.CLEAN_DIR / agent, "fetch", {"fetched": count})
        print(agent, "fetched:", count)


def _ingest(args):
    from .data.ingest import ingest_agent
    from .data.stats import record

    for agent in _agents_from(args):
        outcome = ingest_agent(agent)
        record(config.CLEAN_DIR / agent, "ingest", outcome)
        print(agent, dict(outcome))


def _prepare(args):
    from .data.prepare import prepare_agent
    from .data.stats import record

    for agent in _agents_from(args):
        outcome = prepare_agent(agent)
        record(config.CLEAN_DIR / agent, "prepare", outcome)
        print(agent, dict(outcome))


def _verify(args):
    from .data.stats import record
    from .data.transcribe import verify_agent

    for agent in _agents_from(args):
        outcome = verify_agent(agent, model_size=args.model)
        record(config.CLEAN_DIR / agent, "verify", outcome)
        print(agent, dict(outcome))


def _audit(_args):
    from .data.stats import audit_table

    print(audit_table())


def _transcribe(args):
    from .data.transcribe import transcribe_agent

    for agent in _agents_from(args):
        print(agent, "lines kept:", transcribe_agent(agent, model_size=args.model))


def _manifests(_args):
    from .data.manifest import build_manifests

    print(dict(build_manifests()))


def _references(args):
    from .tts.references import build_references

    for agent in _agents_from(args):
        print(agent, build_references(agent))


def _speak(args):
    import soundfile as sf

    from .tts.engine import XttsEngine

    engine = XttsEngine()
    samples, rate = engine.synthesize(require_agent(args.agent), args.text)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"{args.agent}.wav"
    sf.write(out, samples, rate)
    print("wrote", out)


def _evaluate(args):
    from .evaluation.run import run_evaluation

    print(run_evaluation(lines_per_agent=args.lines))


def _serve(args):
    import uvicorn

    from .serve.api import create_app
    from .tts.engine import XttsEngine

    uvicorn.run(create_app(XttsEngine()), host=args.host, port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="valorant_voice")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, handler, help_text, agent=False):
        sub = commands.add_parser(name, help=help_text)
        if agent:
            sub.add_argument("--agent", help="one agent id; default is every agent with a data/raw folder")
        sub.set_defaults(handler=handler)
        return sub

    add("fetch", _fetch, "download voice lines from the public Hugging Face dataset into data/raw", agent=True)
    add("ingest", _ingest, "clean raw clips into data/clean", agent=True)
    add("prepare", _prepare, "pair clean clips with the source text, drop empty and repeated lines", agent=True)
    add("verify", _verify, "keep lines whose text Whisper hears in the clip", agent=True).add_argument(
        "--model", default="small.en"
    )
    add("audit", _audit, "print the per-agent table of lines kept, minutes and drops")
    add("transcribe", _transcribe, "Whisper transcripts for clean clips", agent=True).add_argument(
        "--model", default="small.en"
    )
    add("manifests", _manifests, "write train, val and test manifests")
    add("references", _references, "choose reference clips per agent", agent=True)
    speak = add("speak", _speak, "synthesize one line to outputs/")
    speak.add_argument("agent")
    speak.add_argument("text")
    add("evaluate", _evaluate, "measure similarity, WER and speed on the test lines").add_argument(
        "--lines", type=int, default=10
    )
    serve = add("serve", _serve, "run the API and the web page")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
