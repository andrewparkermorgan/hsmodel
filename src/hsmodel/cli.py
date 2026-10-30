"""
Command-line interface:

    hsmodel simulate     simulate crossover data under the model
    hsmodel preprocess   convert empirical crossover data to the dataset format
    hsmodel infer        fit the model to a dataset with SVI
    hsmodel nuts         fit the model to a dataset with NUTS (full Bayesian)

`hsmodel COMMAND --help` describes each command's options.  The same commands
run as `python -m hsmodel COMMAND`.
"""
import argparse
import sys

from . import __version__, infer, nuts_command, preprocess, simulate
from .errors import CommandError

COMMANDS = {
	"simulate": (simulate, "simulate crossover data under the hierarchical Housworth-Stahl model"),
	"preprocess": (preprocess, "convert empirical crossover data into the dataset format"),
	"infer": (infer, "fit the model to a dataset with stochastic variational inference"),
	"nuts": (nuts_command, "fit the model to a dataset with NUTS (full Bayesian; slow)"),
}


def build_parser():
	ap = argparse.ArgumentParser(prog="hsmodel", description=__doc__,
	                             formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
	sub = ap.add_subparsers(dest="command", metavar="COMMAND", required=True)
	for name, (module, summary) in COMMANDS.items():
		p = sub.add_parser(name, help=summary, description=module.__doc__,
		                   formatter_class=argparse.RawDescriptionHelpFormatter)
		module.add_arguments(p)
		p.set_defaults(_run=module.run, _parser=p)
	return ap


def main(argv=None):
	args = build_parser().parse_args(argv)
	try:
		return args._run(args)
	except CommandError as e:
		args._parser.error(str(e))          # usage message, exit status 2


if __name__ == "__main__":
	sys.exit(main())
