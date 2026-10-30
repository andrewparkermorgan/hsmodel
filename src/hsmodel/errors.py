"""Exceptions reported by the command-line interface."""


class CommandError(ValueError):
	"""Bad command-line arguments: reported as a usage error (exit status 2)."""


class InputError(ValueError):
	"""A problem with an input file, reported with the file and line (exit status 1)."""
