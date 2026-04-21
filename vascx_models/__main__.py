from .runtime import configure_runtime_environment

configure_runtime_environment()

from .cli import cli


if __name__ == "__main__":
    cli()
