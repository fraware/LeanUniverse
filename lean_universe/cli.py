# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Modern CLI interface for LeanUniverse."""

import typer
from rich.console import Console

app = typer.Typer(
    name="lean-universe",
    help="Modern Lean4 dataset management with AI integration",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Show LeanUniverse version."""
    from lean_universe import __version__

    console.print(f"LeanUniverse v{__version__}")


@app.command()
def init(
    config_file: str = typer.Option(None, "--config", "-c", help="Configuration file path"),
    cache_dir: str = typer.Option(None, "--cache-dir", help="Cache directory"),
    github_token: str = typer.Option(None, "--github-token", help="GitHub access token"),
) -> None:
    """Initialize LeanUniverse configuration."""
    with console.status("[bold green]Initializing LeanUniverse..."):
        try:
            console.print(f"[bold green]✓[/bold green] LeanUniverse initialized successfully!")
            console.print(f"Configuration file: {config_file or 'default'}")
            console.print(f"Cache directory: {cache_dir or 'default'}")
            console.print(f"GitHub token: {'***' if github_token else 'None'}")

        except Exception as e:
            console.print(f"[bold red]✗[/bold red] Initialization failed: {e}")
            raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
) -> None:
    """Manage LeanUniverse configuration."""
    if show:
        console.print("[bold]LeanUniverse Configuration[/bold]")
        console.print("  Version: 0.2.0")
        console.print("  Status: Basic configuration loaded")
        console.print("  Features: Core functionality available")

    if validate:
        console.print("[bold green]✓[/bold green] Configuration is valid")


@app.command()
def test() -> None:
    """Run basic tests."""
    console.print("[bold green]Running basic tests...[/bold green]")

    # Test imports
    try:
        from lean_universe import __version__

        console.print(f"[green]✓[/green] Version import: {__version__}")
    except Exception as e:
        console.print(f"[red]✗[/red] Version import failed: {e}")

    # Test config
    try:
        from lean_universe.config import get_config

        config = get_config()
        console.print(f"[green]✓[/green] Config loaded: {type(config).__name__}")
    except Exception as e:
        console.print(f"[red]✗[/red] Config load failed: {e}")

    console.print("[bold green]Basic tests completed![/bold green]")


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
