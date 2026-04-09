from typer.testing import CliRunner
from rottengenizdat.cli import app

runner = CliRunner()


class TestCLI:
    def test_help_shows_banner(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "rottengenizdat" in result.output.lower() or "rotten" in result.output.lower()

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app)
        assert result.exit_code == 0
