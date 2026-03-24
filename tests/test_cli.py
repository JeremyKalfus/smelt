from smelt.cli import main


def test_cli_main(capsys) -> None:
    code = main([])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.strip() == "smelt bootstrap ready"
