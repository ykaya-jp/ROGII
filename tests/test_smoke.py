"""Smoke tests — verify rogii imports."""


def test_import_package():
    import rogii as pkg

    assert pkg.__version__
