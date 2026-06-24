def test_import():
    import stokify

    assert hasattr(stokify, "__version__")


def test_version_is_string():
    from stokify import __version__

    assert isinstance(__version__, str)
