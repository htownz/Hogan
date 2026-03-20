"""Quick dependency check."""
try:
    import pandas  # noqa: F401
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
    import dotenv  # noqa: F401
    print("Core deps: OK")
except ImportError as e:
    print(f"Missing: {e}")

try:
    from hogan_bot.storage import get_connection  # noqa: F401
    print("hogan_bot: OK")
except ImportError as e:
    print(f"hogan_bot import failed: {e}")
