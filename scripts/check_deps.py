"""Quick dependency check."""
try:
    import pandas, numpy, sklearn, dotenv  # noqa: F401
    print("Core deps: OK")
except ImportError as e:
    print(f"Missing: {e}")

try:
    from hogan_bot.storage import get_connection
    print("hogan_bot: OK")
except ImportError as e:
    print(f"hogan_bot import failed: {e}")
