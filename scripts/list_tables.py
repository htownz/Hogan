import sqlite3
c = sqlite3.connect("data/hogan.db")
rows = c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
for r in rows:
    print(r[0])
c.close()
