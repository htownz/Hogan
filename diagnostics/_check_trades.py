"""Quick diagnostic: why are we not trading?"""
import sqlite3

conn = sqlite3.connect("data/hogan.db")
conn.row_factory = sqlite3.Row

# 1. Paper trades
pt = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
print(f"Paper trades: {pt}")

# 2. Decision log — discover columns first
dl_cols = [r[1] for r in conn.execute("PRAGMA table_info(decision_log)").fetchall()]
print(f"\nDecision log columns: {dl_cols}")
dl = conn.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]
print(f"Decision log entries: {dl}")

if dl > 0:
    # Show last 10 entries
    recent = conn.execute("SELECT * FROM decision_log ORDER BY rowid DESC LIMIT 10").fetchall()
    print("  Last 10 decision_log entries:")
    for r in recent:
        parts = []
        for col in dl_cols:
            val = r[col]
            if val is not None:
                parts.append(f"{col}={val}")
        print(f"    {', '.join(parts[:8])}")

    # Action distribution in decision_log
    act_col = None
    for c in ["action", "final_action", "signal"]:
        if c in dl_cols:
            act_col = c
            break
    if act_col:
        acts = conn.execute(
            f"SELECT {act_col}, COUNT(*) as cnt FROM decision_log GROUP BY {act_col} ORDER BY cnt DESC"
        ).fetchall()
        print(f"\n  Decision log action distribution ({act_col}):")
        for r in acts:
            print(f"    {r[0]}: {r[1]}")

# 3. Swarm decisions
sd_cols = [r[1] for r in conn.execute("PRAGMA table_info(swarm_decisions)").fetchall()]
print(f"\nSwarm decisions columns: {sd_cols}")
sd = conn.execute("SELECT COUNT(*) FROM swarm_decisions").fetchone()[0]
print(f"Swarm decisions total: {sd}")

if sd > 0:
    rows = conn.execute(
        "SELECT final_action, vetoed, COUNT(*) as cnt, "
        "AVG(final_conf) as avg_conf, AVG(agreement) as avg_agree, AVG(entropy) as avg_ent "
        "FROM swarm_decisions GROUP BY final_action, vetoed ORDER BY cnt DESC"
    ).fetchall()
    print("  Breakdown:")
    for r in rows:
        print(f"    action={r['final_action']:>6} vetoed={r['vetoed']} n={r['cnt']:>5} "
              f"conf={r['avg_conf']:.4f} agree={r['avg_agree']:.4f} ent={r['avg_ent']:.4f}")

    # Check mode distribution
    if "mode" in sd_cols:
        modes = conn.execute(
            "SELECT mode, COUNT(*) as cnt FROM swarm_decisions GROUP BY mode"
        ).fetchall()
        print(f"\n  Swarm mode distribution:")
        for r in modes:
            print(f"    {r['mode']}: {r['cnt']}")

    # Last 5 decisions
    safe_cols = "final_action, vetoed, final_conf, agreement, entropy, block_reasons_json"
    if "mode" in sd_cols:
        safe_cols += ", mode"
    if "regime" in sd_cols:
        safe_cols += ", regime"
    recent_sd = conn.execute(
        f"SELECT {safe_cols} FROM swarm_decisions ORDER BY rowid DESC LIMIT 5"
    ).fetchall()
    print("\n  Last 5 swarm decisions:")
    for r in recent_sd:
        reasons = r["block_reasons_json"] or "[]"
        mode = r["mode"] if "mode" in sd_cols else "?"
        regime = r["regime"] if "regime" in sd_cols and r["regime"] else "?"
        print(f"    action={r['final_action']:>6} vetoed={r['vetoed']} "
              f"conf={r['final_conf']:.3f} agree={r['agreement']:.3f} "
              f"ent={r['entropy']:.3f} mode={mode} regime={regime}")
        if r["vetoed"] and reasons != "[]":
            print(f"      reasons: {reasons[:150]}")

    # Non-hold decisions
    wt = conn.execute(
        "SELECT COUNT(*) FROM swarm_decisions WHERE final_action != 'hold'"
    ).fetchone()[0]
    print(f"\n  Non-hold swarm decisions: {wt}")

# 4. Agent votes
av_cols = [r[1] for r in conn.execute("PRAGMA table_info(swarm_agent_votes)").fetchall()]
print(f"\nAgent votes columns: {av_cols}")
av_count = conn.execute("SELECT COUNT(*) FROM swarm_agent_votes").fetchone()[0]
print(f"Agent votes total: {av_count}")

if av_count > 0:
    # Find the ID column
    id_col = "swarm_decision_id" if "swarm_decision_id" in av_cols else "decision_id"
    if id_col not in av_cols:
        # Try to find any ID-like column
        for c in av_cols:
            if "id" in c.lower() and c != "agent_id":
                id_col = c
                break

    # Agent action distribution
    acts = conn.execute(
        "SELECT agent_id, action, COUNT(*) as cnt FROM swarm_agent_votes "
        "GROUP BY agent_id, action ORDER BY agent_id, cnt DESC"
    ).fetchall()
    print("  Agent action distribution:")
    for r in acts:
        print(f"    {r['agent_id']:>25}: {r['action']:>6} x{r['cnt']}")

    # Agent veto distribution
    vetos = conn.execute(
        "SELECT agent_id, SUM(veto) as veto_cnt, COUNT(*) as total "
        "FROM swarm_agent_votes GROUP BY agent_id"
    ).fetchall()
    print("\n  Agent veto rates:")
    for r in vetos:
        rate = r["veto_cnt"] / r["total"] if r["total"] else 0
        print(f"    {r['agent_id']:>25}: {r['veto_cnt']}/{r['total']} ({rate:.0%})")

    # Last decision's votes
    try:
        last_id = conn.execute(f"SELECT MAX({id_col}) FROM swarm_agent_votes").fetchone()[0]
        votes = conn.execute(
            f"SELECT agent_id, action, confidence, veto, veto_reason "
            f"FROM swarm_agent_votes WHERE {id_col} = ?",
            (last_id,),
        ).fetchall()
        print(f"\n  Votes for latest decision (id={last_id}):")
        for v in votes:
            vr = v["veto_reason"] or "-"
            print(f"    {v['agent_id']:>25}: action={v['action']:>6} "
                  f"conf={v['confidence']:.3f} veto={v['veto']} reason={vr[:80]}")
    except Exception as e:
        print(f"  Could not fetch latest votes: {e}")

# 5. Config check — is swarm enabled? is policy_core on?
print("\n--- Config from .env ---")
try:
    from dotenv import dotenv_values
    env = dotenv_values(".env")
    for key in sorted(env.keys()):
        if any(k in key.upper() for k in ["SWARM", "POLICY", "ML_FILTER", "ML_AS_SIZER",
                                            "PAPER", "LIVE", "USE_ML"]):
            print(f"  {key}={env[key]}")
except Exception as e:
    print(f"  .env read error: {e}")

conn.close()
