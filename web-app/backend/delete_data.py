"""Simple data cleanup utility for watermark_registry.db.

Examples:
  python delete_data.py --list
  python delete_data.py --delete-artwork ART-0006
  python delete_data.py --delete-verification VER-0008
  python delete_data.py --delete-selenium
  python delete_data.py --wipe-all --yes
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).parent / "watermark_registry.db"


def connect_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


def list_recent(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    artwork_count = cur.execute("select count(*) from artworks").fetchone()[0]
    verification_count = cur.execute("select count(*) from verifications").fetchone()[0]

    print(f"DB: {DB_PATH}")
    print(f"artworks: {artwork_count}")
    print(f"verifications: {verification_count}")

    print("\nRecent artworks:")
    for row in cur.execute(
        """
        select artwork_id, title, creator_name, registration_date
        from artworks
        order by registration_date desc
        limit 10
        """
    ).fetchall():
        print(row)

    print("\nRecent verifications:")
    for row in cur.execute(
        """
        select verification_id, artwork_id, result_status, verification_date
        from verifications
        order by verification_date desc
        limit 10
        """
    ).fetchall():
        print(row)


def delete_artwork(conn: sqlite3.Connection, artwork_id: str) -> None:
    cur = conn.cursor()
    cur.execute("delete from verifications where artwork_id = ?", (artwork_id,))
    deleted_verifications = cur.rowcount
    cur.execute("delete from artworks where artwork_id = ?", (artwork_id,))
    deleted_artworks = cur.rowcount
    conn.commit()
    print(
        f"Deleted artwork_id={artwork_id}: artworks={deleted_artworks}, related_verifications={deleted_verifications}"
    )


def delete_verification(conn: sqlite3.Connection, verification_id: str) -> None:
    cur = conn.cursor()
    cur.execute("delete from verifications where verification_id = ?", (verification_id,))
    deleted = cur.rowcount
    conn.commit()
    print(f"Deleted verification_id={verification_id}: verifications={deleted}")


def delete_selenium(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        delete from verifications
        where artwork_id in (
            select artwork_id from artworks where creator_name = 'Selenium Runner'
        )
        """
    )
    deleted_verifications = cur.rowcount
    cur.execute("delete from artworks where creator_name = 'Selenium Runner'")
    deleted_artworks = cur.rowcount
    conn.commit()
    print(
        f"Deleted selenium data: artworks={deleted_artworks}, verifications={deleted_verifications}"
    )


def wipe_all(conn: sqlite3.Connection, confirmed: bool) -> None:
    if not confirmed:
        print("Refused. Use --wipe-all --yes to confirm full deletion.")
        return

    cur = conn.cursor()
    cur.execute("delete from verifications")
    deleted_verifications = cur.rowcount
    cur.execute("delete from artworks")
    deleted_artworks = cur.rowcount
    conn.commit()
    print(f"Deleted all data: artworks={deleted_artworks}, verifications={deleted_verifications}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Easy deletion utility for watermark registry data")
    parser.add_argument("--list", action="store_true", help="Show counts and recent records")
    parser.add_argument("--delete-artwork", metavar="ARTWORK_ID", help="Delete one artwork and its related verifications")
    parser.add_argument("--delete-verification", metavar="VERIFICATION_ID", help="Delete one verification record")
    parser.add_argument("--delete-selenium", action="store_true", help="Delete records created by Selenium Runner")
    parser.add_argument("--wipe-all", action="store_true", help="Delete all artworks and verifications")
    parser.add_argument("--yes", action="store_true", help="Confirm dangerous actions like --wipe-all")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not any(
        [
            args.list,
            args.delete_artwork,
            args.delete_verification,
            args.delete_selenium,
            args.wipe_all,
        ]
    ):
        parser.print_help()
        return 0

    with connect_db() as conn:
        if args.list:
            list_recent(conn)
        if args.delete_artwork:
            delete_artwork(conn, args.delete_artwork)
        if args.delete_verification:
            delete_verification(conn, args.delete_verification)
        if args.delete_selenium:
            delete_selenium(conn)
        if args.wipe_all:
            wipe_all(conn, args.yes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
