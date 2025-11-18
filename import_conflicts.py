import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import SessionLocal
from app.models import ConflictIncident

CSV_PATH = "app/data/sri_lanka_elephant_conflict.csv"


def parse_timestamp(x):
    """
    Convert timestamp string → Python datetime.
    Handles formats:
    - 2022-01-01 23:00:00
    - 2022/01/01 23:00
    """
    if isinstance(x, datetime):
        return x

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(str(x).strip(), fmt)
        except:
            pass

    print(f"[WARNING] Could not parse timestamp: {x}")
    return None


def import_conflicts():
    print("📄 Loading CSV...")

    df = pd.read_csv(CSV_PATH)

    print("🗑️ Clearing old data...")
    db: Session = SessionLocal()
    db.query(ConflictIncident).delete()
    db.commit()

    print("📝 Importing rows into database...")

    count = 0
    for _, row in df.iterrows():

        ts = parse_timestamp(row["timestamp"])
        if ts is None:
            continue  # skip bad rows

        incident = ConflictIncident(
            timestamp=ts,
            location=row.get("village_name", "Unknown") or "Unknown",
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            district=row.get("district", "Unknown"),
            elephant_count=int(row.get("elephant_count", 0)),
            incident_type=row.get("incident_type", "unknown"),
            description=row.get("description", "")
        )

        db.add(incident)
        count += 1

    db.commit()
    db.close()

    print(f"✅ Successfully imported {count} conflict incidents.")


if __name__ == "__main__":
    import_conflicts()
