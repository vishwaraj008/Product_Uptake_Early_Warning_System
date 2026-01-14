import mysql.connector
import pandas as pd
from src.db.db_config import DB_CONFIG


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def write_prescriptions(df: pd.DataFrame):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("TRUNCATE TABLE prescriptions")

    insert_sql = """
        INSERT INTO prescriptions
        (date, product, region, units, price_per_unit, revenue, event_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    data = [
        (
            row.date,
            row.product,
            row.region,
            int(row.units),
            float(row.price_per_unit),
            float(row.revenue),
            row.event_type
        )
        for row in df.itertuples(index=False)
    ]

    cursor.executemany(insert_sql, data)
    conn.commit()
    conn.close()


def read_prescriptions(product=None, region=None):
    conn = get_connection()

    query = "SELECT * FROM prescriptions WHERE 1=1"
    params = []

    if product:
        query += " AND product = %s"
        params.append(product)

    if region:
        query += " AND region = %s"
        params.append(region)

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df
