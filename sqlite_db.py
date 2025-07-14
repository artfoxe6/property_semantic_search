import sqlite3

from property import Property


class SqliteDB:
    def __init__(self, db_file="property.db"):
        self.conn = sqlite3.connect(db_file)
        self.create_table()

    def __del__(self):
        self.close()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bedrooms INTEGER,
                bathrooms INTEGER,
                carspaces INTEGER,
                floor INTEGER,
                area INTEGER,
                price REAL,
                province TEXT,
                city TEXT,
                district TEXT,
                build_year INTEGER,
                list_at TEXT,
                decoration TEXT,
                type TEXT,
                distance_to_metro INTEGER,
                distance_to_school INTEGER,
                description TEXT
            )
        ''')
        self.conn.commit()

    def add_property(self, prop: Property):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO properties (
                bedrooms, bathrooms, carspaces, floor, area, price, province,
                city, district, build_year, list_at, decoration, type,
                distance_to_metro, distance_to_school, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prop.bedrooms, prop.bathrooms, prop.carspaces, prop.floor, prop.area, prop.price,
            prop.province, prop.city, prop.district, prop.build_year, prop.list_at,
            prop.decoration, prop.type, prop.distance_to_metro, prop.distance_to_school,
            prop.description
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get(self, prop_id: int) -> Property | None:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM properties WHERE id = ?', (prop_id,))
        row = cursor.fetchone()
        if row:
            return Property(*row)
        return None

    def list(self, last_id: int = 0, limit: int = 1000) -> list[Property]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM properties
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
        ''', (last_id, limit))
        rows = cursor.fetchall()
        return [Property(*row) for row in rows]

    def close(self):
        self.conn.close()


