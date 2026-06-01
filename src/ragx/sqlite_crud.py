import sqlite3


class SQLiteCRUD:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def create_table(self, table_name, columns):
        cols = ", ".join(
            [f"{col} {dtype}" for col, dtype in columns.items()]
        )

        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} ({cols})
        """

        self.cursor.execute(query)
        self.conn.commit()

    def insert(self, table_name, data):
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        query = f"""
        INSERT INTO {table_name} ({columns})
        VALUES ({placeholders})
        """

        self.cursor.execute(query, tuple(data.values()))
        self.conn.commit()

        return self.cursor.lastrowid

    def insert_many(self, table_name, data_list):
        """
        data_list = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30}
        ]
        """

        if not data_list:
            return 0

        columns = list(data_list[0].keys())

        column_names = ", ".join(columns)
        placeholders = ", ".join(["?"] * len(columns))

        query = f"""
        INSERT INTO {table_name} ({column_names})
        VALUES ({placeholders})
        """

        values = [
            tuple(item[col] for col in columns)
            for item in data_list
        ]

        self.cursor.executemany(query, values)
        self.conn.commit()

        return self.cursor.rowcount

    def get_all(self, table_name):
        query = f"SELECT * FROM {table_name}"

        self.cursor.execute(query)

        return [dict(row) for row in self.cursor.fetchall()]

    def get_by_id(self, table_name, record_id):
        query = f"""
        SELECT * FROM {table_name}
        WHERE id = ?
        """

        self.cursor.execute(query, (record_id,))

        row = self.cursor.fetchone()

        return dict(row) if row else None

    def get_by_ids(self, table_name, ids):
        """
        ids = [1, 2, 3]
        """
        placeholders = ", ".join(["?"] * len(ids))

        query = f"""
        SELECT * FROM {table_name}
        WHERE id IN ({placeholders})
        """

        self.cursor.execute(query, tuple(ids))

        return [dict(row) for row in self.cursor.fetchall()]

    def get_where(self, table_name, column, value):
        """
        Example:
        get_where("users", "name", "Alice")
        """
        query = f"""
        SELECT * FROM {table_name}
        WHERE {column} = ?
        """

        self.cursor.execute(query, (value,))

        return [dict(row) for row in self.cursor.fetchall()]

    def update(self, table_name, record_id, data):
        set_clause = ", ".join(
            [f"{key} = ?" for key in data.keys()]
        )

        query = f"""
        UPDATE {table_name}
        SET {set_clause}
        WHERE id = ?
        """

        values = list(data.values())
        values.append(record_id)

        self.cursor.execute(query, tuple(values))
        self.conn.commit()

        return self.cursor.rowcount

    def delete(self, table_name, record_id):
        query = f"""
        DELETE FROM {table_name}
        WHERE id = ?
        """

        self.cursor.execute(query, (record_id,))
        self.conn.commit()

        return self.cursor.rowcount

    def close(self):
        self.conn.close()