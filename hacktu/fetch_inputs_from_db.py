import psycopg2
import json

def fetch_inputs():
    # ...existing code for connection parameters...
    conn = psycopg2.connect(
        host="localhost",         # replace with your DB host
        database="your_database", # replace with your DB name
        user="your_user",         # replace with your DB user
        password="your_password"  # replace with your DB password
    )
    cur = conn.cursor()
    cur.execute("SELECT * FROM loan_application;")
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    data = [dict(zip(columns, row)) for row in rows]
    cur.close()
    conn.close()
    return data

if __name__ == "__main__":
    inputs = fetch_inputs()
    with open('db_inputs.json', 'w') as f:
        json.dump(inputs, f, indent=2)
    print("Database inputs have been saved to db_inputs.json")
