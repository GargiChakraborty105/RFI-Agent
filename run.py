import psycopg2

conn = psycopg2.connect("postgres://agentUser:Gargi102#@103.75.161.199:5432/rfi_agent")
cursor = conn.cursor()

# Example query
cursor.execute("SELECT * FROM Sales_data")
result = cursor.fetchall()
print(result)

cursor.close()
conn.close()