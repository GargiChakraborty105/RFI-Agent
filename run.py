import psycopg2

conn = psycopg2.connect("postgresql://alipa:Gargi1002#@dpg-ctohkfd2ng1s73biuvr0-a.oregon-postgres.render.com/rfi_agent")
cursor = conn.cursor()

# Example query
cursor.execute("SELECT * FROM projects")
result = cursor.fetchall()
print(result)

cursor.close()
conn.close()