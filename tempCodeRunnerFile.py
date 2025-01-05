import psycopg2

conn = psycopg2.connect("postgresql://palcode_ai:blaash@dpg-ctohkfd2ng1s73biuvr0-a.oregon-postgres.render.com/rfi_agent")
cursor = conn.cursor()


# Define the query with a parameterized placeholder
query = """
SELECT 
    COUNT(*) AS current_workload
FROM 
    rfis
WHERE 
    status = 'open'
    AND %s = ANY(assignees_id); 
"""

# Define the specific assignee ID you want to query
assignee_id = 137269  # Replace with your desired ID

# Execute the query with the parameter
cursor.execute(query, (assignee_id,))

conn.commit()

project_instance = cursor.fetchall()

print(project_instance)

cursor.close()

conn.close()