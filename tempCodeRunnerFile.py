import psycopg2

conn = psycopg2.connect("postgresql://palcode_ai:blaash@dpg-ctohkfd2ng1s73biuvr0-a.oregon-postgres.render.com/rfi_agent")
cursor = conn.cursor()


# Define the query with a parameterized placeholder
query = """
SELECT 
    ROUND(
        SUM(CASE WHEN updated_at <= due_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) AS historical_performance_score
FROM 
    rfis,
    UNNEST(assignees_id) AS individual_assignee_id
WHERE 
    status = 'closed'
    AND individual_assignee_id = %s -- Filter for the specific user ID
"""

# Define the specific assignee ID you want to query
assignee_id = 137269  # Replace with your desired ID

# Execute the query with the parameter
cursor.execute(query, (assignee_id,))

conn.commit()

project_instance = cursor.fetchall()

print(project_instance[0][0])

cursor.close()

conn.close()