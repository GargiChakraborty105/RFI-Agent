import psycopg2

class Uploader:
    def __init__(self):
        self.conn = psycopg2.connect("postgresql://palcode_ai:blaash@dpg-ctohkfd2ng1s73biuvr0-a.oregon-postgres.render.com/rfi_agent")
        self.cursor = self.conn.cursor()
    def rfi_uploader(self,datas):
        table = "rfis"
        attributes = [
            'id', 'initiated_at', 'subject', 'status', 'priority_name', 'created_at', 'updated_at', 'due_date', 'questions_body', 'assignees_name', 'project_id', 'assignees_id', 'priority_value'
        ]
        for data in datas:
            try:
                values = []
                for x in attributes:
                    print(f'operating on {x}')
                    values.append(data[x])
                    print(type(data[x]))
                
                columns = ', '.join(attributes)
                placeholders = ', '.join(['%s'] * len(attributes))
                values = tuple(values)
                print(len(columns),len(values))
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders});"
                
                self.cursor.execute(query, values)
                
                self.conn.commit()

                print("Data inserted successfully!")
            except psycopg2.IntegrityError as e:
                self.conn.rollback()
                print(f"Already existed data: {e}")
            
            except Exception as e:
                self.conn.rollback()
                print(f"An unexpected error occurred: {e}")


    def projects_uploader(self,datas):
        table = "projects"
        attributes = [
            'id', 'name', 'start_date', 'completion_date', 'address', 'city', 'state_code', 'country_code', 'zip','created_at', 'active','updated_at', 'company_id', 'company_name'
        ]
        for data in datas:
            try:
                values = []
                for x in attributes:
                    print(f'operating on {x}')
                    values.append(data[x])
                    print(type(data[x]))
                
                columns = ', '.join(attributes)
                placeholders = ', '.join(['%s'] * len(attributes))
                values = tuple(values)
                print(len(columns),len(values))
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders});"
                
                self.cursor.execute(query, values)
                
                self.conn.commit()

                print("Data inserted successfully!")
            except psycopg2.IntegrityError as e:
                self.conn.rollback()
                print(f"Already existed data: {e}")
            
            except Exception as e:
                self.conn.rollback()
                print(f"An unexpected error occurred: {e}")
    
    def user_uploader(self, datas):
        table = "user_data"
        attributes = [
            'id', 'name', 'email_address', 'mobile_phone', 'job_title', 'current_workload', 'historical_performance_score', 'company_id'
        ]
        for data in datas:
            try:
                values = []
                print(f'data: {data}')
                for x in attributes:
                    print(f'operating on {x}')
                    values.append(data[x])
                    print(type(data[x]))
                
                columns = ', '.join(attributes)
                placeholders = ', '.join(['%s'] * len(attributes))
                values = tuple(values)
                print(len(columns),len(values))
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders});"
                
                self.cursor.execute(query, values)
                
                self.conn.commit()

                print("Data inserted successfully!")
            except psycopg2.IntegrityError as e:
                print(f'data: {data}')
                self.conn.rollback()
                print(f"Already existed data: {e}")
            
            except Exception as e:
                print(f'data: {data}')
                self.conn.rollback()
                print(f"An unexpected error occurred: {e}")
        
    def close_connection(self):
        self.cursor.close()
        self.conn.close()
        return "Connection Closed Successfully"

class Fetcher:

    def __init__(self):
        self.conn = psycopg2.connect("postgresql://palcode_ai:blaash@dpg-ctohkfd2ng1s73biuvr0-a.oregon-postgres.render.com/rfi_agent")
        self.cursor = self.conn.cursor()

    def list_rfi(self, project_id):
        self.cursor.execute(f"SELECT * FROM rfis WHERE project_id={project_id}")
        rfi_fields = [desc[0] for desc in self.cursor.description]
        rfi_instances = self.cursor.fetchall()
        rfis = []
        for rfi_instance in rfi_instances:
            result = {c: r for c,r in zip(rfi_fields,list(rfi_instance))}
            rfis.append(result)
        self.cursor.execute(f"SELECT * FROM projects WHERE id={project_id}")
        project_fields = [desc[0] for desc in self.cursor.description]
        project_instance = self.cursor.fetchall()
        result = {c:r for c,r in zip(project_fields, project_instance[0])}
        result["rfis"] = rfis
        return result
    
    def list_projects(self, company_id):
        self.cursor.execute(f"SELECT name,id FROM projects WHERE company_id={company_id}")
        project_fields = [desc[0] for desc in self.cursor.description]
        project_instances = self.cursor.fetchall()
        projects = []
        for project_instance in project_instances:
            result = {c: r for c,r in zip(project_fields,list(project_instance))}
            projects.append(result)
        return projects

    def current_workload_calculator(self,user_id):
        query = """
        SELECT 
            COUNT(*) AS current_workload
        FROM 
            rfis
        WHERE 
            status = 'open'
            AND %s = ANY(assignees_id); 
        """

        # Define the specific assignee ID you want to query  # Replace with your desired ID

        # Execute the query with the parameter
        self.cursor.execute(query, (user_id,))

        self.conn.commit()

        current_workload = self.cursor.fetchall()

        return current_workload[0][0]

    def historical_performance_calculator(self,user_id):
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


        # Define the specific assignee ID you want to query  # Replace with your desired ID

        # Execute the query with the parameter
        self.cursor.execute(query, (user_id,))

        self.conn.commit()

        performance = self.cursor.fetchall()

        return performance[0][0]


    def close_connection(self):
        self.cursor.close()
        self.conn.close()
        return "Connection Closed Successfully"


# upload = Fetcher()
# # d = [{
# #             'id' : 103, 
# #             'initiated_at': "28-12-2024",
# #             'subject': "Furnishing", 
# #             'status' : "open", 
# #             'priority': None, 
# #             'created_at': "27-12-2024", 
# #             'updated_at': "29-12-2024",
# #             'due_date': "01-01-2025", 
# #             'questions': None, 
# #             'assignees' : ["Alipa","golu"],
# #             'remaining_time' : None,
# #             'Message' : 'I love You',
# #             'To' : 'Alipa',
# #             'project_id' : 1002
# #     },
# #     {
# #             'id' : 69, 
# #             'initiated_at': "28-12-2024",
# #             'subject': "Furnishing", 
# #             'status' : "open", 
# #             'priority': None, 
# #             'created_at': "27-12-2024", 
# #             'updated_at': "29-12-2024",
# #             'due_date': "01-01-2025", 
# #             'questions': None, 
# #             'assignees' : ["Alipa","golu"],
# #             'project_id' : 1002,
# #             'remaining_time' : None,
# #             'Message' : 'I love You',
# #             'To' : 'Alipa' 
            
# #     }]
# # upload.rfi_uploader(d)
# print(upload.list_rfi(125079))