import psycopg2

class Uploader:
    def __init__(self):
        self.conn = psycopg2.connect("postgresql://alipa:Gargi1002#@dpg-ctohkfd2ng1s73biuvr0-a.oregon-postgres.render.com/rfi_agent")
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
                
                # Create the column names and values dynamically from the dictionary
                columns = ', '.join(attributes)
                placeholders = ', '.join(['%s'] * len(attributes))
                values = tuple(values)
                print(len(columns),len(values))
                # Formulate the INSERT query
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders});"
                
                # Execute the query
                self.cursor.execute(query, values)
                
                # Commit the transaction
                self.conn.commit()

                print("Data inserted successfully!")
            except psycopg2.IntegrityError as e:
            # Rollback the transaction for the failed insert
                self.conn.rollback()
                print(f"Already existed data: {e}")
            
            # except Exception as e:
            #     # Handle other potential exceptions
            #     self.conn.rollback()
            #     print(f"An unexpected error occurred: {e}")


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
                
                # Create the column names and values dynamically from the dictionary
                columns = ', '.join(attributes)
                placeholders = ', '.join(['%s'] * len(attributes))
                values = tuple(values)
                print(len(columns),len(values))
                # Formulate the INSERT query
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders});"
                
                # Execute the query
                self.cursor.execute(query, values)
                
                # Commit the transaction
                self.conn.commit()

                print("Data inserted successfully!")
            except psycopg2.IntegrityError as e:
            # Rollback the transaction for the failed insert
                self.conn.rollback()
                print(f"Already existed data: {e}")
            
            # except Exception as e:
            #     # Handle other potential exceptions
            #     self.conn.rollback()
            #     print(f"An unexpected error occurred: {e}")
        
        
        self.cursor.close()
        self.conn.close()
# upload = Uploader()
# d = [{
#             'id' : 103, 
#             'initiated_at': "28-12-2024",
#             'subject': "Furnishing", 
#             'status' : "open", 
#             'priority': None, 
#             'created_at': "27-12-2024", 
#             'updated_at': "29-12-2024",
#             'due_date': "01-01-2025", 
#             'questions': None, 
#             'assignees' : ["Alipa","golu"],
#             'remaining_time' : None,
#             'Message' : 'I love You',
#             'To' : 'Alipa',
#             'project_id' : 1002
#     },
#     {
#             'id' : 69, 
#             'initiated_at': "28-12-2024",
#             'subject': "Furnishing", 
#             'status' : "open", 
#             'priority': None, 
#             'created_at': "27-12-2024", 
#             'updated_at': "29-12-2024",
#             'due_date': "01-01-2025", 
#             'questions': None, 
#             'assignees' : ["Alipa","golu"],
#             'project_id' : 1002,
#             'remaining_time' : None,
#             'Message' : 'I love You',
#             'To' : 'Alipa' 
            
#     }]
# upload.rfi_uploader(d)