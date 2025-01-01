from flask import Flask, request, jsonify
from utils.sqlOperator import Fetcher

app = Flask(__name__)

@app.route('/rfis', methods=['GET'])
def rfis():
    try:
        project_id = request.args.get('project_id')    #    Query parameter: '{base_url}?project_id={project_id}
        fetch = Fetcher()
        result = fetch.list_rfi(project_id)
        fetch.close_connection()
        return {"status": "success", "data": result}, 200

    except Exception as e:
        print("Error processing webhook:", e)
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/projects', methods=['GET'])
def projects():
    try:
        company_id = request.args.get('company_id')    #    Query parameter: '{base_url}?company_id={company_id}
        fetch = Fetcher()
        result = fetch.list_projects(company_id)
        fetch.close_connection()
        return {"status": "success", "data": result}, 200

    except Exception as e:
        print("Error processing webhook:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
