# and historical_performance er jonye : 
SELECT 
    assignees, 
    ROUND(
        SUM(CASE WHEN updated_at <= due_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) AS historical_performance_score
FROM 
    rfis
WHERE 
    status = 'Closed'
GROUP BY 
    assignees;

# current workload er jonye sql query jodi run kori : 
SELECT 
    assignees, 
    COUNT(*) AS current_workload
FROM 
    rfi_data
WHERE 
    status IN ('Open')
GROUP BY 
    assignees;


#id, name, job_title, current workload, email, historical performance score, current workload