-- 1. Create a slicer for project interest variable
-- 2. Create a slicer for career role
-- 3. Create a slicer for batch number
USE license_status; 
SELECT * from career_interest;
-- 4. Check the count of Gender in percentage
SELECT sum(case when `Gender` = 'Male' then 1 else 0 end)*100/count(*) as Male_Percentage,
       sum(case when `Gender` = 'Female' then 1 else 0 end)*100/count(*) as Female_Percentage
FROM career_interest;


-- 5. Check the % ratio for each project interest and career role
-- CAREER ROLE
SELECT sum(case when `career_role` = 'Data Scientist' then 1 else 0 end)/count(*) as Data_Scientist_Ratio,
       sum(case when `career_role` = 'Data Analyst' then 1 else 0 end)/count(*) as Data_Analyst_Ratio
FROM career_interest;
-- PROJECT INTEREST
SELECT sum(case when `Project_Interest` = 'ML' then 1 else 0 end)/count(*) as ML_Ratio,
       sum(case when `Project_Interest` = 'DL' then 1 else 0 end)/count(*) as DL_Ratio,
       sum(case when `Project_Interest` = 'SQL' then 1 else 0 end)/count(*) as SQL_Ratio,
       sum(case when `Project_Interest` = 'PBI' then 1 else 0 end)/count(*) as PBI_Ratio,
       sum(case when `Project_Interest` = 'NLP' then 1 else 0 end)/count(*) as NLP_Ratio
FROM career_interest;

-- 6. Check the total count of unique responses received
SELECT Count(Distinct RespondentID) Unique_Responses FROM career_Interest;

--7. Check the total responses received
SELECT Count(RespondentID) Total_Responses FROM career_Interest;

--8. Check what are the total batch count
SELECT Count(Distinct Batch_number) Total_Batches FROM career_Interest;

--9. Add your own flavour of ideas in the dashboard and queries
--What are the project preference of students by career role
SELECT career_role, Project_Interest, COUNT(Project_Interest) Count_project_Preference FROM career_Interest
Group By career_role, Project_Interest;

--What are the project preference of students by gender and career role
SELECT career_role, Project_Interest, Gender, COUNT(Project_Interest) Count_project_Preference FROM career_Interest
Group By career_role, Project_Interest, Gender;

--What are the career role of students by gender and career role
SELECT career_role, Gender, COUNT(Career_role) Count_career_role FROM career_Interest
Group By career_role, Gender;

--What are the career role of students by gender and career role
SELECT career_role, COUNT(Career_role) Count_career_role FROM career_Interest
Group By career_role;

--What are the count of project preference
SELECT Project_Interest, COUNT(Project_interest) Count_Project_interest FROM career_Interest
Group By Project_Interest;