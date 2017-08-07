import csv, sqlite3
import pandas as pd

con = sqlite3.connect(":memory:")
cur = con.cursor() #allows Python to execute SQL commands

# process name-key data
df_name = pd.read_csv('data_files/names.txt',  encoding='latin-1', dtype={'county_code': str, 'district_code': str, 'school_code': str}) # upload FRPM data
df_name['cds'] = df_name['county_code'] + df_name['district_code'] + df_name['school_code']
df_name.to_sql('names', con, if_exists='replace', index=False)

# process frpm data
df_frpm = pd.read_csv('data_files/frpm.csv',  encoding='latin-1') # upload FRPM data
df_frpm.to_sql('frpm', con, if_exists='replace', index=False)

# process sbac data
df_sbac = pd.read_csv('data_files/sbac.txt', usecols= ['county_code', 'district_code', 'school_code', 'test_id', 'grade',
                                                 'percent_exceeded', 'percent_met', 'percent_met_and_above',
                                                 'percent_nearly_met', 'percent_not_met', 'students_with_scores'],  encoding='latin-1') # upload SBAC data
df_sbac = df_sbac[pd.notnull(df_sbac['percent_met'])]
df_sbac.to_sql('sbac', con, if_exists='replace', index=False)

# process staff fte/demo
df_dem = pd.read_csv('data_files/staffdemo.txt', sep = '\t',  encoding='latin-1')
df_dem.to_sql('dem', con, if_exists='replace', index=False)
df_fte = pd.read_csv('data_files/staff_fte.txt',sep='\t', encoding='latin-1')
df_fte = df_fte[df_fte['SchoolCode'] != 0]
df_fte.to_sql('fte', con, if_exists='replace', index=False)

cur.execute("CREATE TABLE staffing AS "
            "SELECT fte.DistrictCode as DistrictCode, fte.SchoolCode as SchoolCode, fte.CountyName as CountyName, SUM(FTETeaching)/100 as num_teachers, "
            "SUM(FTEadministrative)/100 as num_admin, SUM(YearsTeaching)/COUNT(YearsTeaching) as avg_exp, "
            "SUM(YearsInDistrict)/COUNT(YearsInDistrict) as avg_yrs_district "
            "FROM fte JOIN dem ON fte.RecID = dem.RecID "
            "GROUP BY fte.DistrictCode, fte.SchoolCode, fte.CountyName")

# process budget data
df_budget = pd.read_csv('data_files/budget.txt', encoding='latin-1')
df_budget = df_budget[df_budget['Function'] == 1000] # only instructional costs, for now
df_budget.to_sql('budget', con, if_exists='replace', index = False)

cur.execute("CREATE TABLE budget_district AS "
            "SELECT DCode, SUM(Value) as total_budget, total_enrollment FROM budget "
            "JOIN (SELECT district_code, SUM(enrollment) as total_enrollment FROM frpm "
            "GROUP BY district_code) ON budget.Dcode = district_code "
            "GROUP BY DCode") # create district budget for instructional costs, per district

# process EL data
df_el = pd.read_csv('data_files/el.txt', sep = '\t',  encoding='latin-1')
df_el.to_sql('el_raw', con, if_exists='replace', index = False)

cur.execute("CREATE TABLE el AS "
            "SELECT CDS, SUM(TOTAL_EL) FROM el_raw "
            "GROUP BY CDS")

# process dropout data
df_dropout = pd.read_csv('data_files/dropouts.txt', sep = '\t',  encoding='latin-1')
df_dropout.to_sql('dropout_raw', con, if_exists='replace', index = False)

cur.execute("CREATE TABLE dropout AS "
            "SELECT CDS_CODE, SUM(DTOT)+ SUM(D7) + SUM(D8) as total_dropouts FROM dropout_raw "
            "GROUP BY CDS_CODE")

# process suspension data
df_susp = pd.read_csv('data_files/susp.txt', sep = '\t',  encoding='latin-1')
df_susp.to_sql('susp_raw', con, if_exists='replace', index = False)

cur.execute("CREATE TABLE susp AS "
            "SELECT Cds, Name, SUM(TOTAL) as total_suspensions FROM susp_raw "
            "GROUP BY Cds")

# make new table
print("Merging Tables")
cur.execute("CREATE TABLE merged AS "
            "SELECT * FROM sbac JOIN frpm ON "
            "(sbac.county_code = frpm.county_code AND sbac.district_code = frpm.district_code "
            "AND sbac.school_code = frpm.school_code) "
            "JOIN names ON (sbac.county_code = names.county_code "
            "AND sbac.district_code = names.district_code AND sbac.school_code = names.school_code) "
            "JOIN staffing ON sbac.school_code = SchoolCode "
            "JOIN budget_district on sbac.district_code = DCode "
            "LEFT JOIN dropout ON dropout.CDS_CODE = names.cds "
            "LEFT JOIN susp ON susp.Cds = names.cds "
            "WHERE educational_option = 'Traditional'")

# filtering elementary school data
print("Processing elementary school data")
cur.execute("CREATE TABLE elementary_school AS "
            "SELECT school_name, CASE WHEN test_id = 1 THEN 'ela' else 'math' END as test_id, "
            "enrollment, CASE WHEN charter = 'Y' then 'True' else 'False' END AS charter, "
            "f_percent, fr_percent, enrollment/num_teachers as student_teacher_ratio, enrollment/num_admin as student_admin_ratio, "
            "avg_exp, avg_yrs_district, CASE WHEN total_dropouts IS NULL then 0 ELSE total_dropouts/enrollment END as dropouts_per_enrollment, "
            "CASE WHEN total_suspensions IS NULL then 0 ELSE total_suspensions/enrollment END as susp_per_enrollment, "
            "total_budget/total_enrollment as district_budget_per_student,"
            "SUM(percent_met_and_above * students_with_scores * 0.01)/SUM(students_with_scores) "
            "as percent_met_and_above FROM merged "
            "WHERE high_grade IN (4,5,6) "
            "GROUP BY school_name, test_id")

# filtering middle school data
print("Processing middle school data")
cur.execute("CREATE TABLE middle_school AS "
            "SELECT school_name, CASE WHEN test_id = 1 THEN 'ela' else 'math' END as test_id, "
            "enrollment, CASE WHEN charter = 'Y' then 'True' else 'False' END AS charter, "
            "f_percent, fr_percent, enrollment/num_teachers as student_teacher_ratio, enrollment/num_admin as student_admin_ratio, "
            "avg_exp, avg_yrs_district, "
            "CASE WHEN total_suspensions IS NULL then 0 ELSE total_suspensions/enrollment END as susp_per_enrollment, "
            "total_budget/total_enrollment as district_budget_per_student,"
            "SUM(percent_met_and_above * students_with_scores * 0.01)/SUM(students_with_scores) "
            "as percent_met_and_above FROM merged "
            "WHERE low_grade IN (4,5,6) AND high_grade in (7,8)"
            "GROUP BY school_name, test_id")

# filtering high school data
print("Processing high school data")
cur.execute("CREATE TABLE high_school AS "
            "SELECT school_name, CASE WHEN test_id = 1 THEN 'ela' else 'math' END as test_id, "
            "enrollment, CASE WHEN charter = 'Y' then 'True' else 'False' END AS charter, "
            "f_percent, fr_percent, enrollment/num_teachers as student_teacher_ratio, enrollment/num_admin as student_admin_ratio, "
            "avg_exp, avg_yrs_district, CASE WHEN total_dropouts IS NULL then 0 ELSE total_dropouts/enrollment END as dropouts_per_enrollment, "
            "CASE WHEN total_suspensions IS NULL then 0 ELSE total_suspensions/enrollment END as susp_per_enrollment, "
            "total_budget/total_enrollment as district_budget_per_student,"
            "SUM(percent_met_and_above * students_with_scores * 0.01)/SUM(students_with_scores) "
            "as percent_met_and_above FROM merged "
            "WHERE high_grade = 12 "
            "GROUP BY school_name, test_id")

# write to csv
elem_school = cur.execute("SELECT * FROM elementary_school")
with open("elem_school.csv", "wb") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([i[0] for i in elem_school.description])
    csv_writer.writerows(elem_school)

middle_school = cur.execute("SELECT * FROM middle_school")
with open("middle_school.csv", "wb") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([i[0] for i in middle_school.description])
    csv_writer.writerows(middle_school)

high_school = cur.execute("SELECT * FROM high_school")
with open("high_school.csv", "wb") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([i[0] for i in high_school.description])
    csv_writer.writerows(high_school)

# print
print ("fetchall:")
result = cur.fetchall()
for r in result:
    print(r)
