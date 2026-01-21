SELECT * FROM M_SQL_25S1_25_09_29
WHERE COL1 = 'A' AND COL2 = 'B';
-- This SQL query selects all columns from the table M_SQL_25S1_25_09_29
-- where the value in COL1 is 'A' and the value in COL2 is 'B'.
-- It is used to filter records based on specific criteria in these two columns.      

create database sth;
use sth;
create table t1(
    'w1' INT PRIMARY KEY,
    'w2' VARCHAR(20),
    'w3' DATE
    'log' DECIMAL(10,2),
    'flag' BOOLEAN,
    'major' VARCHAR(50) 
    'i1' INT NOT NULL,
    'i2' INT DEFAULT 0,
    'i3' INT UNIQUE,
    'i4' INT AUTO_INCREMENT,
    'l' INT AUTO_ONCREMENT PRIMARY KEY,
    'l2' INT UNIQUE,
    'l3' INT NOT NULL
    'o2' INT DEFAULT 0
);

insert into t1 values(1,'hello','2023-09-29',100.50,true,'CS',10,0,20,NULL,30,40,50,0);
insert into t1 values(2,'world','2023-09-30',200.75
,false,'Math',15,0,25,NULL,35,45,55,0);

describe t1;
DROP TABLE t1;
ALTER TABLE t1 ADD gap INT DEFAULT 0;
ALTER TABLE t1 DROP COLUMN gap;
ALTER TABLE t1 MODIFY COLUMNS w2 VARCHAR(40);
