drop table if exists NV_product_id;
CREATE TABLE NV_product_id (select product_class_id, product_subcategory from product_class 
where product_department='Seafood' 
or product_department='Dairy' 
or product_department='Produce'
or product_department='Egg'
or product_department='Deli');
 
DELIMITER $$
DROP TABLE IF EXISTS NVProductlogs;
DROP PROCEDURE IF EXISTS `NV_product` $$
CREATE PROCEDURE `NV_product`()
BEGIN
	DECLARE no_more_products int default 0;
	DECLARE productCID int(11);
	DECLARE cur CURSOR FOR SELECT product_class_id FROM NV_product_id;
	DECLARE CONTINUE HANDLER FOR NOT FOUND SET no_more_products=1; 
	CREATE TABLE NVProductlogs (  
	Id int(11) NOT NULL AUTO_INCREMENT,  
	ProductID int(11) NOT NULL,
	Recy INT,
	LowFat INT,
	PRIMARY KEY (Id)  
	);

	open cur;
	FETCH cur INTO productCID;
	REPEAT
		INSERT INTO NVProductlogs (ProductID,Recy,LowFat) (
		select product_id, recyclable_package, low_fat 
		from product 
		where product_class_id=productCID);
	FETCH cur INTO productCID;
	UNTIL no_more_products=1 END REPEAT;  
	CLOSE cur;
END $$
DELIMITER;
CALL NV_product();

DELIMITER $$
DROP TABLE IF EXISTS NVProductSales1997;
DROP PROCEDURE IF EXISTS `NV_product_sales_1997` $$
CREATE PROCEDURE `NV_product_sales_1997`()
BEGIN
	DECLARE no_more_products1997 INT DEFAULT 0;
	DECLARE productID1997 INT(11);
	DECLARE recycle INT(11);
	DECLARE lowFat INT(11);
	DECLARE cur1997 CURSOR FOR SELECT ProductID, Recy, LowFat FROM NVProductlogs;
	DECLARE CONTINUE HANDLER FOR NOT FOUND SET no_more_products1997=1;
	
	CREATE TABLE NVProductSales1997(
	Id int(11) NOT NULL AUTO_INCREMENT,
	Time_ID int(11),
	Store_ID int(11),
	Product_ID int(11),
	Recycle int(11),
	Low_Fat int(11),
	Promotion_ID int(11),
	Unit_Sales int(11),
	The_Day VARCHAR(255),
	Week_of_Year int(11),
	Month_of_Year int(11),
	PRIMARY KEY (Id)
	);
	
	open cur1997;
	FETCH cur1997 INTO productID1997,recycle,lowFat;
	REPEAT
		INSERT INTO NVproductSales1997 (Time_ID, Store_ID, Product_ID, Promotion_ID, Unit_Sales) (
		SELECT time_id, store_id, product_id, promotion_id, unit_sales 
		FROM sales_fact_1997
		WHERE product_id=productID1997);
		FETCH cur1997 INTO productID1997,recycle,lowFat;
	UNTIL no_more_products1997=1 END REPEAT;
	CLOSE cur1997;
END $$
DELIMITER;
CALL NV_product_sales_1997;

UPDATE nvproductsales1997
SET Week_of_Year=(
SELECT week_of_year 
FROM time_by_day 
where time_by_day.time_id=nvproductsales1997.Time_ID);

UPDATE nvproductsales1997
SET Month_of_Year=(
SELECT month_of_year 
FROM time_by_day 
where time_by_day.time_id=nvproductsales1997.Time_ID);

DELIMITER $$
DROP TABLE IF EXISTS NVProductSales1997_2;
DROP PROCEDURE IF EXISTS `NV_product_sales_1997_2` $$
CREATE PROCEDURE `NV_product_sales_1997_2`()
BEGIN
	DECLARE no_more_products1997_2 INT DEFAULT 0;
	DECLARE productID1997 INT(11);
	DECLARE recycle INT(11);
	DECLARE lowFat INT(11);
	DECLARE cur1997_2 CURSOR FOR SELECT ProductID, Recy, LowFat FROM NVProductlogs;
	DECLARE CONTINUE HANDLER FOR NOT FOUND SET no_more_products1997_2=1;
	
	CREATE TABLE NVProductSales1997_2(
	ID int(11) NOT NULL AUTO_INCREMENT,
	MONTH_OF_YEAR int(11),
	PRODUCT_ID int(11),
	RECYCLE int(11),
	LOW_FAT int(11),
	TOTAL_UNIT_SALES int(11),
	PRIMARY KEY (Id)
	);
	
	open cur1997_2;
	FETCH cur1997_2 INTO productID1997,recycle,lowFat;
	REPEAT
		INSERT INTO NVproductSales1997_2 (MONTH_OF_YEAR, PRODUCT_ID, TOTAL_UNIT_SALES) (
		SELECT Month_of_Year, Product_ID, sum(Unit_Sales) as total
		FROM nvproductsales1997 
		WHERE Product_ID=productID1997 
		GROUP BY Month_of_Year);
		FETCH cur1997_2 INTO productID1997,recycle,lowFat;
	UNTIL no_more_products1997_2=1 END REPEAT;
	CLOSE cur1997_2;
END $$
DELIMITER;
CALL NV_product_sales_1997_2;
-- DROP TABLE IF EXISTS NVProductSales1997;

UPDATE nvproductsales1997_2 
SET RECYCLE=(
SELECT Recy 
FROM nvproductlogs 
where nvproductlogs.ProductID=nvproductsales1997_2.Product_ID);

UPDATE nvproductsales1997_2 
SET LOW_FAT=(
SELECT LowFat 
FROM nvproductlogs 
where nvproductlogs.ProductID=nvproductsales1997_2.Product_ID);


DELIMITER $$
DROP TABLE IF EXISTS NVProductSales1998;
DROP PROCEDURE IF EXISTS `NV_product_sales_1998` $$
CREATE PROCEDURE `NV_product_sales_1998`()
BEGIN
	DECLARE no_more_products1998 INT DEFAULT 0;
	DECLARE productID1998 INT(11);
	DECLARE recycle INT(11);
	DECLARE lowFat INT(11);
	DECLARE cur1998 CURSOR FOR SELECT ProductID, Recy, LowFat FROM NVProductlogs;
	DECLARE CONTINUE HANDLER FOR NOT FOUND SET no_more_products1998=1;
	
	CREATE TABLE NVProductSales1998(
	Id int(11) NOT NULL AUTO_INCREMENT,
	Time_ID int(11),
	Store_ID int(11),
	Product_ID int(11),
	Recycle int(11),
	Low_Fat int(11),
	Promotion_ID int(11),
	Unit_Sales int(11),
	The_Day VARCHAR(255),
	Week_of_Year int(11),
	Month_of_Year int(11),
	PRIMARY KEY (Id)
	);
	
	open cur1998;
	FETCH cur1998 INTO productID1998,recycle,lowFat;
	REPEAT
		INSERT INTO NVproductSales1998 (Time_ID, Store_ID, Product_ID, Promotion_ID, Unit_Sales) (
		SELECT time_id, store_id, product_id, promotion_id, unit_sales 
		FROM sales_fact_1998
		WHERE product_id=productID1998);
		FETCH cur1998 INTO productID1998,recycle,lowFat;
	UNTIL no_more_products1998=1 END REPEAT;
	CLOSE cur1998;
END $$
DELIMITER;
CALL NV_product_sales_1998;

UPDATE nvproductsales1998
SET Week_of_Year=(
SELECT week_of_year 
FROM time_by_day 
where time_by_day.time_id=nvproductsales1998.Time_ID);

UPDATE nvproductsales1998
SET Month_of_Year=(
SELECT month_of_year 
FROM time_by_day 
where time_by_day.time_id=nvproductsales1998.Time_ID);

UPDATE nvproductsales1998
SET RECYCLE=(
SELECT Recy 
FROM nvproductlogs 
where nvproductlogs.ProductID=nvproductsales1998.Product_ID);

UPDATE nvproductsales1998 
SET LOW_FAT=(
SELECT LowFat
FROM nvproductlogs
where nvproductlogs.ProductID=nvproductsales1998.Product_ID);

UPDATE nvproductsales1998
SET The_Day=(
SELECT the_day
FROM time_by_day
WHERE time_by_day.time_id=nvproductsales1998.Time_ID
);

-- select * from nvproductsales1998;

DELIMITER $$
DROP TABLE IF EXISTS NVProductSales1998_2;
DROP PROCEDURE IF EXISTS `NV_product_sales_1998_2` $$
CREATE PROCEDURE `NV_product_sales_1998_2`()
BEGIN
	DECLARE no_more_products1998_2 INT DEFAULT 0;
	DECLARE productID1998 INT(11);
	DECLARE recycle INT(11);
	DECLARE lowFat INT(11);
	DECLARE cur1998_2 CURSOR FOR SELECT ProductID, Recy, LowFat FROM NVProductlogs;
	DECLARE CONTINUE HANDLER FOR NOT FOUND SET no_more_products1998_2=1;
	
	CREATE TABLE NVProductSales1998_2(
	ID int(11) NOT NULL AUTO_INCREMENT,
	MONTH_OF_YEAR int(11),
	PRODUCT_ID int(11),
	RECYCLE int(11),
	LOW_FAT int(11),
	TOTAL_UNIT_SALES int(11),
	PRIMARY KEY (Id)
	);
	
	open cur1998_2;
	FETCH cur1998_2 INTO productID1998,recycle,lowFat;
	REPEAT
		INSERT INTO NVproductSales1998_2 (MONTH_OF_YEAR, PRODUCT_ID, TOTAL_UNIT_SALES) (
		SELECT Month_of_Year, Product_ID, sum(Unit_Sales) as total
		FROM nvproductsales1998 
		WHERE Product_ID=productID1998 
		GROUP BY Month_of_Year);
		FETCH cur1998_2 INTO productID1998,recycle,lowFat;
	UNTIL no_more_products1998_2=1 END REPEAT;
	CLOSE cur1998_2;
END $$
DELIMITER;
CALL NV_product_sales_1998_2;
-- DROP TABLE IF EXISTS NVProductSales1998;

UPDATE nvproductsales1998_2 
SET RECYCLE=(
SELECT Recy 
FROM nvproductlogs 
where nvproductlogs.ProductID=nvproductsales1998_2.Product_ID);

UPDATE nvproductsales1998_2 
SET LOW_FAT=(
SELECT LowFat
FROM nvproductlogs
where nvproductlogs.ProductID=nvproductsales1998_2.Product_ID);

select * from nvproductsales1998_2