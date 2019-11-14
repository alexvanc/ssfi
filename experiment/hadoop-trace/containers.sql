drop table if exists `hadoop_containers`;
create table `hadoop_containers`(
        `id` int not null primary key auto_increment,
        `full_name` varchar(511) unique not null,
        `short_name` varchar(255) unique not null,
        `start_time` DATETIME,
        `end_time` DATETIME,
        `fault_type` varchar(255),
        `activation_mode` varchar(255)
);
