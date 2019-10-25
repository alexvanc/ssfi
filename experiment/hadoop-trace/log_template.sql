drop table if exists `hadoop_log_template`;
create table `hadoop_log_template`(
        `id` int not null primary key auto_increment,
        `hash_key` varchar(255) unique not null
);
