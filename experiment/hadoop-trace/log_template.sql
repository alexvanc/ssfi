drop table if exists `hadoop_log_template`;
create table `hadoop_log_template`(
        `id` int not null primary key auto_increment,
        `hash_key` varchar(255) unique not null,
        `level` varchar(255),
        `tmpl_content` text,
        `source_fi` varchar(255),
        `source_file` text,
        `length` int,
        `token` text
);
