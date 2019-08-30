drop table if exists `injection_record_weka`;
create table `injection_record_weka`(
        `id` int not null primary key auto_increment,
        `fault_id` varchar(255) not null,
        `fault_type` varchar(255) not null,
        `activation_mode` varchar(255) not null,
        `package` text,
        `class` text,
        `method` text,
        `variable` varchar(255),
        `variable_type` varchar(255),
        `scope` varchar(255),
        `action` varchar(255),
        `variable_value` varchar(255),
        `activated` tinyint not null default 0,
        `activated_number` int default 0,
        `running_output` text,
        `running_error` text,
        `failure_type` varchar(255),
        `running_time` int,
        `exe_index` int
);
