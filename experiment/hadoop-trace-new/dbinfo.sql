drop table if exists `injection_record_hadoop`;
create table `injection_record_hadoop`(
        `id` int not null primary key auto_increment,
        `fault_id` varchar(255) unique not null,
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
        `activation_time` datetime,
        `activated_number` int default 0,
        `running_output` text,
        `running_error` text,
        `failure_type` varchar(255),
        `running_time` int,
        `last_time` int,
        `start_time` datetime,
        `end_time` datetime,
        `exe_index` int,
        `component` varchar(255),
        `sub_component` text,
        `jar_file` text,
        `error_class` text,
        `error_log` TINYINT,
        `error_time` datetime,
        `error_file` text,
        `with_bug` TINYINT DEFAULT 0,
        `process_tag` varchar(255),
        `error_component` text,
        `resource_bug_flag` TINYINT DEFAULT 0,
        `container_id` VARCHAR(511)
);
alter table `injection_record_hadoop` add column `deep_detect` tinyint;
alter table `injection_record_hadoop` add column `deep_detect2` tinyint;
alter table `injection_record_hadoop` add column `metrics_detect` tinyint;
alter table `injection_record_hadoop` add column `trace_detect` tinyint;
alter table `injection_record_hadoop` add column `deep_detect_latency` int DEFAULT 0 ;
alter table `injection_record_hadoop` add column `deep_detect_time` datetime;
alter table `injection_record_hadoop` add column `deep_class` text;
alter table `injection_record_hadoop` add column `deep_component` varchar(255);
alter table `injection_record_hadoop` add column `report_latency` int DEFAULT 0 ;
