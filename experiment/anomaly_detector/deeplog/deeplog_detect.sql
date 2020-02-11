drop table if exists `deeplog_detect`;
create table `deeplog_detect`(
        `id` int not null primary key auto_increment,
        `hidden_size` int,
        `batch_size` int,
        `num_epochs` int,
        `train_size` int,
        `num_candidates` int,
        `failure_type` varchar(255),
        `false_positive` int,
        `false_negative` int,
        `true_positive` int,
        `true_negative` int,
        `p` float,
        `r` float,
        `f1` float,
        `accuracy` float,
        `false_positivate_rate` float
);
