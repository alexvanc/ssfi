//
//  tracer.h
//
//
//  Created by 杨勇 on 21/06/2017.
//
//
#define _GNU_SOURCE
#ifndef tracer_h
#define tracer_h


#endif /* tracer_h */

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <arpa/inet.h>
#include <sys/timeb.h>
#include <pthread.h>
#include <uuid/uuid.h>
#include <curl/curl.h>
#include <string.h>
#include <sys/un.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <sys/stat.h>
#include <spawn.h>
#include <sys/syscall.h>
#include <limits.h>
#include <sys/sendfile.h>
#include <stdarg.h>

#define MAX_SPACE 1000000
#define LOG_LENGTH 2048
#define ID_LENGTH 38
#define UNKNOWN_FAMILY 99
#define S_SIZE 20

//#define DEBUG 1
#ifdef DEBUG
//#define MESSAGE 1
//#define OTHER_SOCK 1
#define DATA_INFO 1
//#define FILTER 1
#define IP_PORT 1
#define RW_ERR 1
//#define CONNECT 1
//#define SOCK 1
//#define HEARTBEAT 1
#endif

//define request type
#define R_DATABASE 24
#define R_CHECK_SEND 20
#define R_MARK_SEND 21
#define R_CHECK_RECV 22
#define R_MARK_RECV 23
#define R_OPEN_SOCK 25
#define R_MARK_SOCK 26
#define R_THREAD 27
#define R_THREAD_DEP 28
#define R_EVENT 29




//define function type
#define F_SEND 1
#define F_RECV 2
#define F_WRITE 3
#define F_READ 4
#define F_SENDMSG 5
#define F_RECVMSG 6
#define F_SENDTO 7
#define F_RECVFROM 8
#define F_WRITEV 9
#define F_READV 10
#define F_SENDMMSG 11
#define F_RECVMMSG 12
#define F_CONNECT 13
#define F_SOCKET 14
#define F_CLOSE 15
#define F_SEND64 17
#define F_SENDFILE 19
#define F_ACCEPT 20
#define F_FORK 21
#define F_VFORK 22
#define F_PTCREATE 23
#define F_PTJOIN 24




//define finish type
#define SEND_NORMALLY 1
#define SEND_ERROR 2
#define SEND_ID 3
#define SEND_FILTER 4
#define SEND_DISCONN 5
#define SEND_BROKEN 6
#define SEND_FAIL 7
#define SEND_OTHER 8


#define RECV_NORMALLY 9
#define RECV_FILTER 10
#define RECV_LEFT 11
#define RECV_FAIL 12
#define RECV_BYD 13
#define RECV_ID 14
#define RECV_ERROR 15
#define RECV_HEADERR 16
#define RECV_HEADFAIL 17


//#define RECV_ERROR 7
//#define RECV_NORMAL_ST 8
//#define RECV_NORMAL_ERR 9
//#define RECV_NORMAL_ST_ERR 10
//#define RECV_ID_NORMAL 11
//#define RECV_ID_ERR 12
//#define RECV_NORMAL1 13
//#define RECV_NORMAL_ID 14

#define DONE_IP6 15
#define DONE_UNIX 16
#define DONE_OTHER 17




//define buffer operation type
#define S_PUT 1
#define S_GET 2
#define S_RELEASE 3


typedef void*(*START)(void *);

//struct mmsghdr {
//    struct msghdr msg_hdr;  /* Message header */
//    unsigned int  msg_len;  /* Number of received bytes for header */
//};

struct string {
    char *ptr;
    size_t len;
};
struct thread_param{
    char * uuid;
    void * args;
    void *(*start_routine)(void *);
    long int ppid;
    long int pktid;
    unsigned long  int ptid;
    long long int ttime;
};

struct storage{
    int sockfd;
    short used;
    size_t left;
};

typedef ssize_t(*RECV)(int sockfd, void *buf, size_t len, int flags);
typedef ssize_t(*SEND)(int sockfd, const void *buf, size_t len, int flags);
typedef ssize_t(*WRITE)(int fd, const void *buf, size_t count);
typedef ssize_t(*READ)(int fd, void *buf, size_t count);
typedef ssize_t(*SENDMSG)(int sockfd, const struct msghdr *msg, int flags);
typedef ssize_t(*RECVMSG)(int sockfd, struct msghdr * msg, int flags);
typedef ssize_t(*SENDTO)(int socket, const void* buf, size_t buflen, int flags, const struct sockaddr* addr, socklen_t addrlen);
typedef ssize_t(*RECVFROM)(int socket, void* buf, size_t buflen, int flags, struct sockaddr* addr, socklen_t* addrlen);
typedef int(*SENDMMSG)(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags);
typedef int(*RECVMMSG)(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags, const struct timespec *timeout);
typedef ssize_t(*SENDFILE64)(int out_fd, int in_fd, off64_t *offset, size_t count);
typedef ssize_t(*SENDFILE)(int out_fd, int in_fd, off_t *offset, size_t count);;
typedef ssize_t(*READV)(int fd, const struct iovec *iov, int iovcnt);
typedef ssize_t(*WRITEV)(int fd, const struct iovec *iov, int iovcnt);

typedef void *(*DLSYM)(void *handle, const char *symbol);

//socket
typedef int(*SOCKET)(int domain, int type, int protocol);
//connect
typedef int(*CONN)(int socket, const struct sockaddr *addr, socklen_t length);
//accept
typedef int(*ACCEPT)(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
//close
typedef int(*CLOSE)(int fd);

typedef int(*P_CREATE)(pthread_t *thread, const pthread_attr_t *attr,void *(*start_routine) (void *), void *arg);
typedef int(*K_DELETE)(pthread_key_t key);
typedef int(*P_JOIN)(pthread_t thread, void **retval);



typedef pid_t(*FORK)(void);
typedef pid_t(*VFORK)(void);
typedef int(*CLONE)(int (*fn)(void *), void *child_stack,int flags, void *arg, ... /* pid_t *ptid, struct user_desc *tls, pid_t *ctid */ );
typedef long(*SYSCALL)(long number, ...);
typedef int(*SPAWN)(pid_t *pid, const char *path,const posix_spawn_file_actions_t *file_actions,const posix_spawnattr_t *attrp,
char *const argv[], char *const envp[]);
typedef int(*SPAWNP)(pid_t *pid, const char *file,const posix_spawn_file_actions_t *file_actions,const posix_spawnattr_t *attrp,
char *const argv[], char *const envp[]);

typedef int(*EXECL)(const char *path, const char *arg, ...);

typedef int(*EXECLP)(const char *file, const char *arg, ...);

//typedef int(*EXECLE)(const char *path, const char *arg, ..., char *const envp[]);

typedef int(*EXECV)(const char *path, char *const argv[]);

typedef int(*EXECVE)(const char *path, char *const argv[], char *const envp[]);

typedef int(*EXECVP)(const char *file, char *const argv[]);

typedef int(*EXECVPE)(const char *file, char *const argv[], char *const envp[]);

typedef int(*FPRINTF)(FILE *stream, const char *format, ...);
typedef int(*FPUTS)(const char *str, FILE *stream);
typedef size_t(*FWRITE)(const void *ptr, size_t size, size_t nmemb, FILE *stream);
//
//int execl(const char *path, const char *arg, ...);
//
//int execlp(const char *file, const char *arg, ...);
//
//int execle(const char *path, const char *arg, ..., char *const envp[]);
//
//int execv(const char *path, char *const argv[]);
//
//int execvp(const char *file, char *const argv[]);
//
//int execve(const char *path, char *const argv[], char *const envp[]);

//generate 37bytes uuid
char *random_uuid( char*,size_t);

//extract 37bytes uuid from message
//int get_uuid(char result[][ID_LENGTH],const char* buf,int len,char* after_uuid);
int get_uuid(char **result,const char* buf,ssize_t len,char* after_uuid,int sockfd);

//ssize_t check_read(char *uuids, char* buf,size_t count,int sockfd,ssize_t *read_length,char *status);
ssize_t check_read(char *uuids, char* buf,size_t count,int sockfd, ssize_t* length,char* status, int* error);

//unused, message buffer in local db
int push_to_local_database(char*,int,char*,int,pid_t,pthread_t,char*,long long,char,int);

//send the message to the central db directly
int push_to_database(char*,int,char*,int,pid_t,pthread_t,char*,long long,char,long,long,long,int,char,const char *);

//send the message to the central db directly
int push_event_to_database(int ,int ,long long ,pid_t ,pid_t ,pthread_t );

//send the message to the central db directly
int push_thread_db(long int ,long int,pthread_t ,long int,long int,pthread_t,long long );

int push_thread_db2(char *parameter);

//get the thread dependency information
int push_thread_dep(pid_t,pid_t,pthread_t,pthread_t,long long );

ssize_t get_storage(char * buf,size_t count);

//check whether a message contains a job id
int find_job(char* content, const char *message,int length);

//check wether a file descriptor is a socket descriptor
int is_socket(int);

//get local timestamp
long long gettime();

//log infomation to local file
void log_event(char*);

//log critical information to local file, usually a fatal error or a case we didn't deal with
void log_important(char*);

//log message to local file
void log_message(char*,int,const char *);

//malloc memory for multi-uuid
char ** init_uuids(ssize_t m);

void init_context();

//check a socket is a unix socket or inet socket
sa_family_t get_socket_family(int sockfd);

//for experiment
void *getresponse2(void *);

size_t op_storage(int type,int sockfd,size_t left);

//to get the http response
int get_response(char* post_parameter);
void init_string(struct string *s);
extern int errno;
size_t writefunc(void *ptr, size_t size, size_t nmemb, struct string *s);
// char *ptr, size_t size, size_t nmemb, void *userdata

ssize_t check_read_header(char *uuids,int sockfd,size_t*length,int flags);
ssize_t check_read_rest(char *buf,int sockfd,size_t length,size_t count, int flags);
ssize_t check_recvmsg_rest(struct msghdr* msg,int sockfd,size_t length,size_t buf_len,int flags);

//for logenhancement
int check_log(int fd,size_t count);

//border checking
int check_filter(char* on_ip,char* in_ip,int on_port,int in_port);

//eliminate the variable part of uuid
int format_uuid(char uuid[ID_LENGTH],char format_uuid[ID_LENGTH]);
