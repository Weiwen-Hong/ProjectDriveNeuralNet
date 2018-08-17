#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <sys/ioctl.h>

char buffer[64];

int send(int fd, int x, int y)
{
	sprintf(buffer, "%d %d\n", x, y);

	int len = strlen(buffer);
	int n = write(fd, buffer, len);
	if(n!=len)
	{
		printf("Failed to send message\n");
		return -1;
	}
	printf("Sent: %s", buffer);
	usleep(1000);
	return 0;
}

//doesnt work.. needs to tune parameters.
int recieve(int fd)
{
	char b[1];
	int i=0;
	int timeout = 100;
	printf("try reading\n");
	int tempn=read(fd,b,1);
	printf("%d\n",tempn);
	printf("read did run\n");
	do
	{
		int n = read(fd, b, 1);//read 1 char at a time
		printf("%d",n);
		if(n == -1)
		{
			printf("Could not read\n");
			return -1;
		}
		else if(n == 0)
		{
			usleep(1000);
			timeout--;
			if(timeout == 0)
			{
				printf("Timeout reached\n");
				return -1;
			}
			continue;
		}
		buffer[i] = b[0];
		i++;
	}while(b[0]!='\n' && i < 64 && timeout > 0);
	buffer[i]=0;//null terminate

	printf("Recieved: %s",buffer);
	return 0;
}

int main(int argc, char *argv[])
{
	const char *port = "/dev/ttyUSB0";
	//open for Reading and Writing
	//port never becomes the controlling terminal for the process
	//non blocking I/O
	int fd = open(port, O_RDWR | O_NONBLOCK);
	//int fd = open(port, O_RDWR);
	if(fd == -1)
	{
		printf("Failed to open port: %s\n", port);
		return -1;
	}

	struct termios config;

	//get current serial port settings
	if(tcgetattr(fd, &config) < 0)
	{
		printf("Failed to get terminal attributes\n");
		return -1;
	}
	
	//Set baud rate both ways
	cfsetispeed(&config, B9600);
	cfsetospeed(&config, B9600);

	config.c_cflag &= ~CRTSCTS;    
	config.c_cflag |= (CLOCAL | CREAD);                   
	config.c_iflag |= (IGNPAR | IGNCR);                  
	config.c_iflag &= ~(IXON | IXOFF | IXANY);          
	config.c_oflag &= ~OPOST;

	config.c_cflag &= ~CSIZE;            
	config.c_cflag |= CS8;              
	config.c_cflag &= ~PARENB;         
	config.c_iflag &= ~INPCK;         
	config.c_iflag &= ~(ICRNL|IGNCR);
	config.c_cflag &= ~CSTOPB;      
	config.c_iflag |= INPCK;       
	//used to be 0.1
	config.c_cc[VTIME] = 0;  //  1s=10   0.1s=1 *
	//used to be 12
	config.c_cc[VMIN] = 0;

	tcsetattr(fd, TCSANOW, &config);
	//wait for the arduino to reset
	usleep(1000*1000);
	//flush anything already in the serial buffer
	tcflush(fd,TCIFLUSH);
	if(tcsetattr(fd, TCSAFLUSH, &config) < 0)
	{
		printf("Failed to initialize terminal attributes\n");
		return -1;
	}

	if(send(fd, 90, 90)!=0)
	{
		printf("Failed to send message\n");
		return -1;
	}
	usleep(2*1000*1000);//sleep for 2 seconds
        
	if(send(fd, 90, 110)!=0)
	{
		printf("Failed to send message\n");
		return -1;
	}
	usleep(2*1000*1000);//sleep for 2 seconds
	
	if(send(fd, 90, 70)!=0)
	{
		printf("Failed to send message\n");
		return -1;
	}
	usleep(2*1000*1000);//sleep for 2 seconds

	if(send(fd, 90, 90)!=0)
	{
		printf("Failed to send message\n");
		return -1;
	}
	usleep(2*1000*1000);//sleep for 2 seconds

	if(send(fd, 100, 90)!=0)
	{
		printf("Failed to send message\n");
		return -1;
	}
	usleep(2*1000*1000);//sleep for 2 seconds

	if(send(fd, 120, 90)!=0)
	{
		printf("Failed to send message\n");
		return -1;
	}
	usleep(1*1000*1000);//sleep for 1 seconds

	if(send(fd, 90, 90)!=0)
	{
		printf("Failed to send message\n");
		return -1;
	}
	usleep(2*1000*1000);//sleep for 2 seconds

	close(fd);
	return 0;
}
