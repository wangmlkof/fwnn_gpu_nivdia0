TARGET=fwnn_gpu

CFLAGS=-lOpenCL

SRCS=$(wildcard ./src/*.cpp)
INC= -I./header
OBJS =$(notdir $(SRCS:.cpp=.o))

vpath %.o ./build
$(TARGET):$(OBJS)  
	nvcc $(CFLAGS) -o $@ $^  

vpath %.cpp ./src
%.o:%.cpp
	nvcc $(INC) -o build/$(notdir $@) -c $<  

clean:  
	rm -rf $(TARGET) $(OBJS)  
