cmake_minimum_required(VERSION 3.2)
project(IR_DataAugmentation)

set(BOOST_ROOT /usr/include/boost/)


set(CMAKE_CXX_STANDARD 11)

#set(OpenCV_DIR /Users/vladislavtyurin/OpenCV)
find_package(Boost COMPONENTS filesystem system thread REQUIRED)
find_package(OpenCV REQUIRED)


set(SOURCE_FILES main.cpp ImageTransformer.h ImageTransformer.cpp)
add_executable(IR_DataAugmentation ${SOURCE_FILES})
include_directories(/usr/include/)
target_link_libraries(IR_DataAugmentation ${OpenCV_LIBS})

include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(IR_DataAugmentation ${Boost_LIBRARIES})
