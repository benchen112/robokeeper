# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.1)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(toss_motion_control C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1

add_executable(toss_motion_control toss_motion_control.cpp )

pico_set_program_name(toss_motion_control "toss_motion_control")
pico_set_program_version(toss_motion_control "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(toss_motion_control 0)
pico_enable_stdio_usb(toss_motion_control 1)

# Add the standard library to the build
target_link_libraries(toss_motion_control
        pico_stdlib)

# Add the standard include files to the build
target_include_directories(toss_motion_control PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

# Add any user requested libraries
target_link_libraries(toss_motion_control 
    pico_stdlib 
    hardware_uart
    hardware_timer
    hardware_pwm
    hardware_gpio
)
pico_add_extra_outputs(toss_motion_control)

