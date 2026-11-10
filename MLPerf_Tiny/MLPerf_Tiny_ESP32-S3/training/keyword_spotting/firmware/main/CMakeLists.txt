# Modified for EdgeAI (2026): added timer and UART component dependencies.


#
# Main component of TF Micro project 'micro_speech'.
#

idf_component_register(
    SRCS main.cc main_functions.cc constants.cc output_handler.cc model.cc
    PRIV_REQUIRES spi_flash esp_timer esp_driver_uart
    INCLUDE_DIRS "")
